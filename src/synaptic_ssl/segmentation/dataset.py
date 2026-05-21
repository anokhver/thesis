"""Fluorescence-patch dataset with blob pseudo-label masks.

Supports per-patch and full-image pseudo-label modes with optional disk
cache. Also supports a ``precomputed_images`` override so that the
iterative DDeep3M+ pipeline can feed the fused image F(x) from
iteration ``t-1`` as the training input for iteration ``t`` (paper
§3.4). Returns ``(image, mask)`` as ``(C, H, W)`` and ``(1, H, W)`` float32,
or ``(image, mask, loss_mask)`` when a loss-mask source is configured
(see ``precomputed_loss_masks`` or ``soma_dir`` + ``dend_dir``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils_data.patch_dataset import PatchDataset


class PseudoLabelSegDataset(Dataset):
    """Blob pseudo-label dataset with optional disk cache.

    Image lookup order (first hit wins):
        1. ``precomputed_images`` dict (in-memory ``filename -> (C, H, W) array``).
        2. ``np.load(patch_ds.root / filename)`` from disk.

    Mask lookup order (first hit wins):
        1. ``precomputed_masks`` dict (in-memory, ``filename -> (H, W) mask``).
        2. ``cache_dir`` disk cache (filled lazily from per-patch fallback).
        3. Per-patch ``generate_puncta_pseudolabel`` (fallback).

    Loss-mask lookup (Option A: ignore region for puncta training).
    When a source is provided, ``__getitem__`` returns the 3-tuple
    ``(image, mask, loss_mask)``; otherwise the legacy 2-tuple is kept.
    Order (first hit wins):
        1. ``precomputed_loss_masks`` dict (``filename -> (H, W)`` mask).
        2. Derived from existing per-patch soma + dend masks on disk:
           ``loss_mask = dilation(soma | dend, loss_mask_dilate_px)``
           where the tiles live at ``soma_dir / <stem>{soma_suffix}``
           and ``dend_dir / <stem>{dend_suffix}`` (defaults match the
           output of ``scripts/pseudolabels/{soma,dendrite}_from_mip.py``).
    Positive pixels in ``mask`` are always supervised: the constructed
    loss-mask is OR-ed with ``mask > 0`` before being returned. This
    guards against rendered-disk pixels that spill just past the
    dilation boundary.

    To consume pseudo-labels saved by the ``blob_pseudolabels`` notebook,
    eager-load them into a dict and pass as ``precomputed_masks``::

        precomputed = {
            rec["filename"]: np.load(OUTPUT_ROOT / rec["filename"])
            for rec in patch_ds.records
        }
    """

    def __init__(
        self,
        patch_dataset: PatchDataset,
        pseudo_cfg,
        cache_dir: str | Path | None = None,
        transform=None,
        precomputed_masks: dict[str, np.ndarray] | None = None,
        precomputed_images: dict[str, np.ndarray] | None = None,
        precomputed_loss_masks: dict[str, np.ndarray] | None = None,
        soma_dir: str | Path | None = None,
        dend_dir: str | Path | None = None,
        soma_suffix: str = "_soma.npy",
        dend_suffix: str = "_dend.npy",
        loss_mask_dilate_px: int = 4,
    ):
        self.patch_ds = patch_dataset
        self.pseudo_cfg = pseudo_cfg
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed = precomputed_masks
        self.precomputed_images = precomputed_images
        self.precomputed_loss_masks = precomputed_loss_masks
        self.soma_dir = Path(soma_dir) if soma_dir is not None else None
        self.dend_dir = Path(dend_dir) if dend_dir is not None else None
        self.soma_suffix = soma_suffix
        self.dend_suffix = dend_suffix
        self.loss_mask_dilate_px = int(loss_mask_dilate_px)
        derive_from_disk = self.soma_dir is not None and self.dend_dir is not None
        if (self.soma_dir is None) != (self.dend_dir is None):
            raise ValueError(
                "soma_dir and dend_dir must both be set or both be None"
            )
        self._use_loss_mask = (
            self.precomputed_loss_masks is not None or derive_from_disk
        )
        self._derive_loss_mask_from_disk = derive_from_disk
        if derive_from_disk and self.loss_mask_dilate_px > 0:
            from skimage.morphology import disk as _morph_disk

            self._loss_mask_struct = _morph_disk(self.loss_mask_dilate_px)
        else:
            self._loss_mask_struct = None

    def __len__(self) -> int:
        return len(self.patch_ds)

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        rec = self.patch_ds.records[idx]
        stem = Path(rec["filename"]).stem
        return self.cache_dir / f"{stem}_pseudo.npy"

    def _get_raw_numpy(self, idx: int) -> np.ndarray:
        """Return the ``(C, H, W)`` float32 patch.

        ``precomputed_images`` (if provided) takes precedence over the
        on-disk ``.npy``. This is how the iterative pipeline feeds the
        fused image F(x) of iteration ``t-1`` as the training input
        for iteration ``t``. The precomputed array must already be in
        the same channel layout the rest of the pipeline expects (i.e.
        the channels selected by ``patch_ds.channels`` if any).
        """
        rec = self.patch_ds.records[idx]
        name = rec["filename"]
        if self.precomputed_images is not None and name in self.precomputed_images:
            patch = np.asarray(self.precomputed_images[name])
            if patch.ndim != 3:
                raise ValueError(
                    f"precomputed_images[{name!r}] must be (C, H, W); "
                    f"got {patch.shape}"
                )
            return patch.astype(np.float32, copy=False)
        patch = np.load(self.patch_ds.root / name)
        if self.patch_ds.channels is not None:
            patch = patch[self.patch_ds.channels]
        return patch.astype(np.float32)

    @staticmethod
    def _validate_mask(
        mask: np.ndarray,
        name: str,
        expected_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError(
                f"pseudo-label mask for {name!r} must have shape (H, W); "
                f"got {mask.shape}"
            )
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"pseudo-label mask for {name!r} has shape {mask.shape}, "
                f"expected {expected_shape}"
            )
        return mask

    def _get_mask(self, idx: int, patch_np: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` pseudo-label mask, using cache if present."""
        rec = self.patch_ds.records[idx]
        name = rec["filename"]
        expected_shape = patch_np.shape[1:]

        # 1. precomputed from full-image mode
        if self.precomputed is not None and name in self.precomputed:
            return self._validate_mask(self.precomputed[name], name, expected_shape)

        # 2. disk cache
        cp = self._cache_path(idx)
        if cp is not None and cp.exists():
            return self._validate_mask(np.load(cp), name, expected_shape)

        # 3. generate per-patch (fallback)
        from ..pseudolabels.puncta import generate_puncta_pseudolabel

        mask, _intermediates, _stats = generate_puncta_pseudolabel(
            patch_np, self.pseudo_cfg
        )
        if cp is not None:
            np.save(cp, mask)
        return self._validate_mask(mask, name, expected_shape)

    def _get_loss_mask(
        self, idx: int, expected_shape: tuple[int, int]
    ) -> np.ndarray | None:
        """Return ``(H, W)`` loss mask or ``None`` when no source is configured.

        Priority: ``precomputed_loss_masks`` > derive from
        ``soma_dir`` + ``dend_dir`` on disk.
        """
        if not self._use_loss_mask:
            return None
        rec = self.patch_ds.records[idx]
        name = rec["filename"]

        if (
            self.precomputed_loss_masks is not None
            and name in self.precomputed_loss_masks
        ):
            return self._validate_mask(
                self.precomputed_loss_masks[name], name, expected_shape
            )

        if not self._derive_loss_mask_from_disk:
            raise FileNotFoundError(
                f"loss mask for {name!r} not found in precomputed_loss_masks "
                f"and no soma_dir/dend_dir configured"
            )

        stem = Path(name).stem
        soma_p = self.soma_dir / f"{stem}{self.soma_suffix}"
        dend_p = self.dend_dir / f"{stem}{self.dend_suffix}"
        if not soma_p.exists() or not dend_p.exists():
            raise FileNotFoundError(
                f"loss mask source missing for {name!r}: "
                f"soma={soma_p} (exists={soma_p.exists()}) "
                f"dend={dend_p} (exists={dend_p.exists()})"
            )
        soma = self._validate_mask(np.load(soma_p), name, expected_shape).astype(bool)
        dend = self._validate_mask(np.load(dend_p), name, expected_shape).astype(bool)
        struct = soma | dend
        if self._loss_mask_struct is not None:
            from scipy.ndimage import binary_dilation as _bd

            struct = _bd(struct, structure=self._loss_mask_struct)
        return struct.astype(np.uint8)

    def __getitem__(self, idx: int):
        patch_np = self._get_raw_numpy(idx)
        mask_np = self._get_mask(idx, patch_np)
        loss_mask_np = self._get_loss_mask(idx, mask_np.shape)

        # (C, H, W) float32, (1, H, W) float32
        image = torch.from_numpy(patch_np).float()
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)

        loss_mask = None
        if loss_mask_np is not None:
            loss_mask = torch.from_numpy(
                (loss_mask_np > 0) | (mask_np > 0)
            ).float().unsqueeze(0)

        if self.transform is not None:
            if loss_mask is None:
                image, mask = self.transform(image, mask)
            else:
                image, mask, loss_mask = self.transform(image, mask, loss_mask)

        if loss_mask is None:
            return image, mask
        return image, mask, loss_mask


class JointChannelSegDataset(Dataset):
    """Two-channel (PRE, POST) binary-mask pseudo-label dataset.

    Returns ``(image (C, H, W) float32, mask (2, H, W) float32)`` where
    ``mask[0]`` is the PRE puncta mask and ``mask[1]`` is the POST puncta
    mask. Both come from the Spotiflow MIP-mode pseudolabel pipeline
    (``scripts/pseudolabels/puncta_spotiflow_from_mip.py``).

    Two on-disk mask layouts are supported (per-patch is checked first):

    1. **Per-patch** (plan ``§3`` format)::

           <per_patch_mask_dir>/<patch_stem>_pre.npy   (H, W) uint8
           <per_patch_mask_dir>/<patch_stem>_post.npy  (H, W) uint8

       Activated by passing ``per_patch_mask_dir``.

    2. **Full-image MIP** (what currently exists on disk)::

           <full_image_mask_dir>/<session>/<source_stem>_pre.npy   (fH, fW) uint8
           <full_image_mask_dir>/<session>/<source_stem>_post.npy  (fH, fW) uint8

       Activated by passing ``full_image_mask_dir``. The patch tile is
       sliced via ``grid_row``/``grid_col``/``patch_size`` from the
       ``PatchDataset`` record. Full-image arrays are opened with
       ``np.load(mmap_mode='r')`` and cached per source stem so repeated
       accesses to the same source share one mmap handle.

    When *both* are passed the per-patch dir is tried first and the
    full-image dir is the fallback (this is the auto-detect behaviour
    requested for the joint pipeline).

    The session key is derived from the *first* path component of
    ``rec["filename"]`` (i.e. the nested-layout subdir written by
    ``PatchDataset``). The source stem is ``Path(rec["source_npy"]).stem``.

    A ``precomputed_loss_masks`` dict can supervise PRE/POST only inside
    a region of interest (e.g. ``soma ∪ dendrite``); when set,
    ``__getitem__`` returns the 3-tuple ``(image, mask, loss_mask)`` with
    ``loss_mask`` broadcast to ``(2, H, W)``.
    """

    def __init__(
        self,
        patch_dataset: PatchDataset,
        *,
        per_patch_mask_dir: str | Path | None = None,
        full_image_mask_dir: str | Path | None = None,
        transform=None,
        pre_suffix: str = "_pre.npy",
        post_suffix: str = "_post.npy",
        precomputed_loss_masks: dict[str, np.ndarray] | None = None,
    ):
        if per_patch_mask_dir is None and full_image_mask_dir is None:
            raise ValueError(
                "JointChannelSegDataset requires per_patch_mask_dir or "
                "full_image_mask_dir (or both)."
            )
        self.patch_ds = patch_dataset
        self.per_patch_mask_dir = (
            Path(per_patch_mask_dir) if per_patch_mask_dir is not None else None
        )
        self.full_image_mask_dir = (
            Path(full_image_mask_dir) if full_image_mask_dir is not None else None
        )
        self.transform = transform
        self.pre_suffix = pre_suffix
        self.post_suffix = post_suffix
        self.precomputed_loss_masks = precomputed_loss_masks
        # cache of source_key -> (pre_mmap, post_mmap); built lazily
        self._fullimg_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.patch_ds)

    # ------------------------------------------------------------------ image

    def _load_image(self, rec) -> np.ndarray:
        name = rec["filename"]
        img = np.load(self.patch_ds.root / name)
        if self.patch_ds.channels is not None:
            img = img[self.patch_ds.channels]
        return img.astype(np.float32)

    # ------------------------------------------------------------------ masks

    def _load_per_patch(self, rec) -> np.ndarray | None:
        if self.per_patch_mask_dir is None:
            return None
        stem = Path(rec["filename"]).stem
        pre_p = self.per_patch_mask_dir / f"{stem}{self.pre_suffix}"
        post_p = self.per_patch_mask_dir / f"{stem}{self.post_suffix}"
        if not (pre_p.exists() and post_p.exists()):
            return None
        pre = np.load(pre_p)
        post = np.load(post_p)
        return np.stack([pre, post], axis=0).astype(np.float32)

    def _get_fullimg_mmaps(self, rec) -> tuple[np.ndarray, np.ndarray]:
        if self.full_image_mask_dir is None:
            raise FileNotFoundError(
                f"no per-patch mask for {rec['filename']!r} and no "
                f"full_image_mask_dir configured"
            )
        # session is the first path component of the nested-layout filename
        parts = Path(rec["filename"]).parts
        session = parts[0] if len(parts) > 1 else ""
        source_stem = Path(rec["source_npy"]).stem
        key = f"{session}/{source_stem}"
        cached = self._fullimg_cache.get(key)
        if cached is not None:
            return cached
        sess_dir = self.full_image_mask_dir / session if session else self.full_image_mask_dir
        pre_p = sess_dir / f"{source_stem}{self.pre_suffix}"
        post_p = sess_dir / f"{source_stem}{self.post_suffix}"
        if not pre_p.exists() or not post_p.exists():
            raise FileNotFoundError(
                f"full-image mask missing for {key!r}: "
                f"pre={pre_p} (exists={pre_p.exists()}) "
                f"post={post_p} (exists={post_p.exists()})"
            )
        pre = np.load(pre_p, mmap_mode="r")
        post = np.load(post_p, mmap_mode="r")
        self._fullimg_cache[key] = (pre, post)
        return pre, post

    def _load_full_image_tile(self, rec) -> np.ndarray:
        pre_full, post_full = self._get_fullimg_mmaps(rec)
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        ps = int(rec["patch_size"])
        y0, x0 = r * ps, c * ps
        pre = np.asarray(pre_full[y0 : y0 + ps, x0 : x0 + ps])
        post = np.asarray(post_full[y0 : y0 + ps, x0 : x0 + ps])
        if pre.shape != (ps, ps) or post.shape != (ps, ps):
            raise ValueError(
                f"sliced mask tile has wrong shape for {rec['filename']!r}: "
                f"pre={pre.shape} post={post.shape} expected=({ps},{ps})"
            )
        return np.stack([pre, post], axis=0).astype(np.float32)

    def _load_mask(self, rec) -> np.ndarray:
        # 1. per-patch dir, if any
        m = self._load_per_patch(rec)
        if m is not None:
            return m
        # 2. fall back to full-image slice
        return self._load_full_image_tile(rec)

    # ------------------------------------------------------------------ loss mask

    def _load_loss_mask(self, rec, expected_hw) -> np.ndarray | None:
        if self.precomputed_loss_masks is None:
            return None
        name = rec["filename"]
        if name not in self.precomputed_loss_masks:
            raise KeyError(f"precomputed_loss_masks missing entry for {name!r}")
        lm = np.asarray(self.precomputed_loss_masks[name])
        if lm.shape != expected_hw:
            raise ValueError(
                f"loss mask for {name!r} has shape {lm.shape}, "
                f"expected {expected_hw}"
            )
        return lm.astype(np.float32)

    # ------------------------------------------------------------------ getitem

    def __getitem__(self, idx: int):
        rec = self.patch_ds.records[idx]
        img_np = self._load_image(rec)              # (C, H, W) float32
        mask_np = self._load_mask(rec)              # (2, H, W) float32
        H, W = img_np.shape[-2:]
        if mask_np.shape != (2, H, W):
            raise ValueError(
                f"mask shape mismatch for {rec['filename']!r}: "
                f"mask={mask_np.shape} image={img_np.shape}"
            )
        loss_mask_np = self._load_loss_mask(rec, (H, W))

        image = torch.from_numpy(img_np).float()
        mask = torch.from_numpy(mask_np).float()    # (2, H, W)

        loss_mask = None
        if loss_mask_np is not None:
            # broadcast (H, W) -> (2, H, W) so SegTrainTransform applies
            # identical geometry to both channels
            base = ((loss_mask_np > 0) | (mask_np.sum(axis=0) > 0)).astype(np.float32)
            loss_mask = torch.from_numpy(np.stack([base, base], axis=0)).float()

        if self.transform is not None:
            if loss_mask is None:
                image, mask = self.transform(image, mask)
            else:
                image, mask, loss_mask = self.transform(image, mask, loss_mask)

        if loss_mask is None:
            return image, mask
        return image, mask, loss_mask
