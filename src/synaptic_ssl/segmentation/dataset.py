"""Fluorescence-patch dataset with blob pseudo-label masks.

Supports per-patch and full-image pseudo-label modes with optional disk
cache. Also supports a ``precomputed_images`` override so that the
iterative DDeep3M+ pipeline can feed the fused image F(x) from
iteration ``t-1`` as the training input for iteration ``t`` (paper
§3.4). Returns ``(image, mask)`` as ``(C, H, W)`` and ``(1, H, W)`` float32.
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
    ):
        self.patch_ds = patch_dataset
        self.pseudo_cfg = pseudo_cfg
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed = precomputed_masks
        self.precomputed_images = precomputed_images

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

    def __getitem__(self, idx: int):
        patch_np = self._get_raw_numpy(idx)
        mask = self._get_mask(idx, patch_np)

        # (C, H, W) float32, (1, H, W) float32
        image = torch.from_numpy(patch_np).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
