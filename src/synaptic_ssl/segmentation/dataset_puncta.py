"""2-channel PRE/POST puncta dataset for SwinUNETR self-training.

Returns ``(image, target, loss_mask)`` as ``(C, H, W)``, ``(2, H, W)``,
``(1, H, W)`` float32. Channel order in ``target`` is ``[PRE, POST]``,
pinned by :data:`PRE_CHANNEL` / :data:`POST_CHANNEL`.

Per-channel masks live on disk as ``<stem>_pre.npy`` / ``<stem>_post.npy``
(written by ``scripts/pseudolabels/puncta_{from_mip,spotiflow_from_mip}.py``
and overwritten by :func:`refresh.refresh_pseudolabels` at iter-1).

The loss mask is required and derived from ``<stem>_soma.npy`` U
``<stem>_dend.npy`` dilated by ``loss_mask_dilate_px``. Positive target
pixels (PRE U POST) are always OR-ed in to guard against rendered-disk
pixels that spill just past the dilation boundary -- same convention as
:class:`dataset.PseudoLabelSegDataset`.

``precomputed_targets`` (``filename -> (2, H, W) array``) overrides the
on-disk PRE/POST masks. Iter-1 self-training uses this to feed teacher
predictions into the student without rewriting disk between epochs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils_data.patch_dataset import PatchDataset


PRE_CHANNEL = 0
POST_CHANNEL = 1


class PunctaSegDataset(Dataset):
    """2-channel (PRE, POST) puncta dataset with loss-mask gating."""

    def __init__(
        self,
        patch_dataset: PatchDataset,
        *,
        pre_dir: str | Path,
        post_dir: str | Path,
        soma_dir: str | Path,
        dend_dir: str | Path,
        pre_suffix: str = "_pre.npy",
        post_suffix: str = "_post.npy",
        soma_suffix: str = "_soma.npy",
        dend_suffix: str = "_dend.npy",
        loss_mask_dilate_px: int = 4,
        transform=None,
        precomputed_images: dict[str, np.ndarray] | None = None,
        precomputed_targets: dict[str, np.ndarray] | None = None,
    ):
        self.patch_ds = patch_dataset
        self.pre_dir = Path(pre_dir)
        self.post_dir = Path(post_dir)
        self.soma_dir = Path(soma_dir)
        self.dend_dir = Path(dend_dir)
        self.pre_suffix = pre_suffix
        self.post_suffix = post_suffix
        self.soma_suffix = soma_suffix
        self.dend_suffix = dend_suffix
        self.loss_mask_dilate_px = int(loss_mask_dilate_px)
        self.transform = transform
        self.precomputed_images = precomputed_images
        self.precomputed_targets = precomputed_targets

        if self.loss_mask_dilate_px > 0:
            from skimage.morphology import disk as _morph_disk

            self._loss_mask_struct = _morph_disk(self.loss_mask_dilate_px)
        else:
            self._loss_mask_struct = None

    def __len__(self) -> int:
        return len(self.patch_ds)

    def _get_image(self, idx: int) -> np.ndarray:
        rec = self.patch_ds.records[idx]
        name = rec["filename"]
        if self.precomputed_images is not None and name in self.precomputed_images:
            patch = np.asarray(self.precomputed_images[name])
            if patch.ndim != 3:
                raise ValueError(
                    f"precomputed_images[{name!r}] must be (C, H, W); got {patch.shape}"
                )
            return patch.astype(np.float32, copy=False)
        patch = np.load(self.patch_ds.root / name)
        if self.patch_ds.channels is not None:
            patch = patch[self.patch_ds.channels]
        return patch.astype(np.float32, copy=False)

    @staticmethod
    def _load_mask_2d(path: Path, name: str, expected_hw: tuple[int, int]) -> np.ndarray:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(
                f"mask {path} for {name!r} must be (H, W); got {arr.shape}"
            )
        if arr.shape != expected_hw:
            raise ValueError(
                f"mask {path} shape {arr.shape} != image HW {expected_hw}"
            )
        return arr

    def _get_target(self, idx: int, hw: tuple[int, int]) -> np.ndarray:
        rec = self.patch_ds.records[idx]
        name = rec["filename"]
        if (
            self.precomputed_targets is not None
            and name in self.precomputed_targets
        ):
            t = np.asarray(self.precomputed_targets[name])
            if t.shape != (2, *hw):
                raise ValueError(
                    f"precomputed_targets[{name!r}] must be (2, H, W) == {(2, *hw)}; "
                    f"got {t.shape}"
                )
            return (t > 0).astype(np.float32, copy=False)

        stem = Path(name).stem
        pre = self._load_mask_2d(self.pre_dir / f"{stem}{self.pre_suffix}", name, hw)
        post = self._load_mask_2d(
            self.post_dir / f"{stem}{self.post_suffix}", name, hw,
        )
        target = np.zeros((2, *hw), dtype=np.float32)
        target[PRE_CHANNEL] = (pre > 0).astype(np.float32)
        target[POST_CHANNEL] = (post > 0).astype(np.float32)
        return target

    def _get_loss_mask(
        self, idx: int, hw: tuple[int, int], target_np: np.ndarray,
    ) -> np.ndarray:
        rec = self.patch_ds.records[idx]
        name = rec["filename"]
        stem = Path(name).stem
        soma = self._load_mask_2d(
            self.soma_dir / f"{stem}{self.soma_suffix}", name, hw,
        ).astype(bool)
        dend = self._load_mask_2d(
            self.dend_dir / f"{stem}{self.dend_suffix}", name, hw,
        ).astype(bool)
        gate = soma | dend
        if self._loss_mask_struct is not None:
            from scipy.ndimage import binary_dilation

            gate = binary_dilation(gate, structure=self._loss_mask_struct)
        # always supervise positive target pixels (sub-pixel disk spill-over)
        pos_any = (target_np.sum(axis=0) > 0)
        return ((gate | pos_any)).astype(np.uint8)

    def __getitem__(self, idx: int):
        patch_np = self._get_image(idx)
        hw = patch_np.shape[1:]
        target_np = self._get_target(idx, hw)
        loss_mask_np = self._get_loss_mask(idx, hw, target_np)

        image = torch.from_numpy(patch_np).float()
        target = torch.from_numpy(target_np).float()  # (2, H, W)
        loss_mask = torch.from_numpy(loss_mask_np).float().unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            image, target, loss_mask = self.transform(image, target, loss_mask)

        return image, target, loss_mask


def positive_patch_fraction(
    patch_dataset: PatchDataset,
    *,
    pre_dir: str | Path,
    post_dir: str | Path,
    pre_suffix: str = "_pre.npy",
    post_suffix: str = "_post.npy",
    sample_size: int | None = None,
    seed: int = 0,
) -> dict:
    """Fraction of patches with non-empty PRE U POST mask.

    Drives the weighted-sampler decision in ``plan.md`` v2 (the open
    "positive-patch oversampling" item):

      * ``f >= 0.20``  -> no action
      * ``0.10 <= f < 0.20`` -> weighted sampler, target 50% positive per batch
      * ``f < 0.10``   -> upsample positives, optionally downsample empties

    Cheap; reads only the two on-disk masks (not the image patch).
    ``sample_size`` lets you estimate on a stratified subsample.

    Returns ``{any_fraction, pre_fraction, post_fraction, both_fraction,
    n_sampled}``.
    """
    import random

    pre_dir = Path(pre_dir)
    post_dir = Path(post_dir)
    n = len(patch_dataset)
    indices = list(range(n))
    if sample_size is not None and sample_size < n:
        rng = random.Random(seed)
        indices = rng.sample(indices, sample_size)

    n_any = n_pre = n_post = n_both = n_seen = 0
    for i in indices:
        stem = Path(patch_dataset.records[i]["filename"]).stem
        pre_p = pre_dir / f"{stem}{pre_suffix}"
        post_p = post_dir / f"{stem}{post_suffix}"
        if not pre_p.exists() or not post_p.exists():
            continue
        pre_any = bool(np.load(pre_p).any())
        post_any = bool(np.load(post_p).any())
        n_seen += 1
        n_pre += int(pre_any)
        n_post += int(post_any)
        n_any += int(pre_any or post_any)
        n_both += int(pre_any and post_any)

    d = max(1, n_seen)
    return {
        "any_fraction":  n_any / d,
        "pre_fraction":  n_pre / d,
        "post_fraction": n_post / d,
        "both_fraction": n_both / d,
        "n_sampled":     n_seen,
    }

