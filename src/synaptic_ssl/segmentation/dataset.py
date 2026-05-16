"""Fluorescence-patch dataset with blob pseudo-label masks.

Supports per-patch and full-image pseudo-label modes with optional disk
cache. Returns ``(image, mask)`` as ``(C, H, W)`` and ``(1, H, W)`` float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils_data.patch_dataset import PatchDataset


class PseudoLabelSegDataset(Dataset):
    """Blob pseudo-label dataset with optional disk cache.

    Mask lookup order (first hit wins):
        1. ``precomputed_masks`` dict (in-memory, ``filename -> (H, W) mask``).
        2. ``cache_dir`` disk cache (filled lazily from per-patch fallback).
        3. Per-patch ``generate_blob_pseudolabel`` (fallback).

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
    ):
        self.patch_ds = patch_dataset
        self.pseudo_cfg = pseudo_cfg
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.precomputed = precomputed_masks

    def __len__(self) -> int:
        return len(self.patch_ds)

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        rec = self.patch_ds.records[idx]
        stem = Path(rec["filename"]).stem
        return self.cache_dir / f"{stem}_pseudo.npy"

    def _get_raw_numpy(self, idx: int) -> np.ndarray:
        """Return the raw ``(C, H, W)`` float32 patch."""
        rec = self.patch_ds.records[idx]
        patch = np.load(self.patch_ds.root / rec["filename"])
        if self.patch_ds.channels is not None:
            patch = patch[self.patch_ds.channels]
        return patch.astype(np.float32)

    @staticmethod
    def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError(
                f"pseudo-label mask for {name!r} must have shape (H, W); "
                f"got {mask.shape}"
            )
        return mask

    def _get_mask(self, idx: int, patch_np: np.ndarray) -> np.ndarray:
        """Return ``(H, W)`` pseudo-label mask, using cache if present."""
        rec = self.patch_ds.records[idx]
        name = rec["filename"]

        # 1. precomputed from full-image mode
        if self.precomputed is not None and name in self.precomputed:
            return self._validate_mask(self.precomputed[name], name)

        # 2. disk cache
        cp = self._cache_path(idx)
        if cp is not None and cp.exists():
            return self._validate_mask(np.load(cp), name)

        # 3. generate per-patch (fallback)
        from ..pseudolabels.blobs import generate_blob_pseudolabel

        mask, _intermediates, _stats = generate_blob_pseudolabel(
            patch_np, self.pseudo_cfg
        )
        if cp is not None:
            np.save(cp, mask)
        return mask

    def __getitem__(self, idx: int):
        patch_np = self._get_raw_numpy(idx)
        mask = self._get_mask(idx, patch_np)

        # (C, H, W) float32, (1, H, W) float32
        image = torch.from_numpy(patch_np).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
