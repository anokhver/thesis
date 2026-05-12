"""Load fluorescence patches with blob pseudo-labels.

Support per-patch and full-image pseudo-label modes. Optionally cache masks
to disk. Return ``(image, mask)`` as ``(C, H, W)`` and ``(1, H, W)``
float32 tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]  # root/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils_data.patch_dataset import PatchDataset  # noqa: E402


class PseudoLabelSegDataset(Dataset):
    """Generate and cache blob pseudo-labels per patch.

    Use ``precomputed_masks`` (``filename -> (H, W) uint8``) to skip per-patch
    generation. Use ``cache_dir`` to enable ``.npy`` disk caching.
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
        """Load the raw (C, H, W) numpy patch (bypassing tensor conversion)."""
        rec = self.patch_ds.records[idx]
        patch = np.load(self.patch_ds.root / rec["filename"])
        if self.patch_ds.channels is not None:
            patch = patch[self.patch_ds.channels]
        return patch.astype(np.float32)

    def _get_mask(self, idx: int, patch_np: np.ndarray) -> np.ndarray:
        """Return (H, W) uint8 pseudo-label mask, using cache if available."""
        rec = self.patch_ds.records[idx]

        # 1. precomputed from full-image mode
        if self.precomputed is not None and rec["filename"] in self.precomputed:
            return self.precomputed[rec["filename"]]

        # 2. disk cache
        cp = self._cache_path(idx)
        if cp is not None and cp.exists():
            return np.load(cp)

        # 3. generate per-patch (fallback)
        from pseudolabels.blobs import generate_blob_pseudolabel

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
