"""
patch_dataset.py
PyTorch Dataset for loading preprocessed .npy patches from preprocess_patches.py.

Usage:
    from patch_dataset import PatchDataset
    ds = PatchDataset("/path/to/patches")
    patch = ds[0]  # (C, 128, 128) float32 tensor
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """
    Loads pre-extracted .npy patches indexed by index.csv.

    Each patch is (C, H, W) float32 in [0, 1]. 
    No augmentation is applied
    here.

    Args:
        root: directory containing .npy files and index.csv
        channels: which channels to load (default: all). For single-channel
                  training, pass e.g. channels=[0] for presynaptic only.
        transform: optional callable applied to the numpy array before
                   converting to tensor.
    """

    def __init__(
        self,
        root: str | Path,
        channels: list[int] | None = None,
        transform=None,
        exclude_patterns: list[str] | None = None,
    ):
        self.root = Path(root)
        self.channels = channels
        self.transform = transform

        # Read index
        csv_path = self.root / "index.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No index.csv found in {self.root}")

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            self.records = list(reader)

        # Filter out records whose source_image matches any exclude pattern
        if exclude_patterns:
            patterns_upper = [p.upper() for p in exclude_patterns]
            before = len(self.records)
            self.records = [
                r for r in self.records
                if not any(p in r["source_image"].upper() for p in patterns_upper)
            ]
            dropped = before - len(self.records)
            if dropped:
                import logging
                logging.getLogger(__name__).info(
                    f"Excluded {dropped} patches matching {exclude_patterns}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.records[idx]
        patch = np.load(self.root / rec["filename"])  # (C, H, W) float32

        if self.channels is not None:
            patch = patch[self.channels]

        if self.transform is not None:
            patch = self.transform(patch)

        return torch.from_numpy(patch)

    @property
    def num_channels(self) -> int:
        if self.channels is not None:
            return len(self.channels)
        return int(self.records[0]["channels"])

    @property
    def patch_size(self) -> int:
        return int(self.records[0]["patch_size"])