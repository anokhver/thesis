"""Build datasets, dataloaders, and channel statistics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm.auto import tqdm

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]  # root/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils_data.patch_dataset import PatchDataset  # noqa: E402


class TransformedSubset(Dataset):
    """Apply a transform on the fly to every sample of a Subset."""

    def __init__(self, subset, transform: Callable):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, i):
        return self.transform(self.subset[i])


def split_train_val(
    dataset: Dataset,
    val_split: float,
    generator: torch.Generator,
) -> tuple[Subset, Subset]:
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    return random_split(dataset, [n_train, n_val], generator=generator)


def compute_channel_stats(
    subset: Dataset,
    in_channels: int,
    max_samples: int = 4096,
    desc: str = "channel stats",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute streaming per-channel mean/std over a ``(C, H, W)`` subset of ``[0, 1]`` tensors."""
    n = min(len(subset), max_samples)
    sums = torch.zeros(in_channels, dtype=torch.float64)
    sqsums = torch.zeros(in_channels, dtype=torch.float64)
    count = 0
    for i in tqdm(range(n), desc=desc, leave=False):
        x = subset[i].double()
        sums += x.sum(dim=(1, 2))
        sqsums += (x ** 2).sum(dim=(1, 2))
        count += x.shape[1] * x.shape[2]
    mean = (sums / count).float()
    var = (sqsums / count).float() - mean ** 2
    std = var.clamp_min(1e-12).sqrt()
    return mean, std


def build_dataloaders(
    data_root: str | Path,
    *,
    exclude_patterns: list[str] | None,
    val_split: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_transform: Callable,
    val_transform: Callable,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader, Subset, Subset, PatchDataset]:
    """Build a train/val split + augmented dataloaders.

    Returns
    -------
    train_loader, val_loader, train_subset, val_subset, raw_dataset
    """
    raw = PatchDataset(root=data_root, exclude_patterns=exclude_patterns)
    train_subset, val_subset = split_train_val(raw, val_split, generator)
    train_ds = TransformedSubset(train_subset, train_transform)
    val_ds = TransformedSubset(val_subset, val_transform)
    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, train_subset, val_subset, raw
