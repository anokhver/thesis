"""SplitPatchDataset.

A thin wrapper around `data_utils.patch_dataset.PatchDataset` that filters to
an explicit list of patch indices loaded from a v2 split JSON produced by
`notebooks/training_v2/00_data_and_split.ipynb`.

Why a separate module
---------------------
All v2 training notebooks need the same image-level split. Rather than
duplicating the class in every notebook, we factor it out so the split logic
is defined exactly once. The split JSON itself is the source of truth for
which patch indices belong to which fold; this class only enforces the
filter.

Usage
-----
>>> from data_utils.split_patch_dataset import SplitPatchDataset
>>> ds_train = SplitPatchDataset(
...     "data/patches_128",
...     "data/splits/v2_split.json",
...     which="train",
... )
>>> x = ds_train[0]   # (C, H, W) float32 tensor
>>> ds_train.fg_stats["tau"], ds_train.fg_stats["alpha"]
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .patch_dataset import PatchDataset


class SplitPatchDataset(Dataset):
    """Image-level-split patch dataset.

    Parameters
    ----------
    patches_dir
        Directory of .npy patches with an `index.csv` (output of
        `data_utils/preprocess_patches.py`).
    split_json
        Path to a JSON file produced by `00_data_and_split.ipynb`. Must
        contain `train_indices`, `val_indices`, `exclude_patterns`,
        and `fg_stats`.
    which
        Either ``"train"`` or ``"val"``.
    channels
        Optional list of channel indices to keep, forwarded to
        `PatchDataset`.
    transform
        Optional callable applied to each patch.
    """

    def __init__(
        self,
        patches_dir: str | Path,
        split_json: str | Path,
        which: str = "train",
        channels: list[int] | None = None,
        transform=None,
    ):
        if which not in {"train", "val"}:
            raise ValueError(f"which must be 'train' or 'val', got {which!r}")

        meta = json.loads(Path(split_json).read_text())
        self._base = PatchDataset(
            patches_dir,
            channels=channels,
            transform=transform,
            exclude_patterns=meta["exclude_patterns"],
        )
        self._idx = list(meta[f"{which}_indices"])
        self.fg_stats: dict = meta["fg_stats"]
        self.split_name: str = meta["name"]
        self.which: str = which

    def __len__(self) -> int:
        return len(self._idx)

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._base[self._idx[i]]

    @property
    def num_channels(self) -> int:
        return self._base.num_channels

    @property
    def patch_size(self) -> int:
        return self._base.patch_size
