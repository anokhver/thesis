"""Filter ``PatchDataset`` to one side of an image-level split."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .patch_dataset import PatchDataset


class SplitPatchDataset(Dataset):
    """``PatchDataset`` filtered to the train or val image-level split.

    Reads patch indices, exclude patterns, and foreground stats from
    ``split_json``. Raise ValueError if ``which`` is not ``"train"`` or ``"val"``.
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
