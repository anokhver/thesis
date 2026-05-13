"""Load preprocessed ``.npy`` patches as a PyTorch Dataset."""

from __future__ import annotations

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Load pre-extracted ``.npy`` patches indexed by ``index.csv``.

    Keep patches as ``(C, H, W)`` float32 in ``[0, 1]``. Select channels with
    ``channels`` when set. Drop damaged patches and exclude-pattern matches by default.
    Apply ``transform`` before converting to tensor.
    """

    def __init__(
        self,
        root: str | Path,
        channels: list[int] | None = None,
        transform=None,
        exclude_patterns: list[str] | None = None,
        exclude_damaged: bool = True,
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

        # Drop damaged patches flagged by data_utils.damage_detection.
        # The column is optional for backward compatibility with old
        # index.csv files that pre-date the damage check.
        if exclude_damaged and self.records and "damaged" in self.records[0]:
            before = len(self.records)
            self.records = [
                r for r in self.records
                if str(r.get("damaged", "")).strip().lower() not in ("true", "1")
            ]
            dropped = before - len(self.records)
            if dropped:
                import logging
                logging.getLogger(__name__).info(
                    f"Excluded {dropped} damaged patches"
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