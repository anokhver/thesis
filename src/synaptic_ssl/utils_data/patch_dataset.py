"""Load preprocessed ``.npy`` patches as a PyTorch Dataset."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Indexed ``.npy`` patch dataset from ``index.csv``.

    Patches are ``(C, H, W)`` float32 in ``[0, 1]``. Channel subset is taken
    when ``channels`` is set. ``transform`` runs on the numpy array before
    tensor conversion.

    Two exclusion mechanisms are supported (combine OR-wise):

    * ``exclude_patterns``: case-insensitive substring match against
      ``source_image`` / ``source_path`` / ``source_npy`` (e.g. ``KONTROLA``).
    * ``exclude_sources``: **exact-match** against the same three columns or
      against the basename of ``source_path``. Use this with the JSON
      denylist produced by ``scripts/preprocess/score_image_noise.py`` to drop entire
      noise-dominated source images.

    Two on-disk layouts are supported:

    * **Flat**: ``root/index.csv`` + ``root/<filename>.npy``. Behaves as before.
    * **Nested**: ``root/<subdir>/index.csv`` + ``root/<subdir>/<filename>.npy``.
      Triggered automatically when ``root/index.csv`` is absent. Every
      sub-``index.csv`` is loaded and each record's ``filename`` is
      rewritten to ``<subdir>/<filename>`` so ``__getitem__`` works
      unchanged. Useful for streaming the full ``patches_128_from_zip``
      dataset where each acquisition date is its own folder.
    """

    def __init__(
        self,
        root: str | Path,
        channels: list[int] | None = None,
        transform=None,
        exclude_patterns: list[str] | None = None,
        exclude_sources: list[str] | set[str] | None = None,
    ):
        self.root = Path(root)
        self.channels = channels
        self.transform = transform
        log = logging.getLogger(__name__)

        # Read index (flat or nested layout)
        csv_path = self.root / "index.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                self.records = list(reader)
        else:
            sub_indexes = sorted(
                p for p in self.root.glob("*/index.csv") if p.is_file()
            )
            if not sub_indexes:
                raise FileNotFoundError(
                    f"No index.csv found in {self.root} or any subfolder"
                )
            self.records = []
            for sub_csv in sub_indexes:
                subdir = sub_csv.parent.name
                with open(sub_csv, "r") as f:
                    for row in csv.DictReader(f):
                        # Prefix filename with subdir so np.load(root/filename) works.
                        row["filename"] = f"{subdir}/{row['filename']}"
                        self.records.append(row)
            log.info(
                f"PatchDataset: aggregated {len(self.records)} records from "
                f"{len(sub_indexes)} sub-index.csv files under {self.root}"
            )

        # Filter out records whose source path/image matches any exclude pattern.
        # Different tilers write different column names — accept any of them.
        if exclude_patterns:
            patterns_upper = [p.upper() for p in exclude_patterns]
            source_cols = ("source_image", "source_path", "source_npy")
            before = len(self.records)
            self.records = [
                r for r in self.records
                if not any(
                    p in str(r.get(col, "")).upper()
                    for col in source_cols
                    for p in patterns_upper
                )
            ]
            dropped = before - len(self.records)
            if dropped:
                log.info(
                    f"Excluded {dropped} patches matching {exclude_patterns}"
                )

        # Exact-match exclusion against an explicit denylist of source names
        # (typically produced by ``scripts/preprocess/score_image_noise.py``). We match
        # against source_npy / source_image / source_path AND the basename of
        # source_path so that callers can pass either bare ``.npy`` names or
        # full paths.
        if exclude_sources and self.records:
            deny = {str(s).strip() for s in exclude_sources if str(s).strip()}
            if deny:
                before = len(self.records)
                self.records = [
                    r for r in self.records
                    if not (
                        str(r.get("source_npy",   "")).strip() in deny
                        or str(r.get("source_image", "")).strip() in deny
                        or str(r.get("source_path",  "")).strip() in deny
                        or Path(str(r.get("source_path", ""))).name in deny
                    )
                ]
                dropped = before - len(self.records)
                if dropped:
                    log.info(
                        f"Excluded {dropped} patches matching "
                        f"{len(deny)} exclude_sources entries"
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