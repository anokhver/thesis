"""Reassemble full MIPs from tiled ``.npy`` patches; slice arrays back."""

from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_index(patch_root):
    """Load ``index.csv`` and group records by ``image_index``."""
    patch_root = Path(patch_root)
    csv_path = patch_root / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No index.csv in {patch_root}")

    with open(csv_path, "r") as f:
        records = list(csv.DictReader(f))

    by_image = defaultdict(list)
    for r in records:
        by_image[int(r["image_index"])].append(r)

    return by_image


def load_patch_records(
    patch_root,
    exclude_patterns: Sequence[str] | None = None,
) -> list[dict]:
    """Load ``index.csv`` as a flat list, filtering damaged / excluded records.

    Unlike ``_load_index`` this returns a **flat** list (not grouped by image)
    and drops rows where ``damaged`` is truthy or ``source_image`` matches
    any pattern in *exclude_patterns* (case-insensitive substring match).
    """
    patch_root = Path(patch_root)
    csv_path = patch_root / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No index.csv in {patch_root}")

    with open(csv_path, "r") as f:
        records = list(csv.DictReader(f))

    pats = [p.upper() for p in (exclude_patterns or [])]
    return [
        r
        for r in records
        if str(r.get("damaged", "")).strip().lower() not in ("true", "1")
        and not any(p in r["source_image"].upper() for p in pats)
    ]


def reassemble_image(patch_root, image_index, exclude_patterns=None):
    """Reassemble one image from its tiled patches.

    Return ``(full_image, records)`` with ``full_image`` shaped ``(C, H, W)``
    float32. Raise ValueError if ``image_index`` is absent or its source
    matches ``exclude_patterns``.
    """
    patch_root = Path(patch_root)
    by_image = _load_index(patch_root)

    if image_index not in by_image:
        raise ValueError(f"image_index {image_index} not found in index.csv")

    records = by_image[image_index]

    if exclude_patterns:
        patterns_upper = [p.upper() for p in exclude_patterns]
        src = records[0]["source_image"].upper()
        if any(p in src for p in patterns_upper):
            raise ValueError(f"Image {image_index} matches exclude pattern")

    patch_size = int(records[0]["patch_size"])
    n_channels = int(records[0]["channels"])
    max_row = max(int(r["grid_row"]) for r in records) + 1
    max_col = max(int(r["grid_col"]) for r in records) + 1

    full_h = max_row * patch_size
    full_w = max_col * patch_size
    full_image = np.zeros((n_channels, full_h, full_w), dtype=np.float32)

    for rec in records:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        patch = np.load(patch_root / rec["filename"])  # (C, pH, pW)
        y0 = r * patch_size
        x0 = c * patch_size
        full_image[:, y0:y0 + patch_size, x0:x0 + patch_size] = patch

    return full_image, records


def slice_to_patches(full_array, records, patch_size=None):
    """Slice ``(C, H, W)`` or ``(H, W)`` back into per-patch arrays.

    Override ``patch_size`` to ignore the value stored in ``records``.
    Return a dict mapping filename to patch array.
    """
    if patch_size is None:
        patch_size = int(records[0]["patch_size"])

    result = {}
    for rec in records:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        y0 = r * patch_size
        x0 = c * patch_size

        if full_array.ndim == 3:
            patch = full_array[:, y0:y0 + patch_size, x0:x0 + patch_size]
        else:
            patch = full_array[y0:y0 + patch_size, x0:x0 + patch_size]

        result[rec["filename"]] = patch.copy()

    return result


def list_image_indices(patch_root, exclude_patterns=None):
    """Return sorted ``image_index`` values, skipping ``exclude_patterns`` matches."""
    by_image = _load_index(patch_root)
    indices = sorted(by_image.keys())

    if exclude_patterns:
        patterns_upper = [p.upper() for p in exclude_patterns]
        indices = [
            idx for idx in indices
            if not any(
                p in by_image[idx][0]["source_image"].upper()
                for p in patterns_upper
            )
        ]

    return indices


# ---------------------------------------------------------------------------
# Caching image accessor
# ---------------------------------------------------------------------------

class ImageCache:
    """Lazy-reassembly cache for full images and per-position patch slices.

    Reassembles each source image at most once and keeps it in memory.
    Call ``clear()`` to free everything.
    """

    def __init__(
        self,
        patch_root,
        patch_records: list[dict],
    ) -> None:
        self.patch_root = Path(patch_root)
        self.patch_records = patch_records
        self._cache: dict[int, np.ndarray] = {}

        self.image_to_positions: dict[int, list[int]] = defaultdict(list)
        for pos, rec in enumerate(patch_records):
            self.image_to_positions[int(rec["image_index"])].append(pos)
        self.available_image_indices: list[int] = sorted(
            self.image_to_positions
        )

    def get_full_image(self, image_index: int) -> np.ndarray:
        """Reassemble a full ``(C, H, W)`` image once and cache it."""
        image_index = int(image_index)
        if image_index not in self._cache:
            full_image, _ = reassemble_image(self.patch_root, image_index)
            self._cache[image_index] = full_image
        return self._cache[image_index]

    def get_patch(self, pos: int) -> np.ndarray:
        """Slice one ``(C, ps, ps)`` patch out of its cached full image."""
        rec = self.patch_records[pos]
        full = self.get_full_image(int(rec["image_index"]))
        ps = int(rec["patch_size"])
        y0 = int(rec["grid_row"]) * ps
        x0 = int(rec["grid_col"]) * ps
        return full[:, y0 : y0 + ps, x0 : x0 + ps]

    def clear(self) -> None:
        """Drop all cached images."""
        self._cache.clear()
