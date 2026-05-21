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

# Columns we accept as the "which source image did this patch come from" field,
# in priority order. Different tilers write different names:
#   - preprocess_training.py    -> "source_image"
#   - tile_from_mip{,_zip}.py   -> "source_npy" + "source_path"
_SOURCE_COLS = ("source_image", "source_path", "source_npy")

# Priority order used when **grouping** patches by source image (i.e. when
# synthesising ``image_index``). Prefer the most globally-unique field first:
# ``source_path`` is the absolute path / URI and disambiguates files that share
# a basename across acquisition dates; ``source_npy`` is a unique stem within
# the tile_from_mip pipeline; ``source_image`` is only a basename and used as
# a last resort (legacy CSVs).
_UNIQUE_KEY_COLS = ("source_path", "source_npy", "source_image")


def _source_label(rec: dict) -> str:
    """Return the first non-empty source field from ``rec``, or ``''``."""
    for col in _SOURCE_COLS:
        val = rec.get(col)
        if val:
            return str(val)
    return ""


def _unique_key(rec: dict) -> str:
    """Return a stable per-source-image key for grouping patches.

    Tries ``source_path`` -> ``source_npy`` -> ``source_image`` so that two
    raw files sharing a basename across different acquisition dates are
    still treated as separate images.
    """
    for col in _UNIQUE_KEY_COLS:
        val = rec.get(col)
        if val:
            return str(val)
    return ""


def _read_index_rows(patch_root: Path) -> list[dict]:
    """Read flat or nested ``index.csv`` files under ``patch_root``.

    Mirrors :class:`PatchDataset` behaviour:

    * Flat layout: ``<patch_root>/index.csv`` + ``<patch_root>/<filename>.npy``.
    * Nested layout: ``<patch_root>/<subdir>/index.csv`` + patches inside that
      subdir. Each record's ``filename`` is rewritten to ``<subdir>/<filename>``
      so ``np.load(patch_root / filename)`` keeps working.
    """
    flat = patch_root / "index.csv"
    if flat.exists():
        with open(flat, "r") as f:
            return list(csv.DictReader(f))

    sub_indexes = sorted(
        p for p in patch_root.glob("*/index.csv") if p.is_file()
    )
    if not sub_indexes:
        raise FileNotFoundError(
            f"No index.csv in {patch_root} or any subfolder"
        )

    records: list[dict] = []
    for sub_csv in sub_indexes:
        subdir = sub_csv.parent.name
        with open(sub_csv, "r") as f:
            for row in csv.DictReader(f):
                row["filename"] = f"{subdir}/{row['filename']}"
                # Each sub-folder writes its own per-folder image_index
                # starting at 0, so they would collide across folders when
                # aggregated. Drop it here and let `_normalise_records`
                # re-enumerate globally based on the source label.
                row.pop("image_index", None)
                records.append(row)
    return records


def _normalise_records(records: list[dict]) -> list[dict]:
    """In-place add ``image_index`` and ``source_image`` aliases.

    If ``image_index`` is already present and integer-castable on every row,
    it is kept (preserves legacy flat CSVs where the column is authoritative).
    Otherwise unique ``_unique_key(rec)`` values are enumerated in first-seen
    order and assigned sequential integer ids; this disambiguates source
    files that share a basename across acquisition dates. ``source_image``
    is always set to ``_source_label(rec)`` (alias for older callers).
    """
    need_synth = False
    for r in records:
        idx_raw = r.get("image_index")
        if idx_raw is None or str(idx_raw).strip() == "":
            need_synth = True
            break

    if need_synth:
        next_id = 0
        key_to_id: dict[str, int] = {}
        for r in records:
            key = _unique_key(r)
            if key not in key_to_id:
                key_to_id[key] = next_id
                next_id += 1
            r["image_index"] = key_to_id[key]

    for r in records:
        # Make sure the alias used by downstream code always exists.
        if not r.get("source_image"):
            r["source_image"] = _source_label(r)

    return records


def _load_index(patch_root):
    """Load ``index.csv`` (flat or nested) and group records by ``image_index``.

    Records are normalised so each row has both ``image_index`` (int) and
    ``source_image`` (str), regardless of which tiler produced the CSV.
    """
    patch_root = Path(patch_root)
    records = _normalise_records(_read_index_rows(patch_root))

    by_image: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_image[int(r["image_index"])].append(r)

    return by_image


def load_patch_records(
    patch_root,
    exclude_patterns: Sequence[str] | None = None,
) -> list[dict]:
    """Load ``index.csv`` as a flat list, filtering excluded records.

    Unlike ``_load_index`` this returns a **flat** list (not grouped by image)
    and drops rows whose source label matches any pattern in
    *exclude_patterns* (case-insensitive substring match). Supports both
    flat and nested (per-subdir) ``index.csv`` layouts.
    """
    patch_root = Path(patch_root)
    records = _normalise_records(_read_index_rows(patch_root))

    pats = [p.upper() for p in (exclude_patterns or [])]
    if not pats:
        return records
    return [
        r for r in records
        if not any(p in _source_label(r).upper() for p in pats)
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
        src = _source_label(records[0]).upper()
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
                p in _source_label(by_image[idx][0]).upper()
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

    def evict(self, image_index: int) -> None:
        """Drop a single image from the cache."""
        self._cache.pop(int(image_index), None)
