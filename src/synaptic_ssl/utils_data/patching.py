"""Shared patch-tiling primitives.

Canonical patch index schema and the row-major non-overlapping tiler used
by every preprocessing path in this repo (``preprocess_training``,
``tile_from_mip``, ``tile_from_mip_zip``).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


# Canonical ``index.csv`` schema. Order matters: it is the on-disk column
# order. Every writer in this package must use this tuple (directly or via
# ``write_index_csv``) so downstream readers can rely on a stable schema.
INDEX_FIELDS: tuple[str, ...] = (
    "filename",
    "source_image",   # basename of original raw file (e.g. .vsi); '' if unknown
    "source_npy",     # basename of full-MIP .npy that was tiled; '' for raw-input pipelines
    "source_path",    # absolute path / UNC / URI of the original file; '' if unknown
    "image_index",    # stable int per source image, scoped to this index.csv
    "grid_row",
    "grid_col",
    "mean_intensity",
    "channels",
    "patch_size",
)


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Tile ``(C, H, W)`` into row-major non-overlapping ``(N, C, ps, ps)``.

    Trims the bottom/right strip when ``H`` or ``W`` is not divisible by
    ``patch_size``.
    """
    C, H, W = image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size

    image = image[:, : n_rows * patch_size, : n_cols * patch_size]

    patches = image.reshape(C, n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)

    return patches


def write_index_csv(records: Iterable[Mapping[str, object]], output_dir: Path) -> Path:
    """Write ``records`` to ``<output_dir>/index.csv`` using ``INDEX_FIELDS``.

    Extra keys are dropped (``extrasaction="ignore"``). Returns the path
    written. Records are written in iteration order; sort the caller's
    list first if a stable on-disk order is required.
    """
    output_dir = Path(output_dir)
    csv_path = output_dir / "index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(INDEX_FIELDS), extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    return csv_path


__all__ = ["INDEX_FIELDS", "extract_patches", "write_index_csv"]
