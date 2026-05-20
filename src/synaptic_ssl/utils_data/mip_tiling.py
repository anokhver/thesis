"""Tile pre-saved full-MIP ``.npy`` files in a folder into patches.

Input is the output of the no-tile preprocessing pipeline: per-source
``(C, H, W)`` ``.npy`` arrays plus an optional ``index.csv`` mapping each
``.npy`` back to its original raw microscopy file.

The companion ``tile_from_mip_zip`` module does the same for ``.npy``
files stored inside a ``.zip`` archive.
"""
from __future__ import annotations

import csv
import gc
import logging
from pathlib import Path

import numpy as np

from .patching import INDEX_FIELDS, extract_patches, write_index_csv

logger = logging.getLogger(__name__)


def _read_source_map(index_csv: Path) -> dict[str, str]:
    """Map ``filename`` → original ``source_path`` from a sibling ``index.csv``.

    Returns an empty dict if the file is missing. Falls back to
    ``source_image`` when ``source_path`` is empty.
    """
    if not index_csv.exists():
        return {}
    source_map: dict[str, str] = {}
    with index_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fname = row.get("filename", "")
            src = row.get("source_path", "") or row.get("source_image", "")
            if fname and src:
                source_map[fname] = src
    return source_map


def tile_one(
    npy_path: Path,
    output_dir: Path,
    patch_size: int,
    source_path: str,
    image_index: int,
) -> list[dict]:
    """Load one full-MIP ``.npy``, tile it, save patches. Return CSV records.

    A 2-D array is treated as a single-channel image. Arrays smaller than
    ``patch_size`` in either dimension are skipped with a warning.
    Synthetic ``source_path`` values (empty or ``zip://...``) fall back to
    the ``.npy`` stem for ``source_image``.
    """
    mip = np.load(npy_path)  # expected (C, H, W)

    if mip.ndim == 2:
        mip = mip[np.newaxis]
    if mip.ndim != 3:
        logger.warning(f"  Unexpected shape {mip.shape} in {npy_path.name}, skipping.")
        return []

    C, H, W = mip.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    if n_rows == 0 or n_cols == 0:
        logger.warning(
            f"  {npy_path.name}: image ({H}x{W}) smaller than "
            f"patch_size {patch_size}, skipping."
        )
        return []

    patches = extract_patches(mip, patch_size)
    del mip
    gc.collect()

    n_patches = patches.shape[0]
    stem = npy_path.stem
    logger.info(f"  {npy_path.name}: {C}x{H}x{W} -> {n_patches} patches ({n_rows}x{n_cols} grid)")

    if source_path and not source_path.startswith("zip://"):
        source_image = Path(source_path).name
    else:
        source_image = stem

    records: list[dict] = []
    for idx in range(n_patches):
        patch = patches[idx]
        row = idx // n_cols
        col = idx % n_cols
        fname = f"{stem}_r{row:02d}_c{col:02d}.npy"
        np.save(output_dir / fname, patch)
        records.append({
            "filename": fname,
            "source_image": source_image,
            "source_npy": npy_path.name,
            "source_path": source_path,
            "image_index": image_index,
            "grid_row": row,
            "grid_col": col,
            "mean_intensity": float(patch.mean()),
            "channels": C,
            "patch_size": patch_size,
        })

    del patches
    gc.collect()
    return records


def process_dir(input_dir: Path, output_dir: Path, patch_size: int) -> list[dict]:
    """Tile all full-MIP ``.npy`` files in *input_dir*; write patches + ``index.csv``.

    Records are sorted by ``(image_index, grid_row, grid_col)`` before
    being written so the on-disk order is deterministic.
    """
    npy_files = sorted(input_dir.glob("*.npy"))
    npy_files = [f for f in npy_files if f.name != "index.csv"]

    if not npy_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    source_map = _read_source_map(input_dir / "index.csv")

    all_records: list[dict] = []
    for image_index, npy_path in enumerate(npy_files):
        source_path = source_map.get(npy_path.name, str(npy_path.resolve()))
        records = tile_one(npy_path, output_dir, patch_size, source_path, image_index)
        all_records.extend(records)

    all_records.sort(key=lambda r: (r["image_index"], r["grid_row"], r["grid_col"]))

    if all_records:
        csv_path = write_index_csv(all_records, output_dir)
        logger.info(f"  index.csv -> {csv_path}  ({len(all_records)} patches)")
    else:
        logger.warning(f"  No patches produced for {input_dir.name}")

    return all_records


__all__ = ["INDEX_FIELDS", "tile_one", "process_dir"]
