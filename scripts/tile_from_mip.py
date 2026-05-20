#!/usr/bin/env python3
r"""Tile pre-saved full-MIP .npy files into patches.

Takes the output of the no-tile preprocessing pipeline
(``Microscopy_no_patch``) and slices each full ``(C, H, W)`` array into
non-overlapping ``(N, C, ps, ps)`` patches — without touching the raw
microscopy files again.

Output mirrors the standard preprocessing output:
  <output_dir>/
      <stem>_r00_c00.npy
      <stem>_r00_c01.npy
      ...
      index.csv   (filename, source_image, source_npy, source_path,
                   image_index, grid_row, grid_col, mean_intensity,
                   channels, patch_size)

Usage::

    # Single folder:
    python scripts/tile_from_mip.py \
        --input_dir  "Z:\<YOUR_TEMP>\Microscopy_no_patch\SessionName" \
        --output_dir "Z:\<YOUR_TEMP>\Microscopy\SessionName" \
        --patch_size 128

    # All session subfolders under a root:
    python scripts/tile_from_mip.py \
        --input_root  "Z:\<YOUR_TEMP>\Microscopy_no_patch" \
        --output_root "Z:\<YOUR_TEMP>\Microscopy" \
        --patch_size  128

    # Dry run to preview:
    python scripts/tile_from_mip.py \
        --input_root  "Z:\<YOUR_TEMP>\Microscopy_no_patch" \
        --output_root "Z:\<YOUR_TEMP>\Microscopy" \
        --dry_run
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Canonical `index.csv` schema shared by every tiler in this repo
# (preprocess_training.py, scripts/tile_from_mip.py, scripts/tile_from_mip_zip.py).
# Keep this in sync across all three writers.
INDEX_FIELDS: tuple[str, ...] = (
    "filename",
    "source_image",   # basename of original raw file (e.g. .vsi); '' if unknown
    "source_npy",     # basename of full-MIP .npy that was tiled
    "source_path",    # absolute path / UNC / URI of the original file; '' if unknown
    "image_index",    # stable int per source image, scoped to this index.csv
    "grid_row",
    "grid_col",
    "mean_intensity",
    "channels",
    "patch_size",
)


# ---------------------------------------------------------------------------
# Tiling  (same logic as preprocess_training.py)
# ---------------------------------------------------------------------------

def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Tile ``(C, H, W)`` → ``(N, C, ps, ps)``. Trims uneven edges."""
    C, H, W = image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    image = image[:, : n_rows * patch_size, : n_cols * patch_size]
    patches = image.reshape(C, n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)
    return patches.reshape(-1, C, patch_size, patch_size)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def tile_one(
    npy_path: Path,
    output_dir: Path,
    patch_size: int,
    source_path: str,
    image_index: int,
) -> list[dict]:
    """Load one full-MIP .npy, tile it, save patches. Return CSV records."""
    mip = np.load(npy_path)  # expected (C, H, W)

    if mip.ndim == 2:
        mip = mip[np.newaxis]  # (1, H, W)
    if mip.ndim != 3:
        logger.warning(f"  Unexpected shape {mip.shape} in {npy_path.name}, skipping.")
        return []

    C, H, W = mip.shape
    n_rows = H // patch_size
    n_cols = W // patch_size

    if n_rows == 0 or n_cols == 0:
        logger.warning(f"  {npy_path.name}: image ({H}x{W}) smaller than patch_size {patch_size}, skipping.")
        return []

    patches = extract_patches(mip, patch_size)
    del mip
    gc.collect()

    n_patches = patches.shape[0]
    stem = npy_path.stem
    logger.info(f"  {npy_path.name}: {C}×{H}×{W} → {n_patches} patches ({n_rows}×{n_cols} grid)")

    # Derive a human-friendly basename for the original raw file. Fall back
    # to the .npy stem when source_path is empty / a synthetic URI.
    if source_path and not source_path.startswith("zip://"):
        source_image = Path(source_path).name
    else:
        source_image = npy_path.stem

    records = []
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


# ---------------------------------------------------------------------------
# Single-folder processing
# ---------------------------------------------------------------------------

def _tile_one_worker(args: tuple) -> list[dict]:
    """Worker function for parallel tiling (must be top-level for pickling)."""
    npy_path, output_dir, patch_size, source_path, image_index = args
    return tile_one(
        Path(npy_path), Path(output_dir), patch_size, source_path, image_index
    )


def process_dir(
    input_dir: Path, output_dir: Path, patch_size: int, workers: int = 0
) -> list[dict]:
    """Tile all full-MIP .npy files in *input_dir*, write patches + index.csv."""
    npy_files = sorted(input_dir.glob("*.npy"))
    npy_files = [f for f in npy_files if f.name != "index.csv"]

    if not npy_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read original source_path from the sibling index.csv if present
    source_map: dict[str, str] = {}
    index_csv = input_dir / "index.csv"
    if index_csv.exists():
        with open(index_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                src = row.get("source_path", "") or row.get("source_image", "")
                if fname and src:
                    source_map[fname] = src

    # Build work items; image_index is stable per source image (sorted order).
    work_items = [
        (str(npy_path), str(output_dir), patch_size,
         source_map.get(npy_path.name, str(npy_path.resolve())), image_index)
        for image_index, npy_path in enumerate(npy_files)
    ]

    all_records: list[dict] = []

    if workers > 1 and len(npy_files) > 1:
        n_workers = min(workers, len(npy_files))
        logger.info(f"  Tiling {len(npy_files)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_tile_one_worker, item): item[0] for item in work_items}
            for future in as_completed(futures):
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception as e:
                    logger.error(f"  Error processing {futures[future]}: {e}")
    else:
        for item in work_items:
            records = _tile_one_worker(item)
            all_records.extend(records)

    all_records.sort(
        key=lambda r: (r["image_index"], r["grid_row"], r["grid_col"])
    )

    csv_path = output_dir / "index.csv"
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(INDEX_FIELDS), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(f"  index.csv → {csv_path}  ({len(all_records)} patches)")
    else:
        logger.warning(f"  No patches produced for {input_dir.name}")

    return all_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tile full-MIP .npy files into patches without re-reading raw microscopy files."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir", type=Path,
        help="Single folder of full-MIP .npy files.",
    )
    group.add_argument(
        "--input_root", type=Path,
        help="Root containing session subfolders of full-MIP .npy files.",
    )

    parser.add_argument(
        "--output_dir", type=Path,
        help="Output folder for patches (required with --input_dir).",
    )
    parser.add_argument(
        "--output_root", type=Path,
        help="Output root; session subfolders are mirrored (required with --input_root).",
    )
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Patch side length in pixels (default: 128).",
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many session folders (0 = all, --input_root only).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (0 = auto based on CPU count, 1 = sequential).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without tiling.",
    )

    args = parser.parse_args()

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    if args.input_root and not args.output_root:
        parser.error("--output_root is required when using --input_root")

    # Resolve worker count
    workers = args.workers if args.workers > 0 else max(1, os.cpu_count() - 2)

    # ---- single folder mode ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info(f"Input:      {args.input_dir}")
        logger.info(f"Output:     {args.output_dir}")
        logger.info(f"Patch size: {args.patch_size}")
        logger.info(f"Workers:    {workers}")
        if args.dry_run:
            n = len(list(args.input_dir.glob("*.npy")))
            logger.info(f"[DRY RUN] Would tile {n} .npy file(s).")
            return
        process_dir(args.input_dir, args.output_dir, args.patch_size, workers)
        return

    # ---- batch / root mode -------------------------------------------------
    if not args.input_root.is_dir():
        logger.error(f"Input root does not exist: {args.input_root}")
        sys.exit(1)

    folders = sorted(
        d for d in args.input_root.iterdir()
        if d.is_dir() and any(d.glob("*.npy"))
    )
    if not folders:
        logger.error(f"No subfolders with .npy files found in {args.input_root}")
        sys.exit(1)

    if args.max_folders and args.max_folders > 0:
        folders = folders[:args.max_folders]

    logger.info(f"\n{'='*70}")
    logger.info("Tile-from-MIP Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Input root:  {args.input_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Patch size:  {args.patch_size}")
    logger.info(f"Workers:     {workers}")
    logger.info(f"Folders:     {len(folders)}")

    if args.dry_run:
        logger.info("[DRY RUN] Would tile:")
        for d in folders:
            n = len(list(d.glob("*.npy")))
            logger.info(f"  - {d.name}  ({n} .npy file(s))")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
        records = process_dir(folder, args.output_root / folder.name, args.patch_size, workers)
        results[folder.name] = len(records)

    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    for name, n in results.items():
        status = f"{n} patches" if n > 0 else "FAILED / empty"
        logger.info(f"  {name}: {status}")
    logger.info(f"Total: {sum(results.values())} patches across {len(folders)} folder(s)")


if __name__ == "__main__":
    main()
