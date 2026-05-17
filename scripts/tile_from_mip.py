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
      index.csv   (filename, source_npy, source_path, grid_row, grid_col,
                   mean_intensity, channels, patch_size)

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
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    records = []
    for idx in range(n_patches):
        patch = patches[idx]
        row = idx // n_cols
        col = idx % n_cols
        fname = f"{stem}_r{row:02d}_c{col:02d}.npy"
        np.save(output_dir / fname, patch)
        records.append({
            "filename": fname,
            "source_npy": npy_path.name,
            "source_path": source_path,
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

def process_dir(input_dir: Path, output_dir: Path, patch_size: int) -> list[dict]:
    """Tile all full-MIP .npy files in *input_dir*, write patches + index.csv."""
    npy_files = sorted(input_dir.glob("*.npy"))
    # Skip any stray patch files (named img0001_r00_c00.npy style)
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

    all_records: list[dict] = []
    for npy_path in npy_files:
        source_path = source_map.get(npy_path.name, str(npy_path.resolve()))
        records = tile_one(npy_path, output_dir, patch_size, source_path)
        all_records.extend(records)

    all_records.sort(key=lambda r: (r["source_npy"], r["grid_row"], r["grid_col"]))

    csv_path = output_dir / "index.csv"
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
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
        "--dry_run", action="store_true",
        help="Print what would be done without tiling.",
    )

    args = parser.parse_args()

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    if args.input_root and not args.output_root:
        parser.error("--output_root is required when using --input_root")

    # ---- single folder mode ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info(f"Input:      {args.input_dir}")
        logger.info(f"Output:     {args.output_dir}")
        logger.info(f"Patch size: {args.patch_size}")
        if args.dry_run:
            n = len(list(args.input_dir.glob("*.npy")))
            logger.info(f"[DRY RUN] Would tile {n} .npy file(s).")
            return
        process_dir(args.input_dir, args.output_dir, args.patch_size)
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
        records = process_dir(folder, args.output_root / folder.name, args.patch_size)
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
