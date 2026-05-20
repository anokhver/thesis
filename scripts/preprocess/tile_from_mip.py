#!/usr/bin/env python3
r"""Tile pre-saved full-MIP ``.npy`` files into patches.

The input is either:

* a single folder of ``.npy`` files (``--input_dir``),
* a parent folder whose subdirectories each contain ``.npy`` files
  (``--input_root``), or
* a ``.zip`` archive containing one or more ``<date>/*.npy`` subfolders
  (``--zip``); read directly via :mod:`zipfile`, no extraction to disk.

Each full ``(C, H, W)`` MIP is sliced into non-overlapping ``(N, C, ps, ps)``
patches; output mirrors the standard preprocessing layout
(``<output>/<stem>_r00_c00.npy`` plus ``index.csv``).

Library code lives in :mod:`synaptic_ssl.utils_data.mip_tiling` and
:mod:`synaptic_ssl.utils_data.mip_tiling_zip`.

Usage::

    # Single folder
    python scripts/preprocess/tile_from_mip.py \
        --input_dir  "Z:\<YOUR_TEMP>\Microscopy_no_patch\SessionName" \
        --output_dir "Z:\<YOUR_TEMP>\Microscopy\SessionName"

    # All session subfolders under a root
    python scripts/preprocess/tile_from_mip.py \
        --input_root  "Z:\<YOUR_TEMP>\Microscopy_no_patch" \
        --output_root "Z:\<YOUR_TEMP>\Microscopy"

    # Zip archive (every <date> subfolder inside)
    python scripts/preprocess/tile_from_mip.py \
        --zip         /path/to/Microscopy_no_patch.zip \
        --output_root data/patches_128_from_zip \
        --workers     4

    # Limit zip to one date folder
    python scripts/preprocess/tile_from_mip.py \
        --zip         /path/to/Microscopy_no_patch.zip \
        --output_root data/patches_128_from_zip \
        --date        20251030
"""
from __future__ import annotations

import argparse
import logging
import os
import zipfile
from pathlib import Path, PurePosixPath

from synaptic_ssl.utils_data.mip_tiling import process_dir
from synaptic_ssl.utils_data.mip_tiling_zip import (
    list_date_folders, list_npy_in_dir, process_zip_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile full-MIP .npy files into non-overlapping patches. "
            "Source can be a single folder, a parent of session folders, "
            "or a .zip archive."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_dir", type=Path,
                        help="Single folder of full-MIP .npy files.")
    source.add_argument("--input_root", type=Path,
                        help="Root containing session subfolders of full-MIP .npy files.")
    source.add_argument("--zip", type=Path,
                        help="Zip archive containing <date>/*.npy subfolders.")

    parser.add_argument("--output_dir", type=Path,
                        help="Output folder for patches (required with --input_dir).")
    parser.add_argument("--output_root", type=Path,
                        help="Output root; session/date subfolders are mirrored "
                             "(required with --input_root or --zip).")

    parser.add_argument("--patch_size", type=int, default=128,
                        help="Patch side length in pixels (default: 128).")
    parser.add_argument("--max_folders", type=int, default=0,
                        help="Process at most this many subfolders / date folders "
                             "(0 = all; --input_root and --zip only).")

    # Zip-specific
    parser.add_argument("--date", default=None,
                        help="Zip only: process a single date folder (e.g. '20251030').")
    parser.add_argument("--max_files_per_folder", type=int, default=0,
                        help="Zip only: cap .npy files per date folder (0 = all).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Zip only: parallel workers per date folder "
                             "(default: 1; 0 / negative = os.cpu_count()).")

    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be tiled without writing anything.")
    return parser.parse_args()


def _run_input_dir(args: argparse.Namespace) -> None:
    if not args.output_dir:
        raise SystemExit("--output_dir is required when using --input_dir")
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist: {args.input_dir}")
    logger.info(f"Input:      {args.input_dir}")
    logger.info(f"Output:     {args.output_dir}")
    logger.info(f"Patch size: {args.patch_size}")
    if args.dry_run:
        n = len(list(args.input_dir.glob("*.npy")))
        logger.info(f"[DRY RUN] Would tile {n} .npy file(s).")
        return
    process_dir(args.input_dir, args.output_dir, args.patch_size)


def _run_input_root(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("--output_root is required when using --input_root")
    if not args.input_root.is_dir():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    folders = sorted(
        d for d in args.input_root.iterdir()
        if d.is_dir() and any(d.glob("*.npy"))
    )
    if not folders:
        raise SystemExit(f"No subfolders with .npy files found in {args.input_root}")

    if args.max_folders and args.max_folders > 0:
        folders = folders[:args.max_folders]

    logger.info(f"\n{'='*70}")
    logger.info("Tile-from-MIP Configuration (folder mode)")
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

    _print_summary(results)


def _run_zip(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("--output_root is required when using --zip")
    if not args.zip.is_file():
        raise SystemExit(f"Zip file does not exist: {args.zip}")

    workers = max(1, os.cpu_count() or 1) if args.workers <= 0 else args.workers

    with zipfile.ZipFile(args.zip, "r") as zf:
        date_dirs = list_date_folders(zf)
        if not date_dirs:
            raise SystemExit(f"No .npy entries found inside any subfolder of {args.zip}")

        if args.date:
            wanted = args.date.strip("/")
            matched = [
                d for d in date_dirs
                if PurePosixPath(d).name == wanted or d == wanted
            ]
            if not matched:
                raise SystemExit(
                    f"--date '{args.date}' not found. Available: "
                    f"{[PurePosixPath(d).name for d in date_dirs]}"
                )
            date_dirs = matched

        if args.max_folders and args.max_folders > 0:
            date_dirs = date_dirs[:args.max_folders]

        logger.info("=" * 70)
        logger.info("Tile-from-MIP Configuration (zip mode)")
        logger.info("=" * 70)
        logger.info(f"Zip:         {args.zip}")
        logger.info(f"Output root: {args.output_root}")
        logger.info(f"Patch size:  {args.patch_size}")
        logger.info(f"Workers:     {workers}")
        logger.info(f"Date dirs:   {len(date_dirs)}")
        for d in date_dirs:
            n = len(list_npy_in_dir(zf, d))
            logger.info(f"  - {d}  ({n} .npy file(s))")

        if args.dry_run:
            logger.info("[DRY RUN] No files written.")
            return

        args.output_root.mkdir(parents=True, exist_ok=True)
        results: dict[str, int] = {}
        for i, dir_prefix in enumerate(date_dirs, start=1):
            date_name = PurePosixPath(dir_prefix).name
            logger.info(f"\n[{i}/{len(date_dirs)}] {dir_prefix}")
            records = process_zip_dir(
                zf=zf,
                zip_path=args.zip,
                dir_prefix=dir_prefix,
                output_dir=args.output_root / date_name,
                patch_size=args.patch_size,
                max_files=args.max_files_per_folder,
                workers=workers,
            )
            results[dir_prefix] = len(records)

    _print_summary(results)


def _print_summary(results: dict[str, int]) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for name, n in results.items():
        status = f"{n} patches" if n > 0 else "FAILED / empty"
        logger.info(f"  {name}: {status}")
    logger.info(
        f"Total: {sum(results.values())} patches across {len(results)} folder(s)"
    )


def main() -> None:
    args = _parse_args()

    # Warn about zip-only flags used in folder mode (ignored, not blocked).
    folder_mode = args.input_dir is not None or args.input_root is not None
    if folder_mode:
        ignored = []
        if args.date is not None:           ignored.append("--date")
        if args.max_files_per_folder > 0:   ignored.append("--max_files_per_folder")
        if args.workers != 1:               ignored.append("--workers")
        if ignored:
            logger.warning(
                f"Ignoring zip-only flag(s) in folder mode: {', '.join(ignored)}"
            )

    if args.input_dir is not None:
        _run_input_dir(args)
    elif args.input_root is not None:
        _run_input_root(args)
    else:
        _run_zip(args)


if __name__ == "__main__":
    main()
