#!/usr/bin/env python3
"""Scan all microscopy images and report per-image z start/end.

Unlike session-level metadata extraction, this script visits every file
under ``--input_root`` and emits one CSV row per scene. See
:mod:`synaptic_ssl.utils_metadata.z_range` for the z-bounds logic.

Usage::

    python scripts/metadata/scan_z_range_all_images.py \
        --input_root  "<DISK>:/.../Microscopy" \
        --output_csv  "<DISK>:/.../z_ranges.csv"
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from synaptic_ssl.utils_metadata.z_range import (
    DEFAULT_EXTENSIONS, find_image_files, scan_file, write_z_range_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan all microscopy files and report per-image z start/end.",
    )
    parser.add_argument("--input_root", type=Path, required=True,
                        help="Root directory to scan recursively.")
    parser.add_argument("--output_csv", type=Path, required=True,
                        help="Path to output CSV file.")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS),
                        help="File extensions to include (default: .vsi .czi .ets .tif .tiff).")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Process at most this many files (0 = all).")
    args = parser.parse_args()

    if not args.input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {args.input_root}")

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions)
    files = find_image_files(args.input_root, exts)

    if args.max_files > 0:
        files = files[:args.max_files]

    logger.info(f"Found {len(files)} file(s) to scan")
    if not files:
        write_z_range_csv([], args.output_csv)
        logger.info(f"No matching files. Wrote empty CSV to: {args.output_csv}")
        return

    all_rows: list[dict] = []
    for idx, path in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] {path}")
        rows = scan_file(path)
        all_rows.extend(rows)

    write_z_range_csv(all_rows, args.output_csv)

    n_ok = sum(1 for r in all_rows if not r.get("error"))
    n_err = sum(1 for r in all_rows if r.get("error"))
    logger.info(f"Wrote {len(all_rows)} row(s) to {args.output_csv}")
    logger.info(f"Rows ok: {n_ok}, rows with error: {n_err}")


if __name__ == "__main__":
    main()
