#!/usr/bin/env python3
r"""Extract per-session microscopy metadata from a folder tree.

One representative file is sampled per acquisition session (subfolder)
since microscope settings don't change within a session. See
:mod:`synaptic_ssl.utils_metadata` for the extraction and flattening
logic.

Output
------
``<output_root>/metadata.csv``
    One row per session folder with flattened key fields.
``<output_root>/metadata_full.json``
    Full raw metadata for every session (for debugging / auditing).

Usage::

    python scripts/metadata/batch_extract_metadata.py \
        --input_root  "<DISK>:\<PI_FOLDER>\ActiveProjects\Microscopy" \
        --output_root "<DISK>:\<YOUR_TEMP>\Microscopy_meta"

    python scripts/metadata/batch_extract_metadata.py \
        --input_root  "<DISK>:\<PI_FOLDER>\ActiveProjects\Microscopy" \
        --output_root "<DISK>:\<YOUR_TEMP>\Microscopy_meta" \
        --max_folders 1 --dry_run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from synaptic_ssl.utils_metadata.flatten import CSV_FIELDS_BASE
from synaptic_ssl.utils_metadata.sessions import (
    find_session_folders, pick_file_by_exts, process_session,
)
from synaptic_ssl.utils_metadata.extractors import OEX_EXTS, VSI_EXTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-session metadata from microscopy folders."
    )
    parser.add_argument("--input_root", type=Path, required=True,
                        help="Root directory containing acquisition session subfolders.")
    parser.add_argument("--output_root", type=Path, required=True,
                        help="Directory where metadata.csv and metadata_full.json are written.")
    parser.add_argument("--max_folders", type=int, default=0,
                        help="Process at most this many session folders (0 = all).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without extracting anything.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively scan nested subfolders for session folders.")

    args = parser.parse_args()

    logger.info(f"\n{'='*70}")
    logger.info("Metadata Extraction Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Input root:  {args.input_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Max folders: {args.max_folders if args.max_folders > 0 else 'all'}")
    logger.info(f"Recursive:   {args.recursive}")

    logger.info(f"\nScanning for session folders in {args.input_root} ...")
    folders = find_session_folders(args.input_root, recursive=args.recursive)

    if not folders:
        logger.error("No session folders found!")
        sys.exit(1)

    if args.max_folders and args.max_folders > 0:
        folders = folders[:args.max_folders]
        logger.info(f"Limiting to first {len(folders)} folder(s) due to --max_folders")

    logger.info(f"Found {len(folders)} session folder(s).\n")

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for folder in folders:
            oex = pick_file_by_exts(folder, OEX_EXTS)
            vsi = pick_file_by_exts(folder, VSI_EXTS)
            rel = folder.relative_to(args.input_root)
            logger.info(
                f"  - {rel}  OEX={oex.name if oex else 'none'}  "
                f"VSI={vsi.name if vsi else 'none'}"
            )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict] = []
    full_json: list[dict] = []
    succeeded = 0

    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
        result = process_session(folder)
        if result is None:
            full_json.append({
                "session_folder": folder.name,
                "session_path": str(folder),
                "_error": "no file found",
            })
            continue
        csv_rows.append(result["csv_row"])
        full_json.append({
            "session_folder": folder.name,
            "session_path": str(folder),
            "sources": result["sources"],
            "metadata": result["raw_metadata"],
        })
        succeeded += 1
        logger.info(f"  [OK] {folder.name}")

    # CSV: base fields first, then any auto-collected vendor / format
    # columns (oex_*, bf_*, tif_*).
    csv_path = args.output_root / "metadata.csv"
    if csv_rows:
        known = set(CSV_FIELDS_BASE)
        prefixes = ("oex_", "bf_", "tif_")
        extra_cols = sorted({
            k for row in csv_rows for k in row
            if k not in known and k.startswith(prefixes)
        })
        fieldnames = list(CSV_FIELDS_BASE) + extra_cols
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
    logger.info(f"\nCSV  → {csv_path}  ({len(csv_rows)} rows)")

    json_path = args.output_root / "metadata_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"JSON → {json_path}")

    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Succeeded: {succeeded}/{len(folders)}")
    if succeeded < len(folders):
        succeeded_paths = {str(r["session_path"]) for r in csv_rows}
        failed = [str(f) for f in folders if str(f) not in succeeded_paths]
        for path in failed:
            logger.info(f"  - {path}")


if __name__ == "__main__":
    main()
