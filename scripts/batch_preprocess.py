#!/usr/bin/env python3
r"""Batch preprocess microscopy folders from mounted disk to local output.

Usage::

    python scripts/batch_preprocess.py \\
        --input_root "Z:\Brejtr\ActiveProjects\Microscopy" \\
        --output_root "Z:\anokhver_temp\Microscopy" \\
        --max_folders 1 \\
        --max_files_per_folder 1 \\
        --patch_size 128 \\
        --workers 2

    # Or use a config file:
    python scripts/batch_preprocess.py \\
        --input_root "Z:\Brejtr\ActiveProjects\Microscopy" \\
        --output_root "Z:\anokhver_temp\Microscopy" \\
        --config configs/preprocess/patches_128.json \\
        --workers 2

Each subdirectory in input_root is processed and saved to output_root with
the same folder structure.
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_image_folders(root: Path, min_files: int = 1) -> list[Path]:
    """Find all subdirectories in root that contain microscopy files."""
    extensions = {".czi", ".tif", ".tiff", ".ets", ".vsi"}
    folders = []
    
    if not root.is_dir():
        logger.error(f"Input root does not exist: {root}")
        return []
    
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        # Check if folder contains image files
        image_files = [
            f for ext in extensions
            for f in item.glob(f"*{ext}")
        ]
        if len(image_files) >= min_files:
            folders.append(item)
            logger.info(f"  Found: {item.name} ({len(image_files)} images)")
    
    return folders


def _load_config_overrides(config_path: Path) -> dict:
    """Load config values to pass as CLI overrides."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Extract only the parameters that apply (ignore input_dir, output_dir)
    return {
        "patch_size": cfg.get("patch_size"),
        "plow": cfg.get("plow"),
        "phigh": cfg.get("phigh"),
        "file_extensions": cfg.get("file_extensions"),
        "skip_patterns": cfg.get("skip_patterns"),
    }


def process_folder(
    input_folder: Path,
    output_root: Path,
    patch_size: int = 128,
    plow: float = 1.0,
    phigh: float = 99.8,
    file_extensions: list = None,
    skip_patterns: list = None,
    workers: int = 1,
    max_files_per_folder: int = 0,
) -> bool:
    """Run preprocessing on one folder."""
    if file_extensions is None:
        file_extensions = [".czi", ".tif", ".tiff", ".ets", ".vsi"]
    if skip_patterns is None:
        skip_patterns = []
    
    # Create output folder with same name as input
    output_dir = output_root / input_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_folder.name}")
    logger.info(f"  Input:  {input_folder}")
    logger.info(f"  Output: {output_dir}")
    
    # Call preprocess_training.py
    _repo = Path(__file__).resolve().parents[1]
    script_path = _repo / "src" / "synaptic_ssl" / "utils_data" / "preprocess_training.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--input_dir", str(input_folder),
        "--output_dir", str(output_dir),
        "--patch_size", str(patch_size),
        "--plow", str(plow),
        "--phigh", str(phigh),
        "--file_extensions", *file_extensions,
        "--workers", str(workers),
    ]

    if max_files_per_folder and max_files_per_folder > 0:
        cmd.extend(["--max_files", str(max_files_per_folder)])
    
    if skip_patterns and skip_patterns != ["None"]:
        cmd.extend(["--skip_patterns", *skip_patterns])
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)

        # Validate that at least one patch record was produced.
        index_csv = output_dir / "index.csv"
        n_records = 0
        if index_csv.exists():
            with open(index_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
            n_records = max(0, len(rows) - 1)

        if n_records <= 0:
            logger.error(f"[FAIL] {input_folder.name}: completed but produced 0 records")
            return False

        logger.info(f"[OK] Completed: {input_folder.name} ({n_records} patches)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[FAIL] {input_folder.name} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess microscopy folders from mounted disk to local output."
    )
    parser.add_argument(
        "--input_root", type=Path, required=True,
        help="Root directory on mounted disk (e.g., Z:\\Brejtr\\ActiveProjects\\Microscopy).",
    )
    parser.add_argument(
        "--output_root", type=Path, required=True,
        help="Root directory for output patches (e.g., Z:\\anokhver_temp\\Microscopy).",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON config file for patch_size, percentiles, etc.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Patch size (default: 128).",
    )
    parser.add_argument(
        "--plow", type=float, default=1.0,
        help="Lower percentile (default: 1.0).",
    )
    parser.add_argument(
        "--phigh", type=float, default=99.8,
        help="Upper percentile (default: 99.8).",
    )
    parser.add_argument(
        "--file_extensions", nargs="+",
        default=[".czi", ".tif", ".tiff", ".ets", ".vsi"],
        help="File extensions to process.",
    )
    parser.add_argument(
        "--skip_patterns", nargs="+", default=[],
        help="Skip folders/files matching these patterns.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers per folder (default: 1).",
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many folders (default: 0 = all).",
    )
    parser.add_argument(
        "--max_files_per_folder", type=int, default=0,
        help="Process at most this many images per folder (default: 0 = all).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without actually running.",
    )

    args = parser.parse_args()

    # Load config overrides if provided
    config_overrides = {}
    if args.config is not None:
        config_overrides = _load_config_overrides(args.config)
        logger.info(f"Loaded config: {args.config}")
        logger.info(f"  Overrides: {config_overrides}")

    # Merge config with CLI args (CLI takes precedence)
    patch_size = args.patch_size if args.patch_size != 128 else config_overrides.get("patch_size", 128)
    plow = args.plow if args.plow != 1.0 else config_overrides.get("plow", 1.0)
    phigh = args.phigh if args.phigh != 99.8 else config_overrides.get("phigh", 99.8)
    file_extensions = args.file_extensions if args.file_extensions else config_overrides.get("file_extensions", [".czi", ".tif", ".tiff", ".ets", ".vsi"])
    skip_patterns = args.skip_patterns if args.skip_patterns else config_overrides.get("skip_patterns", [])

    logger.info(f"\n{'='*70}")
    logger.info("Batch Preprocessing Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Input root:  {args.input_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Patch size:  {patch_size}")
    logger.info(f"Percentiles: [{plow}, {phigh}]")
    logger.info(f"Extensions:  {file_extensions}")
    logger.info(f"Workers:     {args.workers}")
    logger.info(f"Max folders: {args.max_folders if args.max_folders > 0 else 'all'}")
    logger.info(f"Max files/folder: {args.max_files_per_folder if args.max_files_per_folder > 0 else 'all'}")
    
    # Find folders to process
    logger.info(f"\nScanning for image folders in {args.input_root}...")
    folders = _find_image_folders(args.input_root)
    
    if not folders:
        logger.error("No image folders found!")
        sys.exit(1)

    if args.max_folders and args.max_folders > 0:
        folders = folders[:args.max_folders]
        logger.info(f"Limiting run to first {len(folders)} folder(s) due to --max_folders")
    
    logger.info(f"Found {len(folders)} folder(s) to process.\n")
    
    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for folder in folders:
            logger.info(f"  - {folder.name}")
        return
    
    # Process each folder
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] Processing: {folder.name}")
        success = process_folder(
            input_folder=folder,
            output_root=args.output_root,
            patch_size=patch_size,
            plow=plow,
            phigh=phigh,
            file_extensions=file_extensions,
            skip_patterns=skip_patterns,
            workers=args.workers,
            max_files_per_folder=args.max_files_per_folder,
        )
        results[folder.name] = success
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    succeeded = sum(1 for v in results.values() if v)
    logger.info(f"Succeeded: {succeeded}/{len(folders)}")
    if succeeded < len(folders):
        logger.info("Failed folders:")
        for name, success in results.items():
            if not success:
                logger.info(f"  - {name}")


if __name__ == "__main__":
    main()
