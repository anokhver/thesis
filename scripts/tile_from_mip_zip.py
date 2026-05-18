#!/usr/bin/env python3
r"""Tile full-MIP ``.npy`` files stored inside a ZIP archive into patches.

Same logic as :mod:`scripts.tile_from_mip`, but the input is a single
``.zip`` archive with the layout::

    Microscopy/
        <date_of_acquisition>/
            <stem>.npy                     # full (C, H, W) MIP arrays
            ...
            index.csv                      # optional source-image mapping

The archive is read directly through :mod:`zipfile`; nothing is extracted
to disk. Each ``.npy`` entry is decoded via ``np.load(io.BytesIO(...))``
and tiled into non-overlapping ``(N, C, ps, ps)`` patches.

Output mirrors the existing patches_128 layout::

    <output_root>/<date_of_acquisition>/
        <stem>_r00_c00.npy
        <stem>_r00_c01.npy
        ...
        index.csv   (filename, source_npy, source_path, grid_row, grid_col,
                     mean_intensity, channels, patch_size)

Usage::

    # Tile every <date> subfolder found in the zip:
    python scripts/tile_from_mip_zip.py \
        --zip        /path/to/Microscopy_no_patch.zip \
        --output_root data/patches_128_from_zip \
        --patch_size 128

    # Limit to one date folder:
    python scripts/tile_from_mip_zip.py \
        --zip        /path/to/Microscopy_no_patch.zip \
        --output_root data/patches_128_from_zip \
        --date       20251030

    # Inspect contents without writing anything:
    python scripts/tile_from_mip_zip.py \
        --zip        /path/to/Microscopy_no_patch.zip \
        --output_root /tmp/ignored \
        --dry_run
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiling  (identical to preprocess_training.py / tile_from_mip.py)
# ---------------------------------------------------------------------------

def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Tile ``(C, H, W)`` -> ``(N, C, ps, ps)``. Trims uneven edges."""
    C, H, W = image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    image = image[:, : n_rows * patch_size, : n_cols * patch_size]
    patches = image.reshape(C, n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)
    patches = patches.reshape(-1, C, patch_size, patch_size)
    return patches


# ---------------------------------------------------------------------------
# Zip helpers
# ---------------------------------------------------------------------------

def _list_date_folders(zf: zipfile.ZipFile) -> list[str]:
    """Return every directory inside *zf* that directly contains ``.npy`` files.

    Works regardless of how the date folders are nested. The leaf name of
    each returned path (``PurePosixPath(d).name``) is used as the output
    subfolder name. Examples::

        Microscopy/20251030/img.npy           -> "Microscopy/20251030"
        Microscopy_no_patch/20251030/img.npy  -> "Microscopy_no_patch/20251030"
        20251030/img.npy                      -> "20251030"
        img.npy                               -> (skipped: no parent dir)
    """
    npy_dirs: set[str] = set()
    for name in zf.namelist():
        p = PurePosixPath(name)
        if p.suffix.lower() != ".npy":
            continue
        parent = p.parent.as_posix()
        if parent and parent != ".":
            npy_dirs.add(parent)
    return sorted(npy_dirs)


def _list_npy_in_dir(zf: zipfile.ZipFile, dir_prefix: str) -> list[str]:
    """Return ``.npy`` member names directly inside ``dir_prefix/``."""
    prefix = dir_prefix.rstrip("/") + "/"
    out: list[str] = []
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        if "/" in rest or not rest:
            continue
        if rest.lower().endswith(".npy"):
            out.append(name)
    return sorted(out)


def _read_source_map(zf: zipfile.ZipFile, dir_prefix: str) -> dict[str, str]:
    """Load ``<dir_prefix>/index.csv`` and map filename -> source_path."""
    candidate = f"{dir_prefix.rstrip('/')}/index.csv"
    if candidate not in zf.namelist():
        return {}
    source_map: dict[str, str] = {}
    with zf.open(candidate) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        for row in csv.DictReader(text):
            fname = row.get("filename", "")
            src = row.get("source_path", "") or row.get("source_image", "")
            if fname and src:
                source_map[fname] = src
    return source_map


def _load_npy_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Decode an ``.npy`` member into an ndarray without touching disk."""
    with zf.open(member) as f:
        data = f.read()
    return np.load(io.BytesIO(data), allow_pickle=False)


# ---------------------------------------------------------------------------
# Tile one full-MIP array
# ---------------------------------------------------------------------------

def tile_one(
    mip: np.ndarray,
    member_name: str,
    output_dir: Path,
    patch_size: int,
    source_path: str,
) -> list[dict]:
    """Tile a single in-memory MIP array and write patches to *output_dir*."""
    if mip.ndim == 2:
        mip = mip[np.newaxis]
    if mip.ndim != 3:
        logger.warning(
            f"  Unexpected shape {mip.shape} in {member_name}, skipping."
        )
        return []

    C, H, W = mip.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    if n_rows == 0 or n_cols == 0:
        logger.warning(
            f"  {member_name}: image ({H}x{W}) smaller than "
            f"patch_size {patch_size}, skipping."
        )
        return []

    patches = extract_patches(mip, patch_size)
    del mip
    gc.collect()

    n_patches = patches.shape[0]
    stem = PurePosixPath(member_name).stem
    logger.info(
        f"  {PurePosixPath(member_name).name}: "
        f"{C}x{H}x{W} -> {n_patches} patches ({n_rows}x{n_cols} grid)"
    )

    records: list[dict] = []
    for idx in range(n_patches):
        patch = patches[idx]
        row = idx // n_cols
        col = idx % n_cols
        fname = f"{stem}_r{row:02d}_c{col:02d}.npy"
        np.save(output_dir / fname, patch)
        records.append({
            "filename": fname,
            "source_npy": PurePosixPath(member_name).name,
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
# Worker (multiprocessing) — must be at module level to be picklable
# ---------------------------------------------------------------------------

def _tile_member_worker(
    zip_path: str,
    member: str,
    output_dir_str: str,
    patch_size: int,
    source_path: str,
) -> list[dict]:
    """Open the zip in the worker, load one ``.npy`` member, tile, save."""
    output_dir = Path(output_dir_str)
    with zipfile.ZipFile(zip_path, "r") as zf:
        mip = _load_npy_from_zip(zf, member)
    return tile_one(mip, member, output_dir, patch_size, source_path)


# ---------------------------------------------------------------------------
# Process one <date> folder inside the zip
# ---------------------------------------------------------------------------

def process_zip_dir(
    zf: zipfile.ZipFile,
    zip_path: Path,
    dir_prefix: str,
    output_dir: Path,
    patch_size: int,
    max_files: int = 0,
    workers: int = 1,
) -> list[dict]:
    """Tile every ``.npy`` directly under *dir_prefix* in *zf*.

    If *workers* > 1 a :class:`ProcessPoolExecutor` is used and each worker
    opens its own :class:`zipfile.ZipFile` handle on *zip_path*.
    """
    members = _list_npy_in_dir(zf, dir_prefix)
    if not members:
        logger.warning(f"No .npy entries under {dir_prefix}/ in zip")
        return []

    if max_files and max_files > 0:
        members = members[:max_files]
        logger.info(f"  Limiting to first {len(members)} .npy file(s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = _read_source_map(zf, dir_prefix)

    all_records: list[dict] = []

    # Effective worker count: bounded by len(members) so we don't spin idle procs.
    n_workers = max(1, min(int(workers), len(members)))

    if n_workers == 1:
        for member in members:
            member_filename = PurePosixPath(member).name
            source_path = source_map.get(member_filename, f"zip://{member}")
            try:
                mip = _load_npy_from_zip(zf, member)
            except Exception as e:  # pragma: no cover
                logger.error(f"  Failed to load {member}: {e}")
                continue
            all_records.extend(
                tile_one(mip, member, output_dir, patch_size, source_path)
            )
    else:
        logger.info(f"  Using {n_workers} parallel worker(s) for {len(members)} file(s)")
        tasks = []
        for member in members:
            member_filename = PurePosixPath(member).name
            source_path = source_map.get(member_filename, f"zip://{member}")
            tasks.append(
                (str(zip_path), member, str(output_dir), patch_size, source_path)
            )
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_member = {
                ex.submit(_tile_member_worker, *t): t[1] for t in tasks
            }
            for fut in as_completed(future_to_member):
                member = future_to_member[fut]
                try:
                    all_records.extend(fut.result())
                except Exception as e:  # pragma: no cover
                    logger.error(f"  Worker failed on {member}: {e}")

    all_records.sort(
        key=lambda r: (r["source_npy"], r["grid_row"], r["grid_col"])
    )

    csv_path = output_dir / "index.csv"
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
        logger.info(
            f"  index.csv -> {csv_path} ({len(all_records)} patches)"
        )
    else:
        logger.warning(f"  No patches produced for {dir_prefix}")

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile full-MIP .npy files stored inside a .zip archive into "
            "non-overlapping patches without extracting the archive."
        )
    )
    parser.add_argument(
        "--zip", type=Path, required=True,
        help="Path to the input .zip archive.",
    )
    parser.add_argument(
        "--output_root", type=Path, required=True,
        help=(
            "Output root directory. Each <date> subfolder found in the zip "
            "is mirrored as <output_root>/<date>/."
        ),
    )
    parser.add_argument(
        "--date", default=None,
        help=(
            "Optional: process only this date folder (e.g. '20251030'). "
            "By default all date folders found in the zip are processed."
        ),
    )
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Patch side length in pixels (default: 128).",
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many date folders (0 = all).",
    )
    parser.add_argument(
        "--max_files_per_folder", type=int, default=0,
        help="Process at most this many .npy files per date folder (0 = all).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel processes used to tile .npy files within a "
            "date folder (default: 1 = sequential). Use 0 or a negative "
            "value to auto-pick os.cpu_count()."
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List what would be tiled without writing anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.zip.is_file():
        logger.error(f"Zip file does not exist: {args.zip}")
        sys.exit(1)

    with zipfile.ZipFile(args.zip, "r") as zf:
        date_dirs = _list_date_folders(zf)
        if not date_dirs:
            logger.error(
                f"No .npy entries found inside any subfolder of {args.zip}"
            )
            sys.exit(1)

        if args.date:
            # Match a user-supplied --date against the discovered prefixes.
            wanted = args.date.strip("/")
            matched = [
                d for d in date_dirs
                if PurePosixPath(d).name == wanted or d == wanted
            ]
            if not matched:
                logger.error(
                    f"--date '{args.date}' not found. Available: "
                    f"{[PurePosixPath(d).name for d in date_dirs]}"
                )
                sys.exit(1)
            date_dirs = matched

        if args.max_folders and args.max_folders > 0:
            date_dirs = date_dirs[:args.max_folders]

        # Resolve worker count: <=0 means "auto".
        if args.workers <= 0:
            workers = max(1, os.cpu_count() or 1)
        else:
            workers = args.workers

        logger.info("=" * 70)
        logger.info("Tile-from-MIP (zip) Configuration")
        logger.info("=" * 70)
        logger.info(f"Zip:         {args.zip}")
        logger.info(f"Output root: {args.output_root}")
        logger.info(f"Patch size:  {args.patch_size}")
        logger.info(f"Workers:     {workers}")
        logger.info(f"Date dirs:   {len(date_dirs)}")
        for d in date_dirs:
            n = len(_list_npy_in_dir(zf, d))
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

    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for name, n in results.items():
        status = f"{n} patches" if n > 0 else "FAILED / empty"
        logger.info(f"  {name}: {status}")
    logger.info(
        f"Total: {sum(results.values())} patches across "
        f"{len(results)} folder(s)"
    )


if __name__ == "__main__":
    main()
