#!/usr/bin/env python3
"""Scan all microscopy images and report per-image z start/end.

Unlike session-level metadata extraction, this script visits every file under
``--input_root`` and emits one CSV row per scene.

Priority for z range:
1) OME Plane PositionZ metadata (if present)
2) ``physical_pixel_sizes.Z`` with ``size_z`` (start assumed 0)

Usage:
    python scripts/scan_z_range_all_images.py \
        --input_root  "<DISK>:/.../Microscopy" \
        --output_csv  "<DISK>:/.../z_ranges.csv"
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

from aicsimageio import AICSImage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_EXTENSIONS = (".vsi", ".czi", ".ets", ".tif", ".tiff")


def find_image_files(input_root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return all files under *input_root* matching microscopy extensions."""
    files: list[Path] = []
    ext_set = {e.lower() for e in extensions}
    for path in input_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            files.append(path)
    return sorted(files)


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_ome_plane_positions_z(metadata: Any, scene_index: int) -> list[float]:
    """Extract OME Plane PositionZ values for one scene/image if available."""
    try:
        images = getattr(metadata, "images", None)
        if not images:
            return []
        if scene_index >= len(images):
            return []
        image = images[scene_index]
        pixels = getattr(image, "pixels", None)
        if pixels is None:
            return []
        planes = getattr(pixels, "planes", None) or []

        z_positions: list[float] = []
        for plane in planes:
            pos_z = getattr(plane, "position_z", None)
            if pos_z is not None:
                try:
                    z_positions.append(float(pos_z))
                except Exception:
                    continue
        return z_positions
    except Exception:
        return []


def _compute_z_bounds(img: AICSImage, scene_index: int) -> dict[str, Any]:
    """Return z bounds and provenance for the current image scene."""
    dims = img.dims
    size_z = _safe_int(getattr(dims, "Z", 1), default=1)

    pps = img.physical_pixel_sizes
    z_step_um = pps.Z if pps is not None else None
    if z_step_um is not None:
        try:
            z_step_um = float(z_step_um)
        except Exception:
            z_step_um = None

    z_positions = _extract_ome_plane_positions_z(getattr(img, "metadata", None), scene_index)
    if z_positions:
        z_start_um = min(z_positions)
        z_end_um = max(z_positions)
        method = "ome_plane_position_z"
    elif z_step_um is not None:
        z_start_um = 0.0
        z_end_um = (max(size_z, 1) - 1) * z_step_um
        method = "physical_pixel_size_z"
    else:
        z_start_um = None
        z_end_um = None
        method = "unavailable"

    return {
        "size_z": size_z,
        "z_step_um": z_step_um,
        "z_start_um": z_start_um,
        "z_end_um": z_end_um,
        "z_range_method": method,
    }


def scan_file(path: Path) -> list[dict[str, Any]]:
    """Return one output row per scene in *path*."""
    rows: list[dict[str, Any]] = []

    try:
        img = AICSImage(str(path))
    except Exception as exc:
        return [{
            "file_path": str(path),
            "file_name": path.name,
            "file_suffix": path.suffix.lower(),
            "scene_index": None,
            "scene_name": None,
            "size_z": None,
            "z_step_um": None,
            "z_start_um": None,
            "z_end_um": None,
            "z_range_method": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }]

    try:
        scenes = list(img.scenes)
        if not scenes:
            scenes = [img.current_scene]

        for i, scene in enumerate(scenes):
            try:
                img.set_scene(scene)
                bounds = _compute_z_bounds(img, i)
                rows.append({
                    "file_path": str(path),
                    "file_name": path.name,
                    "file_suffix": path.suffix.lower(),
                    "scene_index": i,
                    "scene_name": str(scene),
                    "size_z": bounds["size_z"],
                    "z_step_um": bounds["z_step_um"],
                    "z_start_um": bounds["z_start_um"],
                    "z_end_um": bounds["z_end_um"],
                    "z_range_method": bounds["z_range_method"],
                    "error": None,
                })
            except Exception as exc:
                rows.append({
                    "file_path": str(path),
                    "file_name": path.name,
                    "file_suffix": path.suffix.lower(),
                    "scene_index": i,
                    "scene_name": str(scene),
                    "size_z": None,
                    "z_step_um": None,
                    "z_start_um": None,
                    "z_end_um": None,
                    "z_range_method": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        try:
            img.close()
        except Exception:
            pass

    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "file_path",
        "file_name",
        "file_suffix",
        "scene_index",
        "scene_name",
        "size_z",
        "z_step_um",
        "z_start_um",
        "z_end_um",
        "z_range_method",
        "error",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan all microscopy files and report per-image z start/end.",
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory to scan recursively.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="File extensions to include (default: .vsi .czi .ets .tif .tiff).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Process at most this many files (0 = all).",
    )
    args = parser.parse_args()

    if not args.input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {args.input_root}")

    exts = tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions)
    files = find_image_files(args.input_root, exts)

    if args.max_files > 0:
        files = files[:args.max_files]

    logger.info(f"Found {len(files)} file(s) to scan")
    if not files:
        write_csv([], args.output_csv)
        logger.info(f"No matching files. Wrote empty CSV to: {args.output_csv}")
        return

    all_rows: list[dict[str, Any]] = []
    for idx, path in enumerate(files, start=1):
        logger.info(f"[{idx}/{len(files)}] {path}")
        rows = scan_file(path)
        all_rows.extend(rows)

    write_csv(all_rows, args.output_csv)

    n_ok = sum(1 for r in all_rows if not r.get("error"))
    n_err = sum(1 for r in all_rows if r.get("error"))
    logger.info(f"Wrote {len(all_rows)} row(s) to {args.output_csv}")
    logger.info(f"Rows ok: {n_ok}, rows with error: {n_err}")


if __name__ == "__main__":
    main()
