"""Per-image Z-bounds scan over a microscopy folder tree.

Unlike :mod:`.sessions` (one row per acquisition session), this scans
every file under a root and reports one row per scene with z start/end
in micrometres.

Priority for the z range:
  1. OME ``Plane.PositionZ`` (per-slice physical position).
  2. ``physical_pixel_sizes.Z`` × ``(size_z - 1)``, with start at 0.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_EXTENSIONS: tuple[str, ...] = (".vsi", ".czi", ".ets", ".tif", ".tiff")


def find_image_files(input_root: Path, extensions: tuple[str, ...] = DEFAULT_EXTENSIONS) -> list[Path]:
    """Return every file under *input_root* whose suffix is in *extensions*."""
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


def extract_ome_plane_positions_z(metadata: Any, scene_index: int) -> list[float]:
    """Extract OME ``Plane.PositionZ`` values for one scene, if present."""
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


def compute_z_bounds(img, scene_index: int) -> dict[str, Any]:
    """Return z bounds + provenance for the current scene of an AICSImage."""
    dims = img.dims
    size_z = _safe_int(getattr(dims, "Z", 1), default=1)

    pps = img.physical_pixel_sizes
    z_step_um = pps.Z if pps is not None else None
    if z_step_um is not None:
        try:
            z_step_um = float(z_step_um)
        except Exception:
            z_step_um = None

    z_positions = extract_ome_plane_positions_z(getattr(img, "metadata", None), scene_index)
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
    """Return one row per scene in *path*; a single error row on load failure."""
    from aicsimageio import AICSImage  # type: ignore

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
                bounds = compute_z_bounds(img, i)
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


Z_RANGE_FIELDS: tuple[str, ...] = (
    "file_path", "file_name", "file_suffix",
    "scene_index", "scene_name",
    "size_z", "z_step_um", "z_start_um", "z_end_um", "z_range_method",
    "error",
)


def write_z_range_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    """Write per-scene *rows* to *output_csv* using :data:`Z_RANGE_FIELDS`."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Z_RANGE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "DEFAULT_EXTENSIONS",
    "Z_RANGE_FIELDS",
    "find_image_files",
    "extract_ome_plane_positions_z",
    "compute_z_bounds",
    "scan_file",
    "write_z_range_csv",
]
