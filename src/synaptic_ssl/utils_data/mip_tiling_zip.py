"""Tile full-MIP ``.npy`` files stored inside a ``.zip`` archive into patches.

Same logic as :mod:`synaptic_ssl.utils_data.mip_tiling`, but input is a
single ``.zip`` archive with the layout::

    <prefix>/<date_of_acquisition>/<stem>.npy
    <prefix>/<date_of_acquisition>/index.csv   (optional)

The archive is read directly through :mod:`zipfile`; nothing is extracted
to disk. Each ``.npy`` entry is decoded via ``np.load(io.BytesIO(...))``.
"""
from __future__ import annotations

import csv
import gc
import io
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import numpy as np

from .patching import INDEX_FIELDS, extract_patches, write_index_csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Zip helpers
# ---------------------------------------------------------------------------

def list_date_folders(zf: zipfile.ZipFile) -> list[str]:
    """Return every directory inside *zf* that directly contains ``.npy`` files.

    Works regardless of how the date folders are nested. The leaf name of
    each returned path (``PurePosixPath(d).name``) is used as the output
    subfolder name.
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


def list_npy_in_dir(zf: zipfile.ZipFile, dir_prefix: str) -> list[str]:
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


def read_source_map(zf: zipfile.ZipFile, dir_prefix: str) -> dict[str, str]:
    """Load ``<dir_prefix>/index.csv`` from *zf* and map ``filename`` → ``source_path``."""
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


def load_npy_from_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Decode an ``.npy`` member into an ndarray without touching disk."""
    with zf.open(member) as f:
        data = f.read()
    return np.load(io.BytesIO(data), allow_pickle=False)


# ---------------------------------------------------------------------------
# Tile one full-MIP array
# ---------------------------------------------------------------------------

def tile_array(
    mip: np.ndarray,
    member_name: str,
    output_dir: Path,
    patch_size: int,
    source_path: str,
    image_index: int,
) -> list[dict]:
    """Tile a single in-memory MIP array and write patches to *output_dir*.

    A 2-D array is treated as a single-channel image. Arrays smaller than
    ``patch_size`` in either dimension are skipped with a warning.
    """
    if mip.ndim == 2:
        mip = mip[np.newaxis]
    if mip.ndim != 3:
        logger.warning(f"  Unexpected shape {mip.shape} in {member_name}, skipping.")
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

    if source_path and not source_path.startswith("zip://"):
        source_image = PurePosixPath(source_path.replace("\\", "/")).name
    else:
        source_image = stem
    source_npy = PurePosixPath(member_name).name

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
            "source_npy": source_npy,
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
# Multiprocessing worker — must be at module level to be picklable. Each
# worker opens its own ZipFile handle; sharing one across processes is
# unsafe.
# ---------------------------------------------------------------------------

def _tile_member_worker(
    zip_path: str,
    member: str,
    output_dir_str: str,
    patch_size: int,
    source_path: str,
    image_index: int,
) -> list[dict]:
    output_dir = Path(output_dir_str)
    with zipfile.ZipFile(zip_path, "r") as zf:
        mip = load_npy_from_zip(zf, member)
    return tile_array(
        mip, member, output_dir, patch_size, source_path, image_index,
    )


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

    When ``workers > 1`` a :class:`ProcessPoolExecutor` is used and each
    worker opens its own :class:`zipfile.ZipFile` handle on *zip_path*.
    """
    members = list_npy_in_dir(zf, dir_prefix)
    if not members:
        logger.warning(f"No .npy entries under {dir_prefix}/ in zip")
        return []

    if max_files and max_files > 0:
        members = members[:max_files]
        logger.info(f"  Limiting to first {len(members)} .npy file(s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = read_source_map(zf, dir_prefix)

    all_records: list[dict] = []
    n_workers = max(1, min(int(workers), len(members)))

    if n_workers == 1:
        for image_index, member in enumerate(members):
            member_filename = PurePosixPath(member).name
            source_path = source_map.get(member_filename, f"zip://{member}")
            try:
                mip = load_npy_from_zip(zf, member)
            except Exception as e:  # pragma: no cover
                logger.error(f"  Failed to load {member}: {e}")
                continue
            all_records.extend(
                tile_array(
                    mip, member, output_dir, patch_size,
                    source_path, image_index,
                )
            )
    else:
        logger.info(f"  Using {n_workers} parallel worker(s) for {len(members)} file(s)")
        tasks = []
        for image_index, member in enumerate(members):
            member_filename = PurePosixPath(member).name
            source_path = source_map.get(member_filename, f"zip://{member}")
            tasks.append((
                str(zip_path), member, str(output_dir), patch_size,
                source_path, image_index,
            ))
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

    all_records.sort(key=lambda r: (r["image_index"], r["grid_row"], r["grid_col"]))

    if all_records:
        csv_path = write_index_csv(all_records, output_dir)
        logger.info(f"  index.csv -> {csv_path} ({len(all_records)} patches)")
    else:
        logger.warning(f"  No patches produced for {dir_prefix}")

    return all_records


__all__ = [
    "INDEX_FIELDS",
    "list_date_folders",
    "list_npy_in_dir",
    "read_source_map",
    "load_npy_from_zip",
    "tile_array",
    "process_zip_dir",
]
