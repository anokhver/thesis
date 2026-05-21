"""Scan a folder tree for acquisition sessions and extract per-session metadata.

A "session" is a leaf directory containing one or more microscopy files.
:func:`process_session` picks one ``.oex`` and one ``.vsi``/``.ets``/``.tif``
per folder (mirroring acquisition convention: every file in a session
shares the same microscope settings), extracts raw metadata via
:mod:`.extractors`, and flattens both via :mod:`.flatten` into a single
CSV row.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from .extractors import (
    ETS_EXTS, OEX_EXTS, PRIORITY_EXTENSIONS, TIF_EXTS, VSI_EXTS,
    extract_file_metadata,
)
from .flatten import flatten_ets, flatten_oex, flatten_tif, flatten_vsi

logger = logging.getLogger(__name__)


def find_session_folders(root: Path, recursive: bool = False) -> list[Path]:
    """Return sorted subdirectories of *root* containing any microscopy file.

    With ``recursive=False`` (default) only direct children are scanned.
    """
    extensions = set(PRIORITY_EXTENSIONS)
    folders: list[Path] = []

    if not root.is_dir():
        logger.error(f"Input root does not exist: {root}")
        return []

    iter_dirs: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    for item in sorted(iter_dirs):
        if not item.is_dir():
            continue
        has_file = any(True for ext in extensions for _ in item.glob(f"*{ext}"))
        if has_file:
            folders.append(item)
            rel = item.relative_to(root)
            logger.info(f"  Found session: {rel}")

    return folders


def pick_representative_file(folder: Path) -> Path | None:
    """Return the first file matching the highest-priority extension."""
    for ext in PRIORITY_EXTENSIONS:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def pick_file_by_exts(folder: Path, exts: tuple[str, ...]) -> Path | None:
    """Return first file in *folder* matching any of the given extensions."""
    for ext in exts:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def process_session(folder: Path) -> dict[str, Any] | None:
    """Extract metadata from one ``.oex`` and one image file in *folder*.

    Returns a dict with ``csv_row`` (one flat record for the metadata
    CSV), ``raw_metadata`` (nested per-file dict for the full JSON), and
    ``sources`` (list of filenames consulted), or ``None`` if no
    supported file was found.
    """
    img_extensions = {".oex", ".vsi", ".ets", ".tif", ".tiff", ".czi"}
    n_files = sum(1 for ext in img_extensions for _ in folder.glob(f"*{ext}"))

    oex_file = pick_file_by_exts(folder, OEX_EXTS)
    vsi_file = pick_file_by_exts(folder, VSI_EXTS)
    ets_file = pick_file_by_exts(folder, ETS_EXTS)
    tif_file = pick_file_by_exts(folder, TIF_EXTS)

    if not any([oex_file, vsi_file, ets_file, tif_file]):
        logger.warning(f"  [SKIP] {folder.name}: no supported file found")
        return None

    sources: list[str] = []
    raw_by_type: dict[str, Any] = {}

    if oex_file:
        logger.info(f"  OEX: {oex_file.name}")
        raw_oex = extract_file_metadata(oex_file)
        flat_oex = flatten_oex(raw_oex)
        raw_by_type["oex"] = {"file": oex_file.name, "metadata": raw_oex}
        sources.append(oex_file.name)
    else:
        flat_oex = {}

    if vsi_file:
        logger.info(f"  VSI: {vsi_file.name}")
        raw_vsi = extract_file_metadata(vsi_file)
        flat_vsi = flatten_vsi(raw_vsi)
        raw_by_type["vsi"] = {"file": vsi_file.name, "metadata": raw_vsi}
        sources.append(vsi_file.name)
    elif ets_file:
        logger.info(f"  ETS: {ets_file.name}")
        raw_ets = extract_file_metadata(ets_file)
        flat_vsi = flatten_ets(raw_ets)
        raw_by_type["ets"] = {"file": ets_file.name, "metadata": raw_ets}
        sources.append(ets_file.name)
    elif tif_file:
        logger.info(f"  TIF: {tif_file.name}")
        raw_tif = extract_file_metadata(tif_file)
        flat_vsi = flatten_tif(raw_tif)
        raw_by_type["tif"] = {"file": tif_file.name, "metadata": raw_tif}
        sources.append(tif_file.name)
    else:
        flat_vsi = {}

    import json
    row: dict[str, Any] = {
        "session_folder": folder.name,
        "session_path": str(folder),
        "sources_used": json.dumps(sources),
        "n_image_files": n_files,
    }
    # vsi fields first (image geometry), then oex fields (acquisition settings)
    row.update(flat_vsi)
    row.update(flat_oex)

    return {
        "csv_row": row,
        "raw_metadata": raw_by_type,
        "sources": sources,
    }


__all__ = [
    "find_session_folders",
    "pick_representative_file",
    "pick_file_by_exts",
    "process_session",
]
