#!/usr/bin/env python3
r"""Extract per-session microscopy metadata from a folder tree.

One representative file is sampled per acquisition session (subfolder).
Microscope settings do not change within a session, so a single file
carries all the information needed.

Priority order for representative file selection per folder:
  .oex  → .vsi  → .ets  → .tif / .tiff

Output
------
``<output_root>/metadata.csv``
    One row per session folder with flattened key fields.

``<output_root>/metadata_full.json``
    Full raw metadata for every session (for debugging / auditing).

Usage::

    python scripts/batch_extract_metadata.py \
        --input_root  "<DISK>:\\<PI_FOLDER>\ActiveProjects\Microscopy" \
        --output_root "<DISK>:\\<YOUR_TEMP>\Microscopy_meta"

    # Smoke-test with a single folder:
    python scripts/batch_extract_metadata.py \
        --input_root  "<DISK>:\\<PI_FOLDER>\ActiveProjects\Microscopy" \
        --output_root "<DISK>:\\<YOUR_TEMP>\Microscopy_meta" \
        --max_folders 1 --dry_run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Priority order: which file type to prefer for metadata within a session folder.
_PRIORITY = [".oex", ".vsi", ".ets", ".tif", ".tiff"]

# File types to extract per source (always try both oex and vsi)
_OEX_EXTS = [".oex"]
_VSI_EXTS = [".vsi"]
_ETS_EXTS = [".ets"]
_TIF_EXTS = [".tif", ".tiff"]


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

def _find_session_folders(root: Path) -> list[Path]:
    """Return sorted subdirectories of *root* that contain at least one
    microscopy file (.oex / .vsi / .ets / .tif / .tiff)."""
    extensions = {".oex", ".vsi", ".ets", ".tif", ".tiff"}
    folders: list[Path] = []

    if not root.is_dir():
        logger.error(f"Input root does not exist: {root}")
        return []

    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        has_file = any(
            True for ext in extensions for _ in item.glob(f"*{ext}")
        )
        if has_file:
            folders.append(item)
            logger.info(f"  Found session: {item.name}")

    return folders


def _pick_representative_file(folder: Path) -> Path | None:
    """Return the first file matching the highest-priority extension."""
    for ext in _PRIORITY:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def _pick_file_by_exts(folder: Path, exts: list[str]) -> Path | None:
    """Return first file in folder matching any of the given extensions."""
    for ext in exts:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Metadata extraction  (mirrors imaging/olympus_metadata.py)
# ---------------------------------------------------------------------------

def _xml_to_dict(elem: ET.Element) -> Any:
    node: dict[str, Any] = {}
    if elem.attrib:
        node["@attrs"] = dict(elem.attrib)
    text = (elem.text or "").strip()
    children = list(elem)
    if not children:
        if text:
            return text if not node else {**node, "#text": text}
        return node or None
    for child in children:
        tag = child.tag.split("}", 1)[-1]
        value = _xml_to_dict(child)
        if tag in node:
            if not isinstance(node[tag], list):
                node[tag] = [node[tag]]
            node[tag].append(value)
        else:
            node[tag] = value
    if text:
        node["#text"] = text
    return node


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


# --- per-format extractors --------------------------------------------------

def _extract_oex(path: Path) -> dict[str, Any]:
    import zipfile

    info: dict[str, Any] = {"_kind": "oex"}
    if zipfile.is_zipfile(path):
        info["container"] = "zip"
        members: dict[str, Any] = {}
        with zipfile.ZipFile(path) as zf:
            info["zip_members"] = zf.namelist()
            for name in zf.namelist():
                with zf.open(name) as fh:
                    raw = fh.read()
                if name.lower().endswith((".xml", ".oex")) or raw.lstrip().startswith(b"<"):
                    try:
                        members[name] = _xml_to_dict(ET.fromstring(raw))
                    except ET.ParseError as exc:
                        members[name] = {"_parse_error": str(exc)}
                else:
                    members[name] = {"_binary_size": len(raw)}
        info["members"] = members
    else:
        info["container"] = "xml"
        raw = path.read_bytes()
        try:
            info["xml"] = _xml_to_dict(ET.fromstring(raw))
        except ET.ParseError as exc:
            info["_parse_error"] = str(exc)
    return info


def _extract_vsi_via_aicsimageio(path: Path) -> dict[str, Any]:
    from aicsimageio import AICSImage  # type: ignore

    img = AICSImage(str(path))
    try:
        # Select the largest scene (avoids picking the small RGB thumbnail)
        scenes = list(img.scenes)
        if len(scenes) > 1:
            best, best_size = img.current_scene, -1
            for sc in scenes:
                try:
                    img.set_scene(sc)
                except Exception:
                    continue
                d = img.dims
                size = int(d.X) * int(d.Y) * max(1, int(d.Z))
                if size > best_size:
                    best_size, best = size, sc
            img.set_scene(best)
            logger.info(f"  VSI scene selected: {best!r} ({best_size:,} voxels, {len(scenes)} total)")

        out: dict[str, Any] = {
            "reader": type(img.reader).__name__,
            "scenes": scenes,
            "selected_scene": img.current_scene,
            "shape": dict(zip(img.dims.order, img.dims.shape)),
            "dims_order": img.dims.order,
            "dtype": str(img.dtype),
            "physical_pixel_sizes": {
                "Z": img.physical_pixel_sizes.Z,
                "Y": img.physical_pixel_sizes.Y,
                "X": img.physical_pixel_sizes.X,
            },
            "channel_names": list(img.channel_names) if img.channel_names else None,
        }
        md = getattr(img, "metadata", None)
        if isinstance(md, ET.Element):
            out["ome_xml"] = _xml_to_dict(md)
        elif md is not None:
            out["metadata_repr"] = repr(md)[:2000]
        return out
    finally:
        try:
            img.close()
        except Exception:
            pass


def _extract_vsi(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"_kind": "vsi"}
    info["aicsimageio"] = _safe(_extract_vsi_via_aicsimageio, path)
    return info


def _extract_ets(path: Path) -> dict[str, Any]:
    import struct

    info: dict[str, Any] = {"_kind": "ets"}
    with path.open("rb") as fh:
        head = fh.read(64)
    info["magic"] = head[:4].decode("ascii", "replace")
    if len(head) >= 48 and head[:4] == b"SIS0":
        try:
            (_, header_size, version, ndims,
             additional_header_offset, additional_header_size,
             _r1, used_chunk_offset, used_chunk_count, _r2, _r3
             ) = struct.unpack("<4s I I I I I I I I I I", head[:48])
            info["header"] = {
                "version": version,
                "ndims": ndims,
                "used_chunk_count": used_chunk_count,
            }
        except struct.error:
            pass
    return info


def _extract_tif(path: Path) -> dict[str, Any]:
    import tifffile  # type: ignore

    info: dict[str, Any] = {"_kind": "tif"}
    try:
        with tifffile.TiffFile(str(path)) as tf:
            p = tf.pages[0]
            info.update({
                "n_pages": len(tf.pages),
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "axes": p.axes,
                "is_ome": tf.is_ome,
                "byteorder": tf.byteorder,
            })
            if tf.is_ome and tf.ome_metadata:
                try:
                    info["ome_xml"] = _xml_to_dict(ET.fromstring(tf.ome_metadata))
                except Exception:
                    pass
    except Exception as exc:
        info["_error"] = str(exc)
    return info


_EXTRACTORS = {
    ".oex": _extract_oex,
    ".vsi": _extract_vsi,
    ".ets": _extract_ets,
    ".tif": _extract_tif,
    ".tiff": _extract_tif,
}


def extract_file_metadata(path: Path) -> dict[str, Any]:
    """Return raw metadata dict for *path*."""
    fn = _EXTRACTORS.get(path.suffix.lower())
    if fn is None:
        return {"_error": f"unsupported suffix: {path.suffix}"}
    return _safe(fn, path)


# ---------------------------------------------------------------------------
# Flatten raw metadata → one CSV row
# ---------------------------------------------------------------------------

def _flatten_vsi(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    aics = raw.get("aicsimageio", {})
    if "_error" in aics:
        out["metadata_error"] = aics["_error"]
        return out
    shape = aics.get("shape", {})
    out["size_x"]          = shape.get("X")
    out["size_y"]          = shape.get("Y")
    out["size_z"]          = shape.get("Z")
    out["n_channels"]      = shape.get("C")
    out["dtype"]           = aics.get("dtype")
    out["scenes"]          = json.dumps(aics.get("scenes", []))
    out["selected_scene"]  = aics.get("selected_scene")
    pps = aics.get("physical_pixel_sizes", {})
    out["pixel_size_x_um"] = pps.get("X")
    out["pixel_size_y_um"] = pps.get("Y")
    out["pixel_size_z_um"] = pps.get("Z")
    out["channel_names"]   = json.dumps(aics.get("channel_names"))
    return out


def _flatten_oex(raw: dict) -> dict[str, Any]:
    """Flatten all oex XML attributes into oex_<name> columns automatically."""
    out: dict[str, Any] = {"oex_container": raw.get("container")}

    xml = raw.get("xml", {})
    if not xml:
        return out

    # Collect every attribute name→value pair found anywhere in the XML tree.
    collected: dict[str, list] = {}

    def _walk(node):
        if not isinstance(node, dict):
            return
        attrs = node.get("attribute", [])
        if isinstance(attrs, dict):
            attrs = [attrs]
        for attr in (attrs or []):
            if not isinstance(attr, dict):
                continue
            name = attr.get("@attrs", {}).get("name", "")
            if not name:
                continue
            # Try any child key that carries a "val" — handles all XML value types
            # (unsigned, integer, float, double, string, int64, unsigned_long, bool, …)
            for vtype, v in attr.items():
                if vtype.startswith("@") or vtype == "attribute":
                    continue
                val = v.get("@attrs", {}).get("val") if isinstance(v, dict) else str(v)
                if val is not None:
                    collected.setdefault(name, []).append(val)
                    break
        for key, val in node.items():
            if key not in ("@attrs", "attribute"):
                if isinstance(val, list):
                    for item in val:
                        _walk(item)
                else:
                    _walk(val)

    _walk(xml)

    # Write into out: single value → scalar, multiple → JSON list
    for name, values in collected.items():
        col = "oex_" + name.replace(" ", "_").replace("/", "_")
        out[col] = values[0] if len(set(values)) == 1 else json.dumps(values)

    return out


def _flatten_ets(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata_backend": "ets_header"}
    header = raw.get("header", {})
    out["ets_version"] = header.get("version")
    out["ets_ndims"] = header.get("ndims")
    out["ets_chunk_count"] = header.get("used_chunk_count")
    return out


def _flatten_tif(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata_backend": "tifffile"}
    out["n_pages"] = raw.get("n_pages")
    shape = raw.get("shape", [])
    out["shape"] = json.dumps(shape)
    out["dtype"] = raw.get("dtype")
    out["axes"] = raw.get("axes")
    out["is_ome"] = raw.get("is_ome")
    return out


_FLATTENERS = {
    ".vsi": _flatten_vsi,
    ".oex": _flatten_oex,
    ".ets": _flatten_ets,
    ".tif": _flatten_tif,
    ".tiff": _flatten_tif,
}


def flatten_metadata(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Convert raw nested metadata into a flat dict suitable for a CSV row."""
    fn = _FLATTENERS.get(path.suffix.lower(), lambda r: {})
    flat = fn(raw)
    # error passthrough
    if "_error" in raw:
        flat["metadata_error"] = raw["_error"]
        flat.setdefault("metadata_backend", "failed")
    return flat


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def process_session(folder: Path) -> dict[str, Any] | None:
    """Extract metadata from both .oex and .vsi (one of each) per session folder.

    Returns a merged flat dict (one CSV row) or None on total failure.
    """
    img_extensions = {".oex", ".vsi", ".ets", ".tif", ".tiff", ".czi"}
    n_files = sum(1 for ext in img_extensions for _ in folder.glob(f"*{ext}"))

    oex_file = _pick_file_by_exts(folder, _OEX_EXTS)
    vsi_file = _pick_file_by_exts(folder, _VSI_EXTS)
    ets_file = _pick_file_by_exts(folder, _ETS_EXTS)
    tif_file = _pick_file_by_exts(folder, _TIF_EXTS)

    if not any([oex_file, vsi_file, ets_file, tif_file]):
        logger.warning(f"  [SKIP] {folder.name}: no supported file found")
        return None

    sources = []
    raw_by_type: dict[str, Any] = {}

    # Extract oex
    if oex_file:
        logger.info(f"  OEX: {oex_file.name}")
        raw_oex = extract_file_metadata(oex_file)
        flat_oex = _flatten_oex(raw_oex)
        raw_by_type["oex"] = {"file": oex_file.name, "metadata": raw_oex}
        sources.append(oex_file.name)
    else:
        flat_oex = {}

    # Extract vsi
    if vsi_file:
        logger.info(f"  VSI: {vsi_file.name}")
        raw_vsi = extract_file_metadata(vsi_file)
        flat_vsi = _flatten_vsi(raw_vsi)
        raw_by_type["vsi"] = {"file": vsi_file.name, "metadata": raw_vsi}
        sources.append(vsi_file.name)
    elif ets_file:
        logger.info(f"  ETS: {ets_file.name}")
        raw_ets = extract_file_metadata(ets_file)
        flat_vsi = _flatten_ets(raw_ets)
        raw_by_type["ets"] = {"file": ets_file.name, "metadata": raw_ets}
        sources.append(ets_file.name)
    elif tif_file:
        logger.info(f"  TIF: {tif_file.name}")
        raw_tif = extract_file_metadata(tif_file)
        flat_vsi = _flatten_tif(raw_tif)
        raw_by_type["tif"] = {"file": tif_file.name, "metadata": raw_tif}
        sources.append(tif_file.name)
    else:
        flat_vsi = {}

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


# ---------------------------------------------------------------------------
# CSV fieldnames  (superset across all file types)
# ---------------------------------------------------------------------------

_CSV_FIELDS_BASE = [
    "session_folder",
    "session_path",
    "sources_used",
    "n_image_files",
    # from vsi — image geometry
    "size_x",
    "size_y",
    "size_z",
    "n_channels",
    "dtype",
    "scenes",
    "selected_scene",
    "pixel_size_x_um",
    "pixel_size_y_um",
    "pixel_size_z_um",
    "channel_names",
    # from ets (fallback)
    "ets_version",
    "ets_ndims",
    "ets_chunk_count",
    # from tif (fallback)
    "n_pages",
    "shape",
    "axes",
    "is_ome",
    # errors
    "metadata_error",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-session metadata from microscopy folders."
    )
    parser.add_argument(
        "--input_root", type=Path, required=True,
        help="Root directory containing acquisition session subfolders.",
    )
    parser.add_argument(
        "--output_root", type=Path, required=True,
        help="Directory where metadata.csv and metadata_full.json are written.",
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many session folders (0 = all).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without extracting anything.",
    )

    args = parser.parse_args()

    logger.info(f"\n{'='*70}")
    logger.info("Metadata Extraction Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Input root:  {args.input_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Max folders: {args.max_folders if args.max_folders > 0 else 'all'}")

    logger.info(f"\nScanning for session folders in {args.input_root} ...")
    folders = _find_session_folders(args.input_root)

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
            oex = _pick_file_by_exts(folder, _OEX_EXTS)
            vsi = _pick_file_by_exts(folder, _VSI_EXTS)
            logger.info(f"  - {folder.name}  OEX={oex.name if oex else 'none'}  VSI={vsi.name if vsi else 'none'}")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict] = []
    full_json: list[dict] = []
    succeeded = 0

    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
        result = process_session(folder)
        if result is None:
            full_json.append({"session_folder": folder.name, "_error": "no file found"})
            continue
        csv_rows.append(result["csv_row"])
        full_json.append({
            "session_folder": folder.name,
            "sources": result["sources"],
            "metadata": result["raw_metadata"],
        })
        succeeded += 1
        logger.info(f"  [OK] {folder.name}")

    # Write CSV — build fieldnames dynamically: base fields first, then all oex_ columns found
    csv_path = args.output_root / "metadata.csv"
    if csv_rows:
        oex_cols = sorted({k for row in csv_rows for k in row if k.startswith("oex_")})
        fieldnames = _CSV_FIELDS_BASE + oex_cols
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
    logger.info(f"\nCSV  → {csv_path}  ({len(csv_rows)} rows)")

    # Write full JSON
    json_path = args.output_root / "metadata_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"JSON → {json_path}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Succeeded: {succeeded}/{len(folders)}")
    if succeeded < len(folders):
        failed = [f.name for f in folders if not any(r["session_folder"] == f.name for r in csv_rows)]
        for name in failed:
            logger.info(f"  - {name}")


if __name__ == "__main__":
    main()
