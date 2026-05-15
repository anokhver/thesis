#!/usr/bin/env python3
"""Extract metadata from Olympus cellSens files (.vsi, .ets, .oex).

Try every available backend independently. A missing optional dependency
only disables one source.

Usage:
    python extract_olympus_metadata.py <file1> [<file2> ...]

Optional deps: ``tifffile``, ``aicsimageio``, ``bioio``, ``bioio-bioformats``,
``python-bioformats``.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import traceback
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe(fn, *args, **kwargs):
    """Call ``fn`` and return its result, or a dict with error + traceback."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {
            "_error": f"{type(exc).__name__}: {exc}",
            "_traceback": traceback.format_exc(limit=3),
        }


def _file_stat(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "suffix": path.suffix.lower(),
    }


def _xml_to_dict(elem: ET.Element) -> Any:
    """Convert an ElementTree node to a JSON-serialisable dict/str."""
    node: dict[str, Any] = {}
    if elem.attrib:
        node["@attrs"] = dict(elem.attrib)
    text = (elem.text or "").strip()
    children = list(elem)
    if not children:
        if text:
            if not node:
                return text
            node["#text"] = text
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


# ---------------------------------------------------------------------------
# .oex (Olympus experiment file)
# ---------------------------------------------------------------------------
def extract_oex(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"_kind": "oex"}

    # Some .oex files are zipped, others are plain XML.
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
                        members[name] = {"_parse_error": str(exc),
                                         "_preview": raw[:200].decode("utf-8", "replace")}
                else:
                    members[name] = {"_binary_size": len(raw)}
        info["members"] = members
        return info

    # Plain XML file
    info["container"] = "xml"
    raw = path.read_bytes()
    info["size_bytes"] = len(raw)
    try:
        tree = ET.fromstring(raw)
        info["root_tag"] = tree.tag
        info["xml"] = _xml_to_dict(tree)
    except ET.ParseError as exc:
        info["_parse_error"] = str(exc)
        info["_preview"] = raw[:500].decode("utf-8", "replace")
    return info


# ---------------------------------------------------------------------------
# .ets (tiled pixel container)
# ---------------------------------------------------------------------------
def extract_ets(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"_kind": "ets"}

    # ETS files start with the "SIS0" magic and a fixed-size header.
    with path.open("rb") as fh:
        head = fh.read(64)
    info["magic"] = head[:4].decode("ascii", "replace")
    info["header_hex"] = head.hex()

    # Best-effort header decode (layout reverse-engineered from public code such
    # as bio-formats SISReader / pylibETS). Field meaning is approximate.
    if len(head) >= 64 and head[:4] == b"SIS0":
        try:
            (
                magic,
                header_size,
                version,
                ndims,
                additional_header_offset,
                additional_header_size,
                _reserved1,
                used_chunk_offset,
                used_chunk_count,
                _reserved2,
                _reserved3,
            ) = struct.unpack("<4s I I I I I I I I I I", head[:48])
            info["header"] = {
                "header_size": header_size,
                "version": version,
                "ndims": ndims,
                "additional_header_offset": additional_header_offset,
                "additional_header_size": additional_header_size,
                "used_chunk_offset": used_chunk_offset,
                "used_chunk_count": used_chunk_count,
            }
        except struct.error as exc:
            info["header_decode_error"] = str(exc)

    # Try tifffile: some ETS streams contain TIFF-compatible IFDs.
    try:
        import tifffile  # type: ignore

        try:
            with tifffile.TiffFile(str(path)) as tf:
                info["tifffile"] = {
                    "is_ome": tf.is_ome,
                    "byteorder": tf.byteorder,
                    "n_pages": len(tf.pages),
                    "pages": [
                        {
                            "shape": p.shape,
                            "dtype": str(p.dtype),
                            "axes": p.axes,
                            "tags": {t.name: str(t.value)[:200] for t in p.tags},
                        }
                        for p in tf.pages[:5]
                    ],
                }
        except Exception as exc:  # noqa: BLE001
            info["tifffile"] = {"_error": f"{type(exc).__name__}: {exc}"}
    except ImportError:
        info["tifffile"] = {"_skipped": "tifffile not installed"}

    # If a sibling .vsi is present, mention it - the .ets is meaningless alone.
    sibling_vsi = list(path.parent.parent.glob("*.vsi"))
    info["sibling_vsi"] = [str(p) for p in sibling_vsi]
    return info


# ---------------------------------------------------------------------------
# .vsi (Olympus slide container)
# ---------------------------------------------------------------------------
def _vsi_via_aicsimageio(path: Path) -> dict[str, Any]:
    from aicsimageio import AICSImage  # type: ignore

    img = AICSImage(str(path))
    out: dict[str, Any] = {
        "reader": type(img.reader).__name__,
        "scenes": list(img.scenes),
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
    if md is not None:
        if isinstance(md, ET.Element):
            out["ome_xml"] = _xml_to_dict(md)
        else:
            out["metadata_repr"] = repr(md)[:5000]
    return out


def _vsi_via_bioio(path: Path) -> dict[str, Any]:
    from bioio import BioImage  # type: ignore

    img = BioImage(str(path))
    out: dict[str, Any] = {
        "reader": type(img.reader).__name__,
        "scenes": list(img.scenes),
        "shape": dict(zip(img.dims.order, img.dims.shape)),
        "dtype": str(img.dtype),
        "channel_names": list(img.channel_names) if img.channel_names else None,
    }
    md = getattr(img, "metadata", None)
    if md is not None and isinstance(md, ET.Element):
        out["ome_xml"] = _xml_to_dict(md)
    return out


def _vsi_via_bioformats(path: Path) -> dict[str, Any]:
    import bioformats  # type: ignore
    import javabridge  # type: ignore

    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    try:
        ome_xml = bioformats.get_omexml_metadata(str(path))
        ome = bioformats.OMEXML(ome_xml)
        out = {
            "image_count": ome.get_image_count(),
            "first_image_name": ome.image(0).Name,
            "size": {
                "X": ome.image(0).Pixels.SizeX,
                "Y": ome.image(0).Pixels.SizeY,
                "Z": ome.image(0).Pixels.SizeZ,
                "C": ome.image(0).Pixels.SizeC,
                "T": ome.image(0).Pixels.SizeT,
            },
            "pixel_type": ome.image(0).Pixels.PixelType,
            "ome_xml": _xml_to_dict(ET.fromstring(ome_xml)),
        }
        return out
    finally:
        javabridge.kill_vm()


def extract_vsi(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"_kind": "vsi"}
    info["aicsimageio"] = _safe(_vsi_via_aicsimageio, path)
    info["bioio"] = _safe(_vsi_via_bioio, path)
    info["python_bioformats"] = _safe(_vsi_via_bioformats, path)

    # Sibling _<name>_/frame_*.ets files
    stem = path.stem
    candidates = list(path.parent.glob(f"_{stem}_*/frame_*.ets")) + \
                 list(path.parent.glob(f"_{stem}_/frame_*.ets"))
    info["associated_ets"] = [str(p) for p in candidates]
    return info


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
EXTRACTORS = {
    ".oex": extract_oex,
    ".ets": extract_ets,
    ".vsi": extract_vsi,
}


def extract(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_error": f"file not found: {path}"}
    record = {"file": _file_stat(path)}
    fn = EXTRACTORS.get(path.suffix.lower())
    if fn is None:
        record["_error"] = f"unsupported suffix: {path.suffix}"
        return record
    record["metadata"] = _safe(fn, path)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", help=".vsi / .ets / .oex files")
    parser.add_argument("--json", metavar="PATH", help="write full result as JSON")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    results = [extract(Path(os.path.expanduser(f))) for f in args.files]
    text = json.dumps(results, indent=args.indent, default=str, ensure_ascii=False)

    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
        print(f"wrote {args.json} ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
