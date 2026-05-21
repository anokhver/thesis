"""Per-format raw metadata extractors for Olympus / Bio-Formats microscopy.

Each ``extract_<fmt>`` function returns a JSON-safe dict. Heavy readers
(``aicsimageio``, ``tifffile``, ``ome_types``) are imported lazily so
``import synaptic_ssl.utils_metadata`` works without the optional
``imaging`` extra installed.
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Priority order: which file type to prefer for metadata within a session folder.
PRIORITY_EXTENSIONS: tuple[str, ...] = (".oex", ".vsi", ".ets", ".tif", ".tiff")

OEX_EXTS: tuple[str, ...] = (".oex",)
VSI_EXTS: tuple[str, ...] = (".vsi",)
ETS_EXTS: tuple[str, ...] = (".ets",)
TIF_EXTS: tuple[str, ...] = (".tif", ".tiff")


# ---------------------------------------------------------------------------
# JSON-safe converters
# ---------------------------------------------------------------------------

def xml_to_dict(elem: ET.Element) -> Any:
    """Convert an :class:`ElementTree.Element` tree to a nested dict."""
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
        value = xml_to_dict(child)
        if tag in node:
            if not isinstance(node[tag], list):
                node[tag] = [node[tag]]
            node[tag].append(value)
        else:
            node[tag] = value
    if text:
        node["#text"] = text
    return node


def model_to_dict(obj: Any) -> Any:
    """Convert an ome_types / pydantic / nested object to plain JSON-safe data.

    Handles pydantic v2 (``model_dump``), pydantic v1 (``dict``), enums,
    paths, datetimes, and arbitrary objects via ``str()``. Drops keys
    whose value is ``None`` to keep the output compact.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return model_to_dict(obj.model_dump(exclude_none=True, mode="json"))
        except Exception:
            pass
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return model_to_dict(obj.dict(exclude_none=True))
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): model_to_dict(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple, set)):
        return [model_to_dict(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return str(obj)


def safe(fn, *args, **kwargs):
    """Call *fn*; on any exception return ``{"_error": "TypeName: msg"}``."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Per-format extractors
# ---------------------------------------------------------------------------

def extract_oex(path: Path) -> dict[str, Any]:
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
                        members[name] = xml_to_dict(ET.fromstring(raw))
                    except ET.ParseError as exc:
                        members[name] = {"_parse_error": str(exc)}
                else:
                    members[name] = {"_binary_size": len(raw)}
        info["members"] = members
    else:
        info["container"] = "xml"
        raw = path.read_bytes()
        try:
            info["xml"] = xml_to_dict(ET.fromstring(raw))
        except ET.ParseError as exc:
            info["_parse_error"] = str(exc)
    return info


def _extract_vsi_via_aicsimageio(path: Path) -> dict[str, Any]:
    from aicsimageio import AICSImage  # type: ignore

    img = AICSImage(str(path))
    try:
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
            logger.info(
                f"  VSI scene selected: {best!r} ({best_size:,} voxels, {len(scenes)} total)"
            )

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

        # OME metadata: walk the ome_types model into a plain dict so the
        # Instrument/Objective/Detector/LightSource blocks are preserved.
        md = getattr(img, "metadata", None)
        if isinstance(md, ET.Element):
            out["ome_xml"] = xml_to_dict(md)
        elif md is not None:
            out["ome_model"] = safe(model_to_dict, md)
            try:
                out["ome_model_class"] = f"{type(md).__module__}.{type(md).__name__}"
            except Exception:
                pass
            if hasattr(md, "to_xml"):
                try:
                    xml_str = md.to_xml()
                    out["ome_xml"] = xml_to_dict(ET.fromstring(xml_str))
                except Exception:
                    pass

        # Bio-Formats OriginalMetadata: vendor-specific key/value pairs
        # (Olympus VSI exposes ``Objective Name``, ``Objective Working
        # Distance``, ``Camera Model``, ``Stage Position``, etc. here).
        reader = getattr(img, "reader", None)
        for attr in ("metadata", "original_metadata"):
            val = getattr(reader, attr, None)
            if isinstance(val, dict) and val:
                out.setdefault("bf_original_metadata", model_to_dict(val))
                break

        return out
    finally:
        try:
            img.close()
        except Exception:
            pass


def extract_vsi(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"_kind": "vsi"}
    info["aicsimageio"] = safe(_extract_vsi_via_aicsimageio, path)
    return info


def extract_ets(path: Path) -> dict[str, Any]:
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


def extract_tif(path: Path) -> dict[str, Any]:
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
                "is_bigtiff": getattr(tf, "is_bigtiff", False),
                "is_imagej": getattr(tf, "is_imagej", False),
                "is_lsm": getattr(tf, "is_lsm", False),
                "is_micromanager": getattr(tf, "is_micromanager", False),
                "is_scn": getattr(tf, "is_scn", False),
                "is_svs": getattr(tf, "is_svs", False),
            })

            tags: dict[str, Any] = {}
            for tag in p.tags.values():
                name = tag.name
                try:
                    val = tag.value
                except Exception:
                    continue
                # Skip tile/strip offsets (huge arrays, no analysis value).
                if name in ("TileOffsets", "TileByteCounts",
                            "StripOffsets", "StripByteCounts"):
                    continue
                tags[name] = model_to_dict(val)
            info["tif_tags"] = tags

            for attr in ("imagej_metadata", "ome_metadata",
                         "lsm_metadata", "stk_metadata",
                         "scanimage_metadata", "micromanager_metadata",
                         "shaped_metadata", "fluoview_metadata"):
                val = getattr(tf, attr, None)
                if val:
                    if attr == "ome_metadata" and isinstance(val, str):
                        try:
                            info["ome_xml"] = xml_to_dict(ET.fromstring(val))
                        except Exception:
                            info["ome_metadata_raw"] = val[:8000]
                    else:
                        info[attr] = model_to_dict(val)
    except Exception as exc:
        info["_error"] = str(exc)
    return info


_EXTRACTORS = {
    ".oex": extract_oex,
    ".vsi": extract_vsi,
    ".ets": extract_ets,
    ".tif": extract_tif,
    ".tiff": extract_tif,
}


def extract_file_metadata(path: Path) -> dict[str, Any]:
    """Return raw metadata dict for *path*; ``{"_error": ...}`` on unknown suffix."""
    fn = _EXTRACTORS.get(path.suffix.lower())
    if fn is None:
        return {"_error": f"unsupported suffix: {path.suffix}"}
    return safe(fn, path)


__all__ = [
    "PRIORITY_EXTENSIONS",
    "OEX_EXTS", "VSI_EXTS", "ETS_EXTS", "TIF_EXTS",
    "xml_to_dict", "model_to_dict", "safe",
    "extract_oex", "extract_vsi", "extract_ets", "extract_tif",
    "extract_file_metadata",
]
