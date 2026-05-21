"""Flatten raw extractor output into one CSV row per session.

Mirrors the per-format dispatch in :mod:`.extractors`. The
``flatten_*`` functions are best-effort: missing keys become missing
columns; OME hardware blocks emit ``ome_*`` columns; Bio-Formats
``original_metadata`` and Olympus ``oex`` attributes become
``bf_*`` / ``oex_*`` columns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _ome_get_first(obj: Any, *keys: str) -> Any:
    """Walk *obj* by trying each dot-separated key path; return first hit."""
    for path in keys:
        cur = obj
        ok = True
        for part in path.split("."):
            if cur is None:
                ok = False; break
            if part.isdigit() and isinstance(cur, list):
                idx = int(part)
                cur = cur[idx] if idx < len(cur) else None
            elif isinstance(cur, dict):
                cur = cur.get(part)
            else:
                ok = False; break
        if ok and cur is not None:
            return cur
    return None


def flatten_ome_hardware(ome: dict) -> dict[str, Any]:
    """Pull objective / instrument / detector / channel fields from an
    ``ome_types`` model dict into flat ``ome_*`` columns.
    """
    out: dict[str, Any] = {}
    instruments = ome.get("instruments") or []
    if instruments:
        inst = instruments[0]
        mic = inst.get("microscope") or {}
        out["ome_microscope_model"]        = mic.get("model")
        out["ome_microscope_manufacturer"] = mic.get("manufacturer")
        out["ome_microscope_serial"]       = mic.get("serial_number")
        out["ome_microscope_type"]         = mic.get("type")

        objectives = inst.get("objectives") or []
        if objectives:
            obj = objectives[0]
            out["ome_objective_model"]               = obj.get("model")
            out["ome_objective_manufacturer"]        = obj.get("manufacturer")
            out["ome_objective_nominal_magnification"] = obj.get("nominal_magnification")
            out["ome_objective_calibrated_magnification"] = obj.get("calibrated_magnification")
            out["ome_objective_lens_na"]             = obj.get("lens_na")
            out["ome_objective_immersion"]           = obj.get("immersion")
            out["ome_objective_correction"]          = obj.get("correction")
            out["ome_objective_working_distance"]    = obj.get("working_distance")
            out["ome_objective_working_distance_unit"] = obj.get("working_distance_unit")
            out["ome_n_objectives"] = len(objectives)

        detectors = inst.get("detectors") or []
        if detectors:
            det = detectors[0]
            out["ome_detector_model"]        = det.get("model")
            out["ome_detector_manufacturer"] = det.get("manufacturer")
            out["ome_detector_type"]         = det.get("type")
            out["ome_n_detectors"] = len(detectors)

        light_sources = inst.get("light_source_group") or inst.get("light_sources") or []
        if light_sources:
            out["ome_n_light_sources"] = len(light_sources)
            out["ome_light_source_types"] = json.dumps(
                [ls.get("kind") or type(ls).__name__ for ls in light_sources]
            )

    images = ome.get("images") or []
    if images:
        im = images[0]
        out["ome_image_name"]             = im.get("name")
        out["ome_image_acquisition_date"] = im.get("acquisition_date")
        out["ome_image_description"]      = im.get("description")
        pixels = im.get("pixels") or {}
        out["ome_pixels_type"]            = pixels.get("type")
        out["ome_pixels_dim_order"]       = pixels.get("dimension_order")
        out["ome_size_t"]                 = pixels.get("size_t")
        channels = pixels.get("channels") or []
        if channels:
            out["ome_n_channels"] = len(channels)
            out["ome_channel_names"] = json.dumps(
                [c.get("name") for c in channels]
            )
            out["ome_channel_excitation_nm"] = json.dumps(
                [c.get("excitation_wavelength") for c in channels]
            )
            out["ome_channel_emission_nm"] = json.dumps(
                [c.get("emission_wavelength") for c in channels]
            )
            out["ome_channel_illumination_types"] = json.dumps(
                [c.get("illumination_type") for c in channels]
            )
            out["ome_channel_fluors"] = json.dumps(
                [c.get("fluor") for c in channels]
            )
    return {k: v for k, v in out.items() if v is not None}


def flatten_vsi(raw: dict) -> dict[str, Any]:
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

    ome = aics.get("ome_model")
    if isinstance(ome, dict):
        out.update(flatten_ome_hardware(ome))

    bf = aics.get("bf_original_metadata")
    if isinstance(bf, dict):
        # Vendor-specific keys (Olympus VSI uses spaces and slashes):
        # promote known ones, JSON-encode everything else under bf_*.
        for k, v in bf.items():
            col = "bf_" + str(k).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
            out[col] = v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v)
    return out


def flatten_oex(raw: dict) -> dict[str, Any]:
    """Flatten every oex XML attribute into ``oex_<name>`` columns automatically."""
    out: dict[str, Any] = {"oex_container": raw.get("container")}

    xml = raw.get("xml", {})
    if not xml:
        return out

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
            # Try any child key carrying a "val" — handles all XML value
            # types (unsigned, integer, float, double, string, int64,
            # unsigned_long, bool, ...).
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

    for name, values in collected.items():
        col = "oex_" + name.replace(" ", "_").replace("/", "_")
        out[col] = values[0] if len(set(values)) == 1 else json.dumps(values)

    return out


def flatten_ets(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata_backend": "ets_header"}
    header = raw.get("header", {})
    out["ets_version"] = header.get("version")
    out["ets_ndims"] = header.get("ndims")
    out["ets_chunk_count"] = header.get("used_chunk_count")
    return out


def flatten_tif(raw: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata_backend": "tifffile"}
    out["n_pages"] = raw.get("n_pages")
    shape = raw.get("shape", [])
    out["shape"] = json.dumps(shape)
    out["dtype"] = raw.get("dtype")
    out["axes"] = raw.get("axes")
    out["is_ome"] = raw.get("is_ome")
    out["is_bigtiff"] = raw.get("is_bigtiff")
    out["is_imagej"] = raw.get("is_imagej")

    tags = raw.get("tif_tags") or {}
    for name, val in tags.items():
        col = "tif_" + name
        out[col] = val if isinstance(val, (str, int, float, bool)) or val is None \
            else json.dumps(val)

    ome_xml = raw.get("ome_xml")
    if isinstance(ome_xml, dict):
        # Raw XML-as-dict; an ome_types model_dump would be needed for the
        # hardware extraction path. Only record presence here.
        out["tif_ome_xml_present"] = True
    return out


_FLATTENERS = {
    ".vsi": flatten_vsi,
    ".oex": flatten_oex,
    ".ets": flatten_ets,
    ".tif": flatten_tif,
    ".tiff": flatten_tif,
}


def flatten_metadata(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Convert raw nested metadata into a flat dict suitable for a CSV row."""
    fn = _FLATTENERS.get(path.suffix.lower(), lambda r: {})
    flat = fn(raw)
    if "_error" in raw:
        flat["metadata_error"] = raw["_error"]
        flat.setdefault("metadata_backend", "failed")
    return flat


# Canonical CSV column order. Auto-collected oex_/bf_/tif_ columns are
# appended (sorted) after these by the script writing the file.
CSV_FIELDS_BASE: tuple[str, ...] = (
    "session_folder",
    "session_path",
    "sources_used",
    "n_image_files",
    # from vsi — image geometry
    "size_x", "size_y", "size_z",
    "n_channels", "dtype",
    "scenes", "selected_scene",
    "pixel_size_x_um", "pixel_size_y_um", "pixel_size_z_um",
    "channel_names",
    # from vsi — OME hardware
    "ome_microscope_model", "ome_microscope_manufacturer",
    "ome_microscope_serial", "ome_microscope_type",
    "ome_n_objectives",
    "ome_objective_model", "ome_objective_manufacturer",
    "ome_objective_nominal_magnification", "ome_objective_calibrated_magnification",
    "ome_objective_lens_na", "ome_objective_immersion", "ome_objective_correction",
    "ome_objective_working_distance", "ome_objective_working_distance_unit",
    "ome_n_detectors",
    "ome_detector_model", "ome_detector_manufacturer", "ome_detector_type",
    "ome_n_light_sources", "ome_light_source_types",
    "ome_image_name", "ome_image_acquisition_date", "ome_image_description",
    "ome_pixels_type", "ome_pixels_dim_order", "ome_size_t",
    "ome_n_channels", "ome_channel_names",
    "ome_channel_excitation_nm", "ome_channel_emission_nm",
    "ome_channel_illumination_types", "ome_channel_fluors",
    # from ets (fallback)
    "ets_version", "ets_ndims", "ets_chunk_count",
    # from tif (fallback)
    "n_pages", "shape", "axes", "is_ome", "is_bigtiff", "is_imagej",
    # errors
    "metadata_error",
)


__all__ = [
    "flatten_ome_hardware",
    "flatten_vsi", "flatten_oex", "flatten_ets", "flatten_tif",
    "flatten_metadata",
    "CSV_FIELDS_BASE",
]
