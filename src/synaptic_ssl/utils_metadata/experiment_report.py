"""Assemble a full per-experiment report from flagged sources + patch index.

Inputs:
  * An *experiment summary* dict with ``patch_root``, optional
    ``exclude_patterns``, and ``flagged_sources`` (list of source-image
    filenames).
  * Optional Olympus metadata roots to scan for ``.oex``/``.vsi`` files
    that get matched to each source.

Outputs (returned as a nested dict from :func:`build_report`, and
serialized to ``experiment_metadata_full.json`` /
``flagged_sources_metadata.csv`` / ``all_sources_metadata.csv`` by
:func:`write_outputs`):
  * Per-source parsed name fields (via :mod:`.source_names`).
  * Per-source patch-index aggregates (n_patches, mean_intensity, grid
    extent, ...) from ``<patch_root>/index.csv``.
  * Per-source Olympus metadata enrichment (via :mod:`.olympus_matcher`).
  * Experiment-level summary in the ``derived`` block.
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .olympus_matcher import (
    build_metadata_index, collect_metadata_files, enrich_with_olympus_metadata,
)
from .source_names import basename, parse_source_name

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def load_index_by_source(patch_root: Path) -> dict[str, dict[str, Any]]:
    """Read ``patch_root/index.csv`` and aggregate patch records per source.

    Returns ``{source_basename: {n_patches, mean_intensity_*, grid_*,
    channels_values, patch_size_values}}``. Empty if the file is missing
    or has no recognised source column.
    """
    csv_path = patch_root / "index.csv"
    if not csv_path.exists():
        logger.warning(f"No index.csv found at {csv_path}; skipping patch-level enrichment.")
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    source_candidates = ["source_image", "source_npy", "source_path"]
    source_col = next((c for c in source_candidates if c in fieldnames), None)
    if source_col is None:
        logger.warning(
            "index.csv does not contain any of source_image/source_npy/source_path; "
            "skipping patch-level enrichment."
        )
        return {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        src = row.get(source_col, "")
        if not src:
            continue
        grouped[basename(src)].append(row)

    agg: dict[str, dict[str, Any]] = {}
    for src_base, recs in grouped.items():
        n_patches = len(recs)

        mean_vals = [_to_float(r.get("mean_intensity")) for r in recs]
        mean_vals = [v for v in mean_vals if v is not None]

        grid_rows = [_to_float(r.get("grid_row")) for r in recs]
        grid_cols = [_to_float(r.get("grid_col")) for r in recs]
        grid_rows = [int(v) for v in grid_rows if v is not None]
        grid_cols = [int(v) for v in grid_cols if v is not None]

        channels_vals = {_to_float(r.get("channels")) for r in recs}
        channels_vals = sorted({int(v) for v in channels_vals if v is not None})

        patch_size_vals = {_to_float(r.get("patch_size")) for r in recs}
        patch_size_vals = sorted({int(v) for v in patch_size_vals if v is not None})

        agg[src_base] = {
            "n_patches": n_patches,
            "mean_intensity_mean": (sum(mean_vals) / len(mean_vals)) if mean_vals else None,
            "mean_intensity_min": min(mean_vals) if mean_vals else None,
            "mean_intensity_max": max(mean_vals) if mean_vals else None,
            "grid_row_min": min(grid_rows) if grid_rows else None,
            "grid_row_max": max(grid_rows) if grid_rows else None,
            "grid_col_min": min(grid_cols) if grid_cols else None,
            "grid_col_max": max(grid_cols) if grid_cols else None,
            "channels_values": channels_vals,
            "patch_size_values": patch_size_vals,
        }

    logger.info(f"Loaded patch index metadata for {len(agg)} unique sources from {csv_path}")
    return agg


def contains_excluded(source: str, exclude_patterns: list[str]) -> bool:
    if not exclude_patterns:
        return False
    up = source.upper()
    return any(p.upper() in up for p in exclude_patterns)


def source_row(
    parsed: dict[str, Any],
    patch_agg: dict[str, Any] | None,
    is_flagged: bool,
    excluded: bool,
    olympus_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge parsed name + patch aggregates + Olympus metadata into one row."""
    row = dict(parsed)
    row["is_flagged"] = is_flagged
    row["matches_exclude_patterns"] = excluded
    if patch_agg:
        row.update(patch_agg)
    else:
        row.update({
            "n_patches": None,
            "mean_intensity_mean": None,
            "mean_intensity_min": None,
            "mean_intensity_max": None,
            "grid_row_min": None, "grid_row_max": None,
            "grid_col_min": None, "grid_col_max": None,
            "channels_values": [], "patch_size_values": [],
        })

    if olympus_meta is None:
        row.update({
            "olympus_metadata_found": False,
            "olympus_matched_by": None,
            "olympus_candidate_count": 0,
            "olympus_metadata_path": None,
            "olympus_metadata_suffix": None,
            "olympus_metadata_error": None,
            "olympus_metadata": None,
        })
    else:
        metadata_file = olympus_meta.get("metadata_file") or {}
        metadata_path = metadata_file.get("path") if isinstance(metadata_file, dict) else None
        metadata_suffix = metadata_file.get("suffix") if isinstance(metadata_file, dict) else None
        row.update({
            "olympus_metadata_found": bool(olympus_meta.get("found")),
            "olympus_matched_by": olympus_meta.get("matched_by"),
            "olympus_candidate_count": olympus_meta.get("candidate_count"),
            "olympus_metadata_path": metadata_path,
            "olympus_metadata_suffix": metadata_suffix,
            "olympus_metadata_error": olympus_meta.get("error"),
            "olympus_metadata": olympus_meta.get("metadata"),
        })

    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write *rows* to *path* with a stable preferred-field-first column order."""
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return

    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())

    preferred = [
        "source", "source_basename", "source_stem",
        "is_flagged", "matches_exclude_patterns",
        "plate_or_batch", "condition_raw",
        "concentration_value", "concentration_unit",
        "exposure_hours", "replicate", "imaging_mode",
        "acquisition_date_yyyymmdd", "acquisition_run_sequence",
        "n_patches",
        "mean_intensity_mean", "mean_intensity_min", "mean_intensity_max",
        "grid_row_min", "grid_row_max",
        "grid_col_min", "grid_col_max",
        "channels_values", "patch_size_values",
        "condition_tokens",
    ]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if isinstance(out.get("condition_tokens"), list):
                out["condition_tokens"] = "|".join(str(x) for x in out["condition_tokens"])
            if isinstance(out.get("channels_values"), list):
                out["channels_values"] = "|".join(str(x) for x in out["channels_values"])
            if isinstance(out.get("patch_size_values"), list):
                out["patch_size_values"] = "|".join(str(x) for x in out["patch_size_values"])
            if isinstance(out.get("olympus_metadata"), (dict, list)):
                out["olympus_metadata"] = json.dumps(out["olympus_metadata"], ensure_ascii=True)
            writer.writerow(out)


def load_experiment_json(path: Path | None, inline_json: str | None) -> dict[str, Any]:
    if path is None and inline_json is None:
        raise ValueError("Either path or inline_json must be provided.")
    if path is not None and inline_json is not None:
        raise ValueError("Provide only one of path or inline_json.")
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(inline_json or "{}")


def build_report(experiment: dict[str, Any], metadata_roots: list[Path]) -> dict[str, Any]:
    """Assemble the full experiment report dict.

    Top-level keys: ``generated_at_utc``, ``input_experiment``,
    ``derived``, ``flagged_sources_metadata``, ``all_sources_metadata``.
    """
    patch_root_str = str(experiment.get("patch_root", "")).strip()
    patch_root = Path(patch_root_str) if patch_root_str else None
    flagged_sources = list(experiment.get("flagged_sources") or [])
    exclude_patterns = list(experiment.get("exclude_patterns") or [])

    patch_agg_by_base: dict[str, dict[str, Any]] = {}
    if patch_root is not None:
        patch_agg_by_base = load_index_by_source(patch_root)

    metadata_files = collect_metadata_files(metadata_roots) if metadata_roots else []
    metadata_index = (
        build_metadata_index(metadata_files)
        if metadata_files else {
            "stem": defaultdict(list),
            "core": defaultdict(list),
            "stem_norm": defaultdict(list),
            "core_norm": defaultdict(list),
            "seq_key": defaultdict(list),
        }
    )
    extraction_cache: dict[str, dict[str, Any]] = {}

    flagged_rows: list[dict[str, Any]] = []
    for src in flagged_sources:
        parsed = parse_source_name(src)
        base = parsed["source_basename"]
        olympus_meta = (
            enrich_with_olympus_metadata(base, metadata_index, extraction_cache)
            if metadata_files else None
        )
        flagged_rows.append(source_row(
            parsed,
            patch_agg=patch_agg_by_base.get(base),
            is_flagged=True,
            excluded=contains_excluded(base, exclude_patterns),
            olympus_meta=olympus_meta,
        ))

    all_rows: list[dict[str, Any]] = []
    flagged_bases = {r["source_basename"] for r in flagged_rows}

    for base_name, patch_meta in sorted(patch_agg_by_base.items()):
        parsed = parse_source_name(base_name)
        olympus_meta = (
            enrich_with_olympus_metadata(base_name, metadata_index, extraction_cache)
            if metadata_files else None
        )
        all_rows.append(source_row(
            parsed,
            patch_agg=patch_meta,
            is_flagged=(base_name in flagged_bases),
            excluded=contains_excluded(base_name, exclude_patterns),
            olympus_meta=olympus_meta,
        ))

    # Ensure flagged sources absent from index.csv still appear in all_rows.
    existing_bases = {r["source_basename"] for r in all_rows}
    for row in flagged_rows:
        if row["source_basename"] not in existing_bases:
            all_rows.append(row)

    n_flagged_input = len(flagged_sources)
    n_flagged_in_index = sum(
        1 for row in flagged_rows if row.get("n_patches") is not None
    )
    missing_flagged = [
        row["source_basename"]
        for row in flagged_rows
        if row.get("n_patches") is None
    ]
    olympus_found_count = sum(1 for row in all_rows if row.get("olympus_metadata_found"))
    olympus_missing_sources = [
        row["source_basename"]
        for row in all_rows
        if not row.get("olympus_metadata_found")
    ]

    cond_counts: dict[str, int] = defaultdict(int)
    exposure_counts: dict[str, int] = defaultdict(int)
    for row in all_rows:
        cond = row.get("condition_raw") or "<unknown>"
        cond_counts[str(cond)] += 1
        exp_h = row.get("exposure_hours")
        exposure_counts[str(exp_h) if exp_h is not None else "<unknown>"] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_experiment": experiment,
        "derived": {
            "patch_root_exists": bool(patch_root and patch_root.exists()),
            "index_sources_count": len(patch_agg_by_base),
            "metadata_roots": [str(p) for p in metadata_roots],
            "metadata_files_discovered": len(metadata_files),
            "metadata_files_used": len(extraction_cache),
            "sources_with_olympus_metadata": olympus_found_count,
            "sources_missing_olympus_metadata": olympus_missing_sources,
            "flagged_sources_count_input": n_flagged_input,
            "flagged_sources_in_index": n_flagged_in_index,
            "flagged_sources_missing_from_index": missing_flagged,
            "sources_matching_exclude_patterns": [
                row["source_basename"]
                for row in all_rows
                if row.get("matches_exclude_patterns")
            ],
            "condition_counts": dict(sorted(cond_counts.items())),
            "exposure_hours_counts": dict(sorted(exposure_counts.items())),
        },
        "flagged_sources_metadata": flagged_rows,
        "all_sources_metadata": all_rows,
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the full JSON + per-source CSVs to *output_dir*.

    Returns ``(full_json_path, flagged_csv_path, all_csv_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    full_json = output_dir / "experiment_metadata_full.json"
    full_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    flagged_csv = output_dir / "flagged_sources_metadata.csv"
    all_csv = output_dir / "all_sources_metadata.csv"
    write_csv(report["flagged_sources_metadata"], flagged_csv)
    write_csv(report["all_sources_metadata"], all_csv)

    return full_json, flagged_csv, all_csv


__all__ = [
    "load_index_by_source",
    "contains_excluded",
    "source_row",
    "write_csv",
    "load_experiment_json",
    "build_report",
    "write_outputs",
]
