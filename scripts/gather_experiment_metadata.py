#!/usr/bin/env python3
r"""Gather full metadata for a pseudolabel damage-detection experiment.

Reads the experiment summary JSON (``patch_root``, thresholds,
``exclude_patterns``, ``n_images``, ``n_flagged``, ``flagged_sources``) and
adds:

1) Parsed metadata from each source filename
2) Optional patch-level aggregates from ``<patch_root>/index.csv``
3) Experiment-level consistency checks and summary stats

Outputs
-------
``<output_dir>/experiment_metadata_full.json``
    Full nested report with config, derived stats, and all per-source data.

``<output_dir>/flagged_sources_metadata.csv``
    Flat table for only flagged sources.

``<output_dir>/all_sources_metadata.csv``
    Flat table for every source found in the patch index (if available),
    plus flagged-only sources not present in index.csv.

Usage::

    # 1) Save your experiment JSON to a file and run:
    python scripts/gather_experiment_metadata.py \
        --experiment_json /path/to/experiment_summary.json \
        --output_dir /path/to/output

    # 2) Or pass inline JSON directly:
    python scripts/gather_experiment_metadata.py \
        --experiment_json_inline '{"patch_root": "...", "flagged_sources": [...]}' \
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from imaging.extract_olympus_metadata import extract as extract_olympus_file
except ModuleNotFoundError:
    # Support direct execution via: python scripts/gather_experiment_metadata.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from imaging.extract_olympus_metadata import extract as extract_olympus_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_METADATA_EXT_PRIORITY = [".oex", ".vsi"]


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


def _basename(source: str) -> str:
    # Handles both unix and windows style separators.
    return Path(source.replace("\\", "/")).name


def _source_stem(source: str) -> str:
    base = _basename(source)
    return base[:-4] if base.lower().endswith(".npy") else base


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_timestamp_and_seq(stem: str) -> tuple[str | None, int | None]:
    m = re.search(r"_(\d{8})_(\d+)$", stem)
    if not m:
        return None, None
    date_token = m.group(1)
    seq_token = m.group(2)
    try:
        seq = int(seq_token)
    except ValueError:
        seq = None
    return date_token, seq


def _strip_timestamp_suffix(stem: str) -> str:
    m = re.search(r"_(\d{8})_(\d+)$", stem)
    if not m:
        return stem
    return stem[: m.start()]


def _source_match_keys(source: str) -> dict[str, str | None]:
    stem = _source_stem(source)
    core = _strip_timestamp_suffix(stem)
    date_token, seq = _extract_timestamp_and_seq(stem)
    seq_key = f"{date_token}_{seq}" if date_token is not None and seq is not None else None
    return {
        "stem": stem,
        "core": core,
        "stem_norm": _normalize_key(stem),
        "core_norm": _normalize_key(core),
        "seq_key": seq_key,
    }


def _path_match_keys(path: Path) -> dict[str, str | None]:
    stem = path.stem
    core = _strip_timestamp_suffix(stem)
    date_token, seq = _extract_timestamp_and_seq(stem)
    seq_key = f"{date_token}_{seq}" if date_token is not None and seq is not None else None
    return {
        "stem": stem,
        "core": core,
        "stem_norm": _normalize_key(stem),
        "core_norm": _normalize_key(core),
        "seq_key": seq_key,
    }


def _metadata_ext_rank(path: Path) -> int:
    try:
        return _METADATA_EXT_PRIORITY.index(path.suffix.lower())
    except ValueError:
        return len(_METADATA_EXT_PRIORITY) + 1


def _collect_metadata_files(metadata_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in metadata_roots:
        if not root.exists():
            logger.warning(f"Metadata root does not exist: {root}")
            continue
        if root.is_file() and root.suffix.lower() in {".oex", ".vsi"}:
            files.append(root)
            continue
        for ext in _METADATA_EXT_PRIORITY:
            files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files), key=lambda p: (str(p).lower(), p.stat().st_size if p.exists() else 0))
    logger.info(f"Discovered {len(files)} Olympus metadata files under metadata roots")
    return files


def _build_metadata_index(metadata_files: list[Path]) -> dict[str, dict[str, list[Path]]]:
    index: dict[str, dict[str, list[Path]]] = {
        "stem": defaultdict(list),
        "core": defaultdict(list),
        "stem_norm": defaultdict(list),
        "core_norm": defaultdict(list),
        "seq_key": defaultdict(list),
    }

    for path in metadata_files:
        keys = _path_match_keys(path)
        for key_name, key_val in keys.items():
            if key_val:
                index[key_name][key_val].append(path)

    return index


def _pick_best_metadata_candidate(candidates: list[Path], source: str) -> Path:
    source_base = _basename(source)
    source_stem = _source_stem(source)

    def score(path: Path) -> tuple[int, int, int, str]:
        ext_rank = _metadata_ext_rank(path)
        same_stem = 0 if path.stem == source_stem else 1
        same_base = 0 if path.name.startswith(source_stem) else 1
        return (ext_rank, same_stem, same_base, str(path).lower())

    return sorted(candidates, key=score)[0]


def _match_olympus_file_for_source(
    source: str,
    metadata_index: dict[str, dict[str, list[Path]]],
) -> tuple[Path | None, str | None, int]:
    keys = _source_match_keys(source)
    ordered_matchers = ["stem", "core", "stem_norm", "core_norm", "seq_key"]

    for matcher in ordered_matchers:
        key_val = keys.get(matcher)
        if not key_val:
            continue
        candidates = metadata_index[matcher].get(key_val, [])
        if not candidates:
            continue
        picked = _pick_best_metadata_candidate(candidates, source)
        return picked, matcher, len(candidates)

    return None, None, 0


def _extract_metadata_cached(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    path_key = str(path.resolve())
    if path_key in cache:
        return cache[path_key]
    cache[path_key] = extract_olympus_file(path)
    return cache[path_key]


def enrich_with_olympus_metadata(
    source: str,
    metadata_index: dict[str, dict[str, list[Path]]],
    extraction_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matched_path, matched_by, candidate_count = _match_olympus_file_for_source(source, metadata_index)
    if matched_path is None:
        return {
            "found": False,
            "matched_by": None,
            "candidate_count": 0,
            "metadata_file": None,
            "metadata": None,
            "error": "No matching .oex/.vsi file found",
        }

    extracted = _extract_metadata_cached(matched_path, extraction_cache)
    file_meta = extracted.get("file") if isinstance(extracted, dict) else None
    parsed_meta = extracted.get("metadata") if isinstance(extracted, dict) else None
    error = extracted.get("_error") if isinstance(extracted, dict) else None

    return {
        "found": True,
        "matched_by": matched_by,
        "candidate_count": candidate_count,
        "metadata_file": file_meta,
        "metadata": parsed_meta,
        "error": error,
    }


def _parse_exposure_hours(token: str) -> int | None:
    # Supports common variants: 24hod, 48_hod, 24h, 24hr, 24hours.
    m = re.match(r"(?i)^(\d+)[_\-]?(?:hod|h|hr|hrs|hour|hours)$", token)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_concentration(token: str) -> tuple[float | None, str | None]:
    # Supports 10uM, 0.1uM, 1mM, etc.
    m = re.match(r"(?i)^([0-9]+(?:\.[0-9]+)?)([a-z]+)$", token)
    if not m:
        return None, None
    try:
        value = float(m.group(1))
    except ValueError:
        return None, None
    unit = m.group(2)
    return value, unit


def parse_source_name(source: str) -> dict[str, Any]:
    """Best-effort parser for dataset naming convention.

    Works for names like:
      5_PSI_10uM_48hod_10_Multichannel Z-Stack_20251208_270.npy
      5_PSI1h10uM_WO_48hod_1_Multichannel Z-Stack_20251208_241.npy
      8_7_PSYHARMIN24_hod_1_Multichannel Z-Stack_20251219_690.npy
    """
    base = _basename(source)
    stem = _source_stem(base)

    date_token, run_sequence = _extract_timestamp_and_seq(stem)

    core = stem
    if date_token is not None and run_sequence is not None:
        suffix = f"_{date_token}_{run_sequence}"
        if core.endswith(suffix):
            core = core[: -len(suffix)]

    # Remove imaging mode suffix if present.
    imaging_mode = None
    mm = re.search(r"_Multichannel Z-Stack$", core)
    if mm:
        imaging_mode = "Multichannel Z-Stack"
        core = core[: mm.start()]

    tokens = [t for t in core.split("_") if t]

    plate_or_batch: int | None = None
    replicate: str | None = None
    exposure_h: int | None = None
    concentration_value: float | None = None
    concentration_unit: str | None = None

    if tokens and re.fullmatch(r"\d+", tokens[0]):
        plate_or_batch = int(tokens[0])
        tokens = tokens[1:]

    # Replicate is usually the last token before imaging mode.
    if tokens and re.fullmatch(r"\d+[A-Za-z]?", tokens[-1]):
        replicate = tokens[-1]
        tokens = tokens[:-1]

    # Exposure token may be at the end, or split as e.g. "24", "hod".
    if tokens:
        h = _parse_exposure_hours(tokens[-1])
        if h is not None:
            exposure_h = h
            tokens = tokens[:-1]
        elif len(tokens) >= 2:
            joined = f"{tokens[-2]}{tokens[-1]}"
            h2 = _parse_exposure_hours(joined)
            if h2 is not None:
                exposure_h = h2
                tokens = tokens[:-2]

    # Try to find concentration in remaining tokens.
    condition_tokens = list(tokens)
    for i, tok in enumerate(tokens):
        val, unit = _parse_concentration(tok)
        if val is not None:
            concentration_value = val
            concentration_unit = unit
            condition_tokens = tokens[:i] + tokens[i + 1 :]
            break

    return {
        "source": source,
        "source_basename": base,
        "source_stem": stem,
        "plate_or_batch": plate_or_batch,
        "condition_raw": "_".join(condition_tokens) if condition_tokens else None,
        "condition_tokens": condition_tokens,
        "concentration_value": concentration_value,
        "concentration_unit": concentration_unit,
        "exposure_hours": exposure_h,
        "replicate": replicate,
        "imaging_mode": imaging_mode,
        "acquisition_date_yyyymmdd": date_token,
        "acquisition_run_sequence": run_sequence,
    }


def load_index_by_source(patch_root: Path) -> dict[str, dict[str, Any]]:
    """Read ``index.csv`` and aggregate patch-level metadata per source."""
    csv_path = patch_root / "index.csv"
    if not csv_path.exists():
        logger.warning(f"No index.csv found at {csv_path}; skipping patch-level enrichment.")
        return {}

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Different pipelines may use different source columns.
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
        grouped[_basename(src)].append(row)

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


def _contains_excluded(source: str, exclude_patterns: list[str]) -> bool:
    if not exclude_patterns:
        return False
    up = source.upper()
    return any(p.upper() in up for p in exclude_patterns)


def _source_row(
    parsed: dict[str, Any],
    patch_agg: dict[str, Any] | None,
    is_flagged: bool,
    excluded: bool,
    olympus_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    row = dict(parsed)
    row["is_flagged"] = is_flagged
    row["matches_exclude_patterns"] = excluded
    if patch_agg:
        row.update(patch_agg)
    else:
        row.update(
            {
                "n_patches": None,
                "mean_intensity_mean": None,
                "mean_intensity_min": None,
                "mean_intensity_max": None,
                "grid_row_min": None,
                "grid_row_max": None,
                "grid_col_min": None,
                "grid_col_max": None,
                "channels_values": [],
                "patch_size_values": [],
            }
        )

    if olympus_meta is None:
        row.update(
            {
                "olympus_metadata_found": False,
                "olympus_matched_by": None,
                "olympus_candidate_count": 0,
                "olympus_metadata_path": None,
                "olympus_metadata_suffix": None,
                "olympus_metadata_error": None,
                "olympus_metadata": None,
            }
        )
    else:
        metadata_file = olympus_meta.get("metadata_file") or {}
        metadata_path = metadata_file.get("path") if isinstance(metadata_file, dict) else None
        metadata_suffix = metadata_file.get("suffix") if isinstance(metadata_file, dict) else None
        row.update(
            {
                "olympus_metadata_found": bool(olympus_meta.get("found")),
                "olympus_matched_by": olympus_meta.get("matched_by"),
                "olympus_candidate_count": olympus_meta.get("candidate_count"),
                "olympus_metadata_path": metadata_path,
                "olympus_metadata_suffix": metadata_suffix,
                "olympus_metadata_error": olympus_meta.get("error"),
                "olympus_metadata": olympus_meta.get("metadata"),
            }
        )

    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return

    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())

    preferred = [
        "source",
        "source_basename",
        "source_stem",
        "is_flagged",
        "matches_exclude_patterns",
        "plate_or_batch",
        "condition_raw",
        "concentration_value",
        "concentration_unit",
        "exposure_hours",
        "replicate",
        "imaging_mode",
        "acquisition_date_yyyymmdd",
        "acquisition_run_sequence",
        "n_patches",
        "mean_intensity_mean",
        "mean_intensity_min",
        "mean_intensity_max",
        "grid_row_min",
        "grid_row_max",
        "grid_col_min",
        "grid_col_max",
        "channels_values",
        "patch_size_values",
        "condition_tokens",
    ]
    fieldnames = [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            # Keep CSV compact and easy to parse back.
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
        raise ValueError("Either --experiment_json or --experiment_json_inline must be provided.")
    if path is not None and inline_json is not None:
        raise ValueError("Provide only one of --experiment_json or --experiment_json_inline.")

    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(inline_json or "{}")


def build_report(experiment: dict[str, Any], metadata_roots: list[Path]) -> dict[str, Any]:
    patch_root_str = str(experiment.get("patch_root", "")).strip()
    patch_root = Path(patch_root_str) if patch_root_str else None
    flagged_sources = list(experiment.get("flagged_sources") or [])
    exclude_patterns = list(experiment.get("exclude_patterns") or [])

    patch_agg_by_base: dict[str, dict[str, Any]] = {}
    if patch_root is not None:
        patch_agg_by_base = load_index_by_source(patch_root)

    metadata_files = _collect_metadata_files(metadata_roots) if metadata_roots else []
    metadata_index = _build_metadata_index(metadata_files) if metadata_files else {
        "stem": defaultdict(list),
        "core": defaultdict(list),
        "stem_norm": defaultdict(list),
        "core_norm": defaultdict(list),
        "seq_key": defaultdict(list),
    }
    extraction_cache: dict[str, dict[str, Any]] = {}

    flagged_rows: list[dict[str, Any]] = []
    for src in flagged_sources:
        parsed = parse_source_name(src)
        base = parsed["source_basename"]
        olympus_meta = (
            enrich_with_olympus_metadata(base, metadata_index, extraction_cache)
            if metadata_files
            else None
        )
        flagged_rows.append(
            _source_row(
                parsed,
                patch_agg=patch_agg_by_base.get(base),
                is_flagged=True,
                excluded=_contains_excluded(base, exclude_patterns),
                olympus_meta=olympus_meta,
            )
        )

    all_rows: list[dict[str, Any]] = []
    flagged_bases = {r["source_basename"] for r in flagged_rows}

    for base_name, patch_meta in sorted(patch_agg_by_base.items()):
        parsed = parse_source_name(base_name)
        olympus_meta = (
            enrich_with_olympus_metadata(base_name, metadata_index, extraction_cache)
            if metadata_files
            else None
        )
        all_rows.append(
            _source_row(
                parsed,
                patch_agg=patch_meta,
                is_flagged=(base_name in flagged_bases),
                excluded=_contains_excluded(base_name, exclude_patterns),
                olympus_meta=olympus_meta,
            )
        )

    # Ensure flagged sources absent from index.csv are still included in all_rows.
    existing_bases = {r["source_basename"] for r in all_rows}
    for row in flagged_rows:
        if row["source_basename"] not in existing_bases:
            all_rows.append(row)

    n_flagged_input = len(flagged_sources)
    n_flagged_in_index = sum(
        1
        for row in flagged_rows
        if row.get("n_patches") is not None
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

    report = {
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gather full metadata for a specific experiment JSON (including flagged sources)."
    )
    parser.add_argument(
        "--experiment_json",
        type=Path,
        default=None,
        help="Path to JSON file with experiment summary.",
    )
    parser.add_argument(
        "--experiment_json_inline",
        type=str,
        default=None,
        help="Inline JSON string with experiment summary.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where metadata outputs will be written.",
    )
    parser.add_argument(
        "--metadata_root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Root directory to recursively scan for Olympus metadata files (.oex/.vsi). "
            "Repeat the argument to provide multiple roots."
        ),
    )
    args = parser.parse_args()

    experiment = load_experiment_json(args.experiment_json, args.experiment_json_inline)
    report = build_report(experiment, metadata_roots=list(args.metadata_root or []))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_json = args.output_dir / "experiment_metadata_full.json"
    full_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    flagged_csv = args.output_dir / "flagged_sources_metadata.csv"
    all_csv = args.output_dir / "all_sources_metadata.csv"
    _write_csv(report["flagged_sources_metadata"], flagged_csv)
    _write_csv(report["all_sources_metadata"], all_csv)

    logger.info(f"Wrote full report: {full_json}")
    logger.info(f"Wrote flagged CSV: {flagged_csv}")
    logger.info(f"Wrote all-sources CSV: {all_csv}")

    derived = report["derived"]
    logger.info(
        "Summary: index sources=%d, flagged(input)=%d, flagged(in index)=%d, missing=%d",
        derived["index_sources_count"],
        derived["flagged_sources_count_input"],
        derived["flagged_sources_in_index"],
        len(derived["flagged_sources_missing_from_index"]),
    )


if __name__ == "__main__":
    main()
