"""Parse dataset source filenames into experiment metadata fields.

Best-effort parser for the project's naming convention. Handles names
like::

    5_PSI_10uM_48hod_10_Multichannel Z-Stack_20251208_270.npy
    5_PSI1h10uM_WO_48hod_1_Multichannel Z-Stack_20251208_241.npy
    8_7_PSYHARMIN24_hod_1_Multichannel Z-Stack_20251219_690.npy
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def basename(source: str) -> str:
    """Filename component, accepting either ``/`` or ``\\`` separators."""
    return Path(source.replace("\\", "/")).name


def source_stem(source: str) -> str:
    """Strip a trailing ``.npy`` suffix from the basename, if present."""
    base = basename(source)
    return base[:-4] if base.lower().endswith(".npy") else base


def normalize_key(text: str) -> str:
    """Lowercase ``text`` and drop every non-alphanumeric character."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def extract_timestamp_and_seq(stem: str) -> tuple[str | None, int | None]:
    """Return ``(YYYYMMDD, seq)`` from a trailing ``_YYYYMMDD_<int>`` suffix."""
    m = re.search(r"_(\d{8})_(\d+)$", stem)
    if not m:
        return None, None
    date_token = m.group(1)
    try:
        seq = int(m.group(2))
    except ValueError:
        seq = None
    return date_token, seq


def strip_timestamp_suffix(stem: str) -> str:
    m = re.search(r"_(\d{8})_(\d+)$", stem)
    if not m:
        return stem
    return stem[: m.start()]


def source_match_keys(source: str) -> dict[str, str | None]:
    """Lookup keys used to match a source basename against metadata files."""
    stem = source_stem(source)
    core = strip_timestamp_suffix(stem)
    date_token, seq = extract_timestamp_and_seq(stem)
    seq_key = f"{date_token}_{seq}" if date_token is not None and seq is not None else None
    return {
        "stem": stem,
        "core": core,
        "stem_norm": normalize_key(stem),
        "core_norm": normalize_key(core),
        "seq_key": seq_key,
    }


def path_match_keys(path: Path) -> dict[str, str | None]:
    """Same as :func:`source_match_keys` but for a :class:`Path`."""
    stem = path.stem
    core = strip_timestamp_suffix(stem)
    date_token, seq = extract_timestamp_and_seq(stem)
    seq_key = f"{date_token}_{seq}" if date_token is not None and seq is not None else None
    return {
        "stem": stem,
        "core": core,
        "stem_norm": normalize_key(stem),
        "core_norm": normalize_key(core),
        "seq_key": seq_key,
    }


def _parse_exposure_hours(token: str) -> int | None:
    """Parse exposure tokens: ``24hod``, ``48_hod``, ``24h``, ``24hr``, ``24hours``."""
    m = re.match(r"(?i)^(\d+)[_\-]?(?:hod|h|hr|hrs|hour|hours)$", token)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _parse_concentration(token: str) -> tuple[float | None, str | None]:
    """Parse concentration tokens: ``10uM``, ``0.1uM``, ``1mM``, etc."""
    m = re.match(r"(?i)^([0-9]+(?:\.[0-9]+)?)([a-z]+)$", token)
    if not m:
        return None, None
    try:
        value = float(m.group(1))
    except ValueError:
        return None, None
    return value, m.group(2)


def parse_source_name(source: str) -> dict[str, Any]:
    """Return a flat dict of inferred experiment fields for *source*.

    Keys: ``source``, ``source_basename``, ``source_stem``,
    ``plate_or_batch``, ``condition_raw``, ``condition_tokens``,
    ``concentration_value``, ``concentration_unit``, ``exposure_hours``,
    ``replicate``, ``imaging_mode``, ``acquisition_date_yyyymmdd``,
    ``acquisition_run_sequence``.
    """
    base = basename(source)
    stem = source_stem(base)

    date_token, run_sequence = extract_timestamp_and_seq(stem)

    core = stem
    if date_token is not None and run_sequence is not None:
        suffix = f"_{date_token}_{run_sequence}"
        if core.endswith(suffix):
            core = core[: -len(suffix)]

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

    if tokens and re.fullmatch(r"\d+[A-Za-z]?", tokens[-1]):
        replicate = tokens[-1]
        tokens = tokens[:-1]

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

    condition_tokens = list(tokens)
    for i, tok in enumerate(tokens):
        val, unit = _parse_concentration(tok)
        if val is not None:
            concentration_value = val
            concentration_unit = unit
            condition_tokens = tokens[:i] + tokens[i + 1:]
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


__all__ = [
    "basename",
    "source_stem",
    "normalize_key",
    "extract_timestamp_and_seq",
    "strip_timestamp_suffix",
    "source_match_keys",
    "path_match_keys",
    "parse_source_name",
]
