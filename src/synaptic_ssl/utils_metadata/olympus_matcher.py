"""Match source-image basenames to ``.oex`` / ``.vsi`` metadata files on disk.

Backs the ``--metadata_root`` enrichment in ``gather_experiment_metadata``:
given a list of flagged source filenames and one or more roots to scan,
find the best-matching ``.oex``/``.vsi`` for each source and call the
Olympus extractor on it.

The actual file extraction is delegated to
``imaging.extract_olympus_metadata.extract`` (legacy API: returns
``{"file": ..., "metadata": ..., "_error": ...}``). That module is
imported lazily and falls back to a structured error if missing, so this
package can be imported without it.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from .source_names import path_match_keys, source_match_keys, source_stem, basename

logger = logging.getLogger(__name__)


# Same priority used by sessions.py for representative-file selection.
_METADATA_EXT_PRIORITY = [".oex", ".vsi"]


def _metadata_ext_rank(path: Path) -> int:
    try:
        return _METADATA_EXT_PRIORITY.index(path.suffix.lower())
    except ValueError:
        return len(_METADATA_EXT_PRIORITY) + 1


def collect_metadata_files(metadata_roots: list[Path]) -> list[Path]:
    """Recursively scan *metadata_roots* for ``.oex`` and ``.vsi`` files."""
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


def build_metadata_index(metadata_files: list[Path]) -> dict[str, dict[str, list[Path]]]:
    """Build a five-key lookup index over *metadata_files*.

    Keys mirror :func:`source_names.source_match_keys`: ``stem``,
    ``core``, ``stem_norm``, ``core_norm``, ``seq_key``.
    """
    index: dict[str, dict[str, list[Path]]] = {
        "stem": defaultdict(list),
        "core": defaultdict(list),
        "stem_norm": defaultdict(list),
        "core_norm": defaultdict(list),
        "seq_key": defaultdict(list),
    }
    for path in metadata_files:
        keys = path_match_keys(path)
        for key_name, key_val in keys.items():
            if key_val:
                index[key_name][key_val].append(path)
    return index


def pick_best_metadata_candidate(candidates: list[Path], source: str) -> Path:
    """Tie-break among *candidates*: prefer .oex over .vsi, then exact stem."""
    stem = source_stem(source)

    def score(path: Path) -> tuple[int, int, int, str]:
        ext_rank = _metadata_ext_rank(path)
        same_stem = 0 if path.stem == stem else 1
        same_base = 0 if path.name.startswith(stem) else 1
        return (ext_rank, same_stem, same_base, str(path).lower())

    return sorted(candidates, key=score)[0]


def match_olympus_file_for_source(
    source: str,
    metadata_index: dict[str, dict[str, list[Path]]],
) -> tuple[Path | None, str | None, int]:
    """Return ``(picked_path, matcher_name, candidate_count)`` for *source*."""
    keys = source_match_keys(source)
    ordered_matchers = ["stem", "core", "stem_norm", "core_norm", "seq_key"]

    for matcher in ordered_matchers:
        key_val = keys.get(matcher)
        if not key_val:
            continue
        candidates = metadata_index[matcher].get(key_val, [])
        if not candidates:
            continue
        picked = pick_best_metadata_candidate(candidates, source)
        return picked, matcher, len(candidates)

    return None, None, 0


def _load_extract_olympus():
    """Lazy import of the legacy ``imaging.extract_olympus_metadata.extract``.

    Returns the callable, or a sentinel that produces a structured error
    in the same shape so callers don't need to special-case the missing
    backend.
    """
    try:
        from imaging.extract_olympus_metadata import extract  # type: ignore
        return extract
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "imaging.extract_olympus_metadata is unavailable; "
            "Olympus enrichment will be skipped per source. (%s)", msg,
        )

        def _missing(path: Path) -> dict[str, Any]:
            return {
                "file": {"path": str(path), "suffix": path.suffix.lower()},
                "metadata": None,
                "_error": msg,
            }

        return _missing


def extract_metadata_cached(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Memoize Olympus extraction per absolute path within *cache*."""
    path_key = str(path.resolve())
    if path_key in cache:
        return cache[path_key]
    extract = _load_extract_olympus()
    cache[path_key] = extract(path)
    return cache[path_key]


def enrich_with_olympus_metadata(
    source: str,
    metadata_index: dict[str, dict[str, list[Path]]],
    extraction_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Match *source* to an .oex/.vsi file and run the Olympus extractor.

    Returns a dict suitable for the ``olympus_meta`` field consumed by
    :func:`experiment_report.source_row`.
    """
    matched_path, matched_by, candidate_count = match_olympus_file_for_source(
        source, metadata_index,
    )
    if matched_path is None:
        return {
            "found": False,
            "matched_by": None,
            "candidate_count": 0,
            "metadata_file": None,
            "metadata": None,
            "error": "No matching .oex/.vsi file found",
        }

    extracted = extract_metadata_cached(matched_path, extraction_cache)
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


__all__ = [
    "collect_metadata_files",
    "build_metadata_index",
    "pick_best_metadata_candidate",
    "match_olympus_file_for_source",
    "extract_metadata_cached",
    "enrich_with_olympus_metadata",
]
