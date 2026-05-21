"""Treatment-group classification from source-image filenames.

First-match-wins substring patterns loaded from a JSON config so the
same mapping is shared between the notebook and CLI extractors. Place
more specific patterns before more general ones (e.g. ``PSYHARMIN``
before ``PSY``).
"""
from __future__ import annotations

from typing import Iterable, Sequence


def load_group_patterns(path) -> list[tuple[str, str]]:
    """Load ``[(pattern, group_label), ...]`` from a JSON config file.

    Schema::

        {
            "patterns": [
                ["PATTERN_UPPER", "Display group name"],
                ...
            ]
        }

    First-match-wins; list order is significant — place more specific
    patterns before more general ones (e.g. ``PSYHARMIN`` before
    ``PSY``). The loader uppercases the pattern for case-insensitive
    matching.
    """
    import json
    from pathlib import Path
    with open(Path(path)) as f:
        data = json.load(f)
    if "patterns" not in data or not isinstance(data["patterns"], list):
        raise ValueError(
            f"{path}: expected a top-level 'patterns' list of [pattern, label] pairs"
        )
    out: list[tuple[str, str]] = []
    for entry in data["patterns"]:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            raise ValueError(
                f"{path}: each entry in 'patterns' must be a [pattern, label] pair, got {entry!r}"
            )
        pat, label = entry
        out.append((str(pat).upper(), str(label)))
    return out


def classify_image_by_patterns(
    name: str,
    patterns: Sequence[tuple[str, str]],
) -> str | None:
    """First-match-wins lookup of a treatment group from a filename."""
    upper = str(name).upper()
    for pattern, group in patterns:
        if pattern in upper:
            return group
    return None


def build_group_map_from_patterns(
    source_images: Iterable[str],
    patterns: Sequence[tuple[str, str]],
) -> dict[str, str]:
    """Build ``{source_image -> group_label}`` from a pattern list."""
    gm: dict[str, str] = {}
    for img in {str(s) for s in source_images}:
        g = classify_image_by_patterns(img, patterns)
        if g is not None:
            gm[img] = g
    return gm
