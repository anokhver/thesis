"""Shared path safety helpers used by the run-artifact services.

Centralised so that any future security tightening (e.g. handling
junctions, drive-letter normalisation) lives in one place rather
than four near-duplicate `_is_within` implementations.
"""
from __future__ import annotations

from pathlib import Path


def is_within(child: Path, parent: Path) -> bool:
    """True iff ``child`` resolves to a location inside ``parent``.

    Returns False on any I/O or value error from ``resolve()``
    (long paths, missing permissions, traversal artifacts, …) so
    callers can use this as a pure boolean guard.
    """
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True
