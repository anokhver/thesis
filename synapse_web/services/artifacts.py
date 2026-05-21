"""Helpers for surfacing per-run ML artefacts in the UI.

Pipeline writes plots under ``MEDIA_ROOT/runs/<run.id>/output/plots/``.
When that location is itself inside ``MEDIA_ROOT``, Django's dev media
server can serve the images directly. ``list_plots`` returns the same
metadata either way — entries gain a ``url=None`` when the file exists
on disk but is not web-served, so views/templates can show an honest
"plots are on disk but not URL-addressable" message instead of
silently dropping them.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urljoin

from django.conf import settings

logger = logging.getLogger(__name__)

# Stable label overrides for the 5 known plot files run_clustering emits.
_KNOWN_PLOT_LABELS = {
    "umap_clusters": "UMAP coloured by cluster",
    "umap_groups": "UMAP coloured by treatment group",
    "resolution_stability": "Resolution stability",
    "group_heatmap": "Cluster × group heatmap",
    "group_boxplots": "Per-cluster boxplots by group",
}

# Stable categorical palette used for the dominant-cluster chip in the
# per-source-image table. Cluster id -1 (noise) always uses _NOISE_COLOR.
# Chosen for reasonable contrast on the light-purple theme.
_PALETTE = (
    "#7c4dff", "#26a69a", "#ef5350", "#ffb300", "#42a5f5",
    "#ab47bc", "#66bb6a", "#ec407a", "#5c6bc0", "#ffa726",
    "#26c6da", "#8d6e63",
)
_NOISE_COLOR = "#9e9e9e"


# ---------------------------------------------------------------------------
# URL composition
# ---------------------------------------------------------------------------

def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def media_url_for(abs_path: Path) -> str | None:
    """Return the public media URL for ``abs_path`` or ``None``.

    ``None`` means the path exists on disk but lives outside
    ``MEDIA_ROOT`` (e.g. operator set ``RUNS_DIR`` to a different
    location), so it isn't served by Django's dev media handler.
    """
    media_root = Path(settings.MEDIA_ROOT)
    try:
        resolved = Path(abs_path).resolve()
        rel = resolved.relative_to(media_root.resolve())
    except (ValueError, OSError):
        return None
    base = str(settings.MEDIA_URL or "/media/").rstrip("/") + "/"
    return urljoin(base, rel.as_posix())


def _pretty_label(stem: str) -> str:
    if stem in _KNOWN_PLOT_LABELS:
        return _KNOWN_PLOT_LABELS[stem]
    return stem.replace("_", " ").replace("-", " ").strip().title() or stem


# ---------------------------------------------------------------------------
# Plot discovery
# ---------------------------------------------------------------------------

def list_plots(run) -> list[dict]:
    """Return per-plot metadata for ``run.output_dir/plots/*.png``.

    Skips symlinks (both the directory and individual files). Entries
    have ``url=None`` when the file is real on disk but not web-served.
    Sorted by filename for deterministic UI.
    """
    if not run.output_dir:
        return []
    output_dir = Path(run.output_dir)
    plots_dir = output_dir / "plots"
    try:
        if not plots_dir.is_dir() or plots_dir.is_symlink():
            return []
    except OSError:
        return []

    entries: list[dict] = []
    try:
        children = sorted(plots_dir.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []

    for entry in children:
        name = entry.name
        if name.startswith("."):
            continue
        try:
            if entry.is_symlink() or not entry.is_file():
                continue
        except OSError:
            continue
        if entry.suffix.lower() != ".png":
            continue
        try:
            mtime = int(entry.stat().st_mtime)
        except OSError:
            mtime = 0
        url = media_url_for(entry)
        if url is not None:
            # Bust browser caches when a rerun replaces this file.
            url = f"{url}?v={mtime}"
        entries.append({
            "name": entry.stem,
            "filename": name,
            "path": str(entry),
            "url": url,
            "label": _pretty_label(entry.stem),
            "mtime": mtime,
        })
    return entries


# ---------------------------------------------------------------------------
# Cluster colour map
# ---------------------------------------------------------------------------

def cluster_palette(n: int) -> list[str]:
    if n <= 0:
        return []
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def cluster_color_map(stats) -> dict[int, str]:
    """Map each distinct dominant-cluster id seen in ``stats`` to a hex.

    `-1` (noise) always maps to a fixed neutral grey; non-noise ids are
    sorted ascending and cycle the palette so the chip colour is stable
    for a given (run, cluster id) pair.
    """
    seen: set[int] = set()
    for s in stats:
        cid = getattr(s, "dominant_cluster", None)
        if cid is not None:
            seen.add(int(cid))
    non_noise = sorted(c for c in seen if c != -1)
    mapping: dict[int, str] = {
        cid: _PALETTE[i % len(_PALETTE)]
        for i, cid in enumerate(non_noise)
    }
    if -1 in seen:
        mapping[-1] = _NOISE_COLOR
    return mapping


# ---------------------------------------------------------------------------
# Convenience for views
# ---------------------------------------------------------------------------

def categorize_plots(plots: list[dict]) -> dict[str, dict | None]:
    """Index a `list_plots` result by stem for convenient template access."""
    by_name = {p["name"]: p for p in plots}
    return {
        "umap_clusters": by_name.get("umap_clusters"),
        "umap_groups": by_name.get("umap_groups"),
        "resolution_stability": by_name.get("resolution_stability"),
        "group_heatmap": by_name.get("group_heatmap"),
        "group_boxplots": by_name.get("group_boxplots"),
    }


def output_dir_present(run) -> bool:
    if not run.output_dir:
        return False
    try:
        return Path(run.output_dir).is_dir()
    except OSError:
        return False
