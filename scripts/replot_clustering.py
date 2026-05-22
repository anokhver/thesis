#!/usr/bin/env python3
"""Re-render all clustering plots from a saved bundle's ``clustering/`` dir.

Loads only the artefacts that ``scripts/run_clustering.py`` already wrote
(``labels.npy``, ``freq.npy``, ``counts.npy``, ``Z2.npy``, ``Z3.npy``,
``Z2_image.npy``, ``image_names.npy``, ``cluster_ids.npy``,
``patch_labels.csv``, ``image_groups.csv``, ``summary.json``) and re-runs
**only the plotting layer** of :mod:`synaptic_ssl.clustering.viz`. No
embeddings, no Leiden, no statistics are recomputed.

Usage::

    # one bundle
    python scripts/replot_clustering.py \\
        --bundle data/embeddings/<run_name>

    # several bundles in one call
    python scripts/replot_clustering.py \\
        --bundle data/embeddings/A data/embeddings/B data/embeddings/C

    # arbitrary clustering output dir (not under <bundle>/clustering)
    python scripts/replot_clustering.py \\
        --output-dir path/to/clustering

By default plots overwrite ``<clustering_dir>/plots/*.png``. Use
``--plots-subdir`` to send them to a sibling dir
(e.g. ``--plots-subdir plots_v2``) so the originals are kept.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable when run from the repo root.
_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from synaptic_ssl.clustering.viz import (  # noqa: E402
    plot_2d_clusters,
    plot_2d_groups,
    plot_3d_clusters,
    plot_3d_groups,
    plot_group_cluster_heatmap,
    plot_group_frequency_boxplots,
    plot_image_cluster_heatmap,
    plot_image_level_umap,
)


log = logging.getLogger("replot_clustering")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_npy(path: Path):
    return np.load(path, allow_pickle=False) if path.exists() else None


def _resolve_clustering_dir(arg: str | Path) -> Path:
    """Accept either a bundle dir (then add /clustering) or a clustering dir."""
    p = Path(arg).resolve()
    if (p / "labels.npy").exists():
        return p
    if (p / "clustering" / "labels.npy").exists():
        return p / "clustering"
    raise FileNotFoundError(
        f"no labels.npy under {p} or {p / 'clustering'}; "
        f"point --bundle / --output-dir at a clustering output directory"
    )


def _load_kw_q(summary_path: Path) -> np.ndarray | None:
    """Pull per-cluster Kruskal-Wallis corrected p-values from summary.json."""
    if not summary_path.exists():
        return None
    try:
        with summary_path.open() as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("could not parse %s: %s", summary_path, exc)
        return None
    kw = data.get("per_cluster_kw") or {}
    if isinstance(kw, dict):
        q = kw.get("p_values_corrected")
        if q is not None:
            return np.asarray(q, dtype=np.float64)
    return None


def _build_group_map(
    image_groups_csv: Path,
    *,
    pool_qualifiers: bool = False,
) -> dict[str, str]:
    """Load image -> group mapping from ``image_groups.csv``.

    When ``pool_qualifiers`` is True, strip everything after the first
    ``" | "`` so dose / time / washout variants collapse onto their base
    treatment label, e.g.::

        "Control | 24h"                          -> "Control"
        "Control | 48h"                          -> "Control"
        "Psilocin | 10 uM | 48h | 1h pulse"      -> "Psilocin"
        "Norbaeocystin + harmine | 24h"          -> "Norbaeocystin + harmine"

    This is a *plot-time* remap. The on-disk ``image_groups.csv`` and
    ``summary.json`` are not modified, so the original fine-grained
    labels remain available for downstream analyses.
    """
    if not image_groups_csv.exists():
        return {}
    df = pd.read_csv(image_groups_csv)
    # drop empty / NaN group entries -- matches the pipeline's semantics
    df = df[df["group"].notna() & (df["group"].astype(str).str.len() > 0)]
    gmap = {str(r.source_image): str(r.group) for r in df.itertuples()}
    if pool_qualifiers:
        gmap = {img: g.split(" | ", 1)[0].strip() for img, g in gmap.items()}
    return gmap


def _build_group_counts(
    counts: np.ndarray,
    image_names: np.ndarray,
    group_map: dict[str, str],
) -> tuple[np.ndarray, list[str]]:
    """Aggregate per-image counts into a (n_groups, n_clusters) table."""
    groups = sorted(set(group_map.values()))
    if not groups:
        return np.zeros((0, counts.shape[1]), dtype=np.int64), []
    g2i = {g: i for i, g in enumerate(groups)}
    group_counts = np.zeros((len(groups), counts.shape[1]), dtype=np.int64)
    for r, img in enumerate(image_names):
        g = group_map.get(str(img))
        if g is None:
            continue
        group_counts[g2i[g]] += counts[r]
    return group_counts, groups


# ---------------------------------------------------------------------------
# Plot one bundle
# ---------------------------------------------------------------------------
def replot_one(
    clustering_dir: Path, *,
    plots_subdir: str = "plots",
    pool_qualifiers: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    plots_dir = clustering_dir / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] loading arrays", clustering_dir)
    labels       = np.load(clustering_dir / "labels.npy")
    freq         = np.load(clustering_dir / "freq.npy")
    counts       = np.load(clustering_dir / "counts.npy")
    image_names  = np.load(clustering_dir / "image_names.npy", allow_pickle=True)
    cluster_ids  = np.load(clustering_dir / "cluster_ids.npy")

    Z2           = _load_npy(clustering_dir / "Z2.npy")
    Z3           = _load_npy(clustering_dir / "Z3.npy")
    Z2_image     = _load_npy(clustering_dir / "Z2_image.npy")
    Z3_image     = _load_npy(clustering_dir / "Z3_image.npy")
    Z_img_names  = (np.load(clustering_dir / "Z2_image_names.npy",
                            allow_pickle=True)
                    if (clustering_dir / "Z2_image_names.npy").exists()
                    else None)

    # source_images per patch comes from patch_labels.csv (which is row-
    # aligned with labels.npy in run_clustering's writeback).
    patch_csv = clustering_dir / "patch_labels.csv"
    if patch_csv.exists():
        patch_df = pd.read_csv(patch_csv)
        source_images = patch_df["source_image"].to_numpy()
    else:
        source_images = None

    group_map = _build_group_map(
        clustering_dir / "image_groups.csv",
        pool_qualifiers=pool_qualifiers,
    )
    kw_q = _load_kw_q(clustering_dir / "summary.json")
    if pool_qualifiers and kw_q is not None:
        # Pooling changes the group definitions, so a per-cluster q-value
        # computed against the un-pooled groups no longer applies. Drop it
        # rather than annotate misleadingly.
        log.info("pool_qualifiers=True -> discarding stored kw_q (computed pre-pooling)")
        kw_q = None

    log.info(
        "[%s] N=%d patches, K=%d clusters, %d images, %d groups, "
        "pool_qualifiers=%s, kw_q=%s",
        clustering_dir, labels.size, len(cluster_ids), len(image_names),
        len(set(group_map.values())) if group_map else 0,
        pool_qualifiers,
        "yes" if kw_q is not None else "no",
    )

    def _save(fig, name):
        fig.savefig(plots_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ----- per-image cluster heatmap (always available) ------------------
    try:
        ax = plot_image_cluster_heatmap(freq, image_names, cluster_ids)
        _save(ax.figure, "image_cluster_heatmap.png")
    except Exception as exc:
        log.warning("image_cluster_heatmap failed: %s", exc)

    # ----- patch-level UMAP-2D coloured by cluster ----------------------
    if Z2 is not None:
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            plot_2d_clusters(Z2, labels, ax=ax)
            fig.tight_layout()
            _save(fig, "umap_clusters.png")
        except Exception as exc:
            log.warning("umap_clusters failed: %s", exc)

        # patch-level groups (overlaid + faceted)
        if group_map and source_images is not None:
            try:
                fig, ax = plt.subplots(figsize=(7, 6))
                plot_2d_groups(Z2, source_images, group_map, ax=ax)
                fig.tight_layout()
                _save(fig, "umap_groups.png")
            except Exception as exc:
                log.warning("umap_groups failed: %s", exc)
            try:
                fig, _ = plot_2d_groups(
                    Z2, source_images, group_map,
                    facet=True, facet_ncols=4,
                )
                _save(fig, "umap_groups_facet.png")
            except Exception as exc:
                log.warning("umap_groups_facet failed: %s", exc)

    # ----- image-level UMAP-2D (mean embeddings) -------------------------
    if Z2_image is not None and Z_img_names is not None:
        # un-coloured fallback (always works)
        try:
            ax = plot_image_level_umap(freq, image_names)
            _save(ax.figure, "image_level_umap.png")
        except Exception as exc:
            log.warning("image_level_umap failed: %s", exc)

        if group_map:
            try:
                fig, ax = plt.subplots(figsize=(7, 6))
                plot_2d_groups(
                    Z2_image, Z_img_names, group_map, ax=ax,
                    title="UMAP-2D — one point per image (mean embedding)",
                    point_size=30.0, alpha=0.85,
                    max_points_per_group=None,
                )
                fig.tight_layout()
                _save(fig, "umap_groups_image_level.png")
            except Exception as exc:
                log.warning("umap_groups_image_level failed: %s", exc)
            try:
                fig, _ = plot_2d_groups(
                    Z2_image, Z_img_names, group_map,
                    facet=True, facet_ncols=4,
                    point_size=30.0, alpha=0.85,
                    max_points_per_group=None,
                    title="UMAP-2D (image-level) — faceted by group",
                )
                _save(fig, "umap_groups_image_level_facet.png")
            except Exception as exc:
                log.warning("umap_groups_image_level_facet failed: %s", exc)

    # ----- UMAP-3D ------------------------------------------------------
    if Z3 is not None:
        try:
            fig = plt.figure(figsize=(8, 7))
            plot_3d_clusters(Z3, labels, fig=fig)
            fig.tight_layout()
            _save(fig, "umap3d_clusters.png")
        except Exception as exc:
            log.warning("umap3d_clusters failed: %s", exc)

        if group_map and source_images is not None:
            try:
                fig = plt.figure(figsize=(8, 7))
                plot_3d_groups(Z3, source_images, group_map, fig=fig)
                fig.tight_layout()
                _save(fig, "umap3d_groups.png")
            except Exception as exc:
                log.warning("umap3d_groups failed: %s", exc)

        if (
            Z3_image is not None and Z_img_names is not None
            and group_map
        ):
            try:
                fig = plt.figure(figsize=(8, 7))
                plot_3d_groups(
                    Z3_image, Z_img_names, group_map, fig=fig,
                    title="UMAP-3D — one point per image (mean embedding)",
                    point_size=30.0, alpha=0.85,
                    max_points_per_group=None,
                )
                fig.tight_layout()
                _save(fig, "umap3d_groups_image_level.png")
            except Exception as exc:
                log.warning("umap3d_groups_image_level failed: %s", exc)

    # ----- group × cluster heatmap --------------------------------------
    if group_map:
        gc, gn = _build_group_counts(counts, image_names, group_map)
        if gn and gc.size > 0:
            try:
                fig, ax = plt.subplots()
                plot_group_cluster_heatmap(gc, gn, cluster_ids, ax=ax)
                _save(fig, "group_heatmap.png")
            except Exception as exc:
                log.warning("group_heatmap failed: %s", exc)

        # ----- per-cluster boxplots (faceted, with kw q-stars) ----------
        try:
            fig = plot_group_frequency_boxplots(
                freq, image_names, group_map, cluster_ids,
                q_values=kw_q,
            )
            _save(fig, "group_boxplots.png")
        except Exception as exc:
            log.warning("group_boxplots failed: %s", exc)

    log.info("[%s] wrote plots -> %s", clustering_dir, plots_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="Re-render clustering plots from saved arrays "
                    "(no recomputation).",
    )
    p.add_argument(
        "--bundle", "-b", nargs="+", default=None,
        help="One or more bundle directories (or clustering directories). "
             "If a bundle is given, '<bundle>/clustering' is used.",
    )
    p.add_argument(
        "--output-dir", "-o", default=None,
        help="Alternative: an explicit clustering output directory. "
             "Mutually exclusive with --bundle.",
    )
    p.add_argument(
        "--plots-subdir", default="plots",
        help="Subdir under each clustering directory to write PNGs into. "
             "Default: 'plots' (overwrites the existing plots).",
    )
    p.add_argument(
        "--pool-qualifiers", action="store_true",
        help="Plot-time only: strip dose / time / washout qualifiers "
             "(everything after the first ' | ') so e.g. 'Control | 24h', "
             "'Control | 48h', 'Control | 72h' all plot as 'Control'. The "
             "saved image_groups.csv / summary.json are not modified. "
             "Recommend --plots-subdir plots_pooled to keep the original "
             "per-qualifier plots side-by-side.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    if (args.bundle is None) == (args.output_dir is None):
        log.error("specify exactly one of --bundle / --output-dir")
        return 2

    targets = args.bundle if args.bundle is not None else [args.output_dir]
    rc = 0
    for t in targets:
        try:
            cd = _resolve_clustering_dir(t)
            replot_one(
                cd,
                plots_subdir=args.plots_subdir,
                pool_qualifiers=args.pool_qualifiers,
            )
        except Exception as exc:
            log.error("failed for %s: %s", t, exc)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
