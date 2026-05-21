#!/usr/bin/env python
"""Image-level UMAP visualisation for an existing embeddings bundle.

Generates *only* the per-image (mean-pooled) UMAP figures without
re-running the heavy bootstrap / Leiden / statistics pipeline.

Each input image contributes a single point at the UMAP-2D projection
of its mean-pooled patch embedding. This is the visualisation that
matches the group-level statistical tests (MMD, PERMANOVA), where the
image is the statistical unit — see Caicedo et al., *Nat Methods* 2017.

Outputs (written into ``<bundle>/clustering/``):

* ``Z2_image.npy``                       — (n_images, 2) float32
* ``Z2_image_names.npy``                 — row order
* ``plots/umap_groups_image_level.png``  — overlaid scatter
* ``plots/umap_groups_image_level_facet.png`` — one panel per group

Example::

    python scripts/plot_image_level_umap.py \\
        --bundle data/embeddings/simmim_vicreg_pretrain_scratch_*/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from synaptic_ssl.clustering.pipeline import (  # noqa: E402
    ClusterCfg,
    auto_pca_dim,
    build_group_map_from_patterns,
    fit_umap_2d,
    l2_then_pca_whiten,
    load_group_rules,
    per_image_mean_embeddings,
)
from synaptic_ssl.clustering.viz import plot_2d_groups  # noqa: E402

log = logging.getLogger("plot_image_level_umap")


def _build_group_map(df: pd.DataFrame, repo: Path) -> dict[str, str]:
    """Same precedence as run_clustering.py: metadata column > bundle JSON > repo JSON."""
    if "treatment_group" in df.columns:
        tg = df["treatment_group"].astype(str)
        if (tg != "UNKNOWN").any():
            pairs = (
                df[tg != "UNKNOWN"][["source_image", "treatment_group"]]
                .drop_duplicates()
            )
            return {
                str(r.source_image): str(r.treatment_group)
                for r in pairs.itertuples(index=False)
            }
    # JSON fallbacks
    candidates = [
        Path("group_patterns.json"),  # bundle-local, resolved relatively below
        repo / "configs" / "clustering" / "group_patterns.json",
    ]
    for p in candidates:
        if p.is_file():
            rules = load_group_rules(p)
            return build_group_map_from_patterns(
                df["source_image"].tolist(), **rules,
            )
    return {}


def _process_bundle(bundle: Path, *, seed: int = 0) -> None:
    log.info("=== %s ===", bundle)
    emb_p = bundle / "embeddings.npy"
    meta_p = bundle / "metadata.csv"
    if not emb_p.is_file() or not meta_p.is_file():
        log.error("missing embeddings.npy or metadata.csv in %s — skipping", bundle)
        return

    Z = np.load(emb_p)
    df = pd.read_csv(meta_p)
    if len(df) != len(Z):
        raise ValueError(
            f"metadata length {len(df)} != embeddings length {len(Z)}"
        )
    source_images = df["source_image"].astype(str).to_numpy()
    log.info("loaded %d patch embeddings (dim=%d), %d unique images",
             len(Z), Z.shape[1], len(np.unique(source_images)))

    # group map — same precedence as run_clustering.py
    bundle_patterns = bundle / "group_patterns.json"
    if bundle_patterns.is_file():
        # local file takes precedence over repo if metadata column missing
        # (matches run_clustering.py behaviour)
        pass
    group_map = _build_group_map(df, _REPO)
    if not group_map:
        log.warning("no group_map resolved — group-coloured plots will be empty")
    log.info("group_map covers %d images (%d distinct groups)",
             len(group_map), len(set(group_map.values())))

    # L2 + PCA whitening — same preprocessing as run_clustering.py
    cfg = ClusterCfg(seed=seed)
    d_pca = auto_pca_dim(
        Z, target_var=cfg.pca_dim_target_var, hard_cap=cfg.pca_dim_max,
    )
    log.info("auto PCA dim: d=%d (target_var=%.3f)", d_pca, cfg.pca_dim_target_var)
    P, _pca_obj, cum_var = l2_then_pca_whiten(Z, n_components=d_pca, seed=cfg.seed)
    log.info("P: shape=%s, cum_explained_var=%.4f", P.shape, float(cum_var))

    # mean-pool per image
    M, image_names = per_image_mean_embeddings(P, source_images)
    log.info("per-image means: shape=%s", M.shape)

    # UMAP-2D on image means
    log.info("fitting UMAP-2D on %d image means", len(M))
    Z2_image = fit_umap_2d(M, cfg=cfg, seed=cfg.seed)

    # write artifacts
    out_dir = bundle / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Z2_image.npy", Z2_image)
    np.save(out_dir / "Z2_image_names.npy", image_names)

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if group_map:
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            plot_2d_groups(
                Z2_image, image_names, group_map,
                ax=ax,
                title=f"UMAP-2D — image-level mean embeddings\n{bundle.name}",
                point_size=30.0, alpha=0.85,
                max_points_per_group=None,
            )
            fig.tight_layout()
            fig.savefig(plots_dir / "umap_groups_image_level.png", dpi=150)
            log.info("wrote %s", plots_dir / "umap_groups_image_level.png")
        finally:
            plt.close(fig)

        try:
            fig, _ = plot_2d_groups(
                Z2_image, image_names, group_map,
                facet=True, facet_ncols=4,
                point_size=30.0, alpha=0.85,
                max_points_per_group=None,
                title=f"UMAP-2D (image-level) — faceted by group\n{bundle.name}",
            )
            fig.savefig(plots_dir / "umap_groups_image_level_facet.png", dpi=150)
            plt.close(fig)
            log.info("wrote %s", plots_dir / "umap_groups_image_level_facet.png")
        except Exception as exc:
            log.warning("facet plot failed: %s", exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="plot_image_level_umap",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "bundles", nargs="*", type=Path,
        help="bundle directories (must contain embeddings.npy + metadata.csv); "
             "default: every subdirectory of data/embeddings/",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.bundles:
        bundles = [Path(b).resolve() for b in args.bundles]
    else:
        root = _REPO / "data" / "embeddings"
        bundles = sorted(p for p in root.iterdir() if p.is_dir())
        log.info("no bundles given — found %d under %s", len(bundles), root)

    for b in bundles:
        try:
            _process_bundle(b, seed=args.seed)
        except Exception as exc:  # pragma: no cover
            log.exception("bundle %s failed: %s", b, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
