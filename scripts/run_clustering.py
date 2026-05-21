#!/usr/bin/env python
"""End-to-end clustering + group-statistics over an embeddings bundle.

This is a thin orchestrator that wires together the reusable building
blocks in :mod:`synaptic_ssl.clustering` into a headless, CLI-friendly
runner suitable for cluster submission.

Inputs (produced by ``scripts/extract_full_embeddings.py``):

* ``<bundle>/embeddings.npy``       — (N, D) float32 patch embeddings
* ``<bundle>/metadata.csv``         — N rows; columns include ``filename``,
                                     ``source_image``, ``treatment_group``
* ``<bundle>/manifest.json``        — checkpoint / extraction provenance
* ``<bundle>/group_patterns.json``  — (optional) treatment-group patterns

Outputs are written to ``<bundle>/clustering/`` (or ``--output-dir``):

* ``labels.npy``                  — int64 cluster id per patch (-1 = noise)
* ``patch_labels.csv``            — filename, source_image, cluster
* ``freq.npy`` / ``counts.npy``   — (n_images, n_clusters) freq / counts
* ``image_names.npy``             — row order of freq / counts
* ``cluster_ids.npy``             — column order of freq / counts
* ``Z2.npy``                      — UMAP-2D layout (visualisation only)
* ``summary.json``                — full statistical results
* ``README.md``                   — human-readable summary
* ``plots/*.png``                 — figures (suppress with ``--no-plots``)

Example::

    python scripts/run_clustering.py \\
        --bundle data/embeddings/dinov2_pancreas_ep200/ \\
        --control-pattern DMSO --plots
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo-relative imports (works whether installed or run from source).
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from synaptic_ssl.clustering.pipeline import (  # noqa: E402
    ClusterCfg,
    adaptive_leiden_k,
    auto_pca_dim,
    bootstrap_stability,
    build_group_map_from_patterns,
    chi2_independence,
    cluster_medoids,
    cluster_purity_by_image,
    cluster_size_stats,
    fit_umap_2d,
    group_cluster_test,
    image_level_cv,
    internal_validity_indices,
    l2_then_pca_whiten,
    load_group_patterns,
    pairwise_mmd_groups,
    pairwise_permanova_groups,
    per_cluster_kruskal_wallis,
    per_image_cluster_frequencies,
    per_image_mean_embeddings,
    permanova_frequencies,
    permutation_null_ari,
    pick_resolution,
    subtract_control_mean,
)

log = logging.getLogger("run_clustering")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _to_json_safe(obj):
    """Recursively convert numpy/pandas/NamedTuple values to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]
    # NamedTuple
    if hasattr(obj, "_asdict"):
        return _to_json_safe(obj._asdict())
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        return _to_json_safe(asdict(obj))
    return str(obj)


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_json_safe(payload), f, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# Group-map construction
# ---------------------------------------------------------------------------

def build_group_map(
    df: pd.DataFrame,
    *,
    patterns_path: Path | None,
    repo: Path,
) -> tuple[dict[str, str], str]:
    """Return ``(group_map, source_description)``.

    Preference order:
      1. ``metadata.csv:treatment_group`` (already classified by extractor).
      2. ``<bundle>/group_patterns.json``.
      3. ``<repo>/configs/clustering/group_patterns.json``.
    """
    # 1. Use the pre-classified column if present and non-trivial.
    if "treatment_group" in df.columns:
        tg = df["treatment_group"].astype(str)
        non_unknown = tg[tg != "UNKNOWN"]
        if len(non_unknown) > 0:
            # one (image -> group) per source_image
            pairs = (
                df[tg != "UNKNOWN"][["source_image", "treatment_group"]]
                .drop_duplicates()
            )
            gm = {str(r.source_image): str(r.treatment_group)
                  for r in pairs.itertuples(index=False)}
            return gm, "metadata.csv:treatment_group"

    # 2./3. Fall back to pattern files.
    candidates: list[Path] = []
    if patterns_path is not None:
        candidates.append(Path(patterns_path))
    candidates.append(repo / "configs" / "clustering" / "group_patterns.json")

    for p in candidates:
        if p.is_file():
            patterns = load_group_patterns(p)
            gm = build_group_map_from_patterns(df["source_image"].tolist(), patterns)
            return gm, f"patterns:{p}"

    log.warning("No treatment_group column and no group_patterns.json found — "
                "group-level tests will be skipped.")
    return {}, "none"


# ---------------------------------------------------------------------------
# Plotting (lazy import so headless --no-plots works without matplotlib).
# ---------------------------------------------------------------------------

def render_plots(
    *,
    out_dir: Path,
    summary_sweep: dict,
    full_sweep: dict,
    Z2,
    labels,
    source_images,
    group_map,
    group_counts,
    group_names,
    cluster_ids,
    freq,
    image_names,
    kw_q,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from synaptic_ssl.clustering.viz import (
        plot_2d_clusters,
        plot_2d_groups,
        plot_group_cluster_heatmap,
        plot_group_frequency_boxplots,
        plot_resolution_stability,
    )

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        plot_resolution_stability(summary_sweep, full_sweep, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / "resolution_stability.png", dpi=150)
    finally:
        plt.close(fig)

    if Z2 is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            plot_2d_clusters(Z2, labels, ax=ax)
            fig.tight_layout()
            fig.savefig(plots_dir / "umap_clusters.png", dpi=150)
        finally:
            plt.close(fig)

        if group_map:
            fig, ax = plt.subplots(figsize=(7, 6))
            try:
                plot_2d_groups(Z2, source_images, group_map, ax=ax)
                fig.tight_layout()
                fig.savefig(plots_dir / "umap_groups.png", dpi=150)
            finally:
                plt.close(fig)

    if group_names and group_counts is not None and group_counts.size > 0:
        fig, ax = plt.subplots()
        try:
            plot_group_cluster_heatmap(group_counts, group_names, cluster_ids, ax=ax)
            fig.tight_layout()
            fig.savefig(plots_dir / "group_heatmap.png", dpi=150)
        finally:
            plt.close(fig)

    if group_map and freq.shape[0] > 0:
        try:
            fig = plot_group_frequency_boxplots(
                freq, image_names, group_map, cluster_ids,
                q_values=kw_q,
            )
            fig.savefig(plots_dir / "group_boxplots.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("group_boxplots failed: %s", exc)


# ---------------------------------------------------------------------------
# Bundle IO
# ---------------------------------------------------------------------------

def load_bundle(bundle: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    emb_p = bundle / "embeddings.npy"
    meta_p = bundle / "metadata.csv"
    if not emb_p.is_file():
        raise FileNotFoundError(f"missing {emb_p}")
    if not meta_p.is_file():
        raise FileNotFoundError(f"missing {meta_p}")
    Z = np.load(emb_p)
    df = pd.read_csv(meta_p)
    if len(df) != len(Z):
        raise ValueError(
            f"metadata length {len(df)} != embeddings length {len(Z)} in {bundle}"
        )
    manifest = {}
    man_p = bundle / "manifest.json"
    if man_p.is_file():
        with open(man_p) as f:
            manifest = json.load(f)
    return Z, df, manifest


# ---------------------------------------------------------------------------
# Optional extraction step
# ---------------------------------------------------------------------------

def _resolve_run_name(checkpoint: Path) -> str:
    """Mirror the same convention extract_full_embeddings.py uses: bundle
    folder name is the model run directory name (parent of the .pt file)."""
    p = checkpoint.resolve()
    if p.is_dir():
        return p.name
    return p.parent.name


def _ensure_bundle_from_checkpoint(args: argparse.Namespace) -> Path:
    """Run extract_full_embeddings.py if the target bundle is missing.

    Returns the resolved bundle directory.
    """
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    output_root = (Path(args.output_root).resolve() if args.output_root
                   else _REPO / "data" / "embeddings")
    run_name = args.run_name or _resolve_run_name(ckpt)
    bundle = (output_root / run_name).resolve()
    emb_path = bundle / "embeddings.npy"

    if emb_path.is_file() and not args.extract_overwrite:
        log.info("bundle already exists at %s — skipping extraction", bundle)
        return bundle

    extract_script = _REPO / "scripts" / "extract_full_embeddings.py"
    cmd: list[str] = [
        sys.executable, str(extract_script),
        "--checkpoint", str(ckpt),
        "--output-root", str(output_root),
        "--run-name", run_name,
        "--batch-size", str(args.extract_batch_size),
        "--num-workers", str(args.extract_num_workers),
    ]
    if args.data_root:
        cmd += ["--data-root", str(Path(args.data_root).resolve())]
    if args.exclude_dates:
        cmd += ["--exclude-dates", *args.exclude_dates]
    if args.exclude_patterns_file:
        cmd += [
            "--exclude-patterns-file",
            str(Path(args.exclude_patterns_file).resolve()),
        ]
    if args.extract_device:
        cmd += ["--device", args.extract_device]
    if args.extract_overwrite:
        cmd.append("--overwrite")

    log.info("running extraction: %s", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        raise RuntimeError(
            f"extract_full_embeddings.py failed with exit code {rc}"
        )
    if not emb_path.is_file():
        raise RuntimeError(
            f"extraction reported success but {emb_path} is missing"
        )
    return bundle


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    t_start = time.time()

    # Resolve bundle: either explicit --bundle or derived from --checkpoint
    # (in which case we run extraction if the bundle is missing).
    if args.bundle and args.checkpoint:
        log.error("pass either --bundle or --checkpoint, not both")
        return 2
    if not args.bundle and not args.checkpoint:
        log.error("either --bundle or --checkpoint is required")
        return 2

    if args.checkpoint:
        bundle = _ensure_bundle_from_checkpoint(args)
    else:
        bundle = Path(args.bundle).resolve()

    if not bundle.is_dir():
        log.error("bundle not found: %s", bundle)
        return 2

    out_dir = (Path(args.output_dir).resolve() if args.output_dir
               else bundle / "clustering")
    if out_dir.exists() and args.overwrite:
        log.warning("removing existing output dir %s", out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging to file ------------------------------------------------
    log_path = out_dir / "run_clustering.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    log.info("bundle:  %s", bundle)
    log.info("output:  %s", out_dir)

    # ---- 1. Load bundle -------------------------------------------------
    Z, df, manifest = load_bundle(bundle)
    log.info("embeddings: shape=%s dtype=%s", Z.shape, Z.dtype)
    if "source_image" not in df.columns:
        log.error("metadata.csv missing required column 'source_image'")
        return 2
    source_images = df["source_image"].astype(str).to_numpy()
    filenames = df["filename"].astype(str).to_numpy() if "filename" in df.columns \
        else np.array([str(i) for i in range(len(df))])

    # ---- 2. Group map ---------------------------------------------------
    group_map, group_src = build_group_map(
        df, patterns_path=Path(args.group_patterns) if args.group_patterns else None,
        repo=_REPO,
    )
    log.info("group_map: %d images mapped (source=%s)", len(group_map), group_src)

    # ---- 3. Config (CLI overrides) --------------------------------------
    cfg = ClusterCfg()
    overrides: dict = {"seed": args.seed}
    if args.resolutions:
        overrides["leiden_resolutions"] = tuple(
            float(x) for x in args.resolutions.split(",") if x.strip()
        )
    if args.leiden_k is not None:
        overrides["leiden_k"] = int(args.leiden_k)
        overrides["leiden_k_auto"] = False
    if args.pca_dim is not None:
        overrides["pca_dim"] = int(args.pca_dim)
    if args.control_pattern:
        overrides["control_reference_pattern"] = args.control_pattern
    if args.bootstrap_b is not None:
        overrides["bootstrap_b"] = int(args.bootstrap_b)
    if args.bootstrap_frac is not None:
        overrides["bootstrap_frac"] = float(args.bootstrap_frac)
    if args.permutation_p is not None:
        overrides["permutation_p"] = int(args.permutation_p)
    if args.permanova_n is not None:
        overrides["permanova_n_permutations"] = int(args.permanova_n)
    cfg = replace(cfg, **overrides)
    log.info("cfg: %s", _to_json_safe(asdict(cfg)))

    # ---- 4. Preprocessing -----------------------------------------------
    Z_proc = Z
    if cfg.control_reference_pattern:
        log.info("subtracting control mean (pattern=%r)",
                 cfg.control_reference_pattern)
        Z_proc = subtract_control_mean(
            Z_proc, source_images, cfg.control_reference_pattern,
        )

    if cfg.pca_dim is None:
        d_pca = auto_pca_dim(
            Z_proc, target_var=cfg.pca_dim_target_var, hard_cap=cfg.pca_dim_max,
        )
        log.info("auto PCA dim: d=%d (target_var=%.3f, cap=%d)",
                 d_pca, cfg.pca_dim_target_var, cfg.pca_dim_max)
    else:
        d_pca = int(cfg.pca_dim)
        log.info("fixed PCA dim: d=%d", d_pca)

    P, _pca_obj, cum_var = l2_then_pca_whiten(
        Z_proc, n_components=d_pca, seed=cfg.seed,
    )
    log.info("P: shape=%s, cum_explained_var=%.4f", P.shape, float(cum_var))

    # ---- 5. Adaptive k --------------------------------------------------
    k = adaptive_leiden_k(len(P), cfg.leiden_k) if cfg.leiden_k_auto else cfg.leiden_k
    log.info("leiden_k=%d (auto=%s)", k, cfg.leiden_k_auto)

    # ---- 6. Bootstrap stability sweep -----------------------------------
    log.info("bootstrap stability sweep: B=%d frac=%.2f resolutions=%s",
             cfg.bootstrap_b, cfg.bootstrap_frac, cfg.leiden_resolutions)
    summary_sweep, full_sweep = bootstrap_stability(
        P, cfg.leiden_resolutions, k, cfg.bootstrap_b, cfg.bootstrap_frac,
        seed=cfg.seed, source_images=source_images,
    )

    # ---- 7. Pick resolution ---------------------------------------------
    res_pick, mean_ari, std_ari, n_clust = pick_resolution(summary_sweep, full_sweep)
    labels = np.asarray(full_sweep[res_pick], dtype=np.int64)
    log.info("picked resolution=%.3f mean_ari=%.3f (±%.3f) K=%d",
             res_pick, mean_ari, std_ari, n_clust)

    # ---- 8. UMAP-2D (visualisation only) --------------------------------
    Z2 = None
    if not args.no_umap or not args.no_plots:
        log.info("fit UMAP-2D for visualisation")
        Z2 = fit_umap_2d(P, cfg=cfg, seed=cfg.seed)
        np.save(out_dir / "Z2.npy", Z2)

    # ---- 9. Sanity ------------------------------------------------------
    size_stats = cluster_size_stats(labels)
    log.info("size_stats: %s", size_stats)

    log.info("permutation-null ARI (P=%d)", cfg.permutation_p)
    perm_mean, perm_std, perm_max = permutation_null_ari(
        P, labels, cfg=cfg, resolution=res_pick, k=k,
        source_images=source_images,
    )

    log.info("internal validity indices")
    iv = internal_validity_indices(P, labels, seed=cfg.seed)

    log.info("image-level CV (repeats=%d)", cfg.cv_n_repeats)
    cv_med, cv_q25, cv_q75, cv_per = image_level_cv(
        P, source_images, labels, cfg=cfg,
        group_map=group_map if group_map else None,
    )

    log.info("cluster medoids (samples_per_cluster=%d)", cfg.samples_per_cluster)
    medoids = cluster_medoids(
        P, labels, samples_per_cluster=cfg.samples_per_cluster, seed=cfg.seed,
    )

    # ---- 10. Frequencies + within-image tests ---------------------------
    freq, image_names, cluster_ids, counts = per_image_cluster_frequencies(
        labels, source_images,
    )
    log.info("freq matrix: shape=%s (images=%d, clusters=%d)",
             freq.shape, len(image_names), len(cluster_ids))

    chi2_res = chi2_independence(
        counts, cfg.chi2_min_image_patches,
        n_permutations=cfg.chi2_n_permutations, seed=cfg.seed,
    )
    log.info("chi2 independence: stat=%.3f p=%.4g dof=%s dropped=%s method=%s",
             chi2_res[0], chi2_res[1], chi2_res[2], chi2_res[3], chi2_res[4])

    purity_rows = cluster_purity_by_image(labels, source_images)

    # ---- 11. Group-level tests ------------------------------------------
    group_test = None
    permanova = None
    pair_permanova = None
    kw = None
    mmd = None
    group_counts_arr = None
    group_names_list: list[str] = []
    kw_q = None

    if group_map and len(set(group_map.values())) >= 2:
        log.info("group_cluster_test (image-level permutation chi²)")
        group_test = group_cluster_test(
            counts, image_names, cluster_ids, group_map,
            n_permutations=max(cfg.permanova_n_permutations * 10, 9999),
            seed=cfg.seed,
        )
        group_counts_arr = group_test.group_counts
        group_names_list = list(group_test.group_names)

        log.info("permanova_frequencies (metric=%s n=%d)",
                 cfg.permanova_metric, cfg.permanova_n_permutations)
        permanova = permanova_frequencies(
            freq, image_names, group_map,
            metric=cfg.permanova_metric,
            n_permutations=cfg.permanova_n_permutations, seed=cfg.seed,
        )

        log.info("pairwise PERMANOVA (BH-FDR)")
        pair_permanova = pairwise_permanova_groups(
            freq, image_names, group_map,
            metric=cfg.permanova_metric,
            n_permutations=cfg.permanova_n_permutations,
            correction="bh", seed=cfg.seed,
        )

        log.info("per-cluster Kruskal-Wallis (correction=%s)", cfg.kw_correction)
        kw = per_cluster_kruskal_wallis(
            freq, image_names, cluster_ids, group_map,
            correction=cfg.kw_correction,
        )
        kw_q = (np.asarray(kw.p_values_corrected, dtype=np.float64)
                if kw is not None else None)

        log.info("pairwise MMD on per-image mean embeddings")
        M, M_names = per_image_mean_embeddings(P, source_images)
        # align M with image_names if order differs (per_image_mean uses the
        # same sorted-unique order as per_image_cluster_frequencies)
        mmd = pairwise_mmd_groups(
            M, M_names, group_map,
            n_permutations=cfg.permanova_n_permutations,
            correction="bh", seed=cfg.seed,
        )
    else:
        log.info("Skipping group-level tests (no group_map or only one group)")

    # ---- 12. Save core artefacts ----------------------------------------
    log.info("writing artefacts to %s", out_dir)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "freq.npy", freq)
    np.save(out_dir / "counts.npy", counts)
    np.save(out_dir / "image_names.npy", image_names)
    np.save(out_dir / "cluster_ids.npy", cluster_ids)

    patch_df = pd.DataFrame({
        "filename": filenames,
        "source_image": source_images,
        "cluster": labels,
    })
    patch_df.to_csv(out_dir / "patch_labels.csv", index=False)

    # one row per image with mapped group + total patches
    img_rows = []
    image_to_count = {img: int((source_images == img).sum())
                      for img in set(source_images)}
    for img in sorted(set(source_images)):
        img_rows.append({
            "source_image": img,
            "group": group_map.get(img, ""),
            "n_patches": image_to_count[img],
        })
    pd.DataFrame(img_rows).to_csv(out_dir / "image_groups.csv", index=False)

    # ---- 13. Summary JSON -----------------------------------------------
    summary_payload = {
        "schema_version": 1,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "bundle": str(bundle),
        "output_dir": str(out_dir),
        "manifest": manifest,
        "cfg": asdict(cfg),
        "data": {
            "n_patches": int(len(labels)),
            "n_images": int(len(image_names)),
            "embedding_dim": int(Z.shape[1]),
            "pca_dim": int(d_pca),
            "cum_explained_var": float(cum_var),
            "leiden_k": int(k),
            "control_pattern": cfg.control_reference_pattern,
            "group_source": group_src,
            "group_map_size": int(len(group_map)),
            "group_names": group_names_list,
        },
        "resolution_pick": {
            "resolution": float(res_pick),
            "mean_ari": float(mean_ari),
            "std_ari": float(std_ari),
            "n_clusters": int(n_clust),
        },
        "bootstrap_sweep": {
            f"{r:.4f}": {
                "mean_ari": tup[0],
                "std_ari": tup[1],
                "mean_nmi": tup[2] if len(tup) >= 4 else None,
                "std_nmi": tup[3] if len(tup) >= 4 else None,
                "n_clusters": int(len(set(full_sweep[r]))),
            }
            for r, tup in summary_sweep.items()
        },
        "size_stats": size_stats,
        "permutation_null_ari": {
            "mean": perm_mean, "std": perm_std, "max": perm_max,
            "n_permutations": cfg.permutation_p,
        },
        "internal_validity": iv,
        "image_level_cv": {
            "median_ari": cv_med, "q25": cv_q25, "q75": cv_q75,
            "n_splits": len(cv_per),
        },
        "medoids": medoids,
        "chi2_independence": {
            "stat": chi2_res[0], "p": chi2_res[1], "dof": chi2_res[2],
            "n_dropped": chi2_res[3], "method": chi2_res[4],
        },
        "cluster_purity": purity_rows,
        "group_cluster_test": group_test,
        "permanova": permanova,
        "pairwise_permanova": pair_permanova,
        "per_cluster_kw": kw,
        "pairwise_mmd": mmd,
        "wall_time_seconds": time.time() - t_start,
    }
    _dump_json(out_dir / "summary.json", summary_payload)

    # ---- 14. README -----------------------------------------------------
    readme_lines = [
        "# Clustering output",
        "",
        f"- bundle: `{bundle}`",
        f"- created: {summary_payload['created_at']}",
        f"- N = {len(labels)} patches over {len(image_names)} images",
        f"- PCA dim: {d_pca} (cum var = {float(cum_var):.3f})",
        f"- Leiden k = {k}; chosen resolution = {res_pick:.3f}",
        f"- K = {n_clust} clusters; bootstrap ARI = "
        f"{mean_ari:.3f} ± {std_ari:.3f}",
        f"- permutation-null ARI: mean = {perm_mean:.3f}, max = {perm_max:.3f}",
        f"- image-level CV ARI (median): {cv_med:.3f}",
        f"- silhouette = {iv.silhouette:.3f}, "
        f"Davies-Bouldin = {iv.davies_bouldin:.3f}, "
        f"Calinski-Harabasz = {iv.calinski_harabasz:.1f}",
        "",
        "## Files",
        "- `labels.npy` — int64 cluster id per patch (-1 = noise)",
        "- `patch_labels.csv` — filename, source_image, cluster",
        "- `freq.npy`, `counts.npy` — (image × cluster) matrices",
        "- `image_names.npy`, `cluster_ids.npy` — row / column order",
        "- `Z2.npy` — UMAP-2D layout (visualisation only)",
        "- `summary.json` — all statistical results",
        "- `run_clustering.log` — runtime log",
        "- `plots/` — figures (omit with --no-plots)",
        "",
    ]
    if group_test is not None:
        readme_lines += [
            "## Group-level tests",
            f"- groups: {', '.join(group_names_list)}",
            f"- group_cluster_test (image-perm χ²): stat = "
            f"{group_test.statistic:.3f}, p = {group_test.p_value:.4g}",
        ]
        if permanova is not None:
            readme_lines.append(
                f"- PERMANOVA pseudo-F = {permanova.f_statistic:.3f}, "
                f"p = {permanova.p_value:.4g} "
                f"(R² = {permanova.r_squared:.3f})"
            )
        readme_lines.append("")
    (out_dir / "README.md").write_text("\n".join(readme_lines))

    # ---- 15. Plots ------------------------------------------------------
    if not args.no_plots:
        log.info("rendering plots")
        try:
            render_plots(
                out_dir=out_dir,
                summary_sweep=summary_sweep, full_sweep=full_sweep,
                Z2=Z2, labels=labels, source_images=source_images,
                group_map=group_map,
                group_counts=group_counts_arr,
                group_names=group_names_list,
                cluster_ids=cluster_ids,
                freq=freq, image_names=image_names,
                kw_q=kw_q,
            )
        except Exception as exc:  # pragma: no cover — never block writeback
            log.exception("plot rendering failed: %s", exc)

    elapsed = time.time() - t_start
    log.info("Done in %.1fs  →  %s", elapsed, out_dir)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="End-to-end Leiden clustering + group-statistics on an "
                    "embeddings bundle produced by extract_full_embeddings.py.",
    )
    p.add_argument("--bundle", default=None,
                   help="path to an existing bundle directory "
                        "(must contain embeddings.npy + metadata.csv); "
                        "mutually exclusive with --checkpoint")
    p.add_argument("--checkpoint", "-c", default=None,
                   help="path to a model checkpoint (.pt) or a model run dir; "
                        "if given, runs extract_full_embeddings.py first and "
                        "clusters the resulting bundle")
    p.add_argument("--data-root", default=None,
                   help="(extraction) root containing <date>/index.csv "
                        "subfolders; default: data/patches_128_from_zip")
    p.add_argument("--output-root", default=None,
                   help="(extraction) parent for the bundle directory; "
                        "default: data/embeddings")
    p.add_argument("--run-name", default=None,
                   help="(extraction) override bundle folder name "
                        "(default: model run directory name)")
    p.add_argument("--extract-batch-size", type=int, default=128,
                   help="(extraction) DataLoader batch size")
    p.add_argument("--extract-num-workers", type=int, default=4,
                   help="(extraction) DataLoader workers")
    p.add_argument("--exclude-dates", nargs="*", default=[],
                   help="(extraction) skip these date subfolders by name")
    p.add_argument("--exclude-patterns-file", default=None,
                   help="(extraction) JSON file with case-insensitive substring "
                        "patterns to drop from the patch index (matched against "
                        "source_image / source_path / source_npy). Accepts a "
                        "JSON list or an object with key 'exclude_patterns'. "
                        "Pass data/data_analysis/exclude.json to match the "
                        "pretrain configs.")
    p.add_argument("--extract-device", default=None,
                   help="(extraction) cuda / cpu; default: cuda if available")
    p.add_argument("--extract-overwrite", action="store_true",
                   help="(extraction) re-extract even if bundle exists")
    p.add_argument("--output-dir", default=None,
                   help="where to write clustering artefacts "
                        "(default: <bundle>/clustering/)")
    p.add_argument("--group-patterns", default=None,
                   help="JSON file with treatment-group patterns "
                        "(default: <bundle>/group_patterns.json, "
                        "then configs/clustering/group_patterns.json)")
    p.add_argument("--resolutions", default=None,
                   help="comma-separated Leiden resolutions "
                        "(default: ClusterCfg defaults)")
    p.add_argument("--leiden-k", type=int, default=None,
                   help="kNN-graph k (disables auto)")
    p.add_argument("--pca-dim", type=int, default=None,
                   help="fixed PCA dimensionality "
                        "(default: auto via 95%% var, capped at 200)")
    p.add_argument("--control-pattern", default=None,
                   help="substring matched against source_image; "
                        "if set, mean of those patches is subtracted "
                        "before PCA")
    p.add_argument("--bootstrap-b", type=int, default=None,
                   help="bootstrap replicates")
    p.add_argument("--bootstrap-frac", type=float, default=None,
                   help="fraction of images per bootstrap sample")
    p.add_argument("--permutation-p", type=int, default=None,
                   help="permutation-null ARI replicates")
    p.add_argument("--permanova-n", type=int, default=None,
                   help="PERMANOVA / MMD permutations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true",
                   help="skip matplotlib figures (faster, headless-friendly)")
    p.add_argument("--no-umap", action="store_true",
                   help="skip the UMAP-2D layout entirely (implies no UMAP plots)")
    p.add_argument("--overwrite", action="store_true",
                   help="remove existing output-dir before running")
    return p.parse_args(argv)


def _setup_root_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(h)
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    _setup_root_logging()
    args = parse_args(argv)
    try:
        return run(args)
    except Exception:
        log.exception("run_clustering failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
