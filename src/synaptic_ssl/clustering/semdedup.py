"""SemDeDup-style semantic redundancy diagnostics.

L2-normalise → k-means buckets → within-bucket cosine pairs →
ε-sweep over (1 − cos) threshold → connected-component duplicate
groups. Adapted from Abbas et al. (ICLR 2023) for 10K–50K-patch
fluorescence-microscopy datasets. Diagnostic only: no patches are
removed.

Ref: https://github.com/facebookresearch/SemDeDup
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SemDeDupCfg:
    """Configuration for SemDeDup redundancy diagnostics."""

    seed: int = 42
    l2_normalise: bool = True

    # K-means bucketing — SemDeDup uses k=50K for 440M images.
    # For 10K–50K patches, 50–200 is appropriate (Sec 6.1: k is robust).
    n_clusters: int = 50
    kmeans_max_iter: int = 300

    # Multi-k stability check: run several k values to verify robustness.
    stability_k_values: tuple[int, ...] = (50, 100, 200)

    # Dissimilarity thresholds (ε): cosine > 1−ε → semantic duplicate.
    # SemDeDup Fig 3 uses a continuous sweep; we sample representative ε.
    epsilon_thresholds: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2,
    )

    # Guard: maximum cluster size for full pairwise computation.
    # Larger clusters are subsampled for the histogram.
    max_pairwise_cluster_size: int = 2000

    # Optional PCA dimension reduction (no whitening!) for very
    # high-dimensional embeddings. None = skip PCA.
    pca_dim: int | None = None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class ThresholdStats(NamedTuple):
    """Redundancy statistics for a single ε threshold."""
    epsilon: float
    n_duplicate_pairs: int
    n_patches_with_duplicate: int
    fraction_with_duplicate: float
    n_connected_components: int
    n_unique_groups: int           # singletons + one per connected component
    estimated_fraction_remaining: float  # unique_groups / total patches


class ClusterStats(NamedTuple):
    """Per-cluster redundancy summary."""
    cluster_id: int
    size: int
    mean_cosine: float
    median_cosine: float
    p95_cosine: float
    n_duplicate_pairs_at_default_eps: int   # default ε = 0.03


class RedundancyReport(NamedTuple):
    """Full output of the SemDeDup diagnostic pipeline."""
    # Per-threshold sweep
    threshold_stats: list[ThresholdStats]

    # Per-cluster details
    cluster_stats: list[ClusterStats]

    # Global cosine histogram (flattened upper-triangle within each cluster)
    cosine_histogram_edges: np.ndarray   # (n_bins+1,)
    cosine_histogram_counts: np.ndarray  # (n_bins,)

    # Same-image vs cross-image cosine distributions
    same_image_cosines: np.ndarray       # sample of cosines from same source
    cross_image_cosines: np.ndarray      # sample from different sources

    # Per-source-image duplicate contribution
    per_image_duplicate_count: dict[str, int]   # raw count at default ε
    per_image_duplicate_frac: dict[str, float]  # normalised by image size

    # Multi-k stability (if run)
    stability_results: dict[int, float] | None  # k → fraction_remaining@default_ε

    # Metadata
    n_patches: int
    n_clusters: int
    embedding_dim: int
    default_epsilon: float


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _l2_normalise(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)
    return Z / norms


def _optional_pca(Z: np.ndarray, dim: int | None, seed: int) -> np.ndarray:
    """Dimensionality reduction without whitening (preserves cosine geometry)."""
    if dim is None or Z.shape[1] <= dim:
        return Z
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, whiten=False, random_state=seed)
    return pca.fit_transform(Z).astype(np.float32)


def _run_kmeans(Z: np.ndarray, k: int, seed: int,
                max_iter: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """K-means on L2-normalised embeddings. Returns (labels, centroids)."""
    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=k, random_state=seed, max_iter=max_iter, n_init=3,
    )
    labels = km.fit_predict(Z)
    centroids = km.cluster_centers_
    # Normalise centroids for cosine interpretation
    centroids = _l2_normalise(centroids)
    return labels, centroids


def _within_cluster_cosines(
    Z_normed: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    max_size: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Upper-triangle pairwise cosines within one cluster.

    Subsamples if the cluster is larger than ``max_size``.
    Returns 1D array of cosines.
    """
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return np.array([], dtype=np.float32)
    if len(idx) > max_size:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=max_size, replace=False)

    sub = Z_normed[idx]
    # cosine = inner product for L2-normalised vectors
    cos_mat = sub @ sub.T
    # upper triangle (no diagonal)
    triu_idx = np.triu_indices(len(sub), k=1)
    return cos_mat[triu_idx].astype(np.float32)


def _build_duplicate_graph(
    Z_normed: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    max_size: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, dict[int, set[int]]]:
    """Build duplicate-pair graph within each cluster.

    Returns (n_pairs, n_patches_with_dup, n_components, patch_to_component).
    """
    threshold = 1.0 - epsilon
    total_pairs = 0
    patches_with_dup = set()
    # Union-Find for connected components
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    unique_clusters = sorted(set(labels))
    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue

        working_idx = idx
        if len(idx) > max_size:
            if rng is None:
                rng = np.random.default_rng(0)
            working_idx = rng.choice(idx, size=max_size, replace=False)

        sub = Z_normed[working_idx]
        cos_mat = sub @ sub.T
        triu_r, triu_c = np.triu_indices(len(working_idx), k=1)
        dup_mask = cos_mat[triu_r, triu_c] >= threshold

        dup_rows = triu_r[dup_mask]
        dup_cols = triu_c[dup_mask]
        total_pairs += int(dup_mask.sum())

        for r, c_idx in zip(dup_rows, dup_cols):
            a, b = int(working_idx[r]), int(working_idx[c_idx])
            patches_with_dup.add(a)
            patches_with_dup.add(b)
            union(a, b)

    # Count connected components among duplicated patches
    components: dict[int, set[int]] = {}
    for p in patches_with_dup:
        root = find(p)
        components.setdefault(root, set()).add(p)

    return total_pairs, len(patches_with_dup), len(components), components


def _same_vs_cross_image_cosines(
    Z_normed: np.ndarray,
    source_images: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample within-cluster cosines for same-image vs cross-image pairs.

    Calibrates ε thresholds for this dataset.
    """
    rng = np.random.default_rng(seed)
    n = len(Z_normed)
    same, cross = [], []

    for _ in range(n_samples * 2):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        # Only consider within-cluster pairs (as SemDeDup does)
        if labels[i] != labels[j]:
            continue
        cos = float(Z_normed[i] @ Z_normed[j])
        if source_images[i] == source_images[j]:
            if len(same) < n_samples:
                same.append(cos)
        else:
            if len(cross) < n_samples:
                cross.append(cos)
        if len(same) >= n_samples and len(cross) >= n_samples:
            break

    return np.array(same, dtype=np.float32), np.array(cross, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

DEFAULT_EPSILON = 0.03  # SemDeDup uses this as a "tight" threshold (Sec 4.2)


def redundancy_profile(
    Z: np.ndarray,
    source_images: np.ndarray,
    cfg: SemDeDupCfg | None = None,
) -> RedundancyReport:
    """Run full SemDeDup-style redundancy diagnostic.

    Z: (N, D) float32 patch embeddings (raw, NOT PCA-whitened).
    source_images: (N,) source filename per patch.
    cfg: defaults tuned for 10K–50K patches.

    Returns a RedundancyReport with per-threshold stats, per-cluster
    stats, cosine histograms, and per-image duplicate contribution.

    Ref: https://github.com/facebookresearch/SemDeDup
    """
    if cfg is None:
        cfg = SemDeDupCfg()
    rng = np.random.default_rng(cfg.seed)

    # --- Preprocessing ---
    Z = Z.astype(np.float32)
    Z = _optional_pca(Z, cfg.pca_dim, cfg.seed)
    if cfg.l2_normalise:
        Z_normed = _l2_normalise(Z)
    else:
        Z_normed = Z.copy()

    n_patches, emb_dim = Z_normed.shape

    # --- K-means bucketing ---
    k = min(cfg.n_clusters, n_patches // 2)  # sanity: at least 2 per cluster
    labels, centroids = _run_kmeans(Z_normed, k, cfg.seed, cfg.kmeans_max_iter)

    # --- Per-cluster cosine statistics ---
    all_cosines = []
    cluster_stats_list = []
    for c in sorted(set(labels)):
        cos_vals = _within_cluster_cosines(
            Z_normed, labels, c,
            max_size=cfg.max_pairwise_cluster_size, rng=rng,
        )
        all_cosines.append(cos_vals)
        c_size = int((labels == c).sum())
        if len(cos_vals) > 0:
            n_dup = int((cos_vals >= 1.0 - DEFAULT_EPSILON).sum())
            cluster_stats_list.append(ClusterStats(
                cluster_id=c,
                size=c_size,
                mean_cosine=float(np.mean(cos_vals)),
                median_cosine=float(np.median(cos_vals)),
                p95_cosine=float(np.percentile(cos_vals, 95)),
                n_duplicate_pairs_at_default_eps=n_dup,
            ))
        else:
            cluster_stats_list.append(ClusterStats(
                cluster_id=c, size=c_size, mean_cosine=float("nan"),
                median_cosine=float("nan"), p95_cosine=float("nan"),
                n_duplicate_pairs_at_default_eps=0,
            ))

    # --- Global cosine histogram ---
    all_cos = np.concatenate(all_cosines) if all_cosines else np.array([])
    if len(all_cos) > 0:
        hist_counts, hist_edges = np.histogram(all_cos, bins=200, range=(-1, 1))
    else:
        hist_edges = np.linspace(-1, 1, 201)
        hist_counts = np.zeros(200, dtype=np.int64)

    # --- Same-image vs cross-image cosine distributions ---
    same_cos, cross_cos = _same_vs_cross_image_cosines(
        Z_normed, source_images, labels, n_samples=5000, seed=cfg.seed,
    )

    # --- ε threshold sweep ---
    threshold_stats_list = []
    for eps in cfg.epsilon_thresholds:
        n_pairs, n_with_dup, n_comp, components = _build_duplicate_graph(
            Z_normed, labels, eps,
            max_size=cfg.max_pairwise_cluster_size, rng=rng,
        )
        n_unique = (n_patches - n_with_dup) + n_comp
        threshold_stats_list.append(ThresholdStats(
            epsilon=eps,
            n_duplicate_pairs=n_pairs,
            n_patches_with_duplicate=n_with_dup,
            fraction_with_duplicate=n_with_dup / max(n_patches, 1),
            n_connected_components=n_comp,
            n_unique_groups=n_unique,
            estimated_fraction_remaining=n_unique / max(n_patches, 1),
        ))

    # --- Per-image duplicate contribution (at default ε) ---
    _, _, _, components_default = _build_duplicate_graph(
        Z_normed, labels, DEFAULT_EPSILON,
        max_size=cfg.max_pairwise_cluster_size, rng=rng,
    )
    dup_patches = set()
    for comp in components_default.values():
        dup_patches.update(comp)

    img_dup_count: dict[str, int] = {}
    img_total: dict[str, int] = {}
    for i, img in enumerate(source_images):
        img = str(img)
        img_total[img] = img_total.get(img, 0) + 1
        if i in dup_patches:
            img_dup_count[img] = img_dup_count.get(img, 0) + 1
    img_dup_frac = {
        img: img_dup_count.get(img, 0) / max(img_total[img], 1)
        for img in img_total
    }

    # --- Multi-k stability (optional) ---
    stability: dict[int, float] | None = None
    if cfg.stability_k_values and len(cfg.stability_k_values) > 1:
        stability = {}
        for k_val in cfg.stability_k_values:
            k_eff = min(k_val, n_patches // 2)
            if k_eff < 2:
                continue
            lab_k, _ = _run_kmeans(Z_normed, k_eff, cfg.seed, cfg.kmeans_max_iter)
            _, _, _, comp_k = _build_duplicate_graph(
                Z_normed, lab_k, DEFAULT_EPSILON,
                max_size=cfg.max_pairwise_cluster_size, rng=rng,
            )
            dup_k = set()
            for comp in comp_k.values():
                dup_k.update(comp)
            n_unique_k = (n_patches - len(dup_k)) + len(comp_k)
            stability[k_val] = n_unique_k / max(n_patches, 1)

    return RedundancyReport(
        threshold_stats=threshold_stats_list,
        cluster_stats=cluster_stats_list,
        cosine_histogram_edges=hist_edges,
        cosine_histogram_counts=hist_counts,
        same_image_cosines=same_cos,
        cross_image_cosines=cross_cos,
        per_image_duplicate_count=img_dup_count,
        per_image_duplicate_frac=img_dup_frac,
        stability_results=stability,
        n_patches=n_patches,
        n_clusters=k,
        embedding_dim=emb_dim,
        default_epsilon=DEFAULT_EPSILON,
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_retention_curve(report: RedundancyReport, ax=None):
    """Fraction of unique data remaining vs ε (SemDeDup Fig 3a style)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    eps = [s.epsilon for s in report.threshold_stats]
    frac = [s.estimated_fraction_remaining for s in report.threshold_stats]
    ax.plot(eps, frac, "o-", color="C0", linewidth=2)
    ax.set_xlabel("Dissimilarity threshold ε")
    ax.set_ylabel("Estimated fraction remaining")
    ax.set_title("SemDeDup retention curve")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    return ax


def plot_duplicate_fraction(report: RedundancyReport, ax=None):
    """% of patches with ≥1 semantic duplicate vs ε (SemDeDup Fig 3b style)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    eps = [s.epsilon for s in report.threshold_stats]
    frac = [s.fraction_with_duplicate * 100 for s in report.threshold_stats]
    ax.plot(eps, frac, "s-", color="C1", linewidth=2)
    ax.set_xlabel("Dissimilarity threshold ε")
    ax.set_ylabel("% patches with ≥1 duplicate")
    ax.set_title("Patches with semantic duplicates")
    ax.grid(alpha=0.3)
    return ax


def plot_cosine_histogram(report: RedundancyReport, ax=None,
                          xlim: tuple[float, float] = (0.5, 1.0)):
    """Histogram of within-cluster pairwise cosines (SemDeDup Fig 3c style)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    edges = report.cosine_histogram_edges
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = report.cosine_histogram_counts
    ax.bar(centres, counts, width=edges[1] - edges[0], color="C2", alpha=0.7)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count (within-cluster pairs)")
    ax.set_title("Within-cluster pairwise cosine distribution")
    ax.set_xlim(*xlim)
    ax.grid(alpha=0.3)
    return ax


def plot_same_vs_cross_image(report: RedundancyReport, ax=None):
    """Overlaid histograms of same-image vs cross-image cosine similarities."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.5, 1.0, 100)
    if len(report.same_image_cosines) > 0:
        ax.hist(report.same_image_cosines, bins=bins, alpha=0.5,
                label=f"Same image (n={len(report.same_image_cosines)})",
                density=True, color="C3")
    if len(report.cross_image_cosines) > 0:
        ax.hist(report.cross_image_cosines, bins=bins, alpha=0.5,
                label=f"Cross image (n={len(report.cross_image_cosines)})",
                density=True, color="C0")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Same-image vs cross-image cosine (within-cluster pairs)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_cluster_redundancy_heatmap(report: RedundancyReport, ax=None):
    """Per-cluster redundancy summary — sorted by mean cosine."""
    import matplotlib.pyplot as plt
    stats = sorted(report.cluster_stats, key=lambda s: -s.mean_cosine
                   if np.isfinite(s.mean_cosine) else -999)
    cids = [s.cluster_id for s in stats]
    means = [s.mean_cosine for s in stats]
    sizes = [s.size for s in stats]
    n_dups = [s.n_duplicate_pairs_at_default_eps for s in stats]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cids))
    bars = ax.bar(x, means, color="C4", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cids, rotation=90, fontsize=6)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean intra-cluster cosine")
    ax.set_title(f"Per-cluster mean cosine (k={report.n_clusters})")
    ax.grid(alpha=0.3, axis="y")

    # Annotate top-3 most redundant clusters
    for i in range(min(3, len(stats))):
        ax.annotate(
            f"n={sizes[i]}\ndups={n_dups[i]}",
            (x[i], means[i]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7, color="red",
        )
    return ax


def plot_stability(report: RedundancyReport, ax=None):
    """Fraction remaining at default ε across different k values."""
    import matplotlib.pyplot as plt
    if report.stability_results is None:
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    ks = sorted(report.stability_results.keys())
    fracs = [report.stability_results[k] for k in ks]
    ax.plot(ks, fracs, "D-", color="C5", linewidth=2)
    ax.set_xlabel("Number of k-means clusters (k)")
    ax.set_ylabel(f"Fraction remaining (ε={report.default_epsilon})")
    ax.set_title("Multi-k stability check")
    ax.grid(alpha=0.3)
    return ax


def print_summary(report: RedundancyReport):
    """Print compact text summary of the redundancy report."""
    print(f"SemDeDup Redundancy Report")
    print(f"  Patches: {report.n_patches:,}   "
          f"Embedding dim: {report.embedding_dim}   "
          f"K-means k: {report.n_clusters}")
    print()
    print(f"{'ε':>8s}  {'Dup pairs':>10s}  {'Patches w/ dup':>15s}  "
          f"{'% w/ dup':>8s}  {'Components':>11s}  {'Est. remaining':>15s}")
    print("-" * 75)
    for s in report.threshold_stats:
        print(f"{s.epsilon:8.4f}  {s.n_duplicate_pairs:10,d}  "
              f"{s.n_patches_with_duplicate:15,d}  "
              f"{s.fraction_with_duplicate * 100:7.1f}%  "
              f"{s.n_connected_components:11,d}  "
              f"{s.estimated_fraction_remaining * 100:14.1f}%")

    if report.stability_results:
        print()
        print(f"Multi-k stability (ε={report.default_epsilon}):")
        for k_val in sorted(report.stability_results):
            frac = report.stability_results[k_val]
            print(f"  k={k_val:>4d}  →  {frac * 100:.1f}% remaining")

    # Top-5 most duplicate-heavy images
    top_imgs = sorted(report.per_image_duplicate_frac.items(),
                      key=lambda x: -x[1])[:5]
    if top_imgs:
        print()
        print(f"Top-5 images by duplicate fraction (ε={report.default_epsilon}):")
        for img, frac in top_imgs:
            count = report.per_image_duplicate_count.get(img, 0)
            print(f"  {img}: {frac * 100:.1f}% ({count} duplicate patches)")
