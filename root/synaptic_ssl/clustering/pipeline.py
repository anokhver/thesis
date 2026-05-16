"""Patch clustering pipeline for fluorescence-microscopy SSL features.

Pipeline: features → L2 + PCA whiten → UMAP-15 → Leiden → bootstrap
stability → per-image frequency vectors → chi-square test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClusterCfg:
    """Clustering pipeline configuration."""

    seed: int = 42

    # Clustering space — "pca" clusters directly in PCA-whitened space
    # (recommended for N < 5000; avoids UMAP instability at small N,
    # see Chari & Pachter 2023). "umap" uses the legacy UMAP-15D path.
    cluster_space: str = "pca"

    # Preprocessing
    l2_normalise: bool = True
    pca_dim: int | None = None         # None -> auto via RankMe
    pca_dim_target_var: float = 0.95   # auto: smallest d capturing this
    pca_dim_max: int = 200             # auto: hard cap
    tvn_reference_pattern: str | None = None  # substring matched against
                                              # source_image; if set, the
                                              # mean of those patches is
                                              # subtracted before PCA

    # UMAP (clustering geometry — only used when cluster_space="umap")
    umap_n_components: int = 15
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    umap_init: str = "pca"
    umap_n_epochs: int = 500

    # UMAP (visualisation only)
    umap2_n_neighbors: int = 30
    umap2_min_dist: float = 0.1

    # Leiden (primary)
    leiden_k: int = 30
    leiden_k_auto: bool = True         # auto-scale k based on N
    leiden_resolutions: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0)

    # Bootstrap stability
    bootstrap_b: int = 50
    bootstrap_frac: float = 0.7

    # GMM-BIC cross-check
    gmm_k_min: int = 3
    gmm_k_max: int = 25
    gmm_covariance_type: str = "auto"  # "auto", "full", "diag", "tied"
    gmm_pca_dim: int | None = None     # None -> auto (10-20 PCs for GMM)

    # HDBSCAN fallback
    hdb_min_cluster_size: int = 30
    hdb_min_samples: int = 5
    hdb_method: str = "leaf"
    hdb_auto: bool = True              # auto-scale based on N

    # Permutation null
    permutation_p: int = 10

    # Cluster characterization
    samples_per_cluster: int = 8

    # Image-level cross-validation
    cv_n_repeats: int = 20             # repeated grouped half-splits
    cv_k_neighbors: int = 5            # kNN for label transfer

    # Differential frequency test
    chi2_min_image_patches: int = 20   # drop images with fewer patches
    chi2_n_permutations: int = 10_000  # Monte Carlo sims for sparse tables

    # Group-level statistical tests
    mmd_n_permutations: int = 1000
    mmd_gamma: float | None = None      # None -> median heuristic
    mmd_max_patches: int = 5000         # subsample cap per group for kernel matrix
    mmd_correction: str = "bh"          # "bonferroni" or "bh" (Benjamini-Hochberg)
    permanova_n_permutations: int = 999
    permanova_metric: str = "braycurtis"

    # IO
    embedding_cache: str | None = None  # path to .npy cache; None disables


# ---------------------------------------------------------------------------
# Adaptive parameter scaling for small N
# ---------------------------------------------------------------------------

def adaptive_leiden_k(N: int, default_k: int = 30) -> int:
    """Scale Leiden k to maintain meaningful graph sparsity.

    PhenoGraph (Levine et al. 2015) used k=30 at N≈15 000, giving ~0.2%
    connectivity.  At small N the same k produces an over-connected graph
    that washes out local density.  We target ~1.5% connectivity, which
    reproduces the PhenoGraph operating regime for N≈2 000 and gracefully
    scales for larger datasets.
    """
    k = int(round(0.015 * N))
    return max(5, min(default_k, k))


def adaptive_gmm_params(
    N: int, D: int, *, cfg: ClusterCfg,
) -> tuple[int, str, int]:
    """Auto-select (k_max, covariance_type, pca_dim) for GMM-BIC.

    Full covariance in D dims needs D(D+1)/2 + D free parameters per
    component.  We require ≥10 data points per parameter so that BIC
    remains a reliable model-selection criterion (McLachlan & Peel 2000).
    Falls back to diagonal (2D params/component) or spherical (D+1) when
    N is too small for full covariance.
    """
    def _k_max(n_params_per_comp: int) -> int:
        return max(cfg.gmm_k_min, N // (10 * max(1, n_params_per_comp)))

    # Auto-select PCA dim for GMM (10-20 PCs)
    if cfg.gmm_pca_dim is not None:
        d = min(cfg.gmm_pca_dim, D)
    else:
        d = min(20, D)

    # Auto-select covariance type
    if cfg.gmm_covariance_type != "auto":
        cov = cfg.gmm_covariance_type
        full_params = d * (d + 1) // 2 + d
        diag_params = 2 * d
        params = {"full": full_params, "diag": diag_params,
                  "tied": full_params, "spherical": d + 1}[cov]
        k_max = min(cfg.gmm_k_max, _k_max(params))
        return k_max, cov, d

    full_params = d * (d + 1) // 2 + d
    if _k_max(full_params) >= cfg.gmm_k_min:
        return min(cfg.gmm_k_max, _k_max(full_params)), "full", d

    diag_params = 2 * d
    if _k_max(diag_params) >= cfg.gmm_k_min:
        return min(cfg.gmm_k_max, _k_max(diag_params)), "diag", d

    return min(cfg.gmm_k_max, _k_max(d + 1)), "spherical", d


def adaptive_hdbscan_params(
    N: int, default_min_cluster: int = 30,
) -> tuple[int, int]:
    """Auto-scale HDBSCAN (min_cluster_size, min_samples) for dataset size.

    Targets ~1% of N as minimum cluster size (lower-bounded at 10),
    so that the algorithm can discover fine-grained phenotypes without
    classifying too much as noise.
    """
    min_cluster = max(10, min(default_min_cluster, int(0.01 * N)))
    min_samples = min(5, max(2, min_cluster // 3))
    return min_cluster, min_samples


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_patch_embeddings(
    encoder,
    dataset,
    *,
    device,
    batch_size: int = 64,
    num_workers: int = 0,
    cache_path=None,
    force_recompute: bool = False,
):
    """Run frozen encoder over dataset; return globally-pooled features.

    Returns (Z, filenames, source_images, image_indices). Caches to
    ``cache_path`` if set.
    """
    import torch
    from pathlib import Path

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not force_recompute:
        d = np.load(cache_path, allow_pickle=True).item()
        return (d["Z"].astype(np.float32),
                np.asarray(d["filenames"]),
                np.asarray(d["source_images"]),
                np.asarray(d["image_indices"], dtype=np.int64))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(str(device) != "cpu"),
    )
    feats = []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True).contiguous()
            z = encoder(x)[-1].mean(dim=(-2, -1))
            feats.append(z.float().cpu().numpy())
    Z = np.concatenate(feats, axis=0).astype(np.float32)

    records = getattr(dataset, "records", None)
    if records is None:
        # TransformedSubset(subset=Subset(base=PatchDataset)) is the common
        # wrapper chain; resolve it.
        sub = getattr(dataset, "subset", None)
        if sub is None and hasattr(dataset, "indices"):
            sub = dataset
        if sub is not None:
            # sub is the full PatchDataset (no Subset wrapper)
            sub_records = getattr(sub, "records", None)
            if sub_records is not None and not hasattr(sub, "indices"):
                records = sub_records
            else:
                # sub is a Subset: resolve base dataset + indices
                base = getattr(sub, "dataset", None)
                indices = getattr(sub, "indices", None)
                base_records = getattr(base, "records", None)
                if base_records is not None and indices is not None:
                    records = [base_records[int(i)] for i in indices]
    if records is None:
        filenames = np.array([f"row_{i:06d}.npy" for i in range(Z.shape[0])])
        source_images = np.array(["UNKNOWN"] * Z.shape[0])
        image_indices = np.zeros(Z.shape[0], dtype=np.int64)
    else:
        filenames     = np.array([r["filename"]              for r in records])
        source_images = np.array([r["source_image"]          for r in records])
        image_indices = np.array([int(r.get("image_index", 0))
                                   for r in records], dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path,
                {"Z": Z, "filenames": filenames,
                 "source_images": source_images,
                 "image_indices": image_indices},
                allow_pickle=True)
    return Z, filenames, source_images, image_indices


# ---------------------------------------------------------------------------
# Preprocessing: L2 + (TVN) + PCA whitening
# ---------------------------------------------------------------------------

def auto_pca_dim(Z: np.ndarray, target_var: float = 0.95, hard_cap: int = 200) -> int:
    """Choose d such that cumulative explained variance >= target_var.

    Lower-bounded by 5; upper-bounded by hard_cap and min(N-1, D).
    """
    from sklearn.decomposition import PCA
    n_max = int(min(Z.shape[0] - 1, Z.shape[1], hard_cap))
    pca = PCA(n_components=n_max, random_state=0).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(np.searchsorted(cum, target_var) + 1)
    return max(5, min(d, n_max))


def tvn_centre(Z: np.ndarray, source_images: np.ndarray, pattern: str) -> np.ndarray:
    """Subtract mean of patches matching ``pattern`` (TVN; Kraus et al., CVPR 2024).

    Removes plate/well effects before feature analysis.
    Ref: https://github.com/recursionpharma/maes_microscopy
    """
    if not pattern:
        return Z
    pat = pattern.upper()
    mask = np.array([pat in s.upper() for s in source_images])
    if mask.sum() == 0:
        raise ValueError(f"no source_image matched TVN pattern {pattern!r}")
    return Z - Z[mask].mean(axis=0, keepdims=True)


def l2_then_pca_whiten(
    Z: np.ndarray, n_components: int, seed: int = 0,
):
    """L2-normalise then PCA-whiten. Returns ``(P, pca, cum_var)``."""
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    Zn = normalize(Z.astype(np.float32), norm="l2")
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    P = pca.fit_transform(Zn).astype(np.float32)
    cum = float(np.cumsum(pca.explained_variance_ratio_)[-1])
    return P, pca, cum


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def fit_umap(P: np.ndarray, *, cfg: ClusterCfg, seed: int | None = None,
             n_components: int | None = None) -> np.ndarray:
    import umap
    n_comp = n_components if n_components is not None else cfg.umap_n_components
    init = cfg.umap_init
    if init == "pca" and P.shape[1] < n_comp:
        init = "spectral"
    return umap.UMAP(
        n_components=n_comp,
        n_neighbors=cfg.umap_n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        init=init,
        n_epochs=cfg.umap_n_epochs,
        random_state=seed if seed is not None else cfg.seed,
        n_jobs=1,
    ).fit_transform(P)


def fit_umap_2d(P: np.ndarray, *, cfg: ClusterCfg, seed: int | None = None) -> np.ndarray:
    """2D UMAP for visualisation. Larger ``min_dist`` avoids misleading density."""
    import umap
    init = cfg.umap_init
    if init == "pca" and P.shape[1] < 2:
        init = "spectral"
    return umap.UMAP(
        n_components=2,
        n_neighbors=cfg.umap2_n_neighbors,
        min_dist=cfg.umap2_min_dist,
        metric=cfg.umap_metric,
        init=init,
        n_epochs=cfg.umap_n_epochs,
        random_state=seed if seed is not None else cfg.seed,
        n_jobs=1,
    ).fit_transform(P)


# ---------------------------------------------------------------------------
# Leiden + bootstrap
# ---------------------------------------------------------------------------

def knn_igraph(X: np.ndarray, k: int):
    """Build a k-nearest-neighbour graph as an igraph object."""
    from sklearn.neighbors import NearestNeighbors
    import igraph as ig
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(X)
    edges = [(int(i), int(j)) for i in range(len(X)) for j in idx[i, 1:]]
    g = ig.Graph(n=len(X), edges=edges, directed=False)
    g.simplify(combine_edges="ignore")
    return g


def leiden_partition(g, resolution: float, seed: int = 0) -> np.ndarray:
    import leidenalg
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    return np.array(part.membership, dtype=np.int64)


def leiden_sweep(U: np.ndarray, resolutions: Sequence[float], k: int,
                 seed: int = 0):
    g = knn_igraph(U, k)
    return {float(r): leiden_partition(g, r, seed=seed) for r in resolutions}, g


def bootstrap_stability(
    U: np.ndarray, resolutions: Sequence[float], k: int,
    B: int, frac: float, seed: int = 0,
):
    """Bootstrap stability of Leiden partitions.

    70% subsample → re-cluster → 1-NN propagate → ARI vs full partition.
    Hennig (2007); Lange et al. (2004).
    Returns ``({res: (mean_ari, std_ari)}, {res: labels})``.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(U)
    g_full = knn_igraph(U, k)
    full = {float(r): leiden_partition(g_full, r, seed=seed) for r in resolutions}
    aris = {float(r): [] for r in resolutions}
    for b in range(B):
        idx = rng.choice(n, size=int(frac * n), replace=True)
        U_sub = U[idx]
        g_sub = knn_igraph(U_sub, k)
        for r in resolutions:
            lab_sub = leiden_partition(g_sub, r, seed=seed + b)
            knn = KNeighborsClassifier(n_neighbors=1).fit(U_sub, lab_sub)
            propagated = knn.predict(U)
            aris[float(r)].append(adjusted_rand_score(full[float(r)], propagated))
    summary = {r: (float(np.mean(aris[r])), float(np.std(aris[r]))) for r in aris}
    return summary, full


def pick_resolution(summary: dict, full: dict):
    """Highest mean ARI; tie-break: fewer clusters (Occam)."""
    items = [(r, m, s, len(set(full[r]))) for r, (m, s) in summary.items()]
    items.sort(key=lambda t: (-t[1], t[3]))
    return items[0]  # (resolution, mean_ari, std_ari, n_clusters)


# ---------------------------------------------------------------------------
# Cross-check + fallback
# ---------------------------------------------------------------------------

def gmm_bic_scan(X: np.ndarray, k_min: int, k_max: int, seed: int = 0,
                 *, covariance_type: str = "diag", pca_dim: int | None = None):
    """GMM-BIC scan with configurable covariance type and optional PCA.

    When ``pca_dim`` is set and smaller than X.shape[1], the data is first
    reduced via PCA so that the per-component parameter count stays
    manageable relative to N (McLachlan & Peel 2000).
    """
    from sklearn.mixture import GaussianMixture
    if pca_dim is not None and pca_dim < X.shape[1]:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    bics = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type=covariance_type,
            random_state=seed, max_iter=200, reg_covar=1e-4,
        ).fit(X)
        bics.append((k, float(gmm.bic(X))))
    bics.sort(key=lambda t: t[1])
    return bics


def run_hdbscan_leaf(U: np.ndarray, *, min_cluster_size: int, min_samples: int,
                     method: str = "leaf"):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    return clusterer.fit_predict(U)


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------

def gini(values) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * v) / (n * v.sum())) - (n + 1) / n)


def cluster_size_stats(labels: np.ndarray) -> dict:
    mask = labels != -1
    sizes = np.bincount(labels[mask]) if mask.any() else np.array([0])
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    return {
        "k": int(len(sizes)),
        "noise_frac": n_noise / n_total,
        "gini": gini(sizes),
        "max_cluster_frac": float(sizes.max() / n_total) if len(sizes) else 0.0,
        "singleton_frac": float((sizes < 5).mean()) if len(sizes) else 0.0,
    }


def permutation_null_ari(
    P: np.ndarray, labels_real: np.ndarray, *,
    cfg: ClusterCfg, resolution: float,
    k: int | None = None,
):
    """Permutation-null ARI by shuffling PCs and re-clustering.

    When ``cluster_space="pca"`` the kNN graph is built directly on the
    permuted PCA features (no UMAP).  When ``cluster_space="umap"`` the
    legacy path through UMAP is used.

    Witten & Tibshirani (2010). Real ARI should exceed this distribution.
    """
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(cfg.seed)
    effective_k = k if k is not None else cfg.leiden_k
    aris = []
    for it in range(cfg.permutation_p):
        P_perm = P.copy()
        for d in range(P_perm.shape[1]):
            rng.shuffle(P_perm[:, d])
        if cfg.cluster_space == "pca":
            X_clust = P_perm
        else:
            X_clust = fit_umap(P_perm, cfg=cfg, seed=cfg.seed + it)
        g_perm = knn_igraph(X_clust, effective_k)
        lab_perm = leiden_partition(g_perm, resolution, seed=cfg.seed + it)
        aris.append(adjusted_rand_score(labels_real, lab_perm))
    return float(np.mean(aris)), float(np.std(aris)), float(np.max(aris))


def image_level_cv(
    P: np.ndarray, source_images: np.ndarray, labels: np.ndarray, *,
    cfg: ClusterCfg,
):
    """Repeated grouped half-split cross-validation for label transfer.

    Tests whether cluster labels are transferable across held-out images
    via kNN in PCA space — i.e. whether the embedding captures biological
    structure that generalises beyond the training images.  Unlike the
    legacy approach this does **not** re-run UMAP+Leiden on tiny folds
    (which is unreliable at small N).

    Returns ``(median_ari, q25, q75, per_split_aris)``.
    Caicedo et al. (Nat Methods 2017), adapted for small N.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(cfg.seed)
    unique_images = np.array(sorted(set(source_images)))
    n_img = len(unique_images)

    if n_img < 4:
        return float("nan"), float("nan"), float("nan"), []

    aris: list[float] = []
    for rep in range(cfg.cv_n_repeats):
        perm = rng.permutation(n_img)
        half = unique_images[perm[: n_img // 2]]
        mask_train = np.isin(source_images, half)

        if mask_train.sum() < cfg.cv_k_neighbors + 1 or (~mask_train).sum() < 2:
            continue

        knn = KNeighborsClassifier(n_neighbors=cfg.cv_k_neighbors)
        knn.fit(P[mask_train], labels[mask_train])
        pred = knn.predict(P[~mask_train])
        ari = adjusted_rand_score(labels[~mask_train], pred)
        aris.append(float(ari))

    if not aris:
        return float("nan"), float("nan"), float("nan"), []

    return (
        float(np.median(aris)),
        float(np.percentile(aris, 25)),
        float(np.percentile(aris, 75)),
        aris,
    )


# Keep backward-compatible alias
def train_on_imageset_predict_other(
    P: np.ndarray, source_images: np.ndarray, *,
    cfg: ClusterCfg, resolution: float,
):
    """Deprecated — use :func:`image_level_cv` instead."""
    import warnings
    warnings.warn(
        "train_on_imageset_predict_other is deprecated; use image_level_cv",
        DeprecationWarning, stacklevel=2,
    )
    # Build labels from full-dataset clustering for CV
    k = adaptive_leiden_k(len(P), cfg.leiden_k) if cfg.leiden_k_auto else cfg.leiden_k
    if cfg.cluster_space == "pca":
        X_clust = P
    else:
        X_clust = fit_umap(P, cfg=cfg)
    g = knn_igraph(X_clust, k)
    labels = leiden_partition(g, resolution, seed=cfg.seed)
    med, q25, q75, _ = image_level_cv(P, source_images, labels, cfg=cfg)
    n_a = len(P) // 2
    return med, n_a, len(P) - n_a


# ---------------------------------------------------------------------------
# Cluster characterisation: medoids
# ---------------------------------------------------------------------------

def cluster_medoids(P: np.ndarray, labels: np.ndarray, *,
                    samples_per_cluster: int = 8, seed: int = 0) -> dict:
    """Return medoid index + random samples per cluster.

    Medoid: point closest to cluster mean in PCA-whitened space.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        centre = P[idx].mean(axis=0)
        d = np.linalg.norm(P[idx] - centre, axis=1)
        order = np.argsort(d)
        medoid = int(idx[order[0]])
        extras_count = min(samples_per_cluster - 1, len(idx) - 1)
        if extras_count > 0:
            pick = rng.choice(len(idx), size=extras_count, replace=False)
            extras = [int(idx[i]) for i in pick if int(idx[i]) != medoid]
        else:
            extras = []
        out[int(c)] = {"medoid": medoid, "samples": [medoid] + extras,
                       "size": int(idx.size)}
    return out


# ---------------------------------------------------------------------------
# Treatment-frequency analysis (the thesis-deliverable section)
# ---------------------------------------------------------------------------

def per_image_cluster_frequencies(
    labels: np.ndarray, source_images: np.ndarray, *,
    drop_noise: bool = True,
):
    """Build (n_images, n_clusters) frequency matrix. Rows sum to 1.

    Returns (freq, image_names, cluster_ids, counts).
    """
    if drop_noise:
        mask = labels != -1
        labels = labels[mask]
        source_images = source_images[mask]
    image_names = np.array(sorted(set(source_images)))
    cluster_ids = np.array(sorted(set(labels)))
    cid_to_col = {c: i for i, c in enumerate(cluster_ids)}
    counts = np.zeros((len(image_names), len(cluster_ids)), dtype=np.int64)
    for r, img in enumerate(image_names):
        sub = labels[source_images == img]
        for c in sub:
            counts[r, cid_to_col[int(c)]] += 1
    row_sum = counts.sum(axis=1, keepdims=True).clip(min=1)
    freq = counts / row_sum
    return freq, image_names, cluster_ids, counts


def chi2_independence(
    counts: np.ndarray, min_image_patches: int = 20,
    *, n_permutations: int = 10_000, seed: int = 0,
):
    """Chi-square independence test on (image × cluster) counts.

    Drops images with < ``min_image_patches`` patches.  When the minimum
    expected cell count is < 5 the asymptotic χ² distribution is unreliable
    so we fall back to **Monte Carlo simulation under H₀** — multinomial
    draws per image with the pooled cluster proportions as the null
    (fixed row-margin, shared column probabilities).

    Returns ``(chi2, p, dof, n_dropped, method)`` where *method* is
    ``"asymptotic"`` or ``"monte_carlo"``.
    """
    from scipy.stats import chi2_contingency

    keep = counts.sum(axis=1) >= min_image_patches
    n_dropped = int((~keep).sum())
    table = counts[keep]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan"), 0, n_dropped, "insufficient"

    chi2_obs, p_asymp, dof, expected = chi2_contingency(table)

    if expected.min() >= 5:
        return float(chi2_obs), float(p_asymp), int(dof), n_dropped, "asymptotic"

    # Monte Carlo: simulate tables under independence with fixed row totals
    rng = np.random.default_rng(seed)
    row_sums = table.sum(axis=1)
    col_props = table.sum(axis=0).astype(np.float64)
    col_props /= col_props.sum()

    n_ge = 0
    for _ in range(n_permutations):
        sim = np.vstack([rng.multinomial(int(n_i), col_props) for n_i in row_sums])
        # Skip degenerate tables (all-zero columns)
        if (sim.sum(axis=0) == 0).any():
            continue
        try:
            chi2_sim, _, _, _ = chi2_contingency(sim)
        except ValueError:
            continue
        if chi2_sim >= chi2_obs:
            n_ge += 1
    p_mc = (n_ge + 1) / (n_permutations + 1)
    return float(chi2_obs), float(p_mc), int(dof), n_dropped, "monte_carlo"


def cluster_purity_by_image(
    labels: np.ndarray, source_images: np.ndarray, *, drop_noise: bool = True,
):
    """Per-cluster normalised entropy of the source-image distribution.

    1.0 = perfectly mixed; 0.0 = one image dominates.
    Caicedo et al. (Nat Methods 2017).
    """
    import math
    from scipy.stats import entropy as shannon_entropy
    if drop_noise:
        mask = labels != -1
        labels = labels[mask]
        source_images = source_images[mask]
    n_images = len(set(source_images))
    log_n = math.log(n_images) if n_images > 1 else 1.0
    rows = []
    for c in sorted(set(labels)):
        m = labels == c
        sub = source_images[m]
        _, counts = np.unique(sub, return_counts=True)
        h = float(shannon_entropy(counts / counts.sum()))
        rows.append({
            "cluster": int(c),
            "size": int(m.sum()),
            "n_images_present": int(len(counts)),
            "norm_entropy": h / log_n if log_n > 0 else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Group-level statistical tests
# ---------------------------------------------------------------------------

GroupMap = Mapping[str, str]  # image_name -> group_label


def _bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Returns adjusted p-values (same length as input).
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    adjusted = np.empty(n)
    adjusted[order[-1]] = p[order[-1]]
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(
            adjusted[order[i + 1]],
            p[order[i]] * n / (i + 1),
        )
    return np.clip(adjusted, 0.0, 1.0)


class GroupTestResult(NamedTuple):
    """Result of a group × cluster composition test."""
    statistic: float
    p_value: float
    method_used: str
    group_counts: np.ndarray   # (n_groups, n_clusters)
    group_names: list[str]
    cluster_ids: np.ndarray
    n_images_per_group: dict[str, int]


class MMDResult(NamedTuple):
    """Result of an MMD permutation test between two groups."""
    mmd_squared: float
    p_value: float
    n_permutations: int
    gamma: float
    group_pair: tuple[str, str]
    n_per_group: tuple[int, int]


class PERMANOVAResult(NamedTuple):
    """Result of a PERMANOVA test on per-image frequency vectors."""
    f_statistic: float
    p_value: float
    n_permutations: int
    r_squared: float
    n_per_group: dict[str, int]


def _build_group_counts(
    counts: np.ndarray,
    image_names: np.ndarray,
    group_map: GroupMap,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Aggregate image-level counts into group-level contingency table."""
    groups = sorted(set(group_map.values()))
    g2i = {g: i for i, g in enumerate(groups)}
    group_counts = np.zeros((len(groups), counts.shape[1]), dtype=np.int64)
    n_imgs = {g: 0 for g in groups}
    for r, img in enumerate(image_names):
        g = group_map.get(str(img))
        if g is None:
            continue
        group_counts[g2i[g]] += counts[r]
        n_imgs[g] += 1
    return group_counts, groups, n_imgs


def group_cluster_test(
    counts: np.ndarray,
    image_names: np.ndarray,
    cluster_ids: np.ndarray,
    group_map: GroupMap,
    *,
    min_expected: float = 5.0,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> GroupTestResult:
    """Test cluster composition differs between biological groups.

    Chi² when expected cell counts are sufficient; otherwise permutation
    chi² with image-level label shuffling (avoids pseudoreplication).
    Agresti (2002), ch. 3.
    """
    from scipy.stats import chi2_contingency

    group_counts, groups, n_imgs = _build_group_counts(
        counts, image_names, group_map,
    )
    if group_counts.shape[0] < 2:
        return GroupTestResult(
            float("nan"), float("nan"), "insufficient_groups",
            group_counts, groups, cluster_ids, n_imgs,
        )
    # drop empty clusters
    col_mask = group_counts.sum(axis=0) > 0
    gc = group_counts[:, col_mask]
    cids = cluster_ids[col_mask]
    if gc.shape[1] < 2:
        return GroupTestResult(
            float("nan"), float("nan"), "insufficient_clusters",
            gc, groups, cids, n_imgs,
        )

    chi2_obs, _, dof, expected = chi2_contingency(gc)

    if expected.min() >= min_expected:
        _, p, _, _ = chi2_contingency(gc)
        return GroupTestResult(
            float(chi2_obs), float(p), "chi2", gc, groups, cids, n_imgs,
        )

    # Permutation chi²: shuffle group labels at image level
    rng = np.random.default_rng(seed)
    mapped_images = np.array(
        [img for img in image_names if group_map.get(str(img)) is not None]
    )
    mapped_idx = np.array(
        [i for i, img in enumerate(image_names)
         if group_map.get(str(img)) is not None]
    )
    image_groups = np.array(
        [group_map[str(img)] for img in mapped_images]
    )
    n_ge = 0
    for _ in range(n_permutations):
        perm_groups = image_groups.copy()
        rng.shuffle(perm_groups)
        perm_map = {str(mapped_images[i]): perm_groups[i]
                    for i in range(len(mapped_images))}
        perm_gc, _, _ = _build_group_counts(counts, image_names, perm_map)
        perm_gc = perm_gc[:, col_mask]
        if perm_gc.shape[0] < 2 or (perm_gc.sum(axis=0) == 0).any():
            continue
        chi2_perm, _, _, _ = chi2_contingency(perm_gc)
        if chi2_perm >= chi2_obs:
            n_ge += 1
    p_perm = (n_ge + 1) / (n_permutations + 1)
    return GroupTestResult(
        float(chi2_obs), float(p_perm), "permutation_chi2",
        gc, groups, cids, n_imgs,
    )


def _median_heuristic_gamma(X: np.ndarray, max_pairs: int = 5000) -> float:
    """RBF gamma via median heuristic: gamma = 1 / (2 * median(||x-y||²)).

    Gretton et al. (JMLR 2012), §7.
    """
    from scipy.spatial.distance import pdist
    if len(X) > max_pairs:
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), max_pairs, replace=False)]
    dists_sq = pdist(X, metric="sqeuclidean")
    med = float(np.median(dists_sq))
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def mmd_permutation_test(
    Z: np.ndarray,
    source_images: np.ndarray,
    group_map: GroupMap,
    *,
    gamma: float | None = None,
    n_permutations: int = 1000,
    max_patches: int = 5000,
    seed: int = 0,
    correction: str = "bh",
) -> list[MMDResult]:
    """MMD² permutation test on per-image mean embeddings.

    Per-image means (not raw patches) avoid pseudoreplication.  Gretton
    et al. (JMLR 2012).  Returns one ``MMDResult`` per group pair.

    ``correction`` controls multiple-comparison adjustment:
    ``"bh"`` (Benjamini-Hochberg FDR, default) or ``"bonferroni"``.
    """
    import logging
    from sklearn.metrics.pairwise import rbf_kernel

    log = logging.getLogger(__name__)

    # per-image mean embeddings
    unique_images = sorted(set(source_images))
    img_means = {}
    for img in unique_images:
        mask = source_images == img
        img_means[img] = Z[mask].mean(axis=0)

    groups = sorted(set(group_map.values()))
    group_embeds: dict[str, np.ndarray] = {}
    for g in groups:
        imgs = [img for img in unique_images if group_map.get(str(img)) == g]
        if not imgs:
            continue
        group_embeds[g] = np.array([img_means[img] for img in imgs])

    active_groups = sorted(group_embeds.keys())
    if len(active_groups) < 2:
        return []

    # Warn when group sizes are very small
    for g in active_groups:
        n_g = len(group_embeds[g])
        if n_g < 5:
            log.warning(
                "MMD group %r has only %d image-level observations — "
                "statistical power is very limited", g, n_g,
            )

    # gamma via median heuristic on all image means
    all_means = np.vstack(list(group_embeds.values()))
    used_gamma = gamma if gamma is not None else _median_heuristic_gamma(all_means)

    def _mmd2(X: np.ndarray, Y: np.ndarray) -> float:
        XX = rbf_kernel(X, X, gamma=used_gamma)
        YY = rbf_kernel(Y, Y, gamma=used_gamma)
        XY = rbf_kernel(X, Y, gamma=used_gamma)
        return float(np.mean(XX) + np.mean(YY) - 2 * np.mean(XY))

    rng = np.random.default_rng(seed)
    pairs = [(active_groups[i], active_groups[j])
             for i in range(len(active_groups))
             for j in range(i + 1, len(active_groups))]

    raw_ps: list[float] = []
    raw_results: list[tuple] = []
    for ga, gb in pairs:
        Xa, Xb = group_embeds[ga], group_embeds[gb]
        mmd_obs = _mmd2(Xa, Xb)
        combined = np.concatenate([Xa, Xb], axis=0)
        na = len(Xa)
        n_ge = 0
        for _ in range(n_permutations):
            perm = rng.permutation(len(combined))
            mmd_p = _mmd2(combined[perm[:na]], combined[perm[na:]])
            if mmd_p >= mmd_obs:
                n_ge += 1
        p_raw = (n_ge + 1) / (n_permutations + 1)
        raw_ps.append(p_raw)
        raw_results.append((mmd_obs, ga, gb, len(Xa), len(Xb)))

    # Multiple-comparison correction
    if correction == "bh":
        corrected_ps = _bh_fdr(raw_ps)
    else:
        corrected_ps = np.minimum(1.0, np.array(raw_ps) * len(pairs))

    results = []
    for i, (mmd_obs, ga, gb, na, nb) in enumerate(raw_results):
        results.append(MMDResult(
            mmd_squared=float(mmd_obs),
            p_value=float(corrected_ps[i]),
            n_permutations=n_permutations,
            gamma=float(used_gamma),
            group_pair=(ga, gb),
            n_per_group=(na, nb),
        ))
    return results


def permanova_frequencies(
    freq: np.ndarray,
    image_names: np.ndarray,
    group_map: GroupMap,
    *,
    metric: str = "braycurtis",
    n_permutations: int = 999,
    seed: int = 0,
) -> PERMANOVAResult:
    """PERMANOVA on per-image cluster-frequency vectors.

    One observation per image (no pseudoreplication). Pseudo-F from the
    distance matrix per Anderson (Austral Ecology 2001).
    """
    from scipy.spatial.distance import pdist, squareform

    # map images to groups, keep only those in group_map
    mapped = [(i, group_map[str(img)])
              for i, img in enumerate(image_names)
              if group_map.get(str(img)) is not None]
    if len(mapped) < 3:
        return PERMANOVAResult(
            float("nan"), float("nan"), 0, float("nan"), {},
        )
    idx, grp_labels = zip(*mapped)
    idx = np.array(idx)
    grp_labels = np.array(grp_labels)
    F_sub = freq[idx]

    groups = sorted(set(grp_labels))
    n_per = {g: int((grp_labels == g).sum()) for g in groups}
    if len(groups) < 2 or any(v < 2 for v in n_per.values()):
        return PERMANOVAResult(
            float("nan"), float("nan"), 0, float("nan"), n_per,
        )

    D = squareform(pdist(F_sub, metric=metric))
    N = len(F_sub)

    def _pseudo_f(labels: np.ndarray) -> float:
        """Anderson (2001) pseudo-F from a distance matrix."""
        D_sq = D ** 2
        # total SS
        ss_t = D_sq.sum() / (2 * N)
        # within-group SS
        ss_w = 0.0
        a = len(set(labels))
        for g in set(labels):
            m = labels == g
            n_g = m.sum()
            if n_g < 2:
                continue
            ss_w += D_sq[np.ix_(m, m)].sum() / (2 * n_g)
        ss_a = ss_t - ss_w
        df_a = a - 1
        df_w = N - a
        if df_w == 0 or ss_w == 0:
            return float("inf")
        return (ss_a / df_a) / (ss_w / df_w)

    f_obs = _pseudo_f(grp_labels)
    r_sq = 1.0 - 1.0 / (1.0 + f_obs * (len(groups) - 1) / (N - len(groups))) \
        if np.isfinite(f_obs) else float("nan")

    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(n_permutations):
        perm_labels = rng.permutation(grp_labels)
        if _pseudo_f(perm_labels) >= f_obs:
            n_ge += 1
    p = (n_ge + 1) / (n_permutations + 1)

    return PERMANOVAResult(
        f_statistic=float(f_obs),
        p_value=float(p),
        n_permutations=n_permutations,
        r_squared=float(r_sq),
        n_per_group=n_per,
    )


# ---------------------------------------------------------------------------
# Index.csv writeback
# ---------------------------------------------------------------------------

def write_cluster_columns(
    data_root, filenames: np.ndarray, column_to_labels: dict,
    *, dry_run: bool = False,
) -> dict:
    """Atomic + idempotent. New columns appended; existing overwritten."""
    import os
    from pathlib import Path
    import pandas as pd

    csv_path = Path(data_root) / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise KeyError("expected a 'filename' column in index.csv")
    original_cols = list(df.columns)

    summary = {}
    for col, labels in column_to_labels.items():
        s = pd.Series(np.asarray(labels, dtype=np.int64), index=filenames, name=col)
        aligned = df["filename"].map(s).fillna(-2).astype(np.int64)
        action = "overwrote" if col in df.columns else "appended"
        df[col] = aligned.values
        summary[col] = {
            "action": action,
            "n_assigned": int((aligned != -2).sum()),
            "n_unmatched": int((aligned == -2).sum()),
            "n_clusters": int(len(set(int(v) for v in labels if v != -1))),
        }

    new_cols = [c for c in df.columns if c not in original_cols]
    df = df[[c for c in original_cols if c in df.columns] + new_cols]

    if dry_run:
        return summary
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    return summary
