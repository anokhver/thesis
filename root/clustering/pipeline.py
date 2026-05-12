"""Patch clustering pipeline for fluorescence-microscopy SSL features.

Steps: extract features → L2 + PCA whiten → UMAP-15 → Leiden → bootstrap
stability → per-image frequency vectors → chi-square test.

Key method choices:
* L2 + PCA whitening: Wang & Isola (2020), Mu et al. (2017).
* UMAP-15 not UMAP-2 for clustering geometry: Chari & Pachter (2023).
* Leiden on kNN graph: Traag et al. (2019).
* Bootstrap stability: Hennig (2007), Lange et al. (2004).
* GMM-BIC cross-check: McConville et al. (2019).
* HDBSCAN-leaf fallback: McInnes et al. (2017).
* TVN: Kraus et al. (CVPR 2024).
* Image-level CV: Caicedo et al. (2017).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClusterCfg:
    """Clustering pipeline configuration. Defaults match research-justified recipe."""

    seed: int = 42

    # Preprocessing
    l2_normalise: bool = True
    pca_dim: int | None = None         # None -> auto via RankMe
    pca_dim_target_var: float = 0.95   # auto: smallest d capturing this
    pca_dim_max: int = 200             # auto: hard cap
    tvn_reference_pattern: str | None = None  # substring matched against
                                              # source_image; if set, the
                                              # mean of those patches is
                                              # subtracted before PCA

    # UMAP (clustering geometry)
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
    leiden_resolutions: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0)

    # Bootstrap stability
    bootstrap_b: int = 50
    bootstrap_frac: float = 0.7

    # GMM-BIC cross-check
    gmm_k_min: int = 3
    gmm_k_max: int = 25

    # HDBSCAN fallback
    hdb_min_cluster_size: int = 30
    hdb_min_samples: int = 5
    hdb_method: str = "leaf"

    # Permutation null
    permutation_p: int = 10

    # Cluster characterization
    samples_per_cluster: int = 8

    # Differential frequency test
    chi2_min_image_patches: int = 20  # drop images with fewer patches

    # IO
    embedding_cache: str | None = None  # path to .npy cache; None disables


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
    """Run frozen encoder over dataset, return globally-pooled features.

    Returns (Z, filenames, source_images, image_indices). Caches to ``cache_path`` if set.
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
    """Choose `d` such that cumulative explained variance >= `target_var`.

    Lower-bounded by 5; upper-bounded by `hard_cap` and `min(N-1, D)`.
    """
    from sklearn.decomposition import PCA
    n_max = int(min(Z.shape[0] - 1, Z.shape[1], hard_cap))
    pca = PCA(n_components=n_max, random_state=0).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(np.searchsorted(cum, target_var) + 1)
    return max(5, min(d, n_max))


def tvn_centre(Z: np.ndarray, source_images: np.ndarray, pattern: str) -> np.ndarray:
    """Subtract mean of patches matching ``pattern`` in source_image. TVN (Kraus et al., CVPR 2024).

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
    """Returns ``(P, pca, cum_var)`` with ``P`` whitened to unit variance."""
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

def umap_to_knn_igraph(U: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors
    import igraph as ig
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1).fit(U)
    _, idx = nn.kneighbors(U)
    edges = [(int(i), int(j)) for i in range(len(U)) for j in idx[i, 1:]]
    g = ig.Graph(n=len(U), edges=edges, directed=False)
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
    g = umap_to_knn_igraph(U, k)
    return {float(r): leiden_partition(g, r, seed=seed) for r in resolutions}, g


def bootstrap_stability(
    U: np.ndarray, resolutions: Sequence[float], k: int,
    B: int, frac: float, seed: int = 0,
):
    """Leiden bootstrap stability. Return ``({res: (mean_ari, std_ari)}, {res: labels})``.

    70% subsample → re-cluster → 1-NN propagate → ARI vs full partition.
    Hennig (2007); Lange et al. (2004).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(U)
    g_full = umap_to_knn_igraph(U, k)
    full = {float(r): leiden_partition(g_full, r, seed=seed) for r in resolutions}
    aris = {float(r): [] for r in resolutions}
    for b in range(B):
        idx = rng.choice(n, size=int(frac * n), replace=True)
        U_sub = U[idx]
        g_sub = umap_to_knn_igraph(U_sub, k)
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

def gmm_bic_scan(U: np.ndarray, k_min: int, k_max: int, seed: int = 0):
    from sklearn.mixture import GaussianMixture
    bics = []
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=seed, max_iter=200, reg_covar=1e-4,
        ).fit(U)
        bics.append((k, float(gmm.bic(U))))
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
):
    """Estimate permutation-null ARI by shuffling PCs and rerunning UMAP+Leiden.

    Follow Witten & Tibshirani (2010). Real ARI should exceed this distribution.
    """
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(cfg.seed)
    aris = []
    for it in range(cfg.permutation_p):
        P_perm = P.copy()
        for d in range(P_perm.shape[1]):
            rng.shuffle(P_perm[:, d])
        U_perm = fit_umap(P_perm, cfg=cfg, seed=cfg.seed + it)
        g_perm = umap_to_knn_igraph(U_perm, cfg.leiden_k)
        lab_perm = leiden_partition(g_perm, resolution, seed=cfg.seed + it)
        aris.append(adjusted_rand_score(labels_real, lab_perm))
    return float(np.mean(aris)), float(np.std(aris)), float(np.max(aris))


def train_on_imageset_predict_other(
    P: np.ndarray, source_images: np.ndarray, *,
    cfg: ClusterCfg, resolution: float,
):
    """Run image-level cross-validation by clustering half the images and predicting the rest with kNN.

    Low ARI means clusters do not generalise across biological samples.
    Caicedo et al. (2017).
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(cfg.seed)
    images = np.array(sorted(set(source_images)))
    rng.shuffle(images)
    half = images[: max(1, len(images) // 2)]
    mask_a = np.isin(source_images, half)

    P_a, P_b = P[mask_a], P[~mask_a]
    if len(P_a) < cfg.leiden_k + 5 or len(P_b) < cfg.leiden_k + 5:
        return float("nan"), int(mask_a.sum()), int((~mask_a).sum())

    U_a = fit_umap(P_a, cfg=cfg, seed=cfg.seed)
    U_b = fit_umap(P_b, cfg=cfg, seed=cfg.seed + 1)
    lab_a        = leiden_partition(umap_to_knn_igraph(U_a, cfg.leiden_k), resolution, seed=cfg.seed)
    lab_b_native = leiden_partition(umap_to_knn_igraph(U_b, cfg.leiden_k), resolution, seed=cfg.seed)
    lab_b_pred   = KNeighborsClassifier(n_neighbors=5).fit(P_a, lab_a).predict(P_b)
    ari = float(adjusted_rand_score(lab_b_native, lab_b_pred))
    return ari, int(mask_a.sum()), int((~mask_a).sum())


# ---------------------------------------------------------------------------
# Cluster characterisation: medoids
# ---------------------------------------------------------------------------

def cluster_medoids(P: np.ndarray, labels: np.ndarray, *,
                    samples_per_cluster: int = 8, seed: int = 0) -> dict:
    """Return medoid index + random samples for each cluster.

    Medoid = point closest to cluster mean in PCA-whitened space.
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


def chi2_independence(counts: np.ndarray, min_image_patches: int = 20):
    """Chi-square independence test on (image × cluster) counts.

    Drops images with < ``min_image_patches`` patches. Returns ``(chi2, p, dof, n_dropped)``.
    """
    from scipy.stats import chi2_contingency
    keep = counts.sum(axis=1) >= min_image_patches
    n_dropped = int((~keep).sum())
    table = counts[keep]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), float("nan"), 0, n_dropped
    chi2, p, dof, _ = chi2_contingency(table)
    return float(chi2), float(p), int(dof), n_dropped


def cluster_purity_by_image(
    labels: np.ndarray, source_images: np.ndarray, *, drop_noise: bool = True,
):
    """Compute per-cluster normalised entropy of the source-image distribution.

    ``1.0`` means perfectly mixed. ``0.0`` means one image dominates.
    Caicedo et al. (2017).
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
