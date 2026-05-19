"""Patch-level SSL feature clustering with image-aware statistics.

HPL-inspired (Quiros et al., Nat Commun 15:4596, 2024): SSL patch
embeddings → kNN+Leiden → per-image cluster frequency vectors. HPL
itself clusters directly on the 128-D representations (kNN K=250,
Methods §"Clustering representations"); we add an L2 + PCA-whiten
preprocessing step because our pooled Swin features sit in 768 dims
with measured effective rank around 5–22, so reducing to
``d ≈ eff_rank`` before kNN cuts the noise floor in the Leiden
graph. ``pca_dim=None`` selects ``d`` from the cumulative variance
curve; see :func:`auto_pca_dim`.

HPL's Cox PH / multinomial-logistic readout is replaced by
PERMANOVA on per-image cluster-frequency vectors (multi-group
treatment comparison rather than survival). Every test uses the
source image as the unit of analysis to avoid pseudoreplication
(Lazic et al., BMC Neurosci 11:5, 2010).
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

    # Preprocessing
    l2_normalise: bool = True
    pca_dim: int | None = None         # None -> auto via RankMe-style
                                       # 95% cumulative explained variance
    pca_dim_target_var: float = 0.95   # auto: smallest d capturing this
    pca_dim_max: int = 200             # auto: hard cap
    control_reference_pattern: str | None = None  # substring matched against
                                              # source_image; if set, the
                                              # mean of those patches is
                                              # subtracted before PCA

    # UMAP (2D visualisation only; clustering happens directly in PCA)
    umap_metric: str = "cosine"
    umap_init: str = "pca"
    umap_n_epochs: int = 500
    umap2_n_neighbors: int = 30
    umap2_min_dist: float = 0.1

    # Leiden (primary)
    leiden_k: int = 30
    leiden_k_auto: bool = True         # auto-scale k based on N
    leiden_resolutions: tuple[float, ...] = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0)

    # Bootstrap stability
    bootstrap_b: int = 50
    bootstrap_frac: float = 0.7

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
    permanova_n_permutations: int = 999
    permanova_metric: str = "braycurtis"
    kw_correction: str = "bh"           # per-cluster Kruskal-Wallis multiple-comp

    # IO
    embedding_cache: str | None = None  # path to .npy cache; None disables


# ---------------------------------------------------------------------------
# Adaptive parameter scaling for small N
# ---------------------------------------------------------------------------

def adaptive_leiden_k(N: int, default_k: int = 30) -> int:
    """Scale Leiden k for small datasets.

    Adapts the PhenoGraph code default k=30 (Levine et al., Cell
    162:184-197, 2015) to small N by linear scaling, lower-bounded
    at 5, capped at ``default_k``. Unlike PhenoGraph we omit Jaccard
    weighting and use Leiden instead of Louvain.
    """
    k = int(round(0.015 * N))
    return max(5, min(default_k, k))


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
        # Tolerate heterogeneous index.csv schemas: older zip-tiler outputs
        # only write ``source_npy`` / ``source_path`` and omit ``source_image``
        # and ``image_index``. Fall back to whatever per-source identifier
        # is available, and derive a stable per-image int when missing.
        def _resolve_source_image(r):
            val = (r.get("source_image")
                   or r.get("source_npy")
                   or r.get("source_path"))
            if not val:
                return "UNKNOWN"
            s = str(val).replace("\\", "/")
            return s.rsplit("/", 1)[-1] or s

        filenames     = np.array([r["filename"] for r in records])
        source_images = np.array([_resolve_source_image(r) for r in records])

        parsed_indices: list[int] = []
        needs_derive = False
        for r in records:
            raw = str(r.get("image_index", "")).strip()
            if not raw:
                needs_derive = True
                break
            try:
                parsed_indices.append(int(raw))
            except ValueError:
                needs_derive = True
                break
        if not needs_derive and len(parsed_indices) == len(records):
            image_indices = np.array(parsed_indices, dtype=np.int64)
        else:
            seen: dict[str, int] = {}
            indices_list: list[int] = []
            for s in source_images.tolist():
                if s not in seen:
                    seen[s] = len(seen)
                indices_list.append(seen[s])
            image_indices = np.array(indices_list, dtype=np.int64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path,
                {"Z": Z, "filenames": filenames,
                 "source_images": source_images,
                 "image_indices": image_indices},
                allow_pickle=True)
    return Z, filenames, source_images, image_indices


# ---------------------------------------------------------------------------
# Preprocessing: L2 + (control-mean centering) + PCA whitening
# ---------------------------------------------------------------------------

def auto_pca_dim(Z: np.ndarray, target_var: float = 0.95, hard_cap: int = 200) -> int:
    """Smallest ``d`` with cumulative explained variance >= ``target_var``.

    Lower-bounded by 5, upper-bounded by ``hard_cap`` and ``min(N-1, D)``.

    For pooled CNN / transformer features the effective rank is typically
    far below the nominal dimension; fixed defaults inherited from
    scRNA-seq pipelines (e.g. Scanpy's ``n_comps=50``; Wolf et al.,
    Genome Biol 2018) over-allocate here, so noise PCs get rescaled to
    unit variance by ``PCA(whiten=True)`` and dominate Euclidean kNN
    distances.
    """
    from sklearn.decomposition import PCA
    n_max = int(min(Z.shape[0] - 1, Z.shape[1], hard_cap))
    pca = PCA(n_components=n_max, random_state=0).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(np.searchsorted(cum, target_var) + 1)
    return max(5, min(d, n_max))


def subtract_control_mean(Z: np.ndarray, source_images: np.ndarray, pattern: str) -> np.ndarray:
    """Subtract mean of reference patches matching ``pattern``.

    Control-mean centering: removes additive plate/well offset.
    Inspired by TVN (Ando et al., bioRxiv:161422, 2017) but only
    performs mean subtraction, not the full TVN covariance whitening.
    """
    if not pattern:
        return Z
    pat = pattern.upper()
    mask = np.array([pat in s.upper() for s in source_images])
    if mask.sum() == 0:
        raise ValueError(f"no source_image matched control pattern {pattern!r}")
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
# UMAP (2D visualisation only)
# ---------------------------------------------------------------------------

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
    *, source_images: np.ndarray | None = None,
):
    """Bootstrap stability of Leiden partitions.

    Subsample → re-cluster → 1-NN propagate → ARI vs full partition.
    Adapted from the stability framework of Lange et al. (Neural
    Computation 16:1299-1323, 2004) with ARI replacing their Hamming
    distance. When ``source_images`` is given, resampling is at the
    image-block level to respect the statistical unit; otherwise
    patch-level bootstrap. Returns ``({res: (mean_ari, std_ari)},
    {res: labels})``.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = len(U)
    g_full = knn_igraph(U, k)
    full = {float(r): leiden_partition(g_full, r, seed=seed) for r in resolutions}
    aris = {float(r): [] for r in resolutions}

    image_block = source_images is not None
    if image_block:
        unique_images = np.array(sorted(set(source_images)))
        img_to_idx = {img: np.where(source_images == img)[0]
                      for img in unique_images}
        n_img = len(unique_images)
        n_img_sample = max(2, int(frac * n_img))

    for b in range(B):
        if image_block:
            sampled = rng.choice(unique_images, size=n_img_sample, replace=False)
            idx = np.concatenate([img_to_idx[img] for img in sampled])
        else:
            idx = rng.choice(n, size=int(frac * n), replace=False)
        U_sub = U[idx]
        if len(U_sub) <= k:
            continue
        g_sub = knn_igraph(U_sub, k)
        for r in resolutions:
            lab_sub = leiden_partition(g_sub, r, seed=seed + b)
            knn = KNeighborsClassifier(n_neighbors=1).fit(U_sub, lab_sub)
            propagated = knn.predict(U)
            aris[float(r)].append(adjusted_rand_score(full[float(r)], propagated))
    summary = {r: (float(np.mean(aris[r])) if aris[r] else float("nan"),
                   float(np.std(aris[r])) if aris[r] else float("nan"))
               for r in aris}
    return summary, full


def pick_resolution(summary: dict, full: dict):
    """Highest mean ARI among non-trivial partitions; tie-break by fewer clusters.

    Trivial partitions (K=1 or K=0) are excluded before sorting: with K=1
    the bootstrap ARI is automatically 1.0 because every resampling
    "agrees" on the single label, so a collapsed resolution would
    otherwise win the stability sort.
    """
    import logging

    candidates = [(r, m, s, len(set(full[r]))) for r, (m, s) in summary.items()]
    nontrivial = [t for t in candidates if t[3] >= 2]
    if not nontrivial:
        logging.getLogger(__name__).warning(
            "All resolutions collapsed to K<2; falling back to the lowest "
            "candidate to avoid an empty pick."
        )
        nontrivial = candidates
    nontrivial.sort(key=lambda t: (-t[1], t[3]))
    return nontrivial[0]  # (resolution, mean_ari, std_ari, n_clusters)


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
    source_images: np.ndarray | None = None,
):
    """Permutation-null ARI for clustering quality.

    With ``source_images``: image-mean-swap null. Decompose patches into
    ``image_mean + residual``, permute image-means across images,
    reconstruct, re-cluster, ARI vs real. Preserves multivariate
    covariance and within-image structure but breaks image-level batch
    effects. Without: per-PC shuffle null — each PC column is
    independently permuted, destroying joint structure while preserving
    marginals (Tibshirani et al., JRSS-B 63:411-423, 2001; adapted
    in Witten & Tibshirani, JASA 105:713-726, 2010 §3.2).
    Returns ``(mean_ari, std_ari, max_ari)``.
    """
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(cfg.seed)
    effective_k = k if k is not None else cfg.leiden_k

    image_swap = source_images is not None
    if image_swap:
        unique_images = np.array(sorted(set(source_images)))
        img_to_idx = {img: np.where(source_images == img)[0]
                      for img in unique_images}
        image_means = np.stack(
            [P[img_to_idx[img]].mean(axis=0) for img in unique_images]
        )
        residuals = P.copy()
        for j, img in enumerate(unique_images):
            residuals[img_to_idx[img]] -= image_means[j]

    aris = []
    for it in range(cfg.permutation_p):
        if image_swap:
            perm = rng.permutation(len(unique_images))
            P_perm = np.empty_like(P)
            for j, img in enumerate(unique_images):
                P_perm[img_to_idx[img]] = (
                    image_means[perm[j]] + residuals[img_to_idx[img]]
                )
        else:
            P_perm = P.copy()
            for d in range(P_perm.shape[1]):
                rng.shuffle(P_perm[:, d])
        g_perm = knn_igraph(P_perm, effective_k)
        lab_perm = leiden_partition(g_perm, resolution, seed=cfg.seed + it)
        aris.append(adjusted_rand_score(labels_real, lab_perm))
    return float(np.mean(aris)), float(np.std(aris)), float(np.max(aris))


def image_level_cv(
    P: np.ndarray, source_images: np.ndarray, labels: np.ndarray, *,
    cfg: ClusterCfg,
    group_map: "GroupMap | None" = None,
):
    """Repeated grouped half-split CV for label transfer.

    kNN in PCA space transfers labels across held-out images; tests
    whether the embedding generalises beyond training images. With
    ``group_map``, half-split is stratified by treatment group so
    every group is represented on both sides.
    Returns ``(median_ari, q25, q75, per_split_aris)``.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(cfg.seed)
    unique_images = np.array(sorted(set(source_images)))
    n_img = len(unique_images)

    if n_img < 4:
        return float("nan"), float("nan"), float("nan"), []

    if group_map is not None:
        groups_per_img = np.array(
            [group_map.get(str(img), "__none__") for img in unique_images]
        )
        group_buckets = {
            g: unique_images[groups_per_img == g]
            for g in sorted(set(groups_per_img)) if g != "__none__"
        }

    aris: list[float] = []
    for rep in range(cfg.cv_n_repeats):
        if group_map is not None:
            train_imgs: list = []
            for g, imgs in group_buckets.items():
                if len(imgs) < 2:
                    train_imgs.extend(imgs.tolist())
                    continue
                perm_g = rng.permutation(len(imgs))
                train_imgs.extend(imgs[perm_g[: len(imgs) // 2]].tolist())
            half = np.array(train_imgs)
        else:
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


class PERMANOVAResult(NamedTuple):
    """Result of a PERMANOVA test on per-image frequency vectors."""
    f_statistic: float
    p_value: float
    n_permutations: int
    r_squared: float
    n_per_group: dict[str, int]


class PerClusterTestResult(NamedTuple):
    """Result of per-cluster Kruskal-Wallis with multiple-comparison correction."""
    cluster_ids: np.ndarray
    statistics: np.ndarray         # H per cluster (NaN when undefined)
    p_values_raw: np.ndarray
    p_values_corrected: np.ndarray
    group_names: list[str]
    n_per_group: dict[str, int]
    method: str                    # e.g. "kruskal_wallis_bh"


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
    n_permutations: int = 10_000,
    seed: int = 0,
) -> GroupTestResult:
    """Test cluster composition differs between biological groups.

    χ² statistic with **image-level permutation null**: group labels are
    shuffled across images (not patches) so the test respects the
    statistical unit (Lazic et al., BMC Neurosci 11:5, 2010). The
    asymptotic χ² distribution is not used — its degrees of freedom
    assume independent observations, but patches from the same image
    are not independent, so it would inflate significance.
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

    chi2_obs, _, _, _ = chi2_contingency(gc)

    # Permutation chi²: shuffle group labels at image level
    rng = np.random.default_rng(seed)
    mapped_images = np.array(
        [img for img in image_names if group_map.get(str(img)) is not None]
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


def per_cluster_kruskal_wallis(
    freq: np.ndarray,
    image_names: np.ndarray,
    cluster_ids: np.ndarray,
    group_map: GroupMap,
    *,
    correction: str = "bh",
) -> PerClusterTestResult:
    """Per-cluster Kruskal-Wallis across groups, BH-FDR corrected.

    For each cluster column, tests whether per-image frequency
    distributions differ between biological groups. Unit of analysis
    is the image. Used as a posthoc on top of PERMANOVA to identify
    *which* clusters drive a global difference.

    Multiple-comparison correction: ``"bh"`` (Benjamini & Hochberg,
    J R Stat Soc B 1995) or ``"bonferroni"``. Returns
    :class:`PerClusterTestResult` with NaN entries for clusters where
    the KW statistic is undefined (e.g. all values tied).
    """
    import logging
    from scipy.stats import kruskal

    log = logging.getLogger(__name__)

    mapped = [(i, group_map[str(img)])
              for i, img in enumerate(image_names)
              if group_map.get(str(img)) is not None]
    k_clusters = len(cluster_ids)
    nan_arr = np.full(k_clusters, np.nan)
    if len(mapped) < 3:
        return PerClusterTestResult(
            cluster_ids=cluster_ids,
            statistics=nan_arr.copy(),
            p_values_raw=nan_arr.copy(),
            p_values_corrected=nan_arr.copy(),
            group_names=[],
            n_per_group={},
            method=f"kruskal_wallis_{correction}",
        )
    idx, grp_labels = zip(*mapped)
    idx = np.array(idx)
    grp_labels = np.array(grp_labels)
    F_sub = freq[idx]

    groups = sorted({str(g) for g in grp_labels})
    n_per = {g: int((grp_labels == g).sum()) for g in groups}
    if len(groups) < 2:
        return PerClusterTestResult(
            cluster_ids=cluster_ids,
            statistics=nan_arr.copy(),
            p_values_raw=nan_arr.copy(),
            p_values_corrected=nan_arr.copy(),
            group_names=list(groups),
            n_per_group=n_per,
            method=f"kruskal_wallis_{correction}",
        )

    for g, n in n_per.items():
        if n < 2:
            log.warning(
                "Kruskal-Wallis: group %r has only %d image(s); "
                "per-cluster power is very limited", g, n,
            )

    stats_arr = np.full(k_clusters, np.nan)
    p_raw = np.full(k_clusters, np.nan)
    for k in range(k_clusters):
        samples = [F_sub[grp_labels == g, k] for g in groups]
        try:
            h, p_k = kruskal(*samples)
        except ValueError:
            continue
        if np.isnan(h):
            continue
        stats_arr[k] = float(h)
        p_raw[k] = float(p_k)

    valid = ~np.isnan(p_raw)
    p_corr = np.full(k_clusters, np.nan)
    if valid.any():
        if correction == "bh":
            adj = _bh_fdr(p_raw[valid])
        elif correction == "bonferroni":
            adj = np.minimum(1.0, p_raw[valid] * int(valid.sum()))
        else:
            raise ValueError(f"unknown correction: {correction!r}")
        p_corr[valid] = adj

    return PerClusterTestResult(
        cluster_ids=cluster_ids,
        statistics=stats_arr,
        p_values_raw=p_raw,
        p_values_corrected=p_corr,
        group_names=list(groups),
        n_per_group=n_per,
        method=f"kruskal_wallis_{correction}",
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
