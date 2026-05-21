"""Sanity tests for clustering quality.

Bundle of classical "is this clustering meaningful?" diagnostics:

* :func:`cluster_size_stats` — bookkeeping (K, noise fraction, Gini).
* :func:`permutation_null_ari` — null clustering ARI under structure-
  destroying permutations (Tibshirani et al. 2001; Witten & Tibshirani
  2010 §3.2).
* :func:`image_level_cv` — kNN label-transfer ARI on grouped half-splits
  (Caicedo et al. 2017, adapted).
* :func:`cluster_medoids` — representative patches per cluster.
* :func:`internal_validity_indices` — silhouette (Rousseeuw 1987),
  Davies-Bouldin (1979), Caliński-Harabasz (1974).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import ClusterCfg, GroupMap
from .graph import knn_igraph, leiden_partition


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
    group_map: GroupMap | None = None,
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


class InternalValidity(NamedTuple):
    """Internal cluster-validity indices computed in the clustering space."""
    silhouette: float                # Rousseeuw 1987; higher is better, in [-1, 1]
    davies_bouldin: float            # Davies & Bouldin 1979; lower is better
    calinski_harabasz: float         # Caliński & Harabasz 1974; higher is better
    n_used: int
    n_clusters: int
    method: str


def internal_validity_indices(
    P: np.ndarray, labels: np.ndarray, *,
    drop_noise: bool = True,
    sample_size: int | None = 10_000,
    seed: int = 0,
) -> InternalValidity:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz on a labelled embedding.

    Three classical internal indices, complementary to the bootstrap-ARI
    stability check:

    * **Silhouette** — Rousseeuw, *J. Comput. Appl. Math.* 20:53-65 (1987).
      In [-1, 1]; higher is better. Combines within-cluster compactness
      and between-cluster separation; widely used as a sanity check on
      kNN-graph community detection.
    * **Davies-Bouldin** — Davies & Bouldin, *IEEE TPAMI* 1(2):224-227
      (1979). Lower is better; ratio of within- to between-cluster
      scatter, averaged over clusters.
    * **Calinski-Harabasz** — Caliński & Harabasz, *Commun. Stat.* 3(1):
      1-27 (1974). Higher is better; variance-ratio criterion (between-
      to within-cluster sum-of-squares, scaled by degrees of freedom).

    Operates on ``P`` (the same PCA-whitened space used for Leiden), so
    indices reflect geometry in the clustering space and are not
    distorted by the UMAP-2D layout. Noise points (label == -1) are
    dropped before computing indices; silhouette subsamples to
    ``sample_size`` for tractable runtime at large N.
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )

    if drop_noise:
        mask = labels != -1
        P_use = P[mask]
        labels_use = labels[mask]
    else:
        P_use = P
        labels_use = labels

    n_used = int(P_use.shape[0])
    n_clusters = int(len(set(labels_use)))
    if n_clusters < 2 or n_used < 2:
        return InternalValidity(
            silhouette=float("nan"),
            davies_bouldin=float("nan"),
            calinski_harabasz=float("nan"),
            n_used=n_used,
            n_clusters=n_clusters,
            method="insufficient_clusters",
        )

    rng = np.random.default_rng(seed)
    if sample_size is not None and n_used > sample_size:
        idx = rng.choice(n_used, size=sample_size, replace=False)
        P_sil = P_use[idx]
        labels_sil = labels_use[idx]
        # Re-check that the subsample still has >=2 clusters
        if len(set(labels_sil)) < 2:
            P_sil, labels_sil = P_use, labels_use
        sil_method = f"silhouette_subsampled_{sample_size}"
    else:
        P_sil, labels_sil = P_use, labels_use
        sil_method = "silhouette_full"

    try:
        sil = float(silhouette_score(P_sil, labels_sil, metric="euclidean"))
    except ValueError:
        sil = float("nan")
    try:
        dbi = float(davies_bouldin_score(P_use, labels_use))
    except ValueError:
        dbi = float("nan")
    try:
        chi = float(calinski_harabasz_score(P_use, labels_use))
    except ValueError:
        chi = float("nan")

    return InternalValidity(
        silhouette=sil,
        davies_bouldin=dbi,
        calinski_harabasz=chi,
        n_used=n_used,
        n_clusters=n_clusters,
        method=sil_method,
    )
