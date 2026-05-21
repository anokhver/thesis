"""Gaussian-RBF MMD two-sample test (Gretton et al. 2012) on per-image features.

Operates on the per-image mean-embedding matrix produced by
:func:`synaptic_ssl.clustering.frequencies.per_image_mean_embeddings`,
so the statistical unit is the image (Lazic et al., *BMC Neurosci*
11:5, 2010). Bandwidth uses the median heuristic across all
mapped images so MMD² values across group pairs are on a common
scale; pair p-values are corrected with Benjamini-Hochberg by
default (Benjamini & Hochberg, *JRSS-B* 1995).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .config import GroupMap
from .stats_group import _bh_fdr


def _median_heuristic_sigma(Z: np.ndarray, *,
                            max_samples: int = 1000, seed: int = 0) -> float:
    """RBF bandwidth from the median pairwise distance.

    σ = sqrt(median ||x_i - x_j||² / 2) over the pooled sample. The
    median heuristic is the canonical default for Gaussian-kernel MMD
    (Gretton et al., JMLR 13:723-773, 2012, §8); see also Garreau,
    Jitkrittum & Kanagawa, *arXiv:1707.07269* (2017) for an analysis.
    Subsamples to ``max_samples`` rows to keep the O(N²) pairwise
    distance computation tractable.
    """
    from scipy.spatial.distance import pdist
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]
    if Z.shape[0] < 2:
        return 1.0
    d2 = pdist(Z, metric="sqeuclidean")
    med = float(np.median(d2))
    if med <= 0.0:
        return 1.0
    return float(np.sqrt(med / 2.0))


def mmd2_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Unbiased MMD² estimator with a Gaussian RBF kernel.

    Implementation of Gretton et al. (JMLR 13:723-773, 2012),
    Lemma 6 / Eq. (3). The unbiased estimator drops the diagonal of
    the within-sample Gram matrices so MMD² is unbiased under the
    null but can take small negative values in finite samples.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma * sigma)
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m = X.shape[0]
    n = Y.shape[0]
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    term_xx = Kxx.sum() / (m * (m - 1)) if m > 1 else 0.0
    term_yy = Kyy.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_xy = Kxy.sum() / (m * n) if (m and n) else 0.0
    return float(term_xx + term_yy - 2.0 * term_xy)


def mmd_two_sample_test(
    X: np.ndarray, Y: np.ndarray, *,
    n_permutations: int = 999,
    sigma: float | None = None,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Two-sample permutation test on unbiased Gaussian-RBF MMD².

    Pools ``(X, Y)``, recomputes MMD² under random splits of the same
    sizes, and reports ``(MMD²_obs, p, sigma)``. Default bandwidth via
    the median heuristic on the pooled sample. p = (1 + #{MMD²_perm ≥
    MMD²_obs}) / (B + 1) per Gretton et al. (2012) §6.
    """
    pooled = np.vstack([np.asarray(X), np.asarray(Y)])
    if sigma is None:
        sigma = _median_heuristic_sigma(pooled, seed=seed)
    rng = np.random.default_rng(seed)
    m = X.shape[0]
    mmd_obs = mmd2_unbiased(X, Y, sigma)
    n_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(pooled.shape[0])
        X_p = pooled[perm[:m]]
        Y_p = pooled[perm[m:]]
        if mmd2_unbiased(X_p, Y_p, sigma) >= mmd_obs:
            n_ge += 1
    p = (n_ge + 1) / (n_permutations + 1)
    return float(mmd_obs), float(p), float(sigma)


class PairwiseMMDResult(NamedTuple):
    """Pairwise Gaussian-RBF MMD² between treatment groups."""
    pairs: list[tuple[str, str]]
    mmd2: np.ndarray
    p_values_raw: np.ndarray
    p_values_corrected: np.ndarray
    n_per_group: dict[str, int]
    sigma: float
    n_permutations: int
    method: str


def pairwise_mmd_groups(
    per_image_features: np.ndarray,
    image_names: np.ndarray,
    group_map: GroupMap,
    *,
    n_permutations: int = 999,
    sigma: float | None = None,
    correction: str = "bh",
    seed: int = 0,
) -> PairwiseMMDResult:
    """Pairwise Gaussian-RBF MMD² between groups on per-image vectors.

    For each pair of groups, runs :func:`mmd_two_sample_test` on the
    rows of ``per_image_features`` belonging to those groups, then
    corrects across the C(K, 2) pair p-values with Benjamini-Hochberg
    FDR (``correction='bh'``) or Bonferroni. The kernel bandwidth is
    estimated **once** by the median heuristic on the pooled per-image
    features (all groups), so MMD² values across pairs are on a common
    scale.

    Per Gretton et al. (JMLR 13:723-773, 2012); multiple-comparison
    correction per Benjamini & Hochberg (*JRSS-B* 1995).
    """
    img_to_group = {str(img): group_map.get(str(img)) for img in image_names}
    groups = sorted({g for g in img_to_group.values() if g is not None})
    n_per = {g: int(sum(1 for v in img_to_group.values() if v == g))
             for g in groups}

    method = f"pairwise_mmd_{correction}"
    if len(groups) < 2 or any(v < 2 for v in n_per.values()):
        return PairwiseMMDResult(
            pairs=[], mmd2=np.array([]),
            p_values_raw=np.array([]),
            p_values_corrected=np.array([]),
            n_per_group=n_per,
            sigma=float("nan"),
            n_permutations=n_permutations,
            method=f"{method}_insufficient",
        )

    # Bandwidth from pooled mapped images (a single sigma for all pairs).
    mapped_mask = np.array([img_to_group[str(img)] is not None
                            for img in image_names])
    if sigma is None:
        sigma = _median_heuristic_sigma(
            per_image_features[mapped_mask], seed=seed,
        )

    pairs: list[tuple[str, str]] = []
    mmd_vals: list[float] = []
    p_raw: list[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g_i, g_j = groups[i], groups[j]
            mask_i = np.array([img_to_group[str(img)] == g_i
                               for img in image_names])
            mask_j = np.array([img_to_group[str(img)] == g_j
                               for img in image_names])
            X = per_image_features[mask_i]
            Y = per_image_features[mask_j]
            if X.shape[0] < 2 or Y.shape[0] < 2:
                pairs.append((g_i, g_j))
                mmd_vals.append(float("nan"))
                p_raw.append(float("nan"))
                continue
            mmd_obs, p, _ = mmd_two_sample_test(
                X, Y,
                n_permutations=n_permutations,
                sigma=sigma,
                seed=seed,
            )
            pairs.append((g_i, g_j))
            mmd_vals.append(mmd_obs)
            p_raw.append(p)

    p_raw_arr = np.asarray(p_raw, dtype=np.float64)
    valid = ~np.isnan(p_raw_arr)
    p_corr = np.full_like(p_raw_arr, np.nan)
    if valid.any():
        if correction == "bh":
            p_corr[valid] = _bh_fdr(p_raw_arr[valid])
        elif correction == "bonferroni":
            p_corr[valid] = np.minimum(1.0, p_raw_arr[valid] * int(valid.sum()))
        else:
            raise ValueError(f"unknown correction: {correction!r}")

    return PairwiseMMDResult(
        pairs=pairs,
        mmd2=np.asarray(mmd_vals, dtype=np.float64),
        p_values_raw=p_raw_arr,
        p_values_corrected=p_corr,
        n_per_group=n_per,
        sigma=float(sigma),
        n_permutations=n_permutations,
        method=method,
    )
