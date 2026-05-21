"""Preprocessing for SSL patch features.

Pipeline order matches the notebook: optional control-mean centering →
L2 normalise → PCA-whiten. ``auto_pca_dim`` picks the PCA dimensionality
from the cumulative variance curve so noise PCs are not rescaled to
unit variance by ``PCA(whiten=True)``. ``adaptive_leiden_k`` scales the
kNN k for small datasets (PhenoGraph-style; Levine et al. 2015).
"""
from __future__ import annotations

import numpy as np


def adaptive_leiden_k(N: int, default_k: int = 30) -> int:
    """Scale Leiden k for small datasets.

    Adapts the PhenoGraph code default k=30 (Levine et al., Cell
    162:184-197, 2015) to small N by linear scaling, lower-bounded
    at 5, capped at ``default_k``. Unlike PhenoGraph we omit Jaccard
    weighting and use Leiden instead of Louvain.
    """
    k = int(round(0.015 * N))
    return max(5, min(default_k, k))


def auto_pca_dim(Z: np.ndarray, target_var: float = 0.95, hard_cap: int = 200) -> int:
    """Smallest ``d`` with cumulative explained variance >= ``target_var``.

    Lower-bounded by 5, upper-bounded by ``hard_cap`` and ``min(N-1, D)``.

    Rationale: ``PCA(whiten=True)`` rescales every retained PC to unit
    variance, so keeping PCs that explain near-zero variance would
    promote pure noise to the same scale as informative directions and
    dominate Euclidean kNN distances. The cumulative-variance rule keeps
    only the directions that account for ``target_var`` of the total
    variance (default 95%); the hard cap bounds runtime / memory and
    avoids picking implausibly large ``d`` from a slow-decaying spectrum.
    Both ``target_var`` and ``hard_cap`` are engineering defaults of this
    repo, not values inherited from a specific publication.
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
