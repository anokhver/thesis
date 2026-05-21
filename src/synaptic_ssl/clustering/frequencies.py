"""Per-image cluster-frequency representations and basic image-level tests.

These functions collapse patch-level labels to per-image vectors, which
is the unit of analysis throughout the pipeline (Lazic et al., *BMC
Neurosci* 11:5, 2010; Caicedo et al., *Nat Methods* 2017). The matrix
returned by :func:`per_image_cluster_frequencies` feeds the PERMANOVA /
KW / pairwise tests; :func:`per_image_mean_embeddings` feeds the MMD
two-sample test.
"""
from __future__ import annotations

import numpy as np


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


def per_image_mean_embeddings(
    P: np.ndarray, source_images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate patch features to one mean vector per source image.

    The image is the statistical unit throughout the pipeline (Caicedo
    et al., *Nat Methods* 2017; Lazic et al., *BMC Neurosci* 11:5,
    2010), so any test on raw features must first collapse patches
    within an image. Returns ``(M, image_names)`` with ``M.shape ==
    (n_images, d)`` and rows aligned to ``image_names``.
    """
    image_names = np.array(sorted(set(map(str, source_images))))
    M = np.stack([
        P[np.asarray(source_images) == img].mean(axis=0)
        for img in image_names
    ])
    return M.astype(np.float64), image_names
