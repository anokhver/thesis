"""Group-level statistical tests on per-image cluster-frequency vectors.

The image is the statistical unit throughout (Lazic et al., *BMC
Neurosci* 11:5, 2010), so all tests here consume the per-image
frequency matrix produced by
:func:`synaptic_ssl.clustering.frequencies.per_image_cluster_frequencies`.

Tests in this module:

* :func:`group_cluster_test` — image-permutation χ² across groups.
* :func:`permanova_frequencies` — PERMANOVA on Bray-Curtis distances
  (Anderson, *Austral Ecology* 2001).
* :func:`pairwise_permanova_groups` — pairwise PERMANOVA with BH-FDR
  (Anderson & Walsh 2013; Benjamini & Hochberg 1995).
* :func:`per_cluster_kruskal_wallis` — per-cluster KW posthoc.
* :func:`_bh_fdr` — Benjamini-Hochberg multiple-comparison correction
  (used by several modules).
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from .config import GroupMap


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


class PairwisePERMANOVAResult(NamedTuple):
    """Pairwise PERMANOVA between treatment groups on per-image frequencies."""
    pairs: list[tuple[str, str]]
    f_statistic: np.ndarray
    p_values_raw: np.ndarray
    p_values_corrected: np.ndarray
    n_per_group: dict[str, int]
    metric: str
    n_permutations: int
    method: str


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


def pairwise_permanova_groups(
    freq: np.ndarray,
    image_names: np.ndarray,
    group_map: GroupMap,
    *,
    metric: str = "braycurtis",
    n_permutations: int = 999,
    correction: str = "bh",
    seed: int = 0,
) -> PairwisePERMANOVAResult:
    """Pairwise PERMANOVA with BH-FDR correction over group pairs.

    Each pair test is the standard PERMANOVA pseudo-F (Anderson,
    *Austral Ecology* 26:32-46, 2001) on Bray-Curtis distances between
    per-image cluster-frequency vectors; pairwise multiple-comparison
    correction follows Anderson & Walsh (*Ecological Monographs*
    83:557-574, 2013), with BH-FDR (Benjamini & Hochberg, *JRSS-B*
    57:289-300, 1995) preferred over Bonferroni when several pairs
    are tested.
    """
    img_to_group = {str(img): group_map.get(str(img)) for img in image_names}
    groups = sorted({g for g in img_to_group.values() if g is not None})
    n_per = {g: int(sum(1 for v in img_to_group.values() if v == g))
             for g in groups}

    method = f"pairwise_permanova_{correction}"
    if len(groups) < 2:
        return PairwisePERMANOVAResult(
            pairs=[], f_statistic=np.array([]),
            p_values_raw=np.array([]),
            p_values_corrected=np.array([]),
            n_per_group=n_per,
            metric=metric,
            n_permutations=n_permutations,
            method=f"{method}_insufficient",
        )

    pairs: list[tuple[str, str]] = []
    f_vals: list[float] = []
    p_raw: list[float] = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g_i, g_j = groups[i], groups[j]
            sub_map = {img: g for img, g in img_to_group.items()
                       if g in (g_i, g_j)}
            res = permanova_frequencies(
                freq, image_names, sub_map,
                metric=metric, n_permutations=n_permutations, seed=seed,
            )
            pairs.append((g_i, g_j))
            f_vals.append(res.f_statistic)
            p_raw.append(res.p_value)

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

    return PairwisePERMANOVAResult(
        pairs=pairs,
        f_statistic=np.asarray(f_vals, dtype=np.float64),
        p_values_raw=p_raw_arr,
        p_values_corrected=p_corr,
        n_per_group=n_per,
        metric=metric,
        n_permutations=n_permutations,
        method=method,
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
