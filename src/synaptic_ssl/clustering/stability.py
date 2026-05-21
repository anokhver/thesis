"""Bootstrap stability of Leiden partitions and resolution selection.

ARI follows Lange et al. (*Neural Computation* 16:1299-1323, 2004);
NMI is the Strehl & Ghosh (*JMLR* 3:583-617, 2002) information-theoretic
clustering similarity, reported alongside as a cross-check on ARI.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .graph import knn_igraph, leiden_partition


def bootstrap_stability(
    U: np.ndarray, resolutions: Sequence[float], k: int,
    B: int, frac: float, seed: int = 0,
    *, source_images: np.ndarray | None = None,
):
    """Bootstrap stability of Leiden partitions.

    Subsample → re-cluster → 1-NN propagate → ARI **and** NMI vs the
    full partition. ARI follows Lange et al. (Neural Computation
    16:1299-1323, 2004) with ARI replacing their Hamming distance;
    NMI is the Strehl & Ghosh (JMLR 3:583-617, 2002) information-
    theoretic clustering similarity, reported as a complementary
    metric so a single inflated ARI can be cross-checked.

    When ``source_images`` is given, resampling is at the image-block
    level to respect the statistical unit; otherwise patch-level
    bootstrap. Returns ``({res: (mean_ari, std_ari, mean_nmi,
    std_nmi)}, {res: labels})``.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import normalized_mutual_info_score

    rng = np.random.default_rng(seed)
    n = len(U)
    g_full = knn_igraph(U, k)
    full = {float(r): leiden_partition(g_full, r, seed=seed) for r in resolutions}
    aris = {float(r): [] for r in resolutions}
    nmis = {float(r): [] for r in resolutions}

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
            nmis[float(r)].append(
                normalized_mutual_info_score(full[float(r)], propagated)
            )
    summary = {
        r: (
            float(np.mean(aris[r])) if aris[r] else float("nan"),
            float(np.std(aris[r])) if aris[r] else float("nan"),
            float(np.mean(nmis[r])) if nmis[r] else float("nan"),
            float(np.std(nmis[r])) if nmis[r] else float("nan"),
        )
        for r in aris
    }
    return summary, full


def pick_resolution(summary: dict, full: dict):
    """Highest mean ARI among non-trivial partitions; tie-break by fewer clusters.

    Selection criterion follows Lange et al. (2004) / Hennig (2007):
    ARI-stability is the primary statistic; NMI is reported alongside
    for transparency but is not part of the tie-break. Trivial
    partitions (K=1 or K=0) are excluded before sorting: with K=1 the
    bootstrap ARI is automatically 1.0 because every resampling
    "agrees" on the single label, so a collapsed resolution would
    otherwise win the stability sort.
    """
    import logging

    candidates = [
        (r, m_ari, s_ari, len(set(full[r])))
        for r, (m_ari, s_ari, _m_nmi, _s_nmi) in summary.items()
    ]
    nontrivial = [t for t in candidates if t[3] >= 2]
    if not nontrivial:
        logging.getLogger(__name__).warning(
            "All resolutions collapsed to K<2; falling back to the lowest "
            "candidate to avoid an empty pick."
        )
        nontrivial = candidates
    nontrivial.sort(key=lambda t: (-t[1], t[3]))
    return nontrivial[0]  # (resolution, mean_ari, std_ari, n_clusters)
