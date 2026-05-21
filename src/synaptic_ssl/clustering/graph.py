"""kNN graph construction, Leiden community detection, and 2D UMAP layout.

UMAP here is for visualisation only — clustering runs directly on the
PCA-whitened space to avoid UMAP's well-known density distortion
(see Chari & Pachter, *PLoS Comput Biol* 19:e1011288, 2023). Leiden
follows Traag, Waltman & Van Eck (*Sci Rep* 9:5233, 2019).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import ClusterCfg


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
