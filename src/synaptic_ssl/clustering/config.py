"""Configuration objects and shared type aliases for the clustering pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# ``image_name -> group_label`` (e.g. {"img1.czi": "Psilocybin", ...}).
GroupMap = Mapping[str, str]


@dataclass
class ClusterCfg:
    """Clustering pipeline configuration."""

    seed: int = 42

    # Preprocessing
    l2_normalise: bool = True
    pca_dim: int | None = None         # None -> auto: smallest d such that
                                       # cumulative explained variance >=
                                       # pca_dim_target_var
    pca_dim_target_var: float = 0.95   # auto: smallest d capturing this
    pca_dim_max: int = 200             # auto: hard cap
    control_reference_pattern: str | None = None  # substring matched against
                                              # source_image; if set, the
                                              # mean of those patches is
                                              # subtracted before PCA

    # UMAP (visualisation only; clustering happens directly in PCA)
    umap_metric: str = "cosine"
    umap_init: str = "pca"
    umap_n_epochs: int = 500
    umap2_n_neighbors: int = 30
    umap2_min_dist: float = 0.1
    # 3D layout (alternative visualisation; same neighbours/min_dist defaults
    # as the 2D layout — override per run if you want a denser or sparser
    # global structure).
    umap3_n_neighbors: int = 30
    umap3_min_dist: float = 0.1

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
