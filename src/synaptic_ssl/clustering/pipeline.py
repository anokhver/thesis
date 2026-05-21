"""End-to-end clustering pipeline — composition module.

This file used to hold the full implementation (≈1.5 kLOC). It is now
a thin re-export layer over a set of focused submodules so the whole
pipeline can still be imported from a single name::

    from synaptic_ssl.clustering.pipeline import (
        ClusterCfg, extract_patch_embeddings, l2_then_pca_whiten,
        leiden_sweep, bootstrap_stability, pick_resolution,
        permutation_null_ari, image_level_cv, internal_validity_indices,
        per_image_cluster_frequencies, chi2_independence,
        cluster_purity_by_image, per_image_mean_embeddings,
        group_cluster_test, permanova_frequencies,
        pairwise_permanova_groups, per_cluster_kruskal_wallis,
        pairwise_mmd_groups, load_group_patterns,
        build_group_map_from_patterns, write_cluster_columns,
        ...
    )

Conceptual flow (each line is one block; submodule shown on the right):

1.  configure                           ── :mod:`.config`
2.  extract pooled features             ── :mod:`.embeddings`
3.  L2 + (optional control mean) + PCA  ── :mod:`.preprocess`
4.  build kNN graph + Leiden + UMAP-2D  ── :mod:`.graph`
5.  bootstrap stability + pick res      ── :mod:`.stability`
6.  sanity & internal-validity indices  ── :mod:`.sanity`
7.  per-image cluster frequencies       ── :mod:`.frequencies`
8.  treatment-group classification      ── :mod:`.groups`
9.  group-level frequency tests         ── :mod:`.stats_group`
10. group-level MMD on mean embeddings  ── :mod:`.stats_mmd`
11. writeback cluster columns to CSV    ── :mod:`.writeback`

All public names are re-exported below so existing notebook / CLI
imports (``from synaptic_ssl.clustering.pipeline import …``) continue
to work unchanged.
"""
from __future__ import annotations

# --- 1. Configuration -------------------------------------------------------
from .config import ClusterCfg, GroupMap

# --- 2. Embedding extraction ------------------------------------------------
from .embeddings import extract_patch_embeddings

# --- 3. Preprocessing -------------------------------------------------------
from .preprocess import (
    adaptive_leiden_k,
    auto_pca_dim,
    subtract_control_mean,
    l2_then_pca_whiten,
)

# --- 4. kNN graph + Leiden + UMAP-2D ---------------------------------------
from .graph import (
    knn_igraph,
    leiden_partition,
    leiden_sweep,
    fit_umap_2d,
)

# --- 5. Stability + resolution selection ------------------------------------
from .stability import bootstrap_stability, pick_resolution

# --- 6. Sanity tests + internal cluster-validity indices --------------------
from .sanity import (
    gini,
    cluster_size_stats,
    permutation_null_ari,
    image_level_cv,
    cluster_medoids,
    InternalValidity,
    internal_validity_indices,
)

# --- 7. Per-image cluster frequencies + image-level diagnostics -------------
from .frequencies import (
    per_image_cluster_frequencies,
    chi2_independence,
    cluster_purity_by_image,
    per_image_mean_embeddings,
)

# --- 8. Treatment-group classification --------------------------------------
from .groups import (
    load_group_patterns,
    classify_image_by_patterns,
    build_group_map_from_patterns,
)

# --- 9. Group-level statistical tests on cluster frequencies ----------------
from .stats_group import (
    _bh_fdr,
    GroupTestResult,
    PERMANOVAResult,
    PerClusterTestResult,
    PairwisePERMANOVAResult,
    group_cluster_test,
    permanova_frequencies,
    pairwise_permanova_groups,
    per_cluster_kruskal_wallis,
)

# --- 10. Group-level MMD on per-image mean embeddings -----------------------
from .stats_mmd import (
    PairwiseMMDResult,
    mmd2_unbiased,
    mmd_two_sample_test,
    pairwise_mmd_groups,
)

# --- 11. Writeback ----------------------------------------------------------
from .writeback import write_cluster_columns


__all__ = [
    # config
    "ClusterCfg", "GroupMap",
    # embeddings
    "extract_patch_embeddings",
    # preprocess
    "adaptive_leiden_k", "auto_pca_dim",
    "subtract_control_mean", "l2_then_pca_whiten",
    # graph
    "knn_igraph", "leiden_partition", "leiden_sweep", "fit_umap_2d",
    # stability
    "bootstrap_stability", "pick_resolution",
    # sanity
    "gini", "cluster_size_stats", "permutation_null_ari",
    "image_level_cv", "cluster_medoids",
    "InternalValidity", "internal_validity_indices",
    # frequencies
    "per_image_cluster_frequencies", "chi2_independence",
    "cluster_purity_by_image", "per_image_mean_embeddings",
    # groups
    "load_group_patterns", "classify_image_by_patterns",
    "build_group_map_from_patterns",
    # stats_group
    "GroupTestResult", "PERMANOVAResult", "PerClusterTestResult",
    "PairwisePERMANOVAResult",
    "group_cluster_test", "permanova_frequencies",
    "pairwise_permanova_groups", "per_cluster_kruskal_wallis",
    # stats_mmd
    "PairwiseMMDResult", "mmd2_unbiased", "mmd_two_sample_test",
    "pairwise_mmd_groups",
    # writeback
    "write_cluster_columns",
]
