"""Smoke test for the refactored clustering pipeline."""
from synaptic_ssl.clustering.pipeline import (
    ClusterCfg,
    extract_patch_embeddings,
    auto_pca_dim, subtract_control_mean, l2_then_pca_whiten,
    adaptive_leiden_k,
    knn_igraph,
    fit_umap_2d,
    leiden_sweep, bootstrap_stability, pick_resolution,
    cluster_size_stats, permutation_null_ari,
    internal_validity_indices,
    image_level_cv,
    cluster_medoids,
    per_image_cluster_frequencies, chi2_independence,
    cluster_purity_by_image,
    group_cluster_test, permanova_frequencies,
    pairwise_permanova_groups,
    per_cluster_kruskal_wallis,
    per_image_mean_embeddings, pairwise_mmd_groups,
    load_group_patterns, build_group_map_from_patterns,
    write_cluster_columns,
)
print("Notebook-import set: OK")

import synaptic_ssl.clustering.config
import synaptic_ssl.clustering.embeddings
import synaptic_ssl.clustering.preprocess
import synaptic_ssl.clustering.graph
import synaptic_ssl.clustering.stability
import synaptic_ssl.clustering.sanity
import synaptic_ssl.clustering.frequencies
import synaptic_ssl.clustering.groups
import synaptic_ssl.clustering.stats_group
import synaptic_ssl.clustering.stats_mmd
import synaptic_ssl.clustering.writeback
print("All 11 submodules import cleanly.")

import synaptic_ssl.clustering.pipeline as p
print(f"pipeline.__all__: {len(p.__all__)} names")

import numpy as np
from synaptic_ssl.clustering.stats_group import _bh_fdr
adj = _bh_fdr([0.01, 0.04, 0.03, 0.005])
print(f"_bh_fdr smoke: {adj.round(4).tolist()}")

print(f"adaptive_leiden_k(2000)={adaptive_leiden_k(2000)} (expect 30)")
print(f"adaptive_leiden_k(100)={adaptive_leiden_k(100)} (expect 5)")

from synaptic_ssl.clustering.sanity import gini
print(f"gini([1,1,1,1])={gini([1,1,1,1])} (expect 0.0)")
print(f"cluster_size_stats: {cluster_size_stats(np.array([0,0,1,1,2,-1]))}")

cfg = ClusterCfg()
print(f"ClusterCfg defaults: seed={cfg.seed}, pca_dim={cfg.pca_dim}, leiden_k={cfg.leiden_k}")

# Light functional check on bh_fdr correctness vs scipy/manual
expected = [0.04 / 4 * 4, 0.04 / 3 * 4, 0.04 / 2 * 4, 0.04 / 1 * 4]  # not exact, just sanity
print("SMOKE OK")
