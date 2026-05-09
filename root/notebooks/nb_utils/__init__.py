"""Shared building blocks for the notebooks under ``root/notebooks-new``.

The three task templates (``loaded-weights``, ``from-scratch``,
``clustering``) import everything they need from this package so the
notebooks themselves stay short and editable.
"""

from .config import BaseCfg, DataCfg, ModelCfg, TrainCfg, SSLCfg, dump_config
from .seeding import seed_everything, RNGSnapshot, isolated_rng
from .logging_utils import setup_logger, CSVMetricLogger
from .data import build_dataloaders, compute_channel_stats, TransformedSubset
from .augment import MicroscopyTwoViewTransform, ValSingleViewTransform
from .model import (
    build_swin_encoder,
    SimMIMDecoder,
    MaskToken,
    VICRegProjector,
    build_simmim_vicreg_heads,
    count_params,
)
from .weight_loading import (
    convert_timm_to_swinunetr_state_dict,
    load_pretrained_into_encoder,
)
from .masking import random_block_mask, apply_mask
from .losses import (
    simmim_recon_loss,
    vicreg_terms,
    compute_simmim_vicreg_loss,
    validation_simmim,
)
from .lr_schedule import param_groups_layer_decay, make_warmup_cosine
from .checkpoints import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from .sanity_batch import (
    SANITY_TRAIN_INDICES,
    SANITY_VAL_INDICES,
    fixed_two_view_batch,
    fixed_single_view_batch,
    overfit_on_batch,
)
from .viz import (
    plot_two_views,
    plot_channel_histograms,
    plot_recon_panel,
    plot_loss_curves,
    plot_overfit_curves,
    plot_embedding_2d,
    LOSS_LABELS,
    loss_label,
)
from .post_training import (
    build_run_label,
    eval_recon_batch,
    reload_best_checkpoint,
    post_training_reconstruction,
    plot_post_training_curves,
    embedding_diagnostics,
)
from .clustering import (
    extract_pooled_embeddings,
    effective_rank,
    mean_pairwise_cos,
    reduce_2d,
)
from .clustering_pipeline import (
    ClusterCfg,
    extract_patch_embeddings,
    auto_pca_dim,
    tvn_centre,
    l2_then_pca_whiten,
    fit_umap,
    fit_umap_2d,
    umap_to_knn_igraph,
    leiden_partition,
    leiden_sweep,
    bootstrap_stability,
    pick_resolution,
    gmm_bic_scan,
    run_hdbscan_leaf,
    gini,
    cluster_size_stats,
    permutation_null_ari,
    train_on_imageset_predict_other,
    cluster_medoids,
    per_image_cluster_frequencies,
    chi2_independence,
    cluster_purity_by_image,
    write_cluster_columns,
)
from .clustering_viz import (
    plot_singular_value_spectrum,
    plot_resolution_stability,
    plot_bic_curve,
    plot_2d_clusters,
    plot_cluster_grid,
    plot_image_cluster_heatmap,
    plot_image_level_umap,
    plot_intra_image_entropy,
)
from .pseudolabels import (
    BlobPseudoCfg,
    detect_blobs_log,
    blobs_to_mask,
    meijering_response,
    compute_global_meijering_threshold,
    make_dendrite_mask,
    make_soma_mask,
    make_structural_mask,
    colocalize_intersect,
    score_blob_zscore,
    score_blobs_zscore,
    filter_by_size_shape,
    generate_blob_pseudolabel,
)
from .pseudolabel_viz import (
    show_3channel_grid,
    show_blob_overlay,
    show_scored_blobs,
    show_mask_overlay,
    show_pipeline_stages,
    plot_zscore_histogram,
)

__all__ = [
    "BaseCfg", "DataCfg", "ModelCfg", "TrainCfg", "SSLCfg", "dump_config",
    "seed_everything", "RNGSnapshot", "isolated_rng",
    "setup_logger", "CSVMetricLogger",
    "build_dataloaders", "compute_channel_stats", "TransformedSubset",
    "MicroscopyTwoViewTransform", "ValSingleViewTransform",
    "build_swin_encoder", "SimMIMDecoder", "MaskToken", "VICRegProjector",
    "build_simmim_vicreg_heads", "count_params",
    "convert_timm_to_swinunetr_state_dict", "load_pretrained_into_encoder",
    "random_block_mask", "apply_mask",
    "simmim_recon_loss", "vicreg_terms", "compute_simmim_vicreg_loss",
    "validation_simmim",
    "param_groups_layer_decay", "make_warmup_cosine",
    "save_checkpoint", "load_checkpoint", "find_latest_checkpoint",
    "SANITY_TRAIN_INDICES", "SANITY_VAL_INDICES",
    "fixed_two_view_batch", "fixed_single_view_batch", "overfit_on_batch",
    "plot_two_views", "plot_channel_histograms", "plot_recon_panel",
    "plot_loss_curves", "plot_overfit_curves", "plot_embedding_2d",
    "LOSS_LABELS", "loss_label",
    "build_run_label", "eval_recon_batch", "reload_best_checkpoint",
    "post_training_reconstruction", "plot_post_training_curves",
    "embedding_diagnostics",
    "extract_pooled_embeddings", "effective_rank", "mean_pairwise_cos",
    "reduce_2d",
    "ClusterCfg", "extract_patch_embeddings", "auto_pca_dim", "tvn_centre",
    "l2_then_pca_whiten", "fit_umap", "fit_umap_2d", "umap_to_knn_igraph",
    "leiden_partition", "leiden_sweep", "bootstrap_stability",
    "pick_resolution", "gmm_bic_scan", "run_hdbscan_leaf",
    "gini", "cluster_size_stats", "permutation_null_ari",
    "train_on_imageset_predict_other", "cluster_medoids",
    "per_image_cluster_frequencies", "chi2_independence",
    "cluster_purity_by_image", "write_cluster_columns",
    "plot_singular_value_spectrum", "plot_resolution_stability",
    "plot_bic_curve", "plot_2d_clusters", "plot_cluster_grid",
    "plot_image_cluster_heatmap", "plot_image_level_umap",
    "plot_intra_image_entropy",
    "BlobPseudoCfg", "detect_blobs_log", "blobs_to_mask",
    "meijering_response", "compute_global_meijering_threshold",
    "make_dendrite_mask", "make_soma_mask", "make_structural_mask",
    "colocalize_intersect", "score_blob_zscore", "score_blobs_zscore",
    "filter_by_size_shape", "generate_blob_pseudolabel",
    "show_3channel_grid", "show_blob_overlay", "show_scored_blobs",
    "show_mask_overlay", "show_pipeline_stages", "plot_zscore_histogram",
]
