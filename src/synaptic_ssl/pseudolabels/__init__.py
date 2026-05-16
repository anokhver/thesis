"""Blob-based pseudo-label generation and diagnostic visualisation."""

from .blobs import (
    BlobPseudoCfg,
    smooth_structural_channel,
    detect_blobs_log, blobs_to_mask,
    meijering_response, compute_global_meijering_threshold,
    density_response, compute_global_density_threshold,
    make_density_dendrite_mask,
    make_soma_mask, make_structural_mask,
    score_blobs_zscore,
    filter_by_size_shape,
    generate_blob_pseudolabel,
    compute_fullimage_structural_mask, generate_pseudolabels_fullimage,
)
from .refine import (
    RefineCfg,
    region_grow_from_prob,
    fuse_image_with_prob,
    compute_mask_iou,
    compute_positive_fraction,
)
from .viz import (
    show_3channel_grid, show_blob_overlay, show_scored_blobs,
    show_mask_overlay, show_pipeline_stages, plot_zscore_histogram,
)

__all__ = [
    # blobs
    "BlobPseudoCfg",
    "smooth_structural_channel",
    "detect_blobs_log", "blobs_to_mask",
    "meijering_response", "compute_global_meijering_threshold",
    "density_response", "compute_global_density_threshold",
    "make_density_dendrite_mask",
    "make_soma_mask", "make_structural_mask",
    "score_blobs_zscore",
    "filter_by_size_shape",
    "generate_blob_pseudolabel",
    "compute_fullimage_structural_mask", "generate_pseudolabels_fullimage",
    # refine
    "RefineCfg",
    "region_grow_from_prob", "fuse_image_with_prob",
    "compute_mask_iou", "compute_positive_fraction",
    # viz
    "show_3channel_grid", "show_blob_overlay", "show_scored_blobs",
    "show_mask_overlay", "show_pipeline_stages", "plot_zscore_histogram",
]