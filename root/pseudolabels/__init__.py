"""Blob-based pseudo-label generation and diagnostic visualisation."""

from .blobs import (
    BlobPseudoCfg,
    detect_blobs_log, blobs_to_mask,
    meijering_response, compute_global_meijering_threshold,
    make_dendrite_mask, make_soma_mask, make_structural_mask,
    colocalize_intersect, score_blobs_zscore,
    filter_by_size_shape,
    generate_blob_pseudolabel,
    compute_fullimage_structural_mask, generate_pseudolabels_fullimage,
)
from .viz import (
    show_3channel_grid, show_blob_overlay, show_scored_blobs,
    show_mask_overlay, show_pipeline_stages, plot_zscore_histogram,
)

__all__ = [
    # blobs
    "BlobPseudoCfg",
    "detect_blobs_log", "blobs_to_mask",
    "meijering_response", "compute_global_meijering_threshold",
    "make_dendrite_mask", "make_soma_mask", "make_structural_mask",
    "colocalize_intersect", "score_blobs_zscore",
    "filter_by_size_shape",
    "generate_blob_pseudolabel",
    "compute_fullimage_structural_mask", "generate_pseudolabels_fullimage",
    # viz
    "show_3channel_grid", "show_blob_overlay", "show_scored_blobs",
    "show_mask_overlay", "show_pipeline_stages", "plot_zscore_histogram",
]