"""Blob-based pseudo-label generation and diagnostic visualisation."""

from .blobs import (
    BlobPseudoCfg,
    detect_blobs_log, blobs_to_mask,
    meijering_response, compute_global_meijering_threshold,
    density_response,
    make_density_dendrite_mask,
    coherence_response,
    make_coherence_dendrite_mask,
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
from .soma_fdt import (
    DEFAULT_SOMA_CFG,
    derive_size_params,
    resolve_cfg,
    compute_pseudo_fdt,
    extract_soma_mask,
    run_soma_on_image,
    visualise_soma_mask,
)
from .dendrite_frangi import (
    DEFAULT_DENDRITE_CFG,
    compute_frangi_response,
    extract_dendrite_mask,
    run_dendrite_on_image,
    visualise_dendrite_mask,
)
from .dendrite_frangi import resolve_cfg as resolve_dendrite_cfg
from .viz import (
    show_3channel_grid, show_blob_overlay, show_scored_blobs,
    show_mask_overlay, show_pipeline_stages, plot_zscore_histogram,
)

__all__ = [
    # blobs
    "BlobPseudoCfg",
    "detect_blobs_log", "blobs_to_mask",
    "meijering_response", "compute_global_meijering_threshold",
    "density_response",
    "make_density_dendrite_mask",
    "coherence_response",
    "make_coherence_dendrite_mask",
    "make_soma_mask", "make_structural_mask",
    "score_blobs_zscore",
    "filter_by_size_shape",
    "generate_blob_pseudolabel",
    "compute_fullimage_structural_mask", "generate_pseudolabels_fullimage",
    # refine
    "RefineCfg",
    "region_grow_from_prob", "fuse_image_with_prob",
    "compute_mask_iou", "compute_positive_fraction",
    # soma-fdt
    "DEFAULT_SOMA_CFG",
    "derive_size_params", "resolve_cfg",
    "compute_pseudo_fdt", "extract_soma_mask", "run_soma_on_image",
    "visualise_soma_mask",
    # dendrite-frangi
    "DEFAULT_DENDRITE_CFG",
    "resolve_dendrite_cfg",
    "compute_frangi_response", "extract_dendrite_mask",
    "run_dendrite_on_image", "visualise_dendrite_mask",
    # viz
    "show_3channel_grid", "show_blob_overlay", "show_scored_blobs",
    "show_mask_overlay", "show_pipeline_stages", "plot_zscore_histogram",
]