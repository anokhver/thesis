"""Pseudo-label generators: soma (FDT), dendrite (Frangi), puncta (LoG).

Live pipeline (scripts/{soma,dendrite,puncta}_from_mip.py): soma_fdt ->
dendrite_frangi -> puncta. ``viz`` mirrors the per-step inspection from
``notebooks/pseudolabels/puncta_detection.ipynb``.

Alternative soma detector (scale-normalised LoG) in ``soma_log``.
"""

from .puncta import (
    PunctaCfg,
    DEFAULT_PUNCTA_CFG_PRE, DEFAULT_PUNCTA_CFG_POST, DEFAULT_NEAR_DILATE_PX,
    detect_puncta_log, puncta_to_mask,
    score_puncta_zscore,
    filter_by_size_shape,
    derive_zscore_floors,
    detect_puncta_channel,
    restrict_puncta_to_near,
)
from .puncta import resolve_cfg as resolve_puncta_cfg
from .viz import (
    visualise_structural_overview,
    visualise_puncta_channel,
    visualise_puncta_pair,
    visualise_puncta_full,
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
    prune_spurs,
    run_dendrite_on_image,
    visualise_dendrite_mask,
)
from .dendrite_frangi import resolve_cfg as resolve_dendrite_cfg
from .soma_log import (
    derive_log_sigmas,
    resolve_log_cfg,
    detect_log_blobs,
    filter_log_blobs_by_intensity,
    render_log_blob_mask,
    run_log_on_image,
    visualise_log_mask,
)

__all__ = [
    # puncta
    "PunctaCfg",
    "DEFAULT_PUNCTA_CFG_PRE", "DEFAULT_PUNCTA_CFG_POST",
    "DEFAULT_NEAR_DILATE_PX",
    "resolve_puncta_cfg",
    "detect_puncta_log", "puncta_to_mask",
    "score_puncta_zscore",
    "filter_by_size_shape",
    "derive_zscore_floors",
    "detect_puncta_channel",
    "restrict_puncta_to_near",
    # puncta viz
    "visualise_structural_overview",
    "visualise_puncta_channel",
    "visualise_puncta_pair",
    "visualise_puncta_full",
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
    "prune_spurs",
    "run_dendrite_on_image", "visualise_dendrite_mask",
    # soma-log (alternative LoG-based soma detector)
    "derive_log_sigmas", "resolve_log_cfg",
    "detect_log_blobs", "filter_log_blobs_by_intensity",
    "render_log_blob_mask", "run_log_on_image", "visualise_log_mask",
]
