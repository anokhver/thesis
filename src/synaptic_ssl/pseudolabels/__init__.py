"""Pseudo-label generators: soma (FDT), dendrite (Frangi), puncta (LoG + Spotiflow).

Live pipeline (scripts/{soma,dendrite,puncta}_from_mip.py): soma_fdt ->
dendrite_frangi -> puncta_log. ``viz`` mirrors the per-step inspection
from ``notebooks/pseudolabels/puncta_detection.ipynb``.

Alternative detectors:
  * soma_log -- scale-normalised LoG soma detector.
  * puncta_spotiflow -- pretrained Spotiflow puncta detector
    (Dominguez Mantes et al., Nat. Methods 2025).
"""

from .puncta_log import (
    PunctaCfg,
    DEFAULT_PUNCTA_CFG_PRE, DEFAULT_PUNCTA_CFG_POST, DEFAULT_NEAR_DILATE_PX,
    detect_puncta_log,
    score_puncta_zscore,
    filter_by_size_shape,
    derive_zscore_floors,
    detect_puncta_channel,
)
from .puncta_log import resolve_cfg as resolve_puncta_cfg
from .puncta_common import puncta_to_mask, restrict_puncta_to_near
from .puncta_spotiflow import (
    SpotiflowPunctaCfg,
    DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW,
    DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW,
    load_model as load_spotiflow_model,
    detect_spots_spotiflow,
    filter_by_intensity as filter_spots_by_intensity,
    derive_intensity_floor as derive_spotiflow_floor,
    spots_to_blobs,
    detect_puncta_channel as detect_puncta_channel_spotiflow,
)
from .puncta_spotiflow import resolve_cfg as resolve_spotiflow_cfg
from .viz import (
    visualise_structural_overview,
    visualise_puncta_channel,
    visualise_puncta_pair,
    visualise_puncta_full,
    visualise_puncta_channel_spotiflow,
    visualise_puncta_pair_spotiflow,
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
    # puncta (LoG)
    "PunctaCfg",
    "DEFAULT_PUNCTA_CFG_PRE", "DEFAULT_PUNCTA_CFG_POST",
    "DEFAULT_NEAR_DILATE_PX",
    "resolve_puncta_cfg",
    "detect_puncta_log",
    "score_puncta_zscore",
    "filter_by_size_shape",
    "derive_zscore_floors",
    "detect_puncta_channel",
    # puncta (common geometry helpers)
    "puncta_to_mask", "restrict_puncta_to_near",
    # puncta (Spotiflow)
    "SpotiflowPunctaCfg",
    "DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW", "DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW",
    "resolve_spotiflow_cfg",
    "load_spotiflow_model",
    "detect_spots_spotiflow",
    "filter_spots_by_intensity",
    "derive_spotiflow_floor",
    "spots_to_blobs",
    "detect_puncta_channel_spotiflow",
    # puncta viz
    "visualise_structural_overview",
    "visualise_puncta_channel",
    "visualise_puncta_pair",
    "visualise_puncta_full",
    "visualise_puncta_channel_spotiflow",
    "visualise_puncta_pair_spotiflow",
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
