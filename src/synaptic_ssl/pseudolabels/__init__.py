"""Pseudo-label generators: soma (FDT), dendrite (Frangi), puncta (LoG).

Live pipeline: ``soma_fdt`` -> ``dendrite_frangi`` -> blob detection
(``blobs.detect_blobs_log`` + ``score_blobs_zscore``). The remaining
exports from ``blobs`` (``make_structural_mask``, the meijering/density/
coherence dendrite branches, ``generate_blob_pseudolabel``,
``generate_pseudolabels_fullimage``, ``compute_global_meijering_threshold``,
``compute_fullimage_structural_mask``) and ``refine`` are legacy:
consumed only by ``notebooks/segmentation/train_swinunetr_pseudolabels*``
and ``segmentation/dataset.py``. New code should not import them.

Alternative soma detector: ``notebooks/pseudolabels/log_soma.ipynb``
(scale-normalised LoG). Standalone, not wired into the CLI scripts.
"""

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
    prune_spurs,
    run_dendrite_on_image,
    visualise_dendrite_mask,
)
from .dendrite_frangi import resolve_cfg as resolve_dendrite_cfg

__all__ = [
    # blobs -- live (used by scripts/{puncta,dendrite,soma}_from_mip.py)
    "BlobPseudoCfg",
    "detect_blobs_log", "blobs_to_mask",
    "score_blobs_zscore",
    "filter_by_size_shape",
    # blobs -- legacy (training notebooks + segmentation.dataset only)
    "meijering_response", "compute_global_meijering_threshold",
    "density_response", "make_density_dendrite_mask",
    "coherence_response", "make_coherence_dendrite_mask",
    "make_soma_mask", "make_structural_mask",
    "generate_blob_pseudolabel",
    "compute_fullimage_structural_mask", "generate_pseudolabels_fullimage",
    # refine -- legacy (DDeep3M+ iterative training notebook only)
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
]