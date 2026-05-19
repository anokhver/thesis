"""Patch dataset, preprocessing, reassembly, splits, and damage detection."""

from .patch_dataset import PatchDataset
from .reassemble import (
    reassemble_image, slice_to_patches, list_image_indices,
    load_patch_records, ImageCache,
)
from .split import SplitPatchDataset
from .preprocess_training import (
    load_image, maximum_intensity_projection, normalize_percentile,
    extract_patches, best_z_slice,
)
from .damage_detection import patch_stats, flag_damaged, attach_stats, damage_summary
from .noise_detection import (
    DEFAULT_HF_ENERGY_FRAC, DEFAULT_HP_VAR_RATIO,
    channel_noise_stats, image_noise_stats, score_image_set,
    summarize_scores, flagged_source_names,
)

__all__ = [
    "PatchDataset",
    "reassemble_image", "slice_to_patches", "list_image_indices",
    "load_patch_records", "ImageCache",
    "SplitPatchDataset",
    "load_image", "maximum_intensity_projection", "normalize_percentile",
    "extract_patches", "best_z_slice",
    "patch_stats", "flag_damaged", "attach_stats", "damage_summary",
    "DEFAULT_HF_ENERGY_FRAC", "DEFAULT_HP_VAR_RATIO",
    "channel_noise_stats", "image_noise_stats", "score_image_set",
    "summarize_scores", "flagged_source_names",
]