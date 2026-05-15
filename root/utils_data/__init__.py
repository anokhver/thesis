"""Patch dataset, preprocessing, reassembly, splits, and damage detection."""

from .patch_dataset import PatchDataset
from .reassemble import reassemble_image, slice_to_patches, list_image_indices
from .split import SplitPatchDataset

from .preprocess_training import (
    load_image, maximum_intensity_projection, normalize_percentile,
    extract_patches, best_z_slice,
)
from .damage_detection import patch_stats, flag_damaged, attach_stats, damage_summary

__all__ = [
    "PatchDataset",
    "reassemble_image", "slice_to_patches", "list_image_indices",
    "SplitPatchDataset",
    "load_image", "maximum_intensity_projection", "normalize_percentile",
    "extract_patches", "best_z_slice",
    "patch_stats", "flag_damaged", "attach_stats", "damage_summary",
]