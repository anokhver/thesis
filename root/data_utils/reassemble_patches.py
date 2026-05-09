"""
reassemble_patches.py
Utility to reconstruct full MIP images from tiled .npy patches
using the index.csv manifest, and to slice results back into patches.

Usage:
    from data_utils.reassemble_patches import reassemble_image, slice_to_patches

    full_img = reassemble_image(patch_root, image_index=0)
    # full_img: (C, H, W) numpy array

    # ... run Frangi on full_img ...

    patch_dict = slice_to_patches(result, patch_root, image_index=0)
    # patch_dict: {filename: (C, pH, pW) array}
"""

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np


def _load_index(patch_root):
    """Load and group index.csv records by image_index."""
    patch_root = Path(patch_root)
    csv_path = patch_root / "index.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No index.csv in {patch_root}")

    with open(csv_path, "r") as f:
        records = list(csv.DictReader(f))

    by_image = defaultdict(list)
    for r in records:
        by_image[int(r["image_index"])].append(r)

    return by_image


def reassemble_image(patch_root, image_index, exclude_patterns=None):
    """Reconstruct a full image from its tiled patches.

    Args:
        patch_root: directory containing .npy patches and index.csv
        image_index: which image to reconstruct (int)
        exclude_patterns: if the image matches any pattern, skip it

    Returns:
        full_image: (C, H, W) float32 numpy array
        records: list of CSV records for this image (for later slicing)
    """
    patch_root = Path(patch_root)
    by_image = _load_index(patch_root)

    if image_index not in by_image:
        raise ValueError(f"image_index {image_index} not found in index.csv")

    records = by_image[image_index]

    if exclude_patterns:
        patterns_upper = [p.upper() for p in exclude_patterns]
        src = records[0]["source_image"].upper()
        if any(p in src for p in patterns_upper):
            raise ValueError(f"Image {image_index} matches exclude pattern")

    patch_size = int(records[0]["patch_size"])
    n_channels = int(records[0]["channels"])
    max_row = max(int(r["grid_row"]) for r in records) + 1
    max_col = max(int(r["grid_col"]) for r in records) + 1

    full_h = max_row * patch_size
    full_w = max_col * patch_size
    full_image = np.zeros((n_channels, full_h, full_w), dtype=np.float32)

    for rec in records:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        patch = np.load(patch_root / rec["filename"])  # (C, pH, pW)
        y0 = r * patch_size
        x0 = c * patch_size
        full_image[:, y0:y0 + patch_size, x0:x0 + patch_size] = patch

    return full_image, records


def slice_to_patches(full_array, records, patch_size=None):
    """Slice a full-image array back into patches matching the CSV grid.

    Args:
        full_array: (C, H, W) or (H, W) array at the same resolution as
                    the reassembled image
        records: list of CSV records (from reassemble_image)
        patch_size: override patch size (default: from records)

    Returns:
        dict mapping filename -> patch array
    """
    if patch_size is None:
        patch_size = int(records[0]["patch_size"])

    result = {}
    for rec in records:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        y0 = r * patch_size
        x0 = c * patch_size

        if full_array.ndim == 3:
            patch = full_array[:, y0:y0 + patch_size, x0:x0 + patch_size]
        else:
            patch = full_array[y0:y0 + patch_size, x0:x0 + patch_size]

        result[rec["filename"]] = patch.copy()

    return result


def list_image_indices(patch_root, exclude_patterns=None):
    """Return sorted list of available image indices."""
    by_image = _load_index(patch_root)
    indices = sorted(by_image.keys())

    if exclude_patterns:
        patterns_upper = [p.upper() for p in exclude_patterns]
        indices = [
            idx for idx in indices
            if not any(
                p in by_image[idx][0]["source_image"].upper()
                for p in patterns_upper
            )
        ]

    return indices
