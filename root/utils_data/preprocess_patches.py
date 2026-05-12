#!/usr/bin/env python3
"""
preprocess_patches.py
Offline preprocessing pipeline for confocal fluorescence microscopy images.

Pipeline:
    raw file → load (C,Z,Y,X) → MIP along Z → per-channel percentile
    normalization → tile into 128×128 patches → filter empty patches → save .npy

Design decisions justified by literature:
    - MIP for Z-handling: standard for puncta analysis (SynBot, SynapseJ)
    - Per-channel independent percentile normalization: Cellpose [1,99],
      StarDist [1,99.8]. We use [1,99.8] with clipping to [0,1] because
      StarDist's approach better preserves bright puncta that occupy <1%
      of image area (the 99th percentile can clip them).
    - 128×128 patches: divides 2304 evenly (18×18 grid), gives ~1024 tokens
      for MAE with patch_size=4, compatible with SwinUNETR (divisible by 32).
    - No background subtraction: DL papers (Cellpose, nnU-Net, CA-MAE) skip it;
      classical puncta tools (SynBot, SynapseJ) use rolling-ball but that is
      for thresholding pipelines, not learned feature extractors.
Usage:
    python preprocess_patches.py \
        --input_dir /path/to/raw_images \
        --output_dir /path/to/patches \
        --patch_size 128 \
        --plow 1.0 \
        --phigh 99.8 \
        --min_mean_intensity 0.01 \
        --file_extensions .czi .tif .tiff .ets .vsi
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """
    Load a microscopy image file and return array with shape (C, Z, Y, X).

    Supports .czi, .tif/.tiff, and .ets formats. Falls back through
    aicsimageio → tifffile in order of preference.

    Returns:
        np.ndarray of shape (C, Z, Y, X), original dtype preserved.
    """
    suffix = path.suffix.lower()

    # Try aicsimageio first (handles .czi, .ets, .tif, and many others)
    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        # AICSImage returns (T, C, Z, Y, X); squeeze T
        data = img.data  # shape: (T, C, Z, Y, X)
        if data.ndim == 5:
            data = data[0]  # drop T dimension → (C, Z, Y, X)
        logger.info(
            f"  Loaded via aicsimageio: shape={data.shape}, dtype={data.dtype}"
        )
        return data
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  aicsimageio failed on {path.name}: {e}")

    # Fallback: tifffile for .tif/.tiff
    if suffix in (".tif", ".tiff"):
        import tifffile
        data = tifffile.imread(str(path))
        logger.info(
            f"  Loaded via tifffile: shape={data.shape}, dtype={data.dtype}"
        )
        # Expect (C, Z, Y, X) or (Z, Y, X). If 3D, add channel dim.
        if data.ndim == 3:
            data = data[np.newaxis]  # (1, Z, Y, X)
        return data

    raise ValueError(
        f"Cannot load {path.name}. Install aicsimageio for .czi/.ets support, "
        f"or provide .tif files."
    )


# ---------------------------------------------------------------------------
# Maximum intensity projection
# ---------------------------------------------------------------------------

def maximum_intensity_projection(volume: np.ndarray) -> np.ndarray:
    """
    Collapse Z-stack via MIP.

    Args:
        volume: (C, Z, Y, X)
    Returns:
        (C, Y, X) float32 MIP image.
    """
    # MIP along Z axis (axis=1)
    mip = volume.max(axis=1).astype(np.float32)
    return mip


def best_z_slice(volume: np.ndarray) -> int:
    """
    Return the z-index with the highest total intensity across all channels.

    Args:
        volume: (C, Z, Y, X)
    Returns:
        int z-index with the maximum summed intensity.
    """
    # Sum over C, Y, X for each z → shape (Z,)
    intensity_per_z = volume.sum(axis=(0, 2, 3))
    return int(np.argmax(intensity_per_z))


# ---------------------------------------------------------------------------
# Per-channel percentile normalization
# ---------------------------------------------------------------------------

def normalize_percentile(
    image: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.8,
) -> np.ndarray:
    """
    Per-channel percentile normalization, clipped to [0, 1].

    For each channel independently:
        1. Compute plow-th and phigh-th percentiles
        2. Linearly rescale so that plow → 0, phigh → 1
        3. Clip to [0, 1]

    This follows the StarDist convention (percentile + clip) rather than
    Cellpose (percentile, no clip). Clipping is safer for MAE pixel-space
    reconstruction loss because it bounds the target range.

    Args:
        image: (C, H, W) float32
        plow: lower percentile (default 1.0)
        phigh: upper percentile (default 99.8)

    Returns:
        (C, H, W) float32 in [0, 1]
    """
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        ch = image[c]
        vmin = np.percentile(ch, plow)
        vmax = np.percentile(ch, phigh)
        denom = vmax - vmin
        if denom < 1e-8:
            # Dead channel: set to zero
            out[c] = 0.0
        else:
            out[c] = np.clip((ch - vmin) / denom, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Tiling into patches
# ---------------------------------------------------------------------------

def extract_patches(
    image: np.ndarray,
    patch_size: int = 128,
) -> np.ndarray:
    """
    Tile a (C, H, W) image into non-overlapping (C, patch_size, patch_size) patches.

    2304 / 128 = 18 patches per axis, 324 patches total per image.
    If H or W is not exactly divisible by patch_size, the rightmost/bottom
    strip is discarded (for 2304 and 128 this never happens).

    Args:
        image: (C, H, W) float32
        patch_size: tile size (default 128)

    Returns:
        (N, C, patch_size, patch_size) float32, where N = n_rows * n_cols
    """
    C, H, W = image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size

    # Trim if not perfectly divisible
    image = image[:, : n_rows * patch_size, : n_cols * patch_size]

    # Reshape via view: (C, n_rows, ps, n_cols, ps) → (n_rows, n_cols, C, ps, ps)
    patches = image.reshape(C, n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)  # (n_rows, n_cols, C, ps, ps)
    patches = patches.reshape(-1, C, patch_size, patch_size)  # (N, C, ps, ps)

    return patches


# ---------------------------------------------------------------------------
# Empty patch filtering
# ---------------------------------------------------------------------------

def is_foreground_patch(
    patch: np.ndarray,
    min_mean_intensity: float = 0.01,
) -> bool:
    """
    Decide whether a patch contains enough signal to be worth training on.

    The criterion is simple: the mean intensity across all channels must
    exceed a threshold. After percentile normalization to [0,1], a mean
    of 0.01 corresponds to ~1% of dynamic range, which filters out patches
    that are pure background (no neurites, no puncta).

    StarDist uses a stricter criterion (90% of patches must contain
    foreground objects), but that requires labels. For self-supervised
    pretraining without labels, a simple intensity threshold is the
    pragmatic choice.

    Args:
        patch: (C, H, W) float32 in [0, 1]
        min_mean_intensity: threshold on mean pixel value

    Returns:
        True if patch passes the filter.
    """
    return float(patch.mean()) >= min_mean_intensity


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_image(
    path: Path,
    output_dir: Path,
    patch_size: int,
    plow: float,
    phigh: float,
    min_mean_intensity: float,
    image_index: int,
) -> list[dict]:
    """
    Full pipeline for one image file.

    Returns:
        List of dicts (one per saved patch) for the CSV index.
    """
    logger.info(f"Processing [{image_index}]: {path.name}")

    # 1. Load raw (C, Z, Y, X)
    volume = load_image(path)
    C, Z, H, W = volume.shape
    logger.info(f"  Raw shape: C={C}, Z={Z}, H={H}, W={W}")

    # 2. MIP along Z → (C, H, W)
    mip = maximum_intensity_projection(volume)
    # Free the full volume immediately
    del volume

    # 3. Percentile normalization per channel → [0, 1]
    mip = normalize_percentile(mip, plow=plow, phigh=phigh)

    # 4. Tile into patches
    patches = extract_patches(mip, patch_size=patch_size)
    n_patches = patches.shape[0]
    n_rows = H // patch_size
    n_cols = W // patch_size
    logger.info(f"  Extracted {n_patches} patches ({n_rows}x{n_cols} grid)")

    # 5. Filter and save
    records = []
    kept = 0
    for patch_idx in range(n_patches):
        patch = patches[patch_idx]  # (C, ps, ps)
        if not is_foreground_patch(patch, min_mean_intensity):
            continue

        row = patch_idx // n_cols
        col = patch_idx % n_cols

        # Filename encodes image index, grid position
        fname = f"img{image_index:04d}_r{row:02d}_c{col:02d}.npy"
        np.save(output_dir / fname, patch)

        records.append({
            "filename": fname,
            "source_image": path.name,
            "image_index": image_index,
            "grid_row": row,
            "grid_col": col,
            "mean_intensity": float(patch.mean()),
            "channels": C,
            "patch_size": patch_size,
        })
        kept += 1

    logger.info(f"  Kept {kept}/{n_patches} patches (filtered {n_patches - kept})")
    return records

