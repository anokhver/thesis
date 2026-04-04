#!/usr/bin/env python3
"""
preprocess_patches.py
Preprocessing pipeline for confocal fluorescence microscopy images.

Pipeline:
    raw file → load (C,Z,Y,X) → MIP along Z → per-channel percentile
    normalization → tile into 128×128 patches → filter empty patches → save .npy

Design:
    - MIP for Z-handling (SynBot, SynapseJ)
    - Per-channel independent percentile normalization (Cellpose [1,99]),
      StarDist [1,99.8]. We use [1,99.8] with clipping to [0,1] because
      StarDist's approach better preserves bright puncta that occupy <1%
      of image area (the 99th percentile can clip them).
    - 128×128 patches
    - No background subtraction
Usage:
    python preprocess_patches.py \
        --input_dir /path/to/raw_images \
        --output_dir /path/to/patches \
        --patch_size 128 \
        --plow 1.0 \
        --phigh 99.8 \
        --min_mean_intensity 0.01 \
        --file_extensions .czi .tif .tiff .ets
"""

import argparse
import csv
import gc
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
    Input: vsi files path

    Returns:
        np.ndarray of shape (C, Z, Y, X), original dtype preserved.
    """

    # Try aicsimageio (handles .czi, .ets, .tif, and many others)
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

    raise ValueError(
        f"Cannot load {path.name}"
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
    del volume
    gc.collect()

    # 3. Percentile normalization per channel → [0, 1]
    mip = normalize_percentile(mip, plow=plow, phigh=phigh)

    # 4. Tile into patches
    patches = extract_patches(mip, patch_size=patch_size)
    del mip
    n_patches = patches.shape[0]
    n_rows = H // patch_size
    n_cols = W // patch_size
    logger.info(f"  Extracted {n_patches} patches ({n_rows}x{n_cols} grid)")

    # 5. Filter and save
    records = []
    kept = 0
    for patch_idx in range(n_patches):
        patch = patches[patch_idx]  # (C, ps, ps)
        # NOTE: Foreground filtering is intentionally disabled but retained for potential future use.
        # To re-enable, uncomment the following two lines:
        # if not is_foreground_patch(patch, min_mean_intensity):
        #     continue

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

    del patches
    gc.collect()
    logger.info(f"  Kept {kept}/{n_patches} patches (filtered {n_patches - kept})")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess confocal microscopy images into normalized patches."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing raw microscopy files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Directory to write .npy patches and index.csv.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=128,
        help="Patch side length in pixels (default: 128).",
    )
    parser.add_argument(
        "--plow", type=float, default=1.0,
        help="Lower percentile for normalization (default: 1.0).",
    )
    parser.add_argument(
        "--phigh", type=float, default=99.8,
        help="Upper percentile for normalization (default: 99.8).",
    )
    parser.add_argument(
        "--min_mean_intensity", type=float, default=0.01,
        help="Minimum mean intensity to keep a patch (default: 0.01).",
    )
    parser.add_argument(
        "--file_extensions", nargs="+", default=[".czi", ".tif", ".tiff", ".ets"],
        help="File extensions to glob for (default: .czi .tif .tiff .ets).",
    )
    parser.add_argument(
        "--skip_patterns", nargs="+", default=["KONTROLA"],
        help="Skip files whose name contains any of these substrings "
             "(case-insensitive). Default: KONTROLA (control samples).",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    skip_patterns = [p.upper() for p in (args.skip_patterns or [])]
    files = sorted(
        f for ext in args.file_extensions
        for f in args.input_dir.glob(f"*{ext}")
        if not any(p in f.name.upper() for p in skip_patterns)
    )
    if skip_patterns:
        logger.info(f"Skipping files matching: {args.skip_patterns}")
    if not files:
        logger.error(
            f"No files found in {args.input_dir} with extensions "
            f"{args.file_extensions}"
        )
        sys.exit(1)

    logger.info(f"Found {len(files)} image files in {args.input_dir}")
    logger.info(
        f"Settings: patch_size={args.patch_size}, "
        f"percentiles=[{args.plow}, {args.phigh}], "
        f"min_mean_intensity={args.min_mean_intensity}"
    )

    all_records = []
    for idx, filepath in enumerate(files):
        try:
            records = process_single_image(
                path=filepath,
                output_dir=args.output_dir,
                patch_size=args.patch_size,
                plow=args.plow,
                phigh=args.phigh,
                min_mean_intensity=args.min_mean_intensity,
                image_index=idx,
            )
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed on {filepath.name}: {e}", exc_info=True)
        gc.collect()

    csv_path = args.output_dir / "index.csv"
    if all_records:
        fieldnames = list(all_records[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

    logger.info(
        f"Done. {len(all_records)} patches saved to {args.output_dir}. "
        f"Index written to {csv_path}."
    )


if __name__ == "__main__":
    main()

