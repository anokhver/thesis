#!/usr/bin/env python3
"""Preprocess confocal microscopy into SSL patches (no background subtraction).

Pipeline: load (C,Z,Y,X) → MIP along Z → per-channel percentile
normalisation → tile into 128x128 patches → save ``.npy``.

Per-channel ``[1, 99.8]``-percentile + clip-to-``[0, 1]`` follows
StarDist (Schmidt & Weigert); clipping bounds the MAE pixel-loss target
range. No background subtraction: DL pipelines (Cellpose, nnU-Net,
CA-MAE) skip it. For pseudolabel preprocessing (with rolling-ball)
see ``preprocess_pseudolabels.py``.

Usage::

    python preprocess_training.py --input_dir <raw> --output_dir <patches>
        [--patch_size 128] [--plow 1.0] [--phigh 99.8]
        [--max_files 1]
        [--file_extensions .czi .tif .tiff .ets .vsi]

Ref: https://github.com/stardist/stardist
Ref: https://github.com/MouseLand/cellpose
Ref: https://github.com/MIC-DKFZ/nnUNet
"""


import argparse
import csv
import gc
import logging
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .patching import INDEX_FIELDS, extract_patches

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _select_largest_scene(img) -> None:
    """Switch ``img`` to its largest scene in place.

    Multi-scene Bio-Formats containers (``.vsi``, mosaic ``.czi``) expose
    a small RGB thumbnail at scene 0. Pick the scene with the largest
    Y·X·Z to land on the pyramid base.
    """
    scenes = list(img.scenes)
    if len(scenes) <= 1:
        return
    best_scene = img.current_scene
    best_size = -1
    for sc in scenes:
        try:
            img.set_scene(sc)
        except Exception:
            continue
        dims = img.dims
        size = int(dims.X) * int(dims.Y) * max(1, int(dims.Z))
        if size > best_size:
            best_size = size
            best_scene = sc
    img.set_scene(best_scene)
    logger.info(
        f"  Selected scene {best_scene!r} ({best_size:,} voxels) "
        f"out of {len(scenes)} available"
    )


def load_image(path: Path) -> np.ndarray:
    """Load a microscopy file. Return ``(C, Z, Y, X)`` original dtype.

    Tries aicsimageio first, then tifffile for ``.tif``/``.tiff``. For
    multi-scene containers the largest scene is selected first.
    """
    suffix = path.suffix.lower()

    # Try aicsimageio first (handles .czi, .ets, .vsi, .tif, and many others)
    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        try:
            _select_largest_scene(img)

            # Ask AICSImage for a canonical C,Z,Y,X tensor. This avoids
            # reader-dependent layouts such as (T, C, Z, Y, X, S).
            data = img.get_image_data("CZYX", T=0)

            if data.ndim != 4:
                raise ValueError(
                    f"Unexpected AICSImage shape after CZYX conversion: {data.shape}"
                )

            # Detach from any Java-backed buffer so closing the reader
            # cannot invalidate the returned array.
            data = np.ascontiguousarray(data)
            logger.info(
                f"  Loaded via aicsimageio: shape={data.shape}, dtype={data.dtype}"
            )
            return data
        finally:
            try:
                img.close()
            except Exception:
                pass
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
    """MIP along Z (axis 1). ``(C, Z, Y, X)`` → ``(C, Y, X)`` float32."""
    mip = volume.max(axis=1).astype(np.float32)
    return mip


def best_z_slice(volume: np.ndarray) -> int:
    """Return the Z-index with the highest summed intensity across C·Y·X."""
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
    """Per-channel ``[plow, phigh]``-percentile rescale to ``[0, 1]`` with clipping.

    StarDist convention (clip). Clipping bounds MAE pixel-target range.
    Dead channels (vmax-vmin < 1e-8) are set to zero.

    Ref: https://github.com/stardist/stardist
    """
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        ch = image[c]
        vmin = np.percentile(ch, plow)
        vmax = np.percentile(ch, phigh)
        denom = vmax - vmin
        if denom < 1e-8:
            out[c] = 0.0
        else:
            out[c] = np.clip((ch - vmin) / denom, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Tiling into patches
# ---------------------------------------------------------------------------
# ``extract_patches`` lives in ``patching`` and is re-exported above.


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_image(
    path: Path,
    output_dir: Path,
    patch_size: int,
    plow: float,
    phigh: float,
    image_index: int,
) -> list[dict]:
    """Run the SSL preprocessing pipeline on one file. Return CSV records."""
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
    gc.collect()
    n_patches = patches.shape[0]
    n_rows = H // patch_size
    n_cols = W // patch_size
    logger.info(f"  Extracted {n_patches} patches ({n_rows}x{n_cols} grid)")

    # 5. Save every patch (no foreground filtering).
    records = []
    for patch_idx in range(n_patches):
        patch = patches[patch_idx]  # (C, ps, ps)

        row = patch_idx // n_cols
        col = patch_idx % n_cols

        fname = f"img{image_index:04d}_r{row:02d}_c{col:02d}.npy"
        np.save(output_dir / fname, patch)

        records.append({
            "filename": fname,
            "source_image": path.name,
            "source_npy": "",
            "source_path": str(path.resolve()),
            "image_index": image_index,
            "grid_row": row,
            "grid_col": col,
            "mean_intensity": float(patch.mean()),
            "channels": C,
            "patch_size": patch_size,
        })

    del patches
    gc.collect()
    logger.info(f"  Saved {n_patches} patches")
    return records


def _process_single_image_worker(args_tuple):
    """Multiprocessing wrapper. Each worker spawns its own JVM."""
    path, output_dir, patch_size, plow, phigh, image_index = args_tuple
    try:
        return process_single_image(
            path=path,
            output_dir=output_dir,
            patch_size=patch_size,
            plow=plow,
            phigh=phigh,
            image_index=image_index,
        )
    except Exception as e:
        logger.error(f"Failed on {path.name}: {e}", exc_info=True)
        return []


def _apply_config_file(args: argparse.Namespace, config_path: Path) -> None:
    """Override *args* with values from a JSON config file.

    Paths in the config are resolved relative to the directory containing
    the config file itself.
    """
    import json as _json
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = _json.load(fh)
    cfg_dir = config_path.resolve().parent
    for key, val in cfg.items():
        if key.startswith("_"):
            continue  # skip comments
        if key in ("input_dir", "output_dir") and val is not None:
            val = Path(val)
            if not val.is_absolute():
                val = (cfg_dir / val).resolve()
        setattr(args, key, val)


def _parse_args() -> argparse.Namespace:
    """Parse CLI args with config-file fallback.

    Resolution order: hard-coded defaults -> config file values ->
    explicit CLI flags. Explicit CLI flags always win.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess confocal microscopy images into normalized patches."
    )
    _script_dir = Path(__file__).resolve().parent
    _default_input = (
        _script_dir / ".." / ".." / ".." / "data" / "Microscopy" / "Microscopy" / "20251030"
    )
    _default_output = _script_dir / ".." / ".." / "data" / "patches_128"

    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a JSON config file. Values in the config override "
             "CLI defaults; explicit CLI flags override the config.",
    )
    parser.add_argument(
        "--input_dir", type=Path, default=_default_input,
        help="Directory containing raw microscopy files "
             f"(default: {_default_input}).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=_default_output,
        help="Directory to write .npy patches and index.csv "
             f"(default: {_default_output}).",
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
        "--file_extensions", nargs="+",
        default=[".czi", ".tif", ".tiff", ".ets", ".vsi"],
        help="File extensions to glob for (default: .czi .tif .tiff .ets .vsi).",
    )
    parser.add_argument(
        "--skip_patterns", nargs="+", default=["None"],
        help="Skip files whose name contains any of these substrings "
             "(case-insensitive). Default: None.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). "
             "Each worker spawns its own JVM, so memory usage scales "
             "linearly. 2-4 is a good starting point on Metacentrum.",
    )
    parser.add_argument(
        "--max_files", type=int, default=0,
        help="Process at most this many files from input_dir (default: 0 = all).",
    )

    # Two-pass parse: config supplies defaults, then CLI overrides them.
    # `provided_args` records which flags the user actually typed so we
    # can re-apply them after the config injects its values.
    provided_args = {
        action.dest
        for action in parser._actions
        if any(opt in sys.argv[1:] for opt in action.option_strings)
    }

    args = parser.parse_args()
    if args.config is not None:
        cli_explicit = {k: getattr(args, k) for k in provided_args}
        _apply_config_file(args, args.config)
        for k, v in cli_explicit.items():
            setattr(args, k, v)
    return args


def main():
    args = _parse_args()

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

    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
        logger.info(f"Limiting run to first {len(files)} file(s) due to --max_files")

    logger.info(f"Found {len(files)} image files in {args.input_dir}")
    logger.info(
        f"Settings: patch_size={args.patch_size}, "
        f"percentiles=(plow={args.plow}, phigh={args.phigh}), "
        f"workers={args.workers}"
    )

    work_items = [
        (filepath, args.output_dir, args.patch_size,
         args.plow, args.phigh, idx)
        for idx, filepath in enumerate(files)
    ]

    all_records: list[dict] = []
    n_workers = min(args.workers, len(files))

    if n_workers <= 1:
        # Sequential — no overhead from spawning processes
        for item in work_items:
            all_records.extend(_process_single_image_worker(item))
    else:
        # Parallel — spawn (not fork) so each worker gets a clean JVM
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for records in pool.imap_unordered(
                _process_single_image_worker, work_items
            ):
                all_records.extend(records)

    all_records.sort(
        key=lambda r: (r["image_index"], r["grid_row"], r["grid_col"])
    )

    csv_path = args.output_dir / "index.csv"
    if all_records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(INDEX_FIELDS), extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(all_records)
    else:
        logger.error(
            "No patches were produced. Failing run so callers can detect the issue."
        )
        sys.exit(2)

    logger.info(
        f"Done. {len(all_records)} patches saved to {args.output_dir}. "
        f"Index written to {csv_path}."
    )


if __name__ == "__main__":
    main()

