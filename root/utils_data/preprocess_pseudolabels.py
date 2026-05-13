#!/usr/bin/env python3
"""Preprocess confocal fluorescence into pseudolabel patches.

Pipeline: load (C,Z,Y,X) → MIP along Z → per-channel 2D Gaussian
background subtraction → per-channel percentile normalisation → tile
into 128x128 patches → save ``.npy``.

For SSL pretraining patches (no background subtraction) use
``preprocess_training.py``.

MIP-then-background-subtraction order: SynBot (Savage et al., Cell
Reports Methods 2024). Gaussian high-pass approximation to rolling-ball
(Sternberg, IEEE Computer 1983) — same substitute used in CellProfiler
and scikit-image. Per-channel parameters: Simhal et al.
(Neuroinformatics 2017). Per-channel percentile norm: Cellpose
(Stringer et al., Nature Methods 2021) and StarDist/CSBDeep (Schmidt &
Weigert).

Ref: https://github.com/Eroglu-Lab/Syn_Bot
Ref: https://github.com/GB3Trinity/SynapseJ
Ref: https://github.com/MouseLand/cellpose
Ref: https://github.com/stardist/stardist
Ref: https://github.com/CSBDeep/CSBDeep
Ref: https://github.com/MIC-DKFZ/nnUNet
"""

from __future__ import annotations

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
# Channel-role-aware preprocessing parameters
# ---------------------------------------------------------------------------
# Matches the convention in training/config.py:
#     channel_names = ["pre_synaptic", "post_synaptic", "structural"]
DEFAULT_CHANNEL_ROLES: list[str] = ["pre_synaptic", "post_synaptic", "structural"]

# Per-role preprocessing defaults derived from the synapse / fluorescence
# literature (see citations in module docstring). Applied to the 2D MIP
# (SynBot ordering):
#   - Synaptic channels (sparse puncta, ~2% foreground; Simhal 2017): keep
#     the Gaussian/rolling-ball background subtraction — puncta sit on top
#     of diffuse autofluorescence. Upper percentile pushed to 99.8 so the
#     bright punctum cores are not saturated.
#   - Structural channel (dense neurite/dendrite stain, MAP2-like): SKIP
#     background subtraction — a Gaussian-blur background estimate at
#     sigma≈50 falls inside the signal distribution and the clip-to-zero
#     step then erases legitimate structural staining. Same [1, 99.8]
#     percentile is safe because the signal is broad and the upper tail
#     is not dominated by a few bright punctum pixels.
ROLE_DEFAULTS: dict[str, dict[str, float]] = {
    "pre_synaptic":  {"median_size": 0, "rolling_ball_radius": 50.0, "plow": 1.0, "phigh": 99.8, "smooth_sigma": 0.0},
    "post_synaptic": {"median_size": 0, "rolling_ball_radius": 50.0, "plow": 1.0, "phigh": 99.8, "smooth_sigma": 0.0},
    "synaptic":      {"median_size": 0, "rolling_ball_radius": 50.0, "plow": 1.0, "phigh": 99.8, "smooth_sigma": 0.0},
    "structural":    {"median_size": 0, "rolling_ball_radius": 0.0,  "plow": 1.0, "phigh": 99.8, "smooth_sigma": 2.0},
}


def _resolve_role_or_value(
    value,
    n_channels: int,
    channel_roles: list[str],
    param: str,
):
    """Resolve a CLI value to a per-channel list, defaulting from ``ROLE_DEFAULTS``.

    ``None`` → role defaults. Scalar or length-1 → broadcast. Length-C → as-is.
    """
    if value is None:
        return [ROLE_DEFAULTS[r][param] for r in channel_roles]
    return _to_per_channel_list(value, n_channels, param)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _select_largest_scene(img) -> None:
    """Switch ``img`` to its largest scene in place.

    Multi-scene Bio-Formats containers (``.vsi``, mosaic ``.czi``) expose
    a small RGB thumbnail at scene 0 plus pyramid levels of the real
    acquisition. Pick the scene with the largest Y·X·Z to land on the
    pyramid base instead of the thumbnail.
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
    """Load a microscopy file via aicsimageio. Return ``(C, Z, Y, X)``.

    For multi-scene containers the largest scene is selected first to
    avoid embedded RGB thumbnails. The aicsimageio handle is closed in a
    ``finally`` to release JPype/Bio-Formats Java proxies.
    """
    # Try aicsimageio (handles .czi, .ets, .vsi, .tif, and many others)
    try:
        from aicsimageio import AICSImage
        img = AICSImage(path)
        try:
            _select_largest_scene(img)
            # AICSImage returns (T, C, Z, Y, X); squeeze T
            data = img.data  # shape: (T, C, Z, Y, X)
            if data.ndim == 5:
                data = data[0]  # drop T → (C, Z, Y, X)
            # Detach from any Java-backed buffer so closing the reader
            # below cannot invalidate the returned array.
            data = np.ascontiguousarray(data)
            logger.info(
                f"  Loaded via aicsimageio: shape={data.shape}, dtype={data.dtype}"
            )
            return data
        finally:
            # Bio-Formats / .czi / .ets / .vsi readers hold Java proxies
            # via JPype/scyjava. Drop our handle deterministically so the
            # JVM side can be released by the gc.collect() in the caller
            # — otherwise reader state accumulates across files and the
            # JVM eventually fails to spawn new native threads.
            try:
                img.close()
            except Exception:
                pass
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"  aicsimageio failed on {path.name}: {e}")

    raise ValueError(f"Cannot load {path.name}")


# ---------------------------------------------------------------------------
# Pre-MIP denoising
# ---------------------------------------------------------------------------

def _to_per_channel_list(value, n_channels: int, name: str):
    """Coerce a scalar or sequence into a list of length ``n_channels``."""
    if value is None:
        raise ValueError(f"{name} must not be None at this point.")
    if not isinstance(value, (list, tuple, np.ndarray)):
        return [value] * n_channels
    seq = list(value)
    if len(seq) == 1:
        return [seq[0]] * n_channels
    if len(seq) == n_channels:
        return seq
    raise ValueError(
        f"{name} expects 1 or {n_channels} values, got {len(seq)}: {seq!r}"
    )


def denoise_volume(
    volume: np.ndarray,
    median_size=0,
    rolling_ball_radius: float = 50.0,
) -> np.ndarray:
    """Per-channel per-Z-slice median filter and Gaussian background subtraction.

    Set ``rolling_ball_radius=0`` to skip on a channel. Return ``(C, Z, Y, X)``
    float32. Kept for parity with notebooks that compare pre-MIP vs post-MIP
    denoising; ``process_single_image`` uses ``denoise_mip`` instead.
    """
    C = volume.shape[0]
    median_sizes = _to_per_channel_list(median_size, C, "median_size")
    radii = _to_per_channel_list(rolling_ball_radius, C, "rolling_ball_radius")

    if all(m <= 0 for m in median_sizes) and all(r <= 0 for r in radii):
        return volume.astype(np.float32, copy=False)

    from scipy.ndimage import gaussian_filter as _gaussian_filter
    from scipy.ndimage import median_filter as _median_filter

    out = volume.astype(np.float32, copy=False)

    # Each channel uses its own role-tuned parameters. The Z axis is
    # iterated implicitly via size=(1, m, m) / sigma=(0, r, r).
    # truncate=2.5 (vs scipy's default 4.0) shrinks the Gaussian kernel
    # half-width from 4σ to 2.5σ. With σ=50 that's a 251×251 kernel
    # instead of 401×401 (~2.5× fewer ops). The tail beyond 2.5σ
    # contributes ≤G(2.5σ)≈0.018 to the background estimate and is
    # essentially erased by the subsequent clip-to-zero step.
    for c in range(C):
        m = int(median_sizes[c])
        r = float(radii[c])
        if m > 0:
            out[c] = _median_filter(out[c], size=(1, m, m))
        if r > 0:
            bg = _gaussian_filter(out[c], sigma=(0, r, r), truncate=2.5)
            out[c] = np.clip(out[c] - bg, 0, None)

    return out


def denoise_mip(
    mip: np.ndarray,
    median_size=0,
    rolling_ball_radius: float = 50.0,
) -> np.ndarray:
    """Per-channel 2D Gaussian-high-pass background subtraction on a MIP.

    Optional 2D median filter, then ``image - gaussian(image, sigma)`` per
    channel. ``rolling_ball_radius=0`` (or ``median_size=0``) skips that
    step on a channel. Order follows SynBot (project first, subtract
    background after).

    Args:
        mip: ``(C, H, W)`` array.
        median_size: scalar or length-C list of median window sizes (0 = off).
        rolling_ball_radius: scalar or length-C list of Gaussian sigmas (0 = off).

    Returns:
        ``(C, H, W)`` float32.

    Ref: https://github.com/Eroglu-Lab/Syn_Bot
    """
    if mip.ndim != 3:
        raise ValueError(
            f"denoise_mip expects (C, H, W); got shape {mip.shape}"
        )
    C = mip.shape[0]
    median_sizes = _to_per_channel_list(median_size, C, "median_size")
    radii = _to_per_channel_list(rolling_ball_radius, C, "rolling_ball_radius")

    if all(m <= 0 for m in median_sizes) and all(r <= 0 for r in radii):
        return mip.astype(np.float32, copy=False)

    from scipy.ndimage import gaussian_filter as _gaussian_filter
    from scipy.ndimage import median_filter as _median_filter

    out = mip.astype(np.float32, copy=False)

    # truncate=2.5 (vs scipy's default 4.0) shrinks the Gaussian kernel
    # half-width from 4σ to 2.5σ. With σ=50 that's a 251×251 kernel
    # instead of 401×401. The tail beyond 2.5σ contributes ≤G(2.5σ)≈0.018
    # and is essentially erased by the subsequent clip-to-zero step.
    for c in range(C):
        m = int(median_sizes[c])
        r = float(radii[c])
        if m > 0:
            out[c] = _median_filter(out[c], size=(m, m))
        if r > 0:
            bg = _gaussian_filter(out[c], sigma=r, truncate=2.5)
            out[c] = np.clip(out[c] - bg, 0, None)

    return out


def smooth_mip(
    mip: np.ndarray,
    smooth_sigma=0,
) -> np.ndarray:
    """Per-channel Gaussian low-pass smoothing on a MIP.

    Fills pixelation gaps and connects fragmented structures (e.g. patchy
    dendrites or somas on the structural channel).  This is a *low-pass*
    blur — conceptually opposite to ``denoise_mip`` which is high-pass
    background subtraction.  ``smooth_sigma=0`` skips that channel.

    Applied after background subtraction but before percentile
    normalisation so that the blur operates on physical intensities.

    Args:
        mip: ``(C, H, W)`` array.
        smooth_sigma: scalar or length-C list of Gaussian sigmas (0 = off).

    Returns:
        ``(C, H, W)`` float32.
    """
    if mip.ndim != 3:
        raise ValueError(
            f"smooth_mip expects (C, H, W); got shape {mip.shape}"
        )
    C = mip.shape[0]
    sigmas = _to_per_channel_list(smooth_sigma, C, "smooth_sigma")

    if all(s <= 0 for s in sigmas):
        return mip.astype(np.float32, copy=False)

    from scipy.ndimage import gaussian_filter as _gaussian_filter

    out = mip.astype(np.float32, copy=True)
    for c in range(C):
        s = float(sigmas[c])
        if s > 0:
            out[c] = _gaussian_filter(out[c], sigma=s)
    return out


# ---------------------------------------------------------------------------
# Maximum intensity projection
# ---------------------------------------------------------------------------

def maximum_intensity_projection(volume: np.ndarray) -> np.ndarray:
    """MIP along Z (axis 1). ``(C, Z, Y, X)`` → ``(C, Y, X)`` float32."""
    # MIP along Z axis (axis=1)
    mip = volume.max(axis=1)
    return mip.astype(np.float32, copy=False)


def best_z_slice(volume: np.ndarray) -> int:
    """Return the Z-index with the highest summed intensity across C·Y·X."""
    # Sum over C, Y, X for each z → shape (Z,)
    intensity_per_z = volume.sum(axis=(0, 2, 3))
    return int(np.argmax(intensity_per_z))


# ---------------------------------------------------------------------------
# Per-channel percentile normalization
# ---------------------------------------------------------------------------

def normalize_percentile(
    image: np.ndarray,
    plow=1.0,
    phigh=99.8,
) -> np.ndarray:
    """Per-channel percentile rescale to ``[0, 1]`` with clipping.

    Rescale each channel from its ``[plow, phigh]`` percentiles to ``[0, 1]``.
    StarDist/CSBDeep convention. Return ``(C, H, W)`` float32.

    Ref: https://github.com/stardist/stardist
    Ref: https://github.com/CSBDeep/CSBDeep
    Ref: https://github.com/MouseLand/cellpose
    """
    C = image.shape[0]
    plows = _to_per_channel_list(plow, C, "plow")
    phighs = _to_per_channel_list(phigh, C, "phigh")

    out = np.empty_like(image, dtype=np.float32)
    for c in range(C):
        vmin, vmax = np.percentile(image[c], [plows[c], phighs[c]])
        denom = vmax - vmin
        if denom < 1e-8:
            out[c] = 0.0
        else:
            out[c] = np.clip((image[c] - vmin) / denom, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Tiling into patches
# ---------------------------------------------------------------------------

def extract_patches(
    image: np.ndarray,
    patch_size: int = 128,
) -> np.ndarray:
    """Tile ``(C, H, W)`` into non-overlapping ``(N, C, ps, ps)`` patches.

    Trim the bottom/right strip if ``H`` or ``W`` is not divisible by ``patch_size``.
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
# Main pipeline
# ---------------------------------------------------------------------------

def process_single_image(
    path: Path,
    output_dir: Path,
    patch_size: int,
    plow,
    phigh,
    image_index: int,
    channel_roles: Optional[list[str]] = None,
    median_size=None,
    rolling_ball_radius=None,
    smooth_sigma=None,
) -> list[dict]:
    """Run the full pseudolabel-preprocessing pipeline on one image file.

    Per-channel params accept a scalar or length-C list. ``None`` falls back
    to ``ROLE_DEFAULTS`` for the matching role. Return one CSV-index record
    per saved patch.
    """
    logger.info(f"Processing [{image_index}]: {path.name}")

    # 1. Load raw (C, Z, Y, X)
    volume = load_image(path)
    C, Z, H, W = volume.shape
    logger.info(f"  Raw shape: C={C}, Z={Z}, H={H}, W={W}")

    roles = channel_roles if channel_roles is not None else DEFAULT_CHANNEL_ROLES
    if len(roles) != C:
        raise ValueError(
            f"channel_roles has length {len(roles)} but image has C={C}: "
            f"{roles!r}"
        )

    median_sizes_pc = _resolve_role_or_value(median_size, C, roles, "median_size")
    radii_pc        = _resolve_role_or_value(rolling_ball_radius, C, roles, "rolling_ball_radius")
    smooth_pc       = _resolve_role_or_value(smooth_sigma, C, roles, "smooth_sigma")
    plows_pc        = _resolve_role_or_value(plow, C, roles, "plow")
    phighs_pc       = _resolve_role_or_value(phigh, C, roles, "phigh")
    logger.info(
        f"  Per-channel ({roles}): median={median_sizes_pc}, "
        f"radius={radii_pc}, smooth={smooth_pc}, "
        f"plow={plows_pc}, phigh={phighs_pc}"
    )

    # 2. MIP along Z → (C, H, W). Follow SynBot ordering: project first,
    #    then run background subtraction on the 2D projection.
    mip = maximum_intensity_projection(volume)
    del volume
    # Force a cycle-collector pass so JPype/Bio-Formats Java proxies from
    # load_image() are released before we move on to the next file.
    gc.collect()

    # 3. Per-channel rolling-ball background subtraction on the MIP.
    #    Skipped entirely if every channel has median_size=0 and radius=0.
    #    For the typical (synaptic, synaptic, structural) layout this runs
    #    on the synaptic channels only and skips the structural channel —
    #    the structural-channel skip is the role default that prevents the
    #    Gaussian background estimate from erasing dense neurite staining.
    if any(m > 0 for m in median_sizes_pc) or any(r > 0 for r in radii_pc):
        mip = denoise_mip(
            mip,
            median_size=median_sizes_pc,
            rolling_ball_radius=radii_pc,
        )

    # 3b. Per-channel Gaussian low-pass smoothing. Fills pixelation gaps
    #     and connects fragmented dendrites / somas on the structural
    #     channel. Skipped when smooth_sigma=0 (default for synaptic).
    if any(s > 0 for s in smooth_pc):
        mip = smooth_mip(mip, smooth_sigma=smooth_pc)

    # 4. Per-channel percentile normalization → [0, 1]
    mip = normalize_percentile(mip, plow=plows_pc, phigh=phighs_pc)

    # 5. Tile into patches
    patches = extract_patches(mip, patch_size=patch_size)
    del mip
    gc.collect()
    n_patches = patches.shape[0]
    n_rows = H // patch_size
    n_cols = W // patch_size
    logger.info(f"  Extracted {n_patches} patches ({n_rows}x{n_cols} grid)")

    # 6. Save every patch (no foreground filtering, no damage flagging).
    records: list[dict] = []
    for patch_idx in range(n_patches):
        patch = patches[patch_idx]  # (C, ps, ps)

        row = patch_idx // n_cols
        col = patch_idx % n_cols
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

    del patches
    gc.collect()
    logger.info(f"  Saved {n_patches} patches")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess confocal microscopy images into normalized patches."
    )
    _script_dir = Path(__file__).resolve().parent
    _default_input = _script_dir / ".." / ".." / ".." / "data" / "Microscopy" / "Microscopy" / "20251030"
    _default_output = _script_dir / ".." / ".." / "data" / "patches_128_new2"

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
        "--channel_roles", nargs="+", default=DEFAULT_CHANNEL_ROLES,
        choices=list(ROLE_DEFAULTS.keys()),
        help="Role of each channel, in channel order. Roles determine the "
             "default per-channel preprocessing parameters (see "
             "ROLE_DEFAULTS). Default: pre_synaptic post_synaptic structural.",
    )
    parser.add_argument(
        "--plow", type=float, nargs="+", default=None,
        help="Lower percentile for normalization. Pass one value to "
             "broadcast to all channels, or one value per channel. "
             "Default: role-based (1.0 for all roles).",
    )
    parser.add_argument(
        "--phigh", type=float, nargs="+", default=None,
        help="Upper percentile for normalization. Pass one value to "
             "broadcast to all channels, or one value per channel. "
             "Default: role-based (99.8 for all roles).",
    )
    parser.add_argument(
        "--median_size", type=int, nargs="+", default=None,
        help="Kernel size for 2D median filter applied to the MIP. "
             "Pass one value (broadcast) or one per channel. Default: "
             "role-based (0 = off for all roles).",
    )
    parser.add_argument(
        "--rolling_ball_radius", type=float, nargs="+", default=None,
        help="Sigma for Gaussian-bg subtraction applied to the 2D MIP "
             "(post-projection). Pass one value (broadcast) or one per "
             "channel. Default: role-based (50 for synaptic, 0 = off for "
             "structural — Gaussian-bg on a dense structural channel "
             "erases legitimate signal).",
    )
    parser.add_argument(
        "--smooth_sigma", type=float, nargs="+", default=None,
        help="Sigma for Gaussian low-pass smoothing applied to the 2D "
             "MIP after background subtraction but before percentile "
             "normalisation. Fills pixelation gaps and connects "
             "fragmented dendrites/somas. Pass one value (broadcast) or "
             "one per channel. Default: role-based (0 for synaptic, "
             "2.0 for structural).",
    )
    parser.add_argument(
        "--file_extensions", nargs="+", default=[".czi", ".tif", ".tiff", ".ets", ".vsi"],
        help="File extensions to glob for (default: .czi .tif .tiff .ets .vsi). "
             "Multi-scene containers (notably .vsi, and some .czi mosaics) "
             "are loaded by picking the largest scene to avoid the embedded "
             "RGB thumbnail.",
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
        f"channel_roles={args.channel_roles}, "
        f"percentiles=(plow={args.plow}, phigh={args.phigh}), "
        f"median_size={args.median_size}, "
        f"rolling_ball_radius={args.rolling_ball_radius}, "
        f"smooth_sigma={args.smooth_sigma} "
        f"(None ⇒ role-based default from ROLE_DEFAULTS)"
    )

    all_records: list[dict] = []
    common_kwargs = dict(
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        plow=args.plow,
        phigh=args.phigh,
        channel_roles=args.channel_roles,
        median_size=args.median_size,
        rolling_ball_radius=args.rolling_ball_radius,
        smooth_sigma=args.smooth_sigma,
    )

    for idx, filepath in enumerate(files):
        try:
            records = process_single_image(
                path=filepath,
                image_index=idx,
                **common_kwargs,
            )
            all_records.extend(records)
        except Exception as e:
            logger.error(f"Failed on {filepath.name}: {e}", exc_info=True)
        # Per-iteration cleanup so JPype/Bio-Formats Java proxies do not
        # accumulate across files (otherwise the JVM eventually cannot
        # spawn new native threads on Windows).
        gc.collect()

    all_records.sort(
        key=lambda r: (r["image_index"], r["grid_row"], r["grid_col"])
    )

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

