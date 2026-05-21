#!/usr/bin/env python3
"""Generate a before/after preprocessing preview PNG of an ETS z-stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Add thesis/src/ to sys.path so synaptic_ssl imports work without `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from synaptic_ssl.utils_data.preprocess_training import (  # noqa: E402
    load_image,
    maximum_intensity_projection,
    normalize_percentile,
)


def auto_find_image(data_dir: Path) -> Path:
    """Return the first ``.vsi`` or ``.ets`` found under ``data_dir``."""
    for ext in ("*.vsi", "*.ets"):
        for p in sorted(data_dir.rglob(ext)):
            return p
    raise FileNotFoundError(f"No .vsi/.ets files found under {data_dir}")


CHANNEL_COLORS = ["green", "magenta", "cyan", "yellow", "red", "blue"]
CHANNEL_RGB = [
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
]
CHANNEL_CMAPS = [
    LinearSegmentedColormap.from_list(f"black_{name}", [(0, 0, 0), rgb])
    for name, rgb in zip(CHANNEL_COLORS, CHANNEL_RGB)
]


def _rescale_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale ``arr`` to ``[0, 1]`` for imshow. Return zeros if constant."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr.astype(np.float32) - lo) / (hi - lo)


def _composite_rgb(channels: np.ndarray, rescale_each: bool = True) -> np.ndarray:
    """Build an RGB composite from ``(C, H, W)`` channels using channel colors."""
    C, H, W = channels.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(C):
        r, g, b = CHANNEL_RGB[ch % len(CHANNEL_RGB)]
        src = _rescale_for_display(channels[ch]) if rescale_each else channels[ch].astype(np.float32)
        rgb[..., 0] += src * r
        rgb[..., 1] += src * g
        rgb[..., 2] += src * b
    return np.clip(rgb, 0.0, 1.0)


def _pick_random_z_indices(Z: int, n: int = 3, seed: int | None = None) -> list[int]:
    """Pick up to ``n`` distinct random z-indices; reproducible when ``seed`` is set."""
    if Z <= 0:
        return [0]
    if Z <= n:
        return sorted(set(np.linspace(0, Z - 1, Z, dtype=int).tolist()))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(Z, size=n, replace=False).tolist())


def _draw_grid(ax: plt.Axes, H: int, W: int, patch_size: int, color: str = "white", lw: float = 0.4, alpha: float = 0.45) -> None:
    """Overlay patch boundaries for a given patch size."""
    if patch_size <= 0:
        return
    n_rows = H // patch_size
    n_cols = W // patch_size
    for r in range(1, n_rows):
        ax.axhline(r * patch_size - 0.5, color=color, lw=lw, alpha=alpha)
    for c in range(1, n_cols):
        ax.axvline(c * patch_size - 0.5, color=color, lw=lw, alpha=alpha)


def make_preview(
    volume: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.8,
    patch_size: int = 128,
    seed: int | None = None,
    source_name: str = "",
) -> plt.Figure:
    """Build a raw-vs-preprocessed figure from a ``(C, Z, Y, X)`` volume.

    Top block (compact): per-channel raw previews with 3 random z-slices + raw MIP.
    Bottom row (large): normalized channel MIPs + normalized composite with patch grid.
    """
    C, Z, H, W = volume.shape

    z_indices = _pick_random_z_indices(Z, n=3, seed=seed)
    n_z = len(z_indices)

    # --- compute preprocessed MIP ---
    raw_mip = maximum_intensity_projection(volume)            # (C, H, W) float32
    norm_mip = normalize_percentile(raw_mip.copy(), plow=plow, phigh=phigh)  # [0,1]

    norm_composite = _composite_rgb(norm_mip, rescale_each=False)

    # --- figure layout ---
    # Compact raw block per channel at top + one large post-preprocess row at bottom.
    top_cols = n_z + 1
    bottom_cols = C + 1
    ncols = max(top_cols, bottom_cols)
    fig = plt.figure(figsize=(3.6 * ncols, 2.0 * C + 5.8), dpi=150, facecolor="white")
    gs = GridSpec(
        nrows=C + 1,
        ncols=ncols,
        height_ratios=[1.1] * C + [2.0],
        hspace=0.18,
        wspace=0.06,
        top=0.88,
        bottom=0.06,
        left=0.035,
        right=0.985,
    )

    fig.suptitle(
        f"ETS Stack Preview — {source_name}" if source_name else "ETS Stack Preview",
        fontsize=13,
        fontweight="bold",
        y=0.975,
        color="black",
    )

    # Track panel axes to place section labels dynamically (no overlap).
    top_row_axes: list[plt.Axes] = []

    for ch in range(C):
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]

        # --- TOP (compact): raw z-slices and raw MIP ---
        for j, zi in enumerate(z_indices):
            ax = fig.add_subplot(gs[ch, j])
            if ch == 0:
                top_row_axes.append(ax)
            ax.set_facecolor("white")
            ax.imshow(_rescale_for_display(volume[ch, zi]), cmap=cmap, vmin=0, vmax=1)
            ax.set_axis_off()
            if j == 0:
                ax.text(
                    -0.08, 0.5, f"Ch {ch}", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", color=color,
                    ha="right", va="center", rotation=90,
                )

        # --- TOP: raw MIP ---
        ax = fig.add_subplot(gs[ch, n_z])
        if ch == 0:
            top_row_axes.append(ax)
        ax.set_facecolor("white")
        ax.imshow(_rescale_for_display(raw_mip[ch]), cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        if ch == 0:
            ax.set_title("MIP (raw)", fontsize=8, color="0.2")

    # --- BOTTOM (large): normalized channels in one row + composite ---
    bottom_row_axes: list[plt.Axes] = []
    for ch in range(C):
        ax = fig.add_subplot(gs[C, ch])
        bottom_row_axes.append(ax)
        ax.set_facecolor("white")
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]
        ax.imshow(norm_mip[ch], cmap=cmap, vmin=0, vmax=1)
        _draw_grid(ax, H=H, W=W, patch_size=patch_size)
        ax.set_title(f"Ch {ch} normalized", fontsize=11, fontweight="bold", color=color)
        ax.set_axis_off()

    ax = fig.add_subplot(gs[C, C])
    bottom_row_axes.append(ax)
    ax.set_facecolor("white")
    ax.imshow(norm_composite, vmin=0, vmax=1)
    _draw_grid(ax, H=H, W=W, patch_size=patch_size)
    ax.set_title("Composite", fontsize=11, fontweight="bold", color="0.1")
    ax.set_axis_off()

    z_str = ", ".join(str(z) for z in z_indices)
    top_y = max(ax_.get_position().y1 for ax_ in top_row_axes)
    bottom_top_y = max(ax_.get_position().y1 for ax_ in bottom_row_axes)
    bottom_bottom_y = min(ax_.get_position().y0 for ax_ in bottom_row_axes)

    fig.text(
        0.5,
        min(0.95, top_y + 0.04),
        f"Raw data (compact): z = [{z_str}] + MIP",
        ha="center",
        fontsize=11,
        fontstyle="italic",
        color="0.3",
    )
    fig.text(
        0.5,
        max(0.015, bottom_bottom_y - 0.025),
        f"After preprocessing: percentile [{plow}, {phigh}] with patch grid ({patch_size}px)",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="0.3",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Path to .vsi or .ets file (auto-detected if omitted)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: data/imaging_outputs/ets_preview_<input_stem>.png)",
    )
    parser.add_argument("--plow", type=float, default=1.0, help="Lower percentile (default: 1.0)")
    parser.add_argument("--phigh", type=float, default=99.8, help="Upper percentile (default: 99.8)")
    parser.add_argument("--patch-size", type=int, default=128, choices=[32, 128], help="Patch size for grid overlay: 32 or 128 (default: 128)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random z-slice sampling (default: random each run)")
    args = parser.parse_args()

    # Resolve input
    if args.input is None:
        _project_root = Path(__file__).resolve().parent.parent
        data_dir = _project_root / "data" / "Microscopy"
        args.input = auto_find_image(data_dir)
        print(f"Auto-detected: {args.input}")

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    # Resolve output
    if args.output is None:
        output_dir = _REPO / "data" / "imaging_outputs"
        args.output = output_dir / f"ets_preview_{args.input.stem}.png"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input.name} ...")
    volume = load_image(Path(args.input))
    C, Z, H, W = volume.shape
    print(f"  Shape: C={C}, Z={Z}, H={H}, W={W}, dtype={volume.dtype}")
    print(f"  Grid patch size: {args.patch_size}")

    print("Building preview figure ...")
    fig = make_preview(
        volume,
        plow=args.plow,
        phigh=args.phigh,
        patch_size=args.patch_size,
        seed=args.seed,
        source_name=args.input.name,
    )

    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
