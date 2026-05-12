#!/usr/bin/env python3
"""
Generate a before/after preprocessing preview of an ETS z-stack for thesis figures.

Produces a single PNG with two rows:
  - Top row: raw data (3 representative z-slices per channel + MIP)
  - Bottom row: after preprocessing (MIP + percentile normalization)

Usage:
    python scripts/preview_ets_stack.py [--input PATH] [--output PATH]
                                        [--plow 1.0] [--phigh 99.8]

If --input is omitted, auto-discovers the first .ets file under data/Microscopy.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Add project root so we can import data_utils
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_utils.preprocess_patches import (
    load_image,
    maximum_intensity_projection,
    normalize_percentile,
)


def auto_find_image(data_dir: Path) -> Path:
    """Return the first .vsi file found under *data_dir* (falls back to .ets), skipping KONTROLA."""
    for ext in ("*.vsi", "*.ets"):
        for p in sorted(data_dir.rglob(ext)):
            if "KONTROLA" not in p.name.upper():
                return p
    raise FileNotFoundError(f"No .vsi/.ets files found under {data_dir}")


CHANNEL_COLORS = ["green", "magenta", "cyan", "yellow", "red", "blue"]
# Fluorescence-style colormaps: black → channel color
CHANNEL_RGB = [
    (0.0, 1.0, 0.0),   # green
    (1.0, 0.0, 1.0),   # magenta
    (0.0, 1.0, 1.0),   # cyan
    (1.0, 1.0, 0.0),   # yellow
    (1.0, 0.0, 0.0),   # red
    (0.0, 0.0, 1.0),   # blue
]
CHANNEL_CMAPS = [
    LinearSegmentedColormap.from_list(f"black_{name}", [(0, 0, 0), rgb])
    for name, rgb in zip(CHANNEL_COLORS, CHANNEL_RGB)
]


def _rescale_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1] for imshow (handles constant images gracefully)."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr.astype(np.float32) - lo) / (hi - lo)


def _composite_rgb(channels: np.ndarray) -> np.ndarray:
    """Merge (C, H, W) float32 channels into an (H, W, 3) RGB composite."""
    C, H, W = channels.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(C):
        r, g, b = CHANNEL_RGB[ch % len(CHANNEL_RGB)]
        scaled = _rescale_for_display(channels[ch])
        rgb[..., 0] += scaled * r
        rgb[..., 1] += scaled * g
        rgb[..., 2] += scaled * b
    return np.clip(rgb, 0, 1)


def draw_grid(ax, n_rows, n_cols, patch_size, color="white", lw=0.5, alpha=0.6):
    """Draw patch grid lines on an axes."""
    for r in range(1, n_rows):
        ax.axhline(r * patch_size - 0.5, color=color, lw=lw, alpha=alpha)
    for c in range(1, n_cols):
        ax.axvline(c * patch_size - 0.5, color=color, lw=lw, alpha=alpha)


def make_preview(
    volume: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.8,
    source_name: str = "",
    patch_size: int = 128,
) -> plt.Figure:
    """
    Build a before/after figure from a (C, Z, Y, X) volume.

    Layout:
        Top row (small)  – RAW: MIP per channel + composite
        Bottom row (big) – PREPROCESSED: normalized MIP per channel + composite
    """
    C, Z, H, W = volume.shape

    # --- compute preprocessed MIP ---
    raw_mip = maximum_intensity_projection(volume)            # (C, H, W) float32
    norm_mip = normalize_percentile(raw_mip.copy(), plow=plow, phigh=phigh)  # [0,1]

    # --- figure layout ---
    n_panels = C + 1  # per-channel + composite

    fig = plt.figure(figsize=(5 * n_panels, 9), dpi=150, facecolor="white")
    gs = GridSpec(
        nrows=2, ncols=n_panels,
        height_ratios=[1, 2],
        hspace=0.3, wspace=0.1,
        top=0.92, bottom=0.02, left=0.04, right=0.96,
    )

    fig.suptitle(
        f"ETS Stack Preview — {source_name}" if source_name else "ETS Stack Preview",
        fontsize=14, fontweight="bold", y=0.97, color="black",
    )

    # --- TOP ROW: raw MIP (small) ---
    fig.text(0.5, 0.935, "Raw MIP (before preprocessing)", ha="center", fontsize=11, fontstyle="italic", color="0.4")

    for ch in range(C):
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]
        ax = fig.add_subplot(gs[0, ch])
        ax.imshow(_rescale_for_display(raw_mip[ch]), cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        ax.set_title(f"Ch {ch}", fontsize=9, color=color, fontweight="bold")

    # Raw composite
    ax = fig.add_subplot(gs[0, C])
    ax.imshow(_composite_rgb(raw_mip))
    ax.set_axis_off()
    ax.set_title("Composite", fontsize=9, color="black", fontweight="bold")

    # --- BOTTOM ROW: normalized MIP (big) with patch grid overlay ---
    n_grid_rows = H // patch_size
    n_grid_cols = W // patch_size

    for ch in range(C):
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]
        ax = fig.add_subplot(gs[1, ch])
        ax.imshow(norm_mip[ch], cmap=cmap, vmin=0, vmax=1)
        draw_grid(ax, n_grid_rows, n_grid_cols, patch_size)
        ax.set_axis_off()
        ax.set_title(f"Ch {ch} normalized", fontsize=9, color=color, fontweight="bold")

    # Normalized composite
    ax = fig.add_subplot(gs[1, C])
    ax.imshow(_composite_rgb(norm_mip))
    draw_grid(ax, n_grid_rows, n_grid_cols, patch_size)
    ax.set_axis_off()
    ax.set_title("Composite", fontsize=9, color="black", fontweight="bold")

    # Label between the rows
    bottom_row_top = gs.get_grid_positions(fig)[0][1]
    fig.text(0.5, bottom_row_top + 0.015, "After preprocessing", ha="center", fontsize=11, fontstyle="italic", color="0.4")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Path to .vsi or .ets file (auto-detected if omitted)")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (default: ets_preview.png in cwd)")
    parser.add_argument("--plow", type=float, default=1.0, help="Lower percentile (default: 1.0)")
    parser.add_argument("--phigh", type=float, default=99.8, help="Upper percentile (default: 99.8)")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size for grid overlay (default: 128)")
    args = parser.parse_args()

    # Resolve input
    if args.input is None:
        data_dir = PROJECT_ROOT / ".." / ".." / "data" / "Microscopy" / "Microscopy" / "20251030"
        args.input = auto_find_image(data_dir)
        print(f"Auto-detected: {args.input}")

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    # Resolve output
    if args.output is None:
        args.output = Path("outs/ets_preview.png")

    print(f"Loading {args.input.name} ...")
    volume = load_image(Path(args.input))
    C, Z, H, W = volume.shape
    print(f"  Shape: C={C}, Z={Z}, H={H}, W={W}, dtype={volume.dtype}")

    print("Building preview figure ...")
    fig = make_preview(volume, plow=args.plow, phigh=args.phigh, source_name=args.input.name, patch_size=args.patch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
