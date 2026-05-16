#!/usr/bin/env python3
"""Render a reassembled image from ``.npy`` patches with a patch-grid overlay.

Usage:
    python imaging/preview_patch_grid.py [--image_index 0]
        [--patch_root data/patches_128] [--output outs/patch_grid.png]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Add thesis/root/ to sys.path so synaptic_ssl imports work without `pip install -e .`.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "root"))

from synaptic_ssl.utils_data.reassemble import reassemble_image  # noqa: E402

# ---------------------------------------------------------------------------
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
    LinearSegmentedColormap.from_list(f"black_{n}", [(0, 0, 0), rgb])
    for n, rgb in zip(CHANNEL_COLORS, CHANNEL_RGB)
]


def _rescale(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr.astype(np.float32) - lo) / (hi - lo)


def _composite_rgb(channels: np.ndarray) -> np.ndarray:
    C, H, W = channels.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(C):
        r, g, b = CHANNEL_RGB[ch % len(CHANNEL_RGB)]
        s = _rescale(channels[ch])
        rgb[..., 0] += s * r
        rgb[..., 1] += s * g
        rgb[..., 2] += s * b
    return np.clip(rgb, 0, 1)


def draw_grid(ax, n_rows, n_cols, patch_size, color="white", lw=0.5, alpha=0.6):
    """Overlay patch-boundary grid lines on ``ax``."""
    for r in range(1, n_rows):
        ax.axhline(r * patch_size - 0.5, color=color, lw=lw, alpha=alpha)
    for c in range(1, n_cols):
        ax.axvline(c * patch_size - 0.5, color=color, lw=lw, alpha=alpha)


def make_grid_figure(
    full_image: np.ndarray,
    patch_size: int,
    source_name: str = "",
) -> plt.Figure:
    """Render per-channel panels and a composite with patch-grid overlay.

    ``full_image`` is ``(C, H, W)`` float32. ``patch_size`` sets the grid pitch.
    """
    C, H, W = full_image.shape
    n_rows = H // patch_size
    n_cols = W // patch_size

    # C individual channels + 1 composite
    n_panels = C + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), dpi=150, facecolor="white")
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"Patch grid — {source_name}  ({n_rows}×{n_cols} patches of {patch_size}px)"
        if source_name else
        f"Patch grid  ({n_rows}×{n_cols} patches of {patch_size}px)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Per-channel panels
    for ch in range(C):
        ax = axes[ch]
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]
        ax.imshow(_rescale(full_image[ch]), cmap=cmap, vmin=0, vmax=1)
        draw_grid(ax, n_rows, n_cols, patch_size)
        ax.set_title(f"Ch {ch}", fontsize=10, color=color, fontweight="bold")
        ax.set_axis_off()

    # Composite panel
    ax = axes[C]
    ax.imshow(_composite_rgb(full_image))
    draw_grid(ax, n_rows, n_cols, patch_size)
    ax.set_title("Composite", fontsize=10, fontweight="bold")
    ax.set_axis_off()

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--patch_root", type=Path,
                        default=_REPO / "data" / "patches_128",
                        help="Directory with .npy patches and index.csv")
    parser.add_argument("--image_index", type=int, default=0,
                        help="Image index to reassemble (default: 0)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: outs/patch_grid_<idx>.png)")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"outs/patch_grid_{args.image_index:04d}.png")

    print(f"Reassembling image_index={args.image_index} from {args.patch_root} ...")
    full_image, records = reassemble_image(args.patch_root, args.image_index)
    patch_size = int(records[0]["patch_size"])
    source_name = records[0]["source_image"]
    C, H, W = full_image.shape
    print(f"  {source_name}: C={C}, H={H}, W={W}, patch_size={patch_size}")

    fig = make_grid_figure(full_image, patch_size, source_name=source_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
