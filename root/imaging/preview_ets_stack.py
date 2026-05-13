#!/usr/bin/env python3
"""Generate a before/after preprocessing preview PNG of an ETS z-stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from utils_data.preprocess_pseudolabels import (
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
CHANNEL_CMAPS = ["Greens", "RdPu", "Blues", "YlOrBr", "Reds", "PuBu"]


def _rescale_for_display(arr: np.ndarray) -> np.ndarray:
    """Rescale ``arr`` to ``[0, 1]`` for imshow. Return zeros if constant."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr.astype(np.float32) - lo) / (hi - lo)


def make_preview(
    volume: np.ndarray,
    plow: float = 1.0,
    phigh: float = 99.8,
    source_name: str = "",
) -> plt.Figure:
    """Build a raw-vs-preprocessed figure from a ``(C, Z, Y, X)`` volume.

    Top row per channel: raw z-slices at 25/50/75 % and raw MIP.
    Bottom row per channel: ``[plow, phigh]``-percentile-normalised MIP.
    """
    C, Z, H, W = volume.shape

    z_indices = [Z // 4, Z // 2, 3 * Z // 4]
    n_z = len(z_indices)

    # --- compute preprocessed MIP ---
    raw_mip = maximum_intensity_projection(volume)            # (C, H, W) float32
    norm_mip = normalize_percentile(raw_mip.copy(), plow=plow, phigh=phigh)  # [0,1]

    # --- figure layout ---
    # Top: C rows × (n_z slices + 1 MIP) columns   → raw
    # Bottom: C rows × 1 column (normalized MIP)    → preprocessed
    # We'll do 2 groups separated by a small gap.
    top_cols = n_z + 1  # z-slices + raw MIP
    bot_cols = 1        # normalized MIP
    total_cols = max(top_cols, bot_cols)

    fig = plt.figure(figsize=(4 * top_cols, 4 * C * 2 + 1.5), dpi=150)
    gs = GridSpec(
        nrows=C * 2, ncols=top_cols,
        hspace=0.25, wspace=0.12,
        top=0.93, bottom=0.02, left=0.04, right=0.96,
    )

    fig.suptitle(
        f"ETS Stack Preview — {source_name}" if source_name else "ETS Stack Preview",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # Section labels
    fig.text(0.5, 0.945, "Raw data", ha="center", fontsize=12, fontstyle="italic", color="0.3")

    for ch in range(C):
        cmap = CHANNEL_CMAPS[ch % len(CHANNEL_CMAPS)]
        color = CHANNEL_COLORS[ch % len(CHANNEL_COLORS)]

        # --- TOP: raw z-slices ---
        for j, zi in enumerate(z_indices):
            ax = fig.add_subplot(gs[ch, j])
            ax.imshow(_rescale_for_display(volume[ch, zi]), cmap=cmap, vmin=0, vmax=1)
            ax.set_axis_off()
            if ch == 0:
                ax.set_title(f"z = {zi}/{Z}", fontsize=9)
            if j == 0:
                ax.text(
                    -0.08, 0.5, f"Ch {ch}", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", color=color,
                    ha="right", va="center", rotation=90,
                )

        # --- TOP: raw MIP ---
        ax = fig.add_subplot(gs[ch, n_z])
        ax.imshow(_rescale_for_display(raw_mip[ch]), cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        if ch == 0:
            ax.set_title("MIP (raw)", fontsize=9)

        # --- BOTTOM: normalized MIP ---
        ax = fig.add_subplot(gs[C + ch, 0])
        ax.imshow(norm_mip[ch], cmap=cmap, vmin=0, vmax=1)
        ax.set_axis_off()
        if ch == 0:
            ax.set_title(f"MIP normalized\n[{plow}, {phigh}] percentile → [0, 1]", fontsize=9)
        if j == 0 or True:
            ax.text(
                -0.08, 0.5, f"Ch {ch}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", color=color,
                ha="right", va="center", rotation=90,
            )

    # Label the bottom section
    fig.text(0.5, 0.5 - 0.01, "After preprocessing", ha="center", fontsize=12, fontstyle="italic", color="0.3")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Path to .vsi or .ets file (auto-detected if omitted)")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (default: ets_preview.png in cwd)")
    parser.add_argument("--plow", type=float, default=1.0, help="Lower percentile (default: 1.0)")
    parser.add_argument("--phigh", type=float, default=99.8, help="Upper percentile (default: 99.8)")
    args = parser.parse_args()

    # Resolve input
    if args.input is None:
        _project_root = Path(__file__).resolve().parent.parent.parent
        data_dir = _project_root / "data" / "Microscopy"
        args.input = auto_find_image(data_dir)
        print(f"Auto-detected: {args.input}")

    if not args.input.exists():
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    # Resolve output
    if args.output is None:
        args.output = Path("ets_preview.png")

    print(f"Loading {args.input.name} ...")
    volume = load_image(Path(args.input))
    C, Z, H, W = volume.shape
    print(f"  Shape: C={C}, Z={Z}, H={H}, W={W}, dtype={volume.dtype}")

    print("Building preview figure ...")
    fig = make_preview(volume, plow=args.plow, phigh=args.phigh, source_name=args.input.name)

    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {args.output.resolve()}")


if __name__ == "__main__":
    main()
