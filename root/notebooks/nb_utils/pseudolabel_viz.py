"""Diagnostic plots for the blob pseudo-label pipeline.

Companion to `nb_utils.pseudolabels`. Functions return matplotlib axes
or figures so the notebook can compose them into larger panels.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


CHANNEL_NAMES = ("pre-synaptic", "post-synaptic", "structural")
CHANNEL_CMAPS = ("Greens", "Reds", "Blues")


def show_3channel_grid(
    patch: np.ndarray,
    titles: Sequence[str] = CHANNEL_NAMES,
    figsize=(14, 4),
    vmax=0.5,
):
    """patch: (C, H, W). Displays composite + each channel side by side.

    `vmax=0.5` is a sensible default for percentile-normalised fluorescence
    (most signal lives in the lower half of [0, 1]).
    """
    C, H, W = patch.shape
    fig, axes = plt.subplots(1, C + 1, figsize=figsize)
    composite = patch.max(axis=0)
    axes[0].imshow(composite, cmap="gray", vmin=0, vmax=vmax)
    axes[0].set_title("max-projection (composite)")
    for i in range(C):
        cmap = CHANNEL_CMAPS[i] if i < len(CHANNEL_CMAPS) else "gray"
        axes[i + 1].imshow(patch[i], cmap=cmap, vmin=0, vmax=vmax)
        name = titles[i] if i < len(titles) else f"ch{i}"
        axes[i + 1].set_title(f"ch{i} - {name}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    return fig, axes


def show_blob_overlay(image, blobs, ax, color="lime", linewidth=0.8, vmax=0.5):
    """Draw circles for each blob (radius = sqrt(2)*sigma) over `image`."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
    for row, col, sigma in blobs:
        radius = float(np.sqrt(2.0) * sigma)
        ax.add_patch(Circle((col, row), radius, fill=False,
                            edgecolor=color, linewidth=linewidth))
    ax.axis("off")
    return ax


def show_scored_blobs(
    image,
    scored,
    ax,
    vmax=0.5,
    color_kept="lime",
    color_rejected="red",
    show_rejected=True,
):
    """`scored` is a list of dicts (from score_blobs_zscore).
    Lime = z-score above threshold (kept); red = below (rejected)."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
    for s in scored:
        radius = float(np.sqrt(2.0) * s["sigma"])
        if s["kept"]:
            color = color_kept
        else:
            if not show_rejected:
                continue
            color = color_rejected
        ax.add_patch(Circle((s["col"], s["row"]), radius, fill=False,
                            edgecolor=color, linewidth=0.8))
    ax.axis("off")
    return ax


def show_mask_overlay(image, mask, ax, color=(1, 0.2, 0.2), alpha=0.4, vmax=0.5):
    """Semi-transparent coloured fill for `mask` over greyscale `image`."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
    rgba = np.zeros((*mask.shape, 4))
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = (mask > 0).astype(float) * alpha
    ax.imshow(rgba)
    ax.axis("off")
    return ax


def show_pipeline_stages(
    patch,
    intermediates,
    label_mask,
    figsize=(18, 12),
    vmax=0.5,
):
    """3x3 diagnostic figure showing every intermediate stage on one patch."""
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    composite = patch.max(axis=0)
    axes[0, 0].imshow(composite, cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("composite (max over channels)")

    axes[0, 1].imshow(intermediates["meijering_response"], cmap="hot")
    axes[0, 1].set_title("Meijering response (structural)")

    axes[0, 2].imshow(intermediates["near_structural"], cmap="gray")
    axes[0, 2].set_title(
        f"structural mask (dilated) | dendrite={intermediates['dendrite_mask'].mean():.2%}"
        f"  soma={intermediates['soma_mask'].mean():.2%}"
    )

    show_blob_overlay(
        patch[0], intermediates["pre_blobs_kept"],
        axes[1, 0], color="lime", vmax=vmax,
    )
    axes[1, 0].set_title(f"pre LoG kept ({len(intermediates['pre_blobs_kept'])})")

    show_blob_overlay(
        patch[1], intermediates["post_blobs_kept"],
        axes[1, 1], color="lime", vmax=vmax,
    )
    axes[1, 1].set_title(f"post LoG kept ({len(intermediates['post_blobs_kept'])})")

    axes[1, 2].imshow(intermediates["coloc_mask"], cmap="gray")
    axes[1, 2].set_title(f"co-localised (px={int(intermediates['coloc_mask'].sum())})")

    axes[2, 0].imshow(intermediates["shaped_mask"], cmap="gray")
    axes[2, 0].set_title(f"after shape filter (px={int(intermediates['shaped_mask'].sum())})")

    show_mask_overlay(
        composite, intermediates["near_structural"],
        axes[2, 1], color=(0.2, 0.5, 1.0), alpha=0.4, vmax=vmax,
    )
    axes[2, 1].set_title("structural zone over composite")

    show_mask_overlay(
        composite, label_mask,
        axes[2, 2], color=(1.0, 0.2, 0.2), alpha=0.5, vmax=vmax,
    )
    axes[2, 2].set_title(f"FINAL pseudo-label (px={int(label_mask.sum())})")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    return fig, axes


def plot_zscore_histogram(scored_pre, scored_post, threshold, ax=None, bins=50):
    """Histograms of per-blob z-scores so you can pick a threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    pre_z = np.array([s["z"] for s in scored_pre if not np.isnan(s["z"])])
    post_z = np.array([s["z"] for s in scored_post if not np.isnan(s["z"])])
    zmax = max(
        float(pre_z.max()) if pre_z.size else 1.0,
        float(post_z.max()) if post_z.size else 1.0,
    )
    edges = np.linspace(0, max(zmax, threshold * 1.2), bins)
    ax.hist(pre_z, bins=edges, alpha=0.6, label=f"pre  (n={pre_z.size})")
    ax.hist(post_z, bins=edges, alpha=0.6, label=f"post (n={post_z.size})")
    ax.axvline(threshold, color="red", ls="--", label=f"threshold = {threshold}")
    ax.set_xlabel("per-blob z-score")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title("SynQuant-lite z-score distribution")
    ax.legend(); ax.grid(True, alpha=0.3)
    return ax
