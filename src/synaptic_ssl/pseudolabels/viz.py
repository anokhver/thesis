"""Matplotlib visualisation for the live puncta pipeline.

Mirrors the per-channel + structural diagnostic plots used in
``notebooks/pseudolabels/puncta_detection.ipynb`` so callers don't
reimplement them.

Soma and dendrite diagnostic figures live next to their detectors
(``soma_fdt.visualise_soma_mask``, ``dendrite_frangi.visualise_dendrite_mask``);
this module covers the puncta stage and the soma+dend structural gate
that precedes it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .puncta import PunctaCfg


def _on_near_flags(scored: List[dict], near_mask: np.ndarray | None) -> List[bool]:
    if near_mask is None:
        return [True] * len(scored)
    H, W = near_mask.shape
    out = []
    for s in scored:
        rr = int(np.clip(round(s["row"]), 0, H - 1))
        cc = int(np.clip(round(s["col"]), 0, W - 1))
        out.append(bool(near_mask[rr, cc]))
    return out


def visualise_structural_overview(
    structural_image: np.ndarray,
    soma_mask: np.ndarray,
    dendrite_mask: np.ndarray,
    *,
    near_mask: np.ndarray | None = None,
    ax=None,
    vmax: float = 0.3,
    image_alpha: float = 1.0,
    title: str = "",
):
    """Single-panel summary of the structural stage that gates puncta.

    Structural channel as grayscale with semi-transparent overlays:
    soma red, dendrite green, optional near zone blue. The near zone
    is what the puncta restrictor uses as the keep/reject boundary;
    pass it to see the actual gate, omit it to see just soma ∪ dend.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    ax.imshow(structural_image, cmap="gray", vmin=0, vmax=vmax, alpha=image_alpha)
    if near_mask is not None:
        rgba = np.zeros((*near_mask.shape, 4))
        rgba[..., 2] = 1.0
        rgba[..., 3] = near_mask * 0.18
        ax.imshow(rgba)
    if soma_mask.any():
        rgba = np.zeros((*soma_mask.shape, 4))
        rgba[..., 0] = 1.0
        rgba[..., 3] = soma_mask * 0.35
        ax.imshow(rgba)
    if dendrite_mask.any():
        rgba = np.zeros((*dendrite_mask.shape, 4))
        rgba[..., 1] = 1.0
        rgba[..., 3] = dendrite_mask * 0.35
        ax.imshow(rgba)
    if not title:
        parts = [
            f"soma={soma_mask.mean():.2%}",
            f"dend={dendrite_mask.mean():.2%}",
        ]
        if near_mask is not None:
            parts.append(f"near={near_mask.mean():.2%}")
        title = "structural  " + "  ".join(parts)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


def visualise_puncta_channel(
    image: np.ndarray,
    raw: np.ndarray,
    scored: List[dict],
    kept: np.ndarray,
    *,
    near_mask: np.ndarray | None = None,
    color: str = "lime",
    cfg: PunctaCfg | None = None,
    axes=None,
    vmax: float = 0.3,
    title_prefix: str = "",
):
    """3-panel per-channel diagnostic: raw / LoG candidates / kept-vs-rejected.

    Panels:
      0. raw channel (with near_mask in blue if provided).
      1. raw LoG candidates as yellow circles (radius = sqrt(2)*sigma).
      2. scored blobs coloured ``color`` (kept on near_mask), orange
         (kept off near_mask), or red (rejected). When ``scored`` is
         empty (z-scoring off) every raw blob is drawn as ``color``.

    Pass a ``(3,)`` array of matplotlib axes to embed in a larger grid,
    or omit ``axes`` for a standalone ``(1, 3)`` figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = axes[0].figure
    if len(axes) < 3:
        raise ValueError(f"visualise_puncta_channel needs 3 axes; got {len(axes)}")

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=vmax)
    if near_mask is not None:
        rgba = np.zeros((*near_mask.shape, 4))
        rgba[..., 2] = 1.0
        rgba[..., 3] = near_mask * 0.18
        axes[0].imshow(rgba)
    near_hint = "  (near=blue)" if near_mask is not None else ""
    axes[0].set_title(f"{title_prefix}raw{near_hint}".strip())

    axes[1].imshow(image, cmap="gray", vmin=0, vmax=vmax)
    for r, c, s in raw:
        axes[1].add_patch(Circle(
            (c, r), float(np.sqrt(2) * s),
            fill=False, edgecolor="yellow", linewidth=0.6,
        ))
    axes[1].set_title(f"LoG candidates (n={len(raw)})")

    axes[2].imshow(image, cmap="gray", vmin=0, vmax=vmax)
    n_kept_on = n_kept_off = n_rej = 0
    if scored:
        on_flags = _on_near_flags(scored, near_mask)
        for s, on in zip(scored, on_flags):
            if s["kept"]:
                if on:
                    n_kept_on += 1; col = color
                else:
                    n_kept_off += 1; col = "orange"
            else:
                n_rej += 1; col = "red"
            axes[2].add_patch(Circle(
                (s["col"], s["row"]), float(np.sqrt(2) * s["sigma"]),
                fill=False, edgecolor=col, linewidth=0.6,
            ))
    else:
        for r, c, s in kept:
            axes[2].add_patch(Circle(
                (c, r), float(np.sqrt(2) * s),
                fill=False, edgecolor=color, linewidth=0.6,
            ))
        n_kept_on = len(kept)
    thr_txt = f"z>={cfg.zscore_threshold}" if cfg is not None else "kept"
    if near_mask is not None and scored:
        title2 = f"{thr_txt}  on={n_kept_on} off={n_kept_off} rej={n_rej}"
    elif scored:
        title2 = f"{thr_txt}  kept={n_kept_on + n_kept_off} rej={n_rej}"
    else:
        title2 = f"kept (n={n_kept_on})"
    axes[2].set_title(title2)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return fig, axes


def visualise_puncta_pair(
    pre_image: np.ndarray,
    post_image: np.ndarray,
    raw_pre: np.ndarray,
    scored_pre: List[dict],
    kept_pre: np.ndarray,
    raw_post: np.ndarray,
    scored_post: List[dict],
    kept_post: np.ndarray,
    *,
    near_mask: np.ndarray | None = None,
    cfg_pre: PunctaCfg | None = None,
    cfg_post: PunctaCfg | None = None,
    vmax: float = 0.3,
    title: str = "",
):
    """2x3 grid: top row pre (lime), bottom row post (magenta).

    Convenience for tuning notebooks that compare both puncta channels
    side by side; each row is ``visualise_puncta_channel``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    visualise_puncta_channel(
        pre_image, raw_pre, scored_pre, kept_pre,
        near_mask=near_mask, color="lime", cfg=cfg_pre,
        axes=axes[0], vmax=vmax, title_prefix="pre ",
    )
    visualise_puncta_channel(
        post_image, raw_post, scored_post, kept_post,
        near_mask=near_mask, color="magenta", cfg=cfg_post,
        axes=axes[1], vmax=vmax, title_prefix="post ",
    )
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def visualise_puncta_full(
    image: np.ndarray,
    blobs: np.ndarray,
    *,
    color: str = "lime",
    ax=None,
    vmax: float = 0.3,
    image_alpha: float = 1.0,
    figsize: Tuple[float, float] = (14, 14),
    linewidth: float = 0.5,
    title: str = "",
):
    """Full-image overlay of one channel + blob circles. Stitched batches go here."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.imshow(image, cmap="gray", vmin=0, vmax=vmax, alpha=image_alpha)
    for r, c, s in blobs:
        ax.add_patch(Circle(
            (c, r), float(np.sqrt(2) * s),
            fill=False, edgecolor=color, linewidth=linewidth,
        ))
    ax.set_title(title or f"n={len(blobs)}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    return fig, ax


__all__ = [
    "visualise_structural_overview",
    "visualise_puncta_channel",
    "visualise_puncta_pair",
    "visualise_puncta_full",
]
