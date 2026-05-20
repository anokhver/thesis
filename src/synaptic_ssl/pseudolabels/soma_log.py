"""LoG-based soma detector (alternative to FDT).

Scale-normalised Laplacian-of-Gaussian detection on the structural
channel, followed by optional intensity filtering and shape
regularisation.  Reference / comparison pipeline — not wired into the
live pseudolabel scripts (those use ``soma_fdt``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import disk as draw_disk, ellipse as draw_ellipse
from skimage.feature import blob_log
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import (
    closing as morph_closing,
    convex_hull_image,
    dilation,
    disk,
)


# -------------------------------------------------------------------
# Config helpers
# -------------------------------------------------------------------

def derive_log_sigmas(cfg: dict) -> dict:
    """Pixel-unit LoG sigma range from pixel_size + biology diameters."""
    px_um = cfg["pixel_size_nm"] / 1000.0
    min_radius_px = cfg["min_soma_diameter_um"] / 2.0 / px_um
    max_radius_px = cfg["max_soma_diameter_um"] / 2.0 / px_um
    return dict(
        log_min_sigma=min_radius_px / np.sqrt(2.0),
        log_max_sigma=max_radius_px / np.sqrt(2.0),
        min_radius_px=min_radius_px,
        max_radius_px=max_radius_px,
    )


def resolve_log_cfg(cfg: dict) -> dict:
    """Merge derived sigma values into a copy of *cfg*."""
    out = dict(cfg)
    out.update(derive_log_sigmas(cfg))
    return out


# -------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------

def _normalize_for_log(image: np.ndarray) -> np.ndarray:
    """p1/p99 normalisation — makes ``log_threshold`` image-independent."""
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1, 99])
    scale = max(hi - lo, np.finfo(np.float32).eps)
    return np.clip((image - lo) / scale, 0.0, 1.0)


def detect_log_blobs(structural: np.ndarray, cfg: dict) -> np.ndarray:
    """Scale-normalised LoG blob detection.  Returns ``(N, 3)`` ``(row, col, sigma)``."""
    cfg = resolve_log_cfg(cfg)
    img = structural.astype(np.float32)
    if cfg["log_pre_smooth"] > 0:
        img = gaussian_filter(img, sigma=float(cfg["log_pre_smooth"]))
    img_n = _normalize_for_log(img)
    blobs = blob_log(
        img_n,
        min_sigma=float(cfg["log_min_sigma"]),
        max_sigma=float(cfg["log_max_sigma"]),
        num_sigma=int(cfg["log_num_sigma"]),
        threshold=float(cfg["log_threshold"]),
        overlap=float(cfg["log_overlap"]),
        exclude_border=int(cfg["log_exclude_border"]),
    )
    return blobs if blobs.size else np.zeros((0, 3), dtype=np.float64)


def filter_log_blobs_by_intensity(
    blobs: np.ndarray,
    image: np.ndarray,
    pct: Optional[float],
) -> np.ndarray:
    """Keep blobs whose mean disk intensity >= *pct* percentile of *image*."""
    if pct is None or not len(blobs):
        return blobs
    thr = float(np.percentile(image, float(pct)))
    H, W = image.shape
    kept = []
    for row, col, sigma in blobs:
        r_px = max(1, int(round(np.sqrt(2.0) * sigma)))
        rr, cc = draw_disk((int(row), int(col)), r_px, shape=(H, W))
        if image[rr, cc].mean() >= thr:
            kept.append((row, col, sigma))
    return np.asarray(kept, dtype=np.float64).reshape(-1, 3)


def render_log_blob_mask(
    blobs: np.ndarray,
    shape: tuple[int, int],
    cfg: dict,
) -> np.ndarray:
    """Convert ``(N, 3)`` ``(row, col, sigma)`` detections into a binary mask."""
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if not len(blobs):
        return mask

    mode = cfg.get("shape_mode", "circle")

    if mode in ("circle", "raw", "closing", "convex"):
        for row, col, sigma in blobs:
            r_px = max(1, int(round(np.sqrt(2.0) * sigma)))
            rr, cc = draw_disk((int(row), int(col)), r_px, shape=(H, W))
            mask[rr, cc] = True
        if mode == "closing":
            mask = morph_closing(mask, disk(int(cfg.get("closing_radius", 8))))
        elif mode == "convex":
            lbl = cc_label(mask, connectivity=2)
            out = np.zeros_like(mask)
            for prop in regionprops(lbl):
                y0, x0, y1, x1 = prop.bbox
                out[y0:y1, x0:x1] |= convex_hull_image(
                    lbl[y0:y1, x0:x1] == prop.label
                )
            mask = out
    elif mode == "ellipse":
        for row, col, sigma in blobs:
            r_px = max(1.0, float(np.sqrt(2.0) * sigma))
            rr, cc = draw_ellipse(int(row), int(col), r_px, r_px, shape=(H, W))
            mask[rr, cc] = True
    else:
        raise ValueError(
            f"unknown shape_mode: {mode!r}; expected "
            "'circle' | 'ellipse' | 'closing' | 'convex' | 'raw'"
        )

    if cfg.get("dilate_r", 0) > 0 and mask.any():
        mask = dilation(mask, disk(int(cfg["dilate_r"])))
    return mask


# -------------------------------------------------------------------
# End-to-end pipeline
# -------------------------------------------------------------------

def run_log_on_image(image_path, cfg: dict) -> dict:
    """Load a full MIP ``.npy`` and run the LoG pipeline.  Returns result dict."""
    full = np.load(image_path)
    structural = full[cfg["structural_channel"]].astype(np.float32)
    raw_blobs = detect_log_blobs(structural, cfg)
    filtered_blobs = filter_log_blobs_by_intensity(
        raw_blobs, structural, cfg.get("intensity_filter_pct"),
    )
    mask = render_log_blob_mask(filtered_blobs, structural.shape, cfg)
    return dict(
        structural=structural,
        raw_blobs=raw_blobs,
        filtered_blobs=filtered_blobs,
        mask=mask,
        n_raw=len(raw_blobs),
        n_kept=len(filtered_blobs),
        n_blobs_final=int(cc_label(mask).max()),
    )


# -------------------------------------------------------------------
# Visualisation
# -------------------------------------------------------------------

def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    scale = max(hi - lo, np.finfo(np.float32).eps)
    return np.clip((image - lo) / scale, 0.0, 1.0)


def visualise_log_mask(result: dict, cfg: dict, *, axes=None, title_prefix=""):
    """3-panel: structural / raw-blob-circles / final mask overlay."""
    import matplotlib.pyplot as plt

    structural = result["structural"]
    raw_blobs = result["raw_blobs"]
    mask = result["mask"]

    struct_n = _normalize_for_display(structural)
    color = np.asarray(cfg.get("blob_color", (1.0, 0.2, 0.8)), dtype=np.float32)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = axes[0].figure

    axes[0].imshow(struct_n, cmap="gray")
    axes[0].set_title(f"{title_prefix}structural".strip())

    axes[1].imshow(struct_n, cmap="gray")
    for row, col, sigma in raw_blobs:
        r_px = float(np.sqrt(2.0) * sigma)
        circle = plt.Circle((col, row), r_px, fill=False,
                             edgecolor=color, linewidth=0.8)
        axes[1].add_patch(circle)
    axes[1].set_title(
        f"LoG raw  ({result['n_raw']} blobs"
        + (f", {result['n_kept']} kept"
           if cfg.get("intensity_filter_pct") is not None else "")
        + f", thr={cfg['log_threshold']})"
    )

    base = struct_n * cfg.get("overlay_dim", 0.45)
    overlay = np.dstack([base, base, base])
    if mask.any():
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * color
        overlay[dilation(mask, disk(1)) & ~mask] = color
    overlay = np.clip(overlay, 0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"SOMA mask [{cfg.get('shape_mode', 'circle')}] "
        f"({result['n_blobs_final']} blobs, cov={mask.mean():.2%})"
    )

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return fig, axes
