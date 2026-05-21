"""Synapse mask via PRE-POST co-localisation (post-processing only).

Implements plan ``§10``: extract local peaks per channel from the trained
model's heatmap output, match PRE to POST within a radius, and render
co-localised synapse disks at the matched midpoints.

This module is NOT used at training time. It runs on the
``(2, H, W)`` probability map returned by
``sliding_window_predict_multichannel``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynapseColocCfg:
    """Hyperparameters for :func:`build_synapse_mask`."""

    prob_thresh: float = 0.5
    min_distance: int = 2          # local-peak suppression radius (px)
    match_radius_px: int = 5       # PRE<->POST nearest-neighbour cap (px)
    synapse_radius_px: int = 3     # rendered output disk radius (px)


def _prob_to_points(prob: np.ndarray, prob_thresh: float, min_distance: int) -> np.ndarray:
    """Local-peak extraction. Prefer Spotiflow's optimised routine; fall
    back to ``skimage.feature.peak_local_max`` when Spotiflow is unavailable.
    """
    try:
        from spotiflow.utils.peaks import prob_to_points  # type: ignore[import-not-found]

        pts = prob_to_points(
            prob, prob_thresh=prob_thresh, min_distance=min_distance, mode="fast",
        )
        return np.asarray(pts, dtype=np.int32)
    except Exception:
        from skimage.feature import peak_local_max

        pts = peak_local_max(
            prob, min_distance=max(1, int(min_distance)),
            threshold_abs=float(prob_thresh),
        )
        return np.asarray(pts, dtype=np.int32)


def _render_disks(centers: np.ndarray, shape: tuple[int, int], r: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    if centers.size == 0:
        return canvas
    H, W = shape
    rr, cc = np.indices(shape)
    r2 = int(r) ** 2
    for cy, cx in centers:
        cy_i, cx_i = int(round(float(cy))), int(round(float(cx)))
        # bounding box for speed
        y0 = max(0, cy_i - r); y1 = min(H, cy_i + r + 1)
        x0 = max(0, cx_i - r); x1 = min(W, cx_i + r + 1)
        sub_r = rr[y0:y1, x0:x1]
        sub_c = cc[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] |= (
            ((sub_r - cy_i) ** 2 + (sub_c - cx_i) ** 2 <= r2).astype(np.uint8)
        )
    return canvas


def build_synapse_mask(
    prob_2ch: np.ndarray,
    cfg: SynapseColocCfg | None = None,
    *,
    return_points: bool = False,
):
    """Render a binary synapse mask from a 2-channel probability map.

    Parameters
    ----------
    prob_2ch
        ``(2, H, W)`` float32 in ``[0, 1]``. ``prob_2ch[0]`` is PRE,
        ``prob_2ch[1]`` is POST.
    cfg
        :class:`SynapseColocCfg`. Defaults: ``prob_thresh=0.5``,
        ``min_distance=2``, ``match_radius_px=5``, ``synapse_radius_px=3``.
    return_points
        If True, also return the matched ``(N, 2)`` midpoints and the
        per-channel point lists for debugging.

    Returns
    -------
    np.ndarray
        ``(H, W)`` uint8 mask. ``1`` where a PRE peak is within
        ``match_radius_px`` of a POST peak (a putative synapse). The
        disk is drawn at the midpoint between the matched peaks.
    """
    if cfg is None:
        cfg = SynapseColocCfg()

    if prob_2ch.ndim != 3 or prob_2ch.shape[0] != 2:
        raise ValueError(
            f"prob_2ch must have shape (2, H, W); got {prob_2ch.shape}"
        )
    H, W = prob_2ch.shape[1:]

    pre_pts = _prob_to_points(prob_2ch[0], cfg.prob_thresh, cfg.min_distance)
    post_pts = _prob_to_points(prob_2ch[1], cfg.prob_thresh, cfg.min_distance)

    if pre_pts.size == 0 or post_pts.size == 0:
        empty = np.zeros((H, W), dtype=np.uint8)
        if return_points:
            return empty, np.empty((0, 2), dtype=np.float32), pre_pts, post_pts
        return empty

    # Nearest POST for each PRE within match_radius_px. cKDTree returns
    # `inf` for distances beyond `distance_upper_bound`; treat those as misses.
    from scipy.spatial import cKDTree  # local import to keep module import-light

    tree = cKDTree(post_pts.astype(np.float32))
    dists, idx = tree.query(
        pre_pts.astype(np.float32),
        distance_upper_bound=float(cfg.match_radius_px),
    )
    matched = np.isfinite(dists) & (dists < float(cfg.match_radius_px))
    pre_matched = pre_pts[matched]
    post_matched = post_pts[idx[matched]]
    midpoints = (pre_matched.astype(np.float32) + post_matched.astype(np.float32)) / 2.0

    mask = _render_disks(midpoints, (H, W), cfg.synapse_radius_px)
    if return_points:
        return mask, midpoints, pre_pts, post_pts
    return mask


__all__ = ["SynapseColocCfg", "build_synapse_mask"]
