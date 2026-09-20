"""Detector-agnostic puncta helpers shared by `puncta_log` and `puncta_spotiflow`.

Both detectors yield `(N, 3)` blob arrays of `(row, col, sigma)` --
Spotiflow injects a synthetic sigma derived from a desired pixel radius --
so the rendering and structural-gate helpers can stay shared.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.measure import label as cc_label, regionprops


# 252 nm mean lateral resolution at 107 nm/px. This is deliberately
# independent of detected component size to avoid size-dependent chance pairs.
DEFAULT_OVERLAP_MAX_DISTANCE_PX = 252.0 / 107.0


def puncta_to_mask(blobs: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Render blobs as a union of disks (radius = `sqrt(2) * sigma`)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col, sigma in blobs:
        radius = max(1, int(np.round(np.sqrt(2.0) * sigma)))
        rr, cc = draw_disk((int(row), int(col)), radius, shape=shape)
        mask[rr, cc] = 1
    return mask


def restrict_puncta_to_near(
    blobs: np.ndarray,
    near_mask: np.ndarray,
) -> np.ndarray:
    """Keep `(row, col, sigma)` blobs whose rounded centre lies on `near_mask`."""
    if blobs.shape[0] == 0:
        return blobs
    H, W = near_mask.shape
    rr = np.clip(np.round(blobs[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(blobs[:, 1]).astype(int), 0, W - 1)
    return blobs[near_mask[rr, cc].astype(bool)]


def mask_overlap(
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    *,
    min_fraction: float = 0.8,
    max_distance_px: float = DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    rule: str = "fixed_distance",
) -> tuple[np.ndarray, int, int, int]:
    """Return PRE/POST overlaps and their counts.

    By default, pair component centroids with a fixed maximum separation and
    a globally distance-sorted one-to-one assignment. This prevents a large
    component from receiving a larger chance-pairing radius and prevents one
    component from being counted against several components in the other
    channel. ``rule="center_distance"`` and
    ``rule="smaller_component_fraction"`` preserve previous behaviour.
    The returned mask contains the union of qualifying components because the
    distance rule can qualify circles whose rasterized masks do not intersect.
    """
    if pre_mask.shape != post_mask.shape:
        raise ValueError(
            f"PRE/POST mask shape mismatch: {pre_mask.shape} vs {post_mask.shape}"
        )
    if not 0.0 <= min_fraction <= 1.0:
        raise ValueError(f"min_fraction must be in [0, 1], got {min_fraction}")
    if max_distance_px <= 0:
        raise ValueError(
            f"max_distance_px must be positive, got {max_distance_px}"
        )
    if rule not in {"fixed_distance", "center_distance", "smaller_component_fraction"}:
        raise ValueError(
            "rule must be 'fixed_distance', 'center_distance', or "
            "'smaller_component_fraction'"
        )

    pre_labels = cc_label(np.asarray(pre_mask, dtype=bool), connectivity=2)
    post_labels = cc_label(np.asarray(post_mask, dtype=bool), connectivity=2)
    n_pre = int(pre_labels.max())
    n_post = int(post_labels.max())
    overlap = np.zeros(pre_labels.shape, dtype=bool)
    n_pairs = 0
    if n_pre == 0 or n_post == 0:
        return overlap, n_pre, n_post, n_pairs

    pre_props = regionprops(pre_labels)
    post_props = regionprops(post_labels)
    if rule in {"fixed_distance", "center_distance"}:
        pre_centres = np.array([prop.centroid for prop in pre_props])
        post_centres = np.array([prop.centroid for prop in post_props])
        distances = np.linalg.norm(
            pre_centres[:, None, :] - post_centres[None, :, :], axis=2
        )
        if rule == "fixed_distance":
            candidate_pairs = np.argwhere(distances <= max_distance_px)
        else:
            pre_radii = np.sqrt(np.array([prop.area for prop in pre_props]) / np.pi)
            post_radii = np.sqrt(np.array([prop.area for prop in post_props]) / np.pi)
            candidate_pairs = np.argwhere(
                distances < pre_radii[:, None] + post_radii[None, :]
            )
        order = np.argsort(distances[candidate_pairs[:, 0], candidate_pairs[:, 1]])
        used_pre: set[int] = set()
        used_post: set[int] = set()
        for pair_index in order:
            pre_index, post_index = candidate_pairs[pair_index]
            if pre_index in used_pre or post_index in used_post:
                continue
            overlap |= (pre_labels == pre_index + 1) | (post_labels == post_index + 1)
            used_pre.add(int(pre_index))
            used_post.add(int(post_index))
            n_pairs += 1
    else:
        pre_areas = np.bincount(pre_labels.ravel(), minlength=n_pre + 1)
        post_areas = np.bincount(post_labels.ravel(), minlength=n_post + 1)
        pairs = np.stack((pre_labels.ravel(), post_labels.ravel()), axis=1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        if pairs.size == 0:
            return overlap, n_pre, n_post, n_pairs
        pair_ids, intersections = np.unique(pairs, axis=0, return_counts=True)
        for (pre_id, post_id), intersection in zip(pair_ids, intersections):
            fraction = intersection / min(pre_areas[pre_id], post_areas[post_id])
            if fraction >= min_fraction:
                overlap |= (pre_labels == pre_id) | (post_labels == post_id)
                n_pairs += 1
    return overlap, n_pre, n_post, n_pairs
