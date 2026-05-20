"""Detector-agnostic puncta helpers shared by `puncta_log` and `puncta_spotiflow`.

Both detectors yield `(N, 3)` blob arrays of `(row, col, sigma)` --
Spotiflow injects a synthetic sigma derived from a desired pixel radius --
so the rendering and structural-gate helpers can stay shared.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.draw import disk as draw_disk


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
