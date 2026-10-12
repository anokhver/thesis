"""Per-punctum shape morphology from LoG detections.

``ellipse_from_moments`` fits an ellipse to one LoG-detected punctum by
computing image moments of the background-subtracted intensity in a
small window around the detection. This is detector-agnostic geometry
(like ``pseudolabels.puncta_common``) and does not depend on the
segmentation model or any particular detection pipeline.

Ellipse convention: ``a`` / ``b`` are the semi-major / semi-minor axis
lengths in pixels (``a >= b``), derived from the eigenvalues of the
intensity-weighted covariance matrix of the window (same convention as
``skimage.measure.regionprops``: axis length = ``4 * sqrt(eigenvalue)``,
so semi-axis = ``2 * sqrt(eigenvalue)``). ``eccentricity`` follows the
same definition: ``sqrt(1 - (b / a) ** 2)``. ``theta_deg`` is the angle
of the major axis in degrees, measured from the column axis.
"""
from __future__ import annotations

import logging
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import dilation, disk, white_tophat

from synaptic_ssl.clustering.groups import classify_image_by_patterns
from synaptic_ssl.pseudolabels.puncta_common import restrict_puncta_to_near
from synaptic_ssl.pseudolabels.puncta_log import PunctaCfg, detect_puncta_channel

logger = logging.getLogger(__name__)

#: Column order written to the per-punctum CSV (step 3).
PUNCTA_CSV_COLUMNS: tuple[str, ...] = (
    "file_name",
    "group_name",
    "role",
    "channel_index",
    "dendrite_channel_index",
    "x",
    "y",
    "radius_um",
    "a_um",
    "b_um",
    "eccentricity",
    "theta_deg",
    "intensity",
    "sigma_log",
    "dendrite_area_um2",
    "px_size_um",
    "near_dilate_px",
)


def ellipse_from_moments(
    image: np.ndarray,
    row: float,
    col: float,
    sigma: float,
    *,
    k: float = 3.0,
) -> dict | None:
    """Fit an ellipse to one LoG punctum via background-subtracted moments.

    ``image`` must be 2D and background-subtracted (e.g. the white-tophat
    channel used by ``detect_puncta_channel``). The window half-size is
    ``h = max(2, ceil(k * sigma))``, cropped as
    ``image[row - h : row + h + 1, col - h : col + h + 1]``.

    Returns ``None`` if the window would cross the image border, or if
    the background-subtracted weights sum to (near) zero or degenerate
    into a zero-area ellipse.

    Returns a dict with keys ``row``, ``col`` (sub-pixel centroid, in
    absolute image coordinates), ``a``, ``b`` (semi-major / semi-minor
    axis lengths in px, ``a >= b``), ``eccentricity``, ``theta_deg``
    (major-axis angle, degrees), ``intensity`` (peak background-subtracted
    value in the window), and ``sigma`` (echoed back for traceability).
    """
    h = max(2, ceil(k * sigma))
    rc, cc = int(round(row)), int(round(col))
    r0, c0 = rc - h, cc - h
    r1, c1 = rc + h + 1, cc + h + 1
    height, width = image.shape
    if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
        return None

    window = image[r0:r1, c0:c1].astype(np.float64)
    ring = np.concatenate(
        [window[0, :], window[-1, :], window[1:-1, 0], window[1:-1, -1]]
    )
    background = float(np.median(ring))
    weights = np.clip(window - background, 0.0, None)
    m00 = float(weights.sum())
    if m00 <= 1e-9:
        return None

    size = window.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    cy = float((weights * yy).sum() / m00)
    cx = float((weights * xx).sum() / m00)
    dy = yy - cy
    dx = xx - cx
    mu20 = float((weights * dy * dy).sum() / m00)
    mu02 = float((weights * dx * dx).sum() / m00)
    mu11 = float((weights * dy * dx).sum() / m00)

    cov = np.array([[mu20, mu11], [mu11, mu02]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    lam_minor, lam_major = float(eigvals[0]), float(eigvals[1])

    a = 2.0 * np.sqrt(lam_major)
    b = 2.0 * np.sqrt(lam_minor)
    if a <= 1e-9:
        return None
    eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2)))
    # Major-axis eigenvector [vy, vx] (eigh ascending order -> index 1).
    vy, vx = eigvecs[:, 1]
    theta_deg = float(np.degrees(np.arctan2(vy, vx)))

    return dict(
        row=float(r0 + cy),
        col=float(c0 + cx),
        a=float(a),
        b=float(b),
        eccentricity=eccentricity,
        theta_deg=theta_deg,
        intensity=float(weights.max()),
        sigma=float(sigma),
    )


def analyse_image(
    npy_path: Path,
    dend_mask: np.ndarray,
    *,
    pre_channel: int,
    post_channel: int,
    cfg_pre: PunctaCfg,
    cfg_post: PunctaCfg,
    near_dilate_px: int,
    px_size_um: float,
    auto_floors: bool = True,
    k: float = 3.0,
    group_rules: dict | None = None,
    dendrite_channel: int = 2,
) -> pd.DataFrame:
    """Per-punctum morphology for one full-MIP ``.npy``, pre and post roles.

    ``dend_mask`` is the (already-loaded or stitched) dendrite mask for
    ``npy_path``, bool ``(H, W)``. Only the dendrite mask gates puncta
    here; the soma mask is not used. ``group_rules`` is the dict
    returned by ``clustering.groups.load_group_rules`` (or ``None`` to
    leave ``group_name`` empty for every row).

    Returns a ``pd.DataFrame`` with one row per punctum and columns
    ``PUNCTA_CSV_COLUMNS``. An image with zero kept puncta in a role
    contributes no rows for that role (its area is only visible via the
    per-image log line / the images-CSV written by the CLI script).
    """
    full = np.load(npy_path)
    file_name = npy_path.name
    if group_rules is not None:
        group_name = classify_image_by_patterns(file_name, **group_rules) or ""
    else:
        group_name = ""

    near = (
        dilation(dend_mask, disk(int(near_dilate_px)))
        if near_dilate_px > 0
        else dend_mask
    )
    dendrite_area_um2 = float(near.sum()) * px_size_um ** 2

    rows: list[dict] = []
    per_role_counts: dict[str, int] = {}
    for role, channel_index, cfg in (
        ("pre", pre_channel, cfg_pre),
        ("post", post_channel, cfg_post),
    ):
        img = full[channel_index].astype(np.float32)
        _, _, kept, cfg_eff = detect_puncta_channel(img, cfg, auto_floors=auto_floors)
        on_near = restrict_puncta_to_near(kept, near)
        per_role_counts[role] = int(on_near.shape[0])

        tophat_r = int(cfg_eff.intensity_tophat_radius)
        tophat_img = (
            white_tophat(img, footprint=disk(tophat_r)) if tophat_r > 0 else img
        )

        n_dropped = 0
        for row, col, sigma in on_near:
            ellipse = ellipse_from_moments(tophat_img, row, col, sigma, k=k)
            if ellipse is None:
                n_dropped += 1
                continue
            radius_px = sqrt(ellipse["a"] * ellipse["b"])
            rows.append(
                dict(
                    file_name=file_name,
                    group_name=group_name,
                    role=role,
                    channel_index=channel_index,
                    dendrite_channel_index=dendrite_channel,
                    x=ellipse["col"],
                    y=ellipse["row"],
                    radius_um=radius_px * px_size_um,
                    a_um=ellipse["a"] * px_size_um,
                    b_um=ellipse["b"] * px_size_um,
                    eccentricity=ellipse["eccentricity"],
                    theta_deg=ellipse["theta_deg"],
                    intensity=ellipse["intensity"],
                    sigma_log=ellipse["sigma"],
                    dendrite_area_um2=dendrite_area_um2,
                    px_size_um=px_size_um,
                    near_dilate_px=near_dilate_px,
                )
            )
        if n_dropped:
            logger.info(
                "%s: dropped %d/%d %s puncta (ellipse fit failed near border)",
                file_name, n_dropped, on_near.shape[0], role,
            )

    logger.info(
        "%s: n_pre=%d n_post=%d dendrite_area_um2=%.3f",
        file_name, per_role_counts.get("pre", 0), per_role_counts.get("post", 0),
        dendrite_area_um2,
    )
    return pd.DataFrame(rows, columns=list(PUNCTA_CSV_COLUMNS))

