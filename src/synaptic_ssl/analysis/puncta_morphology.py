"""Per-punctum shape morphology from LoG detections.

``fit_ellipse`` fits an elliptical 2D Gaussian (plus a constant
background) to one LoG-detected punctum in a small circular window
around the detection. This is detector-agnostic geometry (like
``pseudolabels.puncta_common``) and does not depend on the
segmentation model or any particular detection pipeline.

Why a Gaussian fit and not image moments: intensity-weighted moments
in a window pick up the positive half of the pixel noise, which on
synthetic spots inflates the size by 20-60 %, makes round spots look
elongated and biases the orientation towards the window diagonals
(+-45 deg). A least-squares Gaussian fit stays unbiased in size and
orientation at the same noise levels (see
``tests/test_puncta_morphology.py``).

Ellipse convention: ``a`` / ``b`` are the semi-major / semi-minor axis
lengths in pixels (``a >= b``), equal to twice the fitted Gaussian
standard deviations (``a = 2 * sigma_major``), i.e. the same scale as
``skimage.measure.regionprops`` (semi-axis = ``2 * sqrt(eigenvalue)``).
``eccentricity`` = ``sqrt(1 - (b / a) ** 2)``. ``theta_deg`` is the
angle of the major axis in degrees, measured from the column axis
towards the row axis, wrapped to ``(-90, 90]``.

Note that for spots close to the optical resolution, noise alone makes
a round spot fit as slightly elongated (eccentricity ~0.3-0.5 at
10-20 % noise), so small eccentricities are not distinguishable from 0.
"""
from __future__ import annotations

import logging
from math import ceil, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
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


def _gaussian_residuals(
    p: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    nb: np.ndarray,
) -> np.ndarray:
    """Elliptical Gaussian + constant background + one round Gaussian per
    neighbour (centre fixed at its LoG position; height and width free)."""
    amp, y0, x0, s_major, s_minor, theta, bg = p[:7]
    ct, st = np.cos(theta), np.sin(theta)
    u = (x - x0) * ct + (y - y0) * st
    w = -(x - x0) * st + (y - y0) * ct
    model = amp * np.exp(-0.5 * (u / s_major) ** 2 - 0.5 * (w / s_minor) ** 2) + bg
    for (ny, nx), n_amp, n_sig in zip(nb, p[7::2], p[8::2]):
        model = model + n_amp * np.exp(-0.5 * ((y - ny) ** 2 + (x - nx) ** 2) / n_sig ** 2)
    return model - v


def fit_ellipse(
    image: np.ndarray,
    row: float,
    col: float,
    sigma: float,
    *,
    k: float = 3.0,
    neighbours: np.ndarray | None = None,
) -> dict | None:
    """Fit an elliptical 2D Gaussian to one LoG punctum.

    ``image`` must be 2D and background-subtracted (e.g. the white-tophat
    channel used by ``detect_puncta_channel``). The fit uses the pixels
    of a circular window of radius ``h = max(2, ceil(k * sigma))`` around
    ``(row, col)``. ``neighbours`` is an optional ``(N, 3)`` array of
    other LoG detections ``(row, col, sigma)`` (the punctum itself may be
    included; it is skipped). Neighbours whose light reaches the window
    are fitted jointly as round Gaussians, so a bright neighbour does not
    stretch or shift the ellipse.

    Returns ``None`` if the window would cross the image border, or if
    the fit fails (no positive peak, a width at its bound, or a centre
    that drifted more than ``h / 2`` px).

    Returns a dict with keys ``row``, ``col`` (sub-pixel fitted centre,
    in absolute image coordinates), ``a``, ``b`` (semi-major / semi-minor
    axis lengths in px, ``a >= b``), ``eccentricity``, ``theta_deg``
    (major-axis angle, degrees, ``(-90, 90]``), ``intensity`` (fitted peak
    height above the local background), and ``sigma`` (the LoG scale,
    echoed back for traceability).
    """
    h = max(2, ceil(k * sigma))
    rc, cc = int(round(row)), int(round(col))
    r0, c0 = rc - h, cc - h
    r1, c1 = rc + h + 1, cc + h + 1
    height, width = image.shape
    if r0 < 0 or c0 < 0 or r1 > height or c1 > width:
        return None

    window = image[r0:r1, c0:c1].astype(np.float64)
    # Offsets from the LoG centre (not the rounded pixel).
    yy, xx = np.mgrid[r0:r1, c0:c1]
    yy = yy - float(row)
    xx = xx - float(col)
    use = (yy ** 2 + xx ** 2) <= h * h
    y, x, v = yy[use], xx[use], window[use]

    nb = np.zeros((0, 3))
    if neighbours is not None and len(neighbours):
        nb = np.asarray(neighbours, dtype=np.float64)[:, :3].copy()
        nb[:, :2] -= (row, col)
        dist = np.hypot(nb[:, 0], nb[:, 1])
        # Skip self; keep neighbours whose 3-sigma disk overlaps the window.
        nb = nb[(dist > 1e-6) & (dist < h + 3.0 * nb[:, 2])]

    bg0 = float(np.percentile(v, 25))
    amp0 = float(v[(y ** 2 + x ** 2) <= max(1.0, sigma) ** 2].max()) - bg0
    if amp0 <= 1e-9:
        return None

    s_lo, s_hi = 0.3, float(2 * h)
    p0 = [amp0, 0.0, 0.0, float(sigma), 0.9 * float(sigma), 0.0, bg0]
    lower = [0.0, -h, -h, s_lo, s_lo, -np.inf, -np.inf]
    upper = [np.inf, h, h, s_hi, s_hi, np.inf, np.inf]
    for _, _, n_sig in nb:
        p0 += [amp0, float(n_sig)]
        lower += [0.0, 0.5 * float(n_sig)]
        upper += [np.inf, 2.0 * float(n_sig)]
    try:
        fit = least_squares(
            _gaussian_residuals, p0, bounds=(lower, upper), args=(y, x, v, nb[:, :2])
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not fit.success:
        return None
    amp, dy, dx, s1, s2, theta, _ = (float(t) for t in fit.x[:7])
    if amp <= 1e-9 or dy * dy + dx * dx > (h / 2) ** 2:
        return None
    if min(s1, s2) <= s_lo * 1.01 or max(s1, s2) >= s_hi * 0.99:
        return None
    if s2 > s1:
        s1, s2 = s2, s1
        theta += np.pi / 2

    a = 2.0 * s1
    b = 2.0 * s2
    theta_deg = float((np.degrees(theta) + 90.0) % 180.0 - 90.0)
    if theta_deg == -90.0:
        theta_deg = 90.0
    return dict(
        row=float(row) + dy,
        col=float(col) + dx,
        a=a,
        b=b,
        eccentricity=float(np.sqrt(max(0.0, 1.0 - (b / a) ** 2))),
        theta_deg=theta_deg,
        intensity=amp,
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
            ellipse = fit_ellipse(tophat_img, row, col, sigma, k=k, neighbours=kept)
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
                "%s: dropped %d/%d %s puncta (at image border or ellipse fit failed)",
                file_name, n_dropped, on_near.shape[0], role,
            )

    logger.info(
        "%s: n_pre=%d n_post=%d dendrite_area_um2=%.3f",
        file_name, per_role_counts.get("pre", 0), per_role_counts.get("post", 0),
        dendrite_area_um2,
    )
    return pd.DataFrame(rows, columns=list(PUNCTA_CSV_COLUMNS))

