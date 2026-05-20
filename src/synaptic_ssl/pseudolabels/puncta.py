"""Puncta pseudo-labels for fluorescence-microscopy MIPs.

Per-channel pipeline: white top-hat -> scale-space LoG -> annular-background
z-score gate -> render disks -> optional shape/size filter -> optional
centre-on-(soma U dendrite) gate.

Annular z-score is a parametric Gaussian approximation of SynQuant's
Wilcoxon SNR test (Wang et al., Bioinformatics 2020). Other building
blocks: scale-normalised LoG (Lindeberg, IJCV 1998), white top-hat
(Pathak et al., Front Cell Dev Biol 2025, sec 2.5.2).

Ref: https://github.com/yu-lab-vt/SynQuant
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import List, Optional, Tuple

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.feature import blob_log
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import disk as morph_disk, white_tophat


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PunctaCfg:
    """Per-channel detection knobs.

    Defaults are conservative; calibrated values for the 107 nm/px
    dataset live in ``DEFAULT_PUNCTA_CFG_PRE`` and
    ``DEFAULT_PUNCTA_CFG_POST`` below.
    """

    # LoG blob detection
    log_min_sigma: float = 0.7
    log_max_sigma: float = 1.8
    log_num_sigma: int = 5
    log_threshold: float = 0.005
    log_overlap: float = 0.5
    log_exclude_border: int = 5  # ~ 3 * log_max_sigma

    # Per-blob annular z-score (local-SNR test).
    # Local-SNR test inspired by SynQuant (Wang et al. 2020): SynQuant uses a
    # Wilcoxon rank-sum statistic against tabulated null moments; this is a
    # parametric Gaussian approximation z = (mu_in - mu_bg) / sigma_bg.
    use_zscore: bool = True
    zscore_inner_radius: int = 3
    zscore_outer_radius: int = 8
    zscore_threshold: float = 2.0
    # Annulus statistic. mean+std collapses in crowded fields where
    # neighbouring puncta inflate both mu_bg and sigma_bg. A robust
    # low-percentile + MAD (SExtractor / SynQuant style) ignores bright
    # contaminants and gives a real "background" estimate.
    zscore_bg_percentile: int = 0         # 0 = mean; e.g. 25 = 25th-percentile
    zscore_bg_robust_scale: bool = False  # True = MAD*1.4826 instead of std
    # Absolute-intensity safety floors. In dark regions sigma_bg collapses
    # toward zero, so any pixel noise gives a huge z; require the raw
    # intensity to clear an absolute floor too. ``detect_puncta_channel``
    # can derive these per-image from the tophat background.
    zscore_sigma_bg_floor: float = 0.0    # clamp sigma_bg from below before dividing
    zscore_min_inner: float = 0.0         # require mu_in >= this absolute intensity
    zscore_min_contrast: float = 0.0      # require mu_in - mu_bg >= this absolute delta

    # White-tophat preprocessing radius (px). Flattens slow-varying
    # background so puncta in dense clusters keep contrast against their
    # local annulus. Choose >= 2 * sqrt(2) * log_max_sigma. 0 = off.
    intensity_tophat_radius: int = 0

    # Shape priors on the rendered LoG-disk-union mask.
    # Same shape-prior form as SynQuant (paraP3D.java: minfill, maxWHratio,
    # area bounds), but with our own values. Default area bounds match the
    # LoG-disk geometry at 107 nm/px (a single LoG disk of max-sigma
    # radius ~ 2.5 px has area ~ 20 px^2), not the physical synapse area;
    # widen + bump log_max_sigma to target the latter.
    min_size: int = 3
    max_size: int = 40
    min_fill: float = 0.5
    max_wh_ratio: float = 4.0


# ---------------------------------------------------------------------------
# Calibrated defaults
# ---------------------------------------------------------------------------
#
# Per-channel overrides on top of the PunctaCfg field defaults. Calibrated
# in notebooks/pseudolabels/puncta_detection.ipynb on the 20251219 dataset
# at 107 nm/px (60x, NA 1.4, 488 nm excitation).
#
# Biology → pixel mapping (Harris & Stevens 1989, J Neurosci 9(8):2982–2997):
#   PRE  (Bassoon/Synaptophysin) bouton: 500–1000 nm EM diameter
#   POST (PSD-95/Homer)          PSD:    200–500 nm  EM diameter
# Convolved with PSF (~200 nm ≈ 1.9 px); observed sizes at 107 nm/px:
#   PRE  ≈ 5–10 px → LoG sigma 1.8–3.4  (disk ∅ ≈ 2·√2·σ ≈ 5.1–9.6 px)
#   POST ≈ 2.7–5 px → LoG sigma 1.0–1.8 (disk ∅ ≈ 2.8–5.1 px)
#
# Channel selection lives in scripts/puncta_from_mip.py (--pre_channel /
# --post_channel); the detection functions take a 2D array, not a stack.

DEFAULT_PUNCTA_CFG_PRE: dict = dict(
    # PRE boutons observed ≈ 5–10 px after PSF convolution.
    log_min_sigma=1.8,
    log_max_sigma=3.4,
    log_num_sigma=5,
    log_threshold=0.025,
    log_overlap=0.5,
    # ~ 3 * max_sigma
    log_exclude_border=10,

    # Annulus: inner ≈ ceil(sqrt(2)*max_sigma), outer ≈ 2*inner.
    use_zscore=True,
    zscore_inner_radius=5,
    zscore_outer_radius=10,
    zscore_threshold=4.0,
    zscore_bg_percentile=25,
    zscore_bg_robust_scale=True,

    # ~ 2 * sqrt(2) * max_sigma
    intensity_tophat_radius=10,

    # pi*(sqrt(2)*sigma)^2: min≈20 at sigma=1.8, max≈73 at sigma=3.4
    min_size=20,
    max_size=75,
    min_fill=0.5,
    max_wh_ratio=2.5,
)


DEFAULT_PUNCTA_CFG_POST: dict = dict(
    # POST PSD observed ≈ 2.7–5 px after PSF convolution.
    log_min_sigma=1.0,
    log_max_sigma=1.8,
    log_num_sigma=4,
    log_threshold=0.030,
    log_overlap=0.5,
    # ~ 3 * max_sigma
    log_exclude_border=6,

    # Tighter annulus for smaller spots.
    use_zscore=True,
    zscore_inner_radius=3,
    zscore_outer_radius=6,
    zscore_threshold=6.0,
    zscore_bg_percentile=25,
    zscore_bg_robust_scale=True,

    # ~ 2 * sqrt(2) * max_sigma
    intensity_tophat_radius=5,

    # pi*(sqrt(2)*sigma)^2: min≈6 at sigma=1.0, max≈20 at sigma=1.8
    min_size=6,
    max_size=22,
    min_fill=0.5,
    max_wh_ratio=2.0,
)


# Dilation (px) applied to the (soma U dendrite) mask before the
# centre-on-mask gate. 4 px ~ 0.43 um at 107 nm/px adds a small
# perisomatic/peridendritic margin without admitting background puncta.
DEFAULT_NEAR_DILATE_PX: int = 4


def resolve_cfg(
    overrides: Optional[dict] = None,
    *,
    channel: str = "pre",
) -> "PunctaCfg":
    """Merge ``overrides`` into ``DEFAULT_PUNCTA_CFG_{PRE,POST}`` and
    return a ``PunctaCfg``.

    ``channel`` selects the base default (``'pre'`` or ``'post'``).
    Unknown keys in ``overrides`` raise ``ValueError`` so a typo'd JSON
    does not pass silently.
    """
    if channel == "pre":
        base = DEFAULT_PUNCTA_CFG_PRE
    elif channel == "post":
        base = DEFAULT_PUNCTA_CFG_POST
    else:
        raise ValueError(f"channel must be 'pre' or 'post', got {channel!r}")
    merged = dict(base)
    merged.update(overrides or {})
    valid = {f.name for f in dataclass_fields(PunctaCfg)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"unknown PunctaCfg field(s) in overrides: {sorted(unknown)}"
        )
    return PunctaCfg(**merged)


# ---------------------------------------------------------------------------
# LoG blob detection (per channel)
# ---------------------------------------------------------------------------

def detect_puncta_log(image: np.ndarray, cfg: PunctaCfg) -> np.ndarray:
    """Scale-space LoG (Lindeberg 1998). Returns ``(N, 3)`` ``(row, col, sigma)``."""
    if image.ndim != 2:
        raise ValueError(f"detect_puncta_log expects 2D, got shape {image.shape}")
    blobs = blob_log(
        image,
        min_sigma=cfg.log_min_sigma,
        max_sigma=cfg.log_max_sigma,
        num_sigma=cfg.log_num_sigma,
        threshold=cfg.log_threshold,
        overlap=cfg.log_overlap,
        exclude_border=cfg.log_exclude_border,
    )
    return blobs if blobs.size else np.zeros((0, 3), dtype=np.float64)


def puncta_to_mask(blobs: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Render LoG blobs as a union of disks (radius = ``sqrt(2) * sigma``)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col, sigma in blobs:
        radius = max(1, int(np.round(np.sqrt(2.0) * sigma)))
        rr, cc = draw_disk((int(row), int(col)), radius, shape=shape)
        mask[rr, cc] = 1
    return mask


# ---------------------------------------------------------------------------
# Per-blob annular-background z-score (local-SNR test)
# ---------------------------------------------------------------------------

def _local_window(image: np.ndarray, row: int, col: int, half: int):
    H, W = image.shape
    r0 = max(0, row - half); r1 = min(H, row + half + 1)
    c0 = max(0, col - half); c1 = min(W, col + half + 1)
    return image[r0:r1, c0:c1], (r0, c0)


def _score_one_puncta_zscore(
    image: np.ndarray,
    row: float,
    col: float,
    sigma: float,
    r_in: int,
    r_out: int,
    *,
    sigma_bg_floor: float = 0.0,
    bg_percentile: int = 0,
    bg_robust_scale: bool = False,
) -> dict:
    """Annular-background z = (mu_in - mu_bg) / sigma_bg for one LoG blob.

    Parametric Gaussian approximation of the SynQuant local-SNR test
    (Wang et al. 2020), which itself uses a Wilcoxon rank-sum statistic
    against tabulated null moments.

    ``sigma_bg_floor`` clamps ``sigma_bg`` from below before dividing,
    so near-uniform dark regions can't blow z up by collapsing the
    denominator.

    ``bg_percentile`` (0 = mean) and ``bg_robust_scale`` (False = std)
    switch from mean+std to a robust percentile + MAD estimator.
    Necessary in crowded fields where the annulus contains other puncta
    and mean+std collapse z even for obviously bright spots.
    """
    rri, cci = int(round(row)), int(round(col))
    half = r_out
    win, (r0, c0) = _local_window(image, rri, cci, half)
    H, W = win.shape
    cy, cx = rri - r0, cci - c0
    yy, xx = np.ogrid[:H, :W]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2

    radius = max(1, int(np.round(np.sqrt(2.0) * sigma)))
    inside = d2 <= radius * radius
    annulus = (d2 >= r_in * r_in) & (d2 < r_out * r_out) & ~inside

    if inside.sum() < 1 or annulus.sum() < 8:
        return {
            "row": float(row), "col": float(col), "sigma": float(sigma),
            "radius": int(radius), "n_inside": int(inside.sum()),
            "mu_in": float("nan"), "mu_bg": float("nan"),
            "sigma_bg": float("nan"),
            "z": float("nan"), "kept": False,
        }
    in_vals = win[inside]
    bg_vals = win[annulus]
    mu_in = float(in_vals.mean())
    med_in = float(np.median(in_vals))
    if bg_percentile and bg_percentile > 0:
        mu_bg = float(np.percentile(bg_vals, bg_percentile))
    else:
        mu_bg = float(bg_vals.mean())
    if bg_robust_scale:
        med_bg = float(np.median(bg_vals))
        sigma_bg_raw = float(1.4826 * np.median(np.abs(bg_vals - med_bg)))
    else:
        sigma_bg_raw = float(bg_vals.std(ddof=1))
    sigma_bg = max(sigma_bg_raw, float(sigma_bg_floor)) + 1e-8
    z = float((mu_in - mu_bg) / sigma_bg)
    return {
        "row": float(row), "col": float(col), "sigma": float(sigma),
        "radius": int(radius), "n_inside": int(inside.sum()),
        "mu_in": mu_in, "med_in": med_in, "mu_bg": mu_bg, "sigma_bg": sigma_bg,
        "z": z, "kept": False,
    }


def score_puncta_zscore(
    image: np.ndarray,
    blobs: np.ndarray,
    cfg: PunctaCfg,
) -> List[dict]:
    """Score every blob and tag ``kept`` against ``cfg.zscore_threshold``.

    A blob is kept iff its z-score clears ``cfg.zscore_threshold`` AND
    its absolute intensities clear the safety floors
    (``zscore_min_inner``, ``zscore_min_contrast``). The floors default
    to 0.0 (off) and exist to stop the dark-void z-score blow-up where
    a tiny ``sigma_bg`` would otherwise let any pixel noise pass.
    """
    out = []
    for row, col, sigma in blobs:
        rec = _score_one_puncta_zscore(
            image, row, col, sigma,
            cfg.zscore_inner_radius,
            cfg.zscore_outer_radius,
            sigma_bg_floor=cfg.zscore_sigma_bg_floor,
            bg_percentile=cfg.zscore_bg_percentile,
            bg_robust_scale=cfg.zscore_bg_robust_scale,
        )
        if np.isnan(rec["z"]):
            rec["kept"] = False
        else:
            rec["kept"] = bool(
                rec["z"] >= cfg.zscore_threshold
                and rec["mu_in"] >= cfg.zscore_min_inner
                and rec["med_in"] >= cfg.zscore_min_inner
                and (rec["mu_in"] - rec["mu_bg"]) >= cfg.zscore_min_contrast
            )
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Shape / size filter (post-rendering)
# ---------------------------------------------------------------------------

def filter_by_size_shape(mask: np.ndarray, cfg: PunctaCfg) -> np.ndarray:
    """Keep CCs passing area, bbox aspect ratio, and bbox fill bounds.

    Same shape-prior form as SynQuant (Wang et al. 2020, ``paraP3D.java``:
    ``minfill``, ``maxWHratio``, min/max area), but with our own values.
    """
    lbl = cc_label(mask, connectivity=2)
    out = np.zeros_like(mask)
    for prop in regionprops(lbl):
        area = prop.area
        if area < cfg.min_size or area > cfg.max_size:
            continue
        bb_h = prop.bbox[2] - prop.bbox[0]
        bb_w = prop.bbox[3] - prop.bbox[1]
        if bb_h == 0 or bb_w == 0:
            continue
        wh_ratio = max(bb_h, bb_w) / max(1, min(bb_h, bb_w))
        if wh_ratio > cfg.max_wh_ratio:
            continue
        fill = area / float(bb_h * bb_w)
        if fill < cfg.min_fill:
            continue
        out[lbl == prop.label] = 1
    return out


# ---------------------------------------------------------------------------
# End-to-end per-channel helpers (live pipeline)
# Visualisation lives in ``pseudolabels.viz``.
# ---------------------------------------------------------------------------

def derive_zscore_floors(detected: np.ndarray) -> Tuple[float, float, float]:
    """Per-image safety floors from the (already tophatted) image bg.

    Returns ``(sigma_bg_floor, min_inner, min_contrast)``. ``bg`` = pixels
    at or below the 30th percentile; ``sigma`` = ``1.4826 * MAD(bg)``;
    floors = ``(sigma, median(bg) + 4*sigma, 4*sigma)``. Returns
    ``(0, 0, 0)`` (floors disabled) on degenerate input.
    """
    finite = detected[np.isfinite(detected)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    p30 = np.percentile(finite, 30)
    bg = finite[finite <= p30]
    if bg.size == 0:
        return 0.0, 0.0, 0.0
    bg_med = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    bg_sigma = 1.4826 * bg_mad
    if not (np.isfinite(bg_med) and np.isfinite(bg_sigma)):
        return 0.0, 0.0, 0.0
    return bg_sigma, bg_med + 4.0 * bg_sigma, 4.0 * bg_sigma


def detect_puncta_channel(
    img2d: np.ndarray,
    cfg: PunctaCfg,
    *,
    auto_floors: bool = True,
) -> Tuple[np.ndarray, List[dict], np.ndarray, PunctaCfg]:
    """Tophat -> (optional auto floors) -> LoG -> z-score on one 2D channel.

    Returns ``(raw[N,3], scored, kept[M,3], cfg_eff)``:
      * ``raw`` — every LoG candidate ``[row, col, sigma]``.
      * ``scored`` — full ``score_puncta_zscore`` records (one per raw
        blob), each carrying ``z, mu_in, mu_bg, sigma_bg, kept, ...``.
        Empty list if z-scoring is off or no candidates.
      * ``kept`` — subset of ``raw`` that survived z + floor gating.
      * ``cfg_eff`` — cfg actually used (per-image floors patched in
        when ``auto_floors=True``); pass this to the viz helpers so
        titles reflect the real thresholds.
    """
    r = int(cfg.intensity_tophat_radius)
    det = white_tophat(img2d, footprint=morph_disk(r)) if r > 0 else img2d
    if auto_floors:
        sg, mi, mc = derive_zscore_floors(det)
        cfg_eff = replace(
            cfg,
            zscore_sigma_bg_floor=sg,
            zscore_min_inner=mi,
            zscore_min_contrast=mc,
        )
    else:
        cfg_eff = cfg
    raw = detect_puncta_log(det, cfg_eff)
    if not cfg_eff.use_zscore or raw.shape[0] == 0:
        return raw, [], raw, cfg_eff
    scored = score_puncta_zscore(det, raw, cfg_eff)
    kept = np.array(
        [[s["row"], s["col"], s["sigma"]] for s in scored if s["kept"]]
    ).reshape(-1, 3)
    return raw, scored, kept, cfg_eff


def restrict_puncta_to_near(
    blobs: np.ndarray,
    near_mask: np.ndarray,
) -> np.ndarray:
    """Keep ``[row, col, sigma]`` blobs whose rounded centre lies on ``near_mask``."""
    if blobs.shape[0] == 0:
        return blobs
    H, W = near_mask.shape
    rr = np.clip(np.round(blobs[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(blobs[:, 1]).astype(int), 0, W - 1)
    return blobs[near_mask[rr, cc].astype(bool)]
