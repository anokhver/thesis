"""Spotiflow-based puncta pseudo-labels for fluorescence-microscopy MIPs.

Per-channel pipeline: Spotiflow `predict` (model.predict, percentile
"auto" normalizer + heatmap NMS) -> optional absolute-intensity floor on
the raw image -> synthetic-sigma adapter -> render disks -> optional
centre-on-(soma U dendrite) gate.

Spotiflow is single-channel and threshold-agnostic (Dominguez Mantes
et al., Nat. Methods 2025, doi:10.1038/s41592-025-02662-x;
https://github.com/weigertlab/spotiflow). The model returns `(N, 2)`
`(y, x)` subpixel coordinates; this module wraps them as `(N, 3)`
`(row, col, sigma)` with a synthetic sigma derived from a user-chosen
pixel radius, so the shared geometry helpers (`puncta_to_mask`,
`restrict_puncta_to_near`) work unchanged.

`spotiflow` is an optional dependency; import lazily at model-load time.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from typing import List, Optional, Tuple

import numpy as np

from .puncta_common import puncta_to_mask, restrict_puncta_to_near  # noqa: F401


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpotiflowPunctaCfg:
    """Per-channel knobs for the Spotiflow detector.

    `prob_thresh=None` uses the model's calibrated optimum (0.5 for the
    `general` checkpoint). `n_tiles=None` lets Spotiflow infer tiling
    from GPU memory; force e.g. `(3, 3)` on small GPUs for a 2304x2304
    MIP. `intensity_floor` is auto-derived per-image when the
    orchestrator runs with `auto_floor=True`.
    """

    # Model
    pretrained_model: str = "general"
    device: str = "auto"            # "auto" | "cuda" | "cpu" | "mps"
    normalizer: str = "auto"        # percentile p1/p99.8 (see spotiflow.utils)

    # Inference
    prob_thresh: Optional[float] = None
    min_distance: int = 1
    exclude_border: bool = False
    n_tiles: Optional[Tuple[int, int]] = None
    verbose: bool = False

    # Rendering. sigma = render_radius_px / sqrt(2) so puncta_to_mask
    # paints a disk of exactly `render_radius_px` pixels.
    render_radius_px: float = 3.0

    # Absolute-intensity floor on the raw (un-normalised) image, applied
    # at the spot centre (mean over a disk of `intensity_inner_radius`).
    # When `auto_floor=True` in the orchestrator, `intensity_floor` is
    # patched to `bg_med + intensity_floor_k * bg_sigma` per image, with
    # bg = pixels at or below the 30th percentile and
    # bg_sigma = 1.4826 * MAD. Same form as `derive_zscore_floors` in
    # `puncta_log`. `intensity_floor_k=0` disables the floor.
    intensity_floor_k: float = 4.0
    intensity_inner_radius: int = 1
    intensity_floor: float = 0.0


# ---------------------------------------------------------------------------
# Calibrated defaults
# ---------------------------------------------------------------------------
#
# Per-channel overrides on top of the SpotiflowPunctaCfg field defaults.
# Render radii match the biology priors used in `puncta_log`
# (Harris & Stevens 1989, J Neurosci 9(8):2982-2997, after PSF
# convolution at 107 nm/px):
#   PRE  boutons ~5-10 px diameter -> render_radius_px=4
#   POST PSD     ~3-5  px diameter -> render_radius_px=2
# `min_distance` is the NMS radius (px) on the model heatmap; match the
# render radius so neighbouring presynaptic boutons don't merge.
# `prob_thresh=None` defers to the `general` checkpoint's calibrated
# optimum (0.5); tighten when you see false positives, relax for faint
# puncta.

DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW: dict = dict(
    prob_thresh=None,
    min_distance=2,
    exclude_border=False,
    render_radius_px=4.0,
    intensity_floor_k=4.0,
    intensity_inner_radius=1,
)


DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW: dict = dict(
    prob_thresh=None,
    min_distance=1,
    exclude_border=False,
    render_radius_px=2.0,
    intensity_floor_k=4.0,
    intensity_inner_radius=1,
)


def resolve_cfg(
    overrides: Optional[dict] = None,
    *,
    channel: str = "pre",
) -> "SpotiflowPunctaCfg":
    """Merge `overrides` into `DEFAULT_PUNCTA_CFG_{PRE,POST}_SPOTIFLOW`
    and return a `SpotiflowPunctaCfg`.

    `channel` selects the base default (`'pre'` or `'post'`). Unknown
    keys in `overrides` raise `ValueError` so a typo'd JSON does not
    pass silently. Mirrors `puncta_log.resolve_cfg`.
    """
    if channel == "pre":
        base = DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW
    elif channel == "post":
        base = DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW
    else:
        raise ValueError(f"channel must be 'pre' or 'post', got {channel!r}")
    merged = dict(base)
    merged.update(overrides or {})
    valid = {f.name for f in dataclass_fields(SpotiflowPunctaCfg)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"unknown SpotiflowPunctaCfg field(s) in overrides: {sorted(unknown)}"
        )
    return SpotiflowPunctaCfg(**merged)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: SpotiflowPunctaCfg):
    """Load a pretrained Spotiflow checkpoint. `spotiflow` is optional.

    `cfg.device` is forwarded as `map_location` so `--device cpu` avoids
    touching GPU memory at load time.
    """
    try:
        from spotiflow.model import Spotiflow
    except ImportError as e:
        raise ImportError(
            "spotiflow is not installed. Install it with "
            "`pip install spotiflow` or `pip install -e .[spotiflow]`."
        ) from e
    return Spotiflow.from_pretrained(
        cfg.pretrained_model,
        map_location=cfg.device,
        verbose=bool(cfg.verbose),
    )


def _disk_offsets(radius: int) -> np.ndarray:
    """Integer (dy, dx) offsets for a closed disk of `radius` pixels."""
    r = int(radius)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    mask = yy * yy + xx * xx <= r * r
    dy, dx = np.where(mask)
    return np.stack([dy - r, dx - r], axis=1)


def _mean_in_disk(
    img2d: np.ndarray,
    y: float,
    x: float,
    offsets: np.ndarray,
) -> float:
    """Mean of `img2d` over the disk centered at the rounded `(y, x)`,
    clipped to image bounds. Returns NaN if no in-bounds pixels.
    """
    H, W = img2d.shape
    rr = int(round(y)); cc = int(round(x))
    ys = rr + offsets[:, 0]
    xs = cc + offsets[:, 1]
    in_bounds = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    if not np.any(in_bounds):
        return float("nan")
    return float(img2d[ys[in_bounds], xs[in_bounds]].mean())


# ---------------------------------------------------------------------------
# Spot detection (per channel)
# ---------------------------------------------------------------------------

def detect_spots_spotiflow(
    img2d: np.ndarray,
    model,
    cfg: SpotiflowPunctaCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Spotiflow on one 2D channel.

    Returns `(points (N, 2), prob (N,))`. `points` are subpixel
    `(y, x)` coordinates; `prob` is the heatmap peak intensity per spot
    (NaN if the build of spotiflow doesn't expose it).
    """
    if img2d.ndim != 2:
        raise ValueError(f"detect_spots_spotiflow expects 2D, got shape {img2d.shape}")
    points, details = model.predict(
        img2d.astype(np.float32),
        prob_thresh=cfg.prob_thresh,
        min_distance=int(cfg.min_distance),
        exclude_border=bool(cfg.exclude_border),
        n_tiles=cfg.n_tiles,
        normalizer=cfg.normalizer,
        device=cfg.device,
        verbose=bool(cfg.verbose),
    )
    points = np.asarray(points, dtype=np.float64)
    prob = np.asarray(
        getattr(details, "prob", np.full(len(points), np.nan)),
        dtype=np.float64,
    )
    return points, prob


# ---------------------------------------------------------------------------
# Intensity gate / geometry adapter
# ---------------------------------------------------------------------------

def filter_by_intensity(
    img2d: np.ndarray,
    points: np.ndarray,
    *,
    min_inner: float,
    radius: int = 1,
) -> np.ndarray:
    """Keep-mask for spots whose mean intensity over a `radius`-disk at the
    rounded centre clears `min_inner`.

    Pure post-hoc gate; subpixel coords are rounded for indexing.
    Returns `np.ones(N, dtype=bool)` when `min_inner <= 0` (disabled).
    """
    if len(points) == 0 or min_inner <= 0:
        return np.ones(len(points), dtype=bool)
    offsets = _disk_offsets(radius)
    keep = np.zeros(len(points), dtype=bool)
    floor = float(min_inner)
    for i, (y, x) in enumerate(points):
        mu = _mean_in_disk(img2d, y, x, offsets)
        keep[i] = np.isfinite(mu) and mu >= floor
    return keep


def derive_intensity_floor(img2d: np.ndarray, k: float) -> Tuple[float, float, float]:
    """Per-image background statistics for the absolute-intensity floor.

    Returns `(bg_med, bg_sigma, floor)` where `bg` = pixels at or below
    the 30th percentile, `bg_sigma` = 1.4826 * MAD, and
    `floor = bg_med + k * bg_sigma`. Returns `(0, 0, 0)` on degenerate
    input. Same statistic as `puncta_log.derive_zscore_floors` (which
    operates on the tophatted image; here we use the raw image since
    Spotiflow runs its own percentile normalisation internally).
    """
    finite = img2d[np.isfinite(img2d)]
    if finite.size == 0 or k <= 0:
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
    return bg_med, bg_sigma, bg_med + float(k) * bg_sigma


def spots_to_blobs(points: np.ndarray, radius_px: float) -> np.ndarray:
    """`(N, 2)` -> `(N, 3)` `(row, col, sigma)` adapter.

    sigma = radius_px / sqrt(2) so `puncta_to_mask` and
    `viz.visualise_puncta_full` paint disks of exactly `radius_px`
    pixels.
    """
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    sigma = float(radius_px) / np.sqrt(2.0)
    out = np.empty((len(points), 3), dtype=np.float64)
    out[:, 0] = points[:, 0]
    out[:, 1] = points[:, 1]
    out[:, 2] = sigma
    return out


# ---------------------------------------------------------------------------
# End-to-end per-channel helper (live pipeline)
# ---------------------------------------------------------------------------

def detect_puncta_channel(
    img2d: np.ndarray,
    cfg: SpotiflowPunctaCfg,
    *,
    model,
    auto_floor: bool = True,
) -> Tuple[np.ndarray, List[dict], np.ndarray, SpotiflowPunctaCfg]:
    """Spotiflow predict -> optional intensity floor -> synthetic-sigma blobs.

    Returns `(raw[N, 3], scored, kept[M, 3], cfg_eff)`, mirroring
    `puncta_log.detect_puncta_channel`:
      * `raw` -- every spotiflow spot as `(row, col, sigma)`.
      * `scored` -- per-spot dicts with `row, col, prob, mu_in,
        floor, kept`. Empty list if no candidates.
      * `kept` -- subset of `raw` that survived the intensity floor.
      * `cfg_eff` -- cfg actually used (`intensity_floor` patched in
        when `auto_floor=True`).
    """
    if auto_floor and cfg.intensity_floor_k > 0:
        _, _, floor = derive_intensity_floor(img2d, cfg.intensity_floor_k)
        cfg_eff = replace(cfg, intensity_floor=floor)
    else:
        cfg_eff = cfg

    points, prob = detect_spots_spotiflow(img2d, model, cfg_eff)
    if len(points) == 0:
        raw = spots_to_blobs(points, cfg_eff.render_radius_px)
        return raw, [], raw, cfg_eff

    keep = filter_by_intensity(
        img2d, points,
        min_inner=cfg_eff.intensity_floor,
        radius=cfg_eff.intensity_inner_radius,
    )

    # Per-spot mean-inner-intensity for the scored records (same disk
    # window as the keep gate).
    offsets = _disk_offsets(cfg_eff.intensity_inner_radius)
    scored: List[dict] = []
    for i, (y, x) in enumerate(points):
        mu_in = _mean_in_disk(img2d, y, x, offsets)
        scored.append({
            "row": float(y), "col": float(x),
            "prob": float(prob[i]) if i < len(prob) else float("nan"),
            "mu_in": mu_in,
            "floor": float(cfg_eff.intensity_floor),
            "kept": bool(keep[i]),
        })

    raw = spots_to_blobs(points, cfg_eff.render_radius_px)
    kept_pts = points[keep]
    kept = spots_to_blobs(kept_pts, cfg_eff.render_radius_px)
    return raw, scored, kept, cfg_eff
