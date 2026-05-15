"""Blob pseudo-label generation for fluorescence-microscopy patches.

Channel convention: 0=presynaptic, 1=postsynaptic, 2=structural/neurite.

Pipeline: scikit-image LoG blobs on pre+post → co-localisation via
intersection after small dilation → restrict to dendrite ∪ soma
structural mask (Meijering ridge + intensity threshold) → shape filter
→ optional per-blob annular z-score (SynQuant-lite).

References:
* Meijering et al. (Cytometry A 2004) — neurite ridge filter.
* Lindeberg (IJCV 1998) — scale-normalised LoG.
* Wang et al. (Bioinformatics 2020) — SynQuant annular z-score.
* Fantuzzo et al. (eNeuro 2017) — puncta size/intensity calibration.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from skimage.draw import disk as draw_disk
from skimage.feature import blob_log
from skimage.filters import gaussian, meijering, threshold_otsu
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import (
    closing as morph_closing,
    dilation,
    disk as morph_disk,
    remove_small_holes,
    remove_small_objects,
    white_tophat,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BlobPseudoCfg:
    """Pseudo-label pipeline configuration.

    Defaults calibrated for 107 nm/px confocal: puncta 2-5 px,
    dendrites 4-18 px.
    """

    # channel roles
    pre_channel: int = 0
    post_channel: int = 1
    structural_channel: int = 2

    # LoG blob detection on pre/post channels
    log_min_sigma: float = 0.7
    log_max_sigma: float = 1.8
    log_num_sigma: int = 5
    log_threshold: float = 0.005
    log_overlap: float = 0.5
    log_exclude_border: int = 5  # ~ 3 * log_max_sigma

    # co-localization (intersect pre with post after a small pre-dilation)
    coloc_dilation: int = 2

    # Meijering-based dendrite mask
    dendrite_sigmas: Sequence[int] = field(
        default_factory=lambda: list(range(2, 10))
    )
    # Threshold on Meijering response. None means "use Otsu of the
    # current patch only" (NOT recommended on patches; see module
    # docstring). Set this from `compute_global_meijering_threshold`.
    dendrite_threshold: float | None = None

    # soma mask (high intensity in the structural channel)
    soma_intensity_percentile: float = 99.0
    # Cultured-neuron somas are ~10-30 um diameter, i.e. ~7000-60000 px^2
    # at 107 nm/px. 5000 px^2 (~ 80 px disk) is a safe lower bound that
    # excludes bright dendrite junctions and isolated bright puncta.
    soma_min_area: int = 5000
    # Binary morphology applied BEFORE the size filter to consolidate
    # fragmented bright regions inside a soma into a single connected
    # component. A percentile threshold alone cuts a soma into several
    # small CCs (the nucleolus is dim, cytoplasm is patchy) so smaller
    # somas drop below ``soma_min_area`` even when the bigger one survives.
    # ``soma_closing_radius`` is the radius of the closing structuring
    # element in pixels (0 = no closing).  ``soma_fill_holes`` plugs the
    # dark nucleolus hole inside the soma.
    soma_closing_radius: int = 0
    soma_fill_holes: bool = False

    # structural-mask post-processing (dendrites U somas, then dilate)
    structural_dilation: int = 4  # ~ 0.43 um -- "near-neurite" zone

    # shape priors on candidate puncta
    min_size: int = 3
    max_size: int = 40
    min_fill: float = 0.5
    max_wh_ratio: float = 4.0

    # structural-channel smoothing (applied before Meijering, NOT before soma)
    # "none" = disabled, "gaussian" = Gaussian blur, "median" = median filter,
    # "tophat" = white top-hat (removes grid background, keeps bright features)
    structural_smooth_method: str = "none"
    structural_smooth_sigma: float = 1.0   # sigma for gaussian
    structural_median_size: int = 3        # kernel size for median (must be odd)
    structural_tophat_radius: int = 15     # disk radius for white top-hat

    # per-blob SynQuant-lite z-score (set use_zscore=False to skip)
    use_zscore: bool = True
    zscore_inner_radius: int = 3
    zscore_outer_radius: int = 8
    zscore_threshold: float = 5.0  # SynQuant default = 10 (strict). 5-7 = friendlier.


# ---------------------------------------------------------------------------
# 1. Structural-channel smoothing
# ---------------------------------------------------------------------------

def smooth_structural_channel(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """Pre-process a 2D structural channel before ridge detection.

    Reduces pixelation / grid artifacts so Meijering captures dendrite
    ridges more cleanly.  Applied before ``make_dendrite_mask``, but NOT
    before ``make_soma_mask`` (percentile thresholds are sensitive to
    background removal).

    Methods:
        * ``"none"``     — passthrough.
        * ``"gaussian"`` — isotropic Gaussian blur (``structural_smooth_sigma``).
        * ``"median"``   — median filter (``structural_median_size``).
        * ``"tophat"``   — morphological white top-hat with a disk of
          ``structural_tophat_radius``.  Extracts bright features (dendrites)
          while removing slowly-varying background and periodic grid
          artifacts.  Best when the structural channel has a visible
          checkerboard/grid pattern from the acquisition.

    Returns the processed image (same shape/dtype).
    """
    if image.ndim != 2:
        raise ValueError(f"smooth_structural_channel expects 2D, got {image.shape}")
    method = cfg.structural_smooth_method
    if method == "none":
        return image
    if method == "gaussian":
        if cfg.structural_smooth_sigma <= 0:
            return image
        return gaussian(
            image,
            sigma=cfg.structural_smooth_sigma,
            preserve_range=True,
        ).astype(image.dtype)
    if method == "median":
        size = cfg.structural_median_size
        if size < 1:
            return image
        if size % 2 == 0:
            raise ValueError(
                f"structural_median_size must be odd, got {size}"
            )
        return median_filter(image, size=size).astype(image.dtype)
    if method == "tophat":
        radius = cfg.structural_tophat_radius
        if radius < 1:
            return image
        return white_tophat(image, morph_disk(radius)).astype(image.dtype)
    raise ValueError(
        f"unknown structural_smooth_method: {method!r}; "
        f"expected 'none', 'gaussian', 'median', or 'tophat'"
    )


# ---------------------------------------------------------------------------
# 2. LoG blob detection (per channel)
# ---------------------------------------------------------------------------

def detect_blobs_log(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """Scale-space LoG blob detection on a 2D channel.

    Return ``(N, 3)`` array of ``(row, col, sigma)``. Scale-normalised LoG
    (Lindeberg, IJCV 1998).

    Ref: https://github.com/scikit-image/scikit-image
    """
    if image.ndim != 2:
        raise ValueError(f"detect_blobs_log expects 2D, got shape {image.shape}")
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


def blobs_to_mask(blobs: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Render LoG blobs as a union of disks (radius = ``sqrt(2) * sigma``)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col, sigma in blobs:
        radius = max(1, int(np.round(np.sqrt(2.0) * sigma)))
        rr, cc = draw_disk((int(row), int(col)), radius, shape=shape)
        mask[rr, cc] = 1
    return mask


# ---------------------------------------------------------------------------
# 3. Structural mask: dendrite (Meijering) U soma (intensity)
# ---------------------------------------------------------------------------

def meijering_response(
    image: np.ndarray,
    sigmas: Iterable[int],
) -> np.ndarray:
    """Multi-scale Meijering ridge response on a 2D structural channel.

    ``black_ridges=False`` for bright-on-dark fluorescence. Meijering et al.
    (Cytometry A 2004).
    """
    return meijering(image, sigmas=list(sigmas), black_ridges=False)


def compute_global_meijering_threshold(
    structural_patches: Sequence[np.ndarray],
    sigmas: Iterable[int],
    method: str = "otsu",
    percentile: float = 90.0,
    cfg: BlobPseudoCfg | None = None,
) -> float:
    """Pool Meijering responses across patches and return a single threshold.

    Stabilises the threshold against per-patch Otsu failure on neurite-free
    patches.  When ``cfg`` is provided, each patch is smoothed before
    computing the Meijering response so the threshold matches the smoothed
    pipeline.
    """
    pooled = []
    sigmas = list(sigmas)
    for ch_img in structural_patches:
        if ch_img.ndim != 2:
            raise ValueError(f"expected 2D patch, got {ch_img.shape}")
        img = smooth_structural_channel(ch_img, cfg) if cfg is not None else ch_img
        r = meijering_response(img, sigmas)
        nz = r[r > 0]
        if nz.size:
            pooled.append(nz)
    if not pooled:
        return 0.0
    pooled = np.concatenate(pooled)
    if method == "otsu":
        return float(threshold_otsu(pooled))
    if method == "percentile":
        return float(np.percentile(pooled, percentile))
    raise ValueError(f"unknown method: {method!r}")


def make_dendrite_mask(
    structural_image: np.ndarray,
    sigmas: Iterable[int],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(response, response > threshold)``."""
    response = meijering_response(structural_image, sigmas)
    return response, response > threshold


def make_soma_mask(
    structural_image: np.ndarray,
    intensity_percentile: float = 99.0,
    min_area: int = 200,
    closing_radius: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """Soma mask from high-intensity connected components in the structural channel.

    Recovers bright solid somas that Meijering misses. Keep CCs with
    area ≥ ``min_area`` after optional closing and hole-fill.
    """
    if structural_image.ndim != 2:
        raise ValueError("make_soma_mask expects 2D")
    thr = float(np.percentile(structural_image, intensity_percentile))
    bright = structural_image > thr
    if not bright.any():
        return np.zeros_like(bright, dtype=bool)
    if closing_radius > 0:
        bright = morph_closing(bright, morph_disk(closing_radius))
    bright = remove_small_objects(bright, min_size=min_area)
    if fill_holes and bright.any():
        bright = remove_small_holes(bright, max_size=min_area)
    return bright


def make_structural_mask(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> dict:
    """Build dendrite ∪ soma mask plus dilated near-neuron zone from ``(C, H, W)``.

    Smoothing (if enabled) is applied to the structural channel before
    Meijering ridge detection but NOT before soma detection (percentile
    thresholds are sensitive to blur).
    """
    if cfg.dendrite_threshold is None:
        raise ValueError(
            "cfg.dendrite_threshold is None -- call "
            "compute_global_meijering_threshold first and assign the result."
        )
    struct_raw = image[cfg.structural_channel]
    struct_smooth = smooth_structural_channel(struct_raw, cfg)
    response, dendrite = make_dendrite_mask(
        struct_smooth, cfg.dendrite_sigmas, cfg.dendrite_threshold
    )
    # soma uses the RAW channel -- percentile thresholds shift under blur
    soma = make_soma_mask(
        struct_raw,
        intensity_percentile=cfg.soma_intensity_percentile,
        min_area=cfg.soma_min_area,
        closing_radius=cfg.soma_closing_radius,
        fill_holes=cfg.soma_fill_holes,
    )
    structural = dendrite | soma
    near = dilation(structural, morph_disk(cfg.structural_dilation))
    return {
        "meijering_response": response,
        "dendrite_mask": dendrite,
        "soma_mask": soma,
        "structural_mask": structural,
        "near_structural": near,
        "smoothed_structural": struct_smooth,
    }


# ---------------------------------------------------------------------------
# 4. Co-localization
# ---------------------------------------------------------------------------

def colocalize_intersect(
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Dilate ``pre_mask`` and intersect with ``post_mask``. Sub-pixel offset tolerance."""
    if dilation_radius > 0:
        pre = dilation(pre_mask.astype(bool), morph_disk(dilation_radius))
    else:
        pre = pre_mask.astype(bool)
    return (pre & post_mask.astype(bool)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. Per-blob z-score (SynQuant-lite, local-window implementation)
# ---------------------------------------------------------------------------

def _local_window(image: np.ndarray, row: int, col: int, half: int):
    H, W = image.shape
    r0 = max(0, row - half); r1 = min(H, row + half + 1)
    c0 = max(0, col - half); c1 = min(W, col + half + 1)
    return image[r0:r1, c0:c1], (r0, c0)


def score_blob_zscore(
    image: np.ndarray,
    row: float,
    col: float,
    sigma: float,
    r_in: int,
    r_out: int,
) -> dict:
    """Annular-background z-score for one LoG blob (SynQuant-lite, local window).

    Wang et al. (Bioinformatics 2020).
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
            "sigma_bg": float("nan"), "z_raw": float("nan"),
            "z": float("nan"), "kept": False,
        }
    in_vals = win[inside]
    bg_vals = win[annulus]
    mu_in = float(in_vals.mean())
    mu_bg = float(bg_vals.mean())
    sigma_bg = float(bg_vals.std(ddof=1)) + 1e-8
    z_raw = (mu_in - mu_bg) / sigma_bg
    n_in = int(inside.sum())
    z = float(z_raw * np.sqrt(2.0 * np.log(max(2, n_in))))
    return {
        "row": float(row), "col": float(col), "sigma": float(sigma),
        "radius": int(radius), "n_inside": n_in,
        "mu_in": mu_in, "mu_bg": mu_bg, "sigma_bg": sigma_bg,
        "z_raw": float(z_raw), "z": z, "kept": False,
    }


def score_blobs_zscore(
    image: np.ndarray,
    blobs: np.ndarray,
    cfg: BlobPseudoCfg,
) -> List[dict]:
    """Score every blob and tag ``kept`` against ``cfg.zscore_threshold``."""
    out = []
    for row, col, sigma in blobs:
        rec = score_blob_zscore(
            image, row, col, sigma,
            cfg.zscore_inner_radius,
            cfg.zscore_outer_radius,
        )
        rec["kept"] = bool(rec["z"] >= cfg.zscore_threshold) if not np.isnan(rec["z"]) else False
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# 6. Shape / size filter
# ---------------------------------------------------------------------------

def filter_by_size_shape(
    mask: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """Drop CCs violating min/max area, aspect ratio, or bbox fill.

    SynQuant-style shape priors. Fantuzzo et al. (eNeuro 2017) size calibration.
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
# 7. End-to-end orchestrator on a single (C, H, W) patch
# ---------------------------------------------------------------------------

def generate_blob_pseudolabel(
    patch: np.ndarray,
    cfg: BlobPseudoCfg,
    precomputed_struct: dict | None = None,
) -> Tuple[np.ndarray, dict, dict]:
    """Run the pseudo-label pipeline on one ``(C, H, W)`` patch.

    Pass ``precomputed_struct`` (a structural-mask slice from the full
    image) to avoid per-patch border artifacts. Return
    ``(label_mask, intermediates, stats)``.
    """
    if patch.ndim != 3:
        raise ValueError(f"patch must be (C, H, W); got {patch.shape}")
    C, H, W = patch.shape
    if C <= max(cfg.pre_channel, cfg.post_channel, cfg.structural_channel):
        raise ValueError(
            f"patch has {C} channels but cfg references "
            f"pre={cfg.pre_channel}, post={cfg.post_channel}, "
            f"structural={cfg.structural_channel}"
        )

    # 1. structural mask (dendrite U soma, dilated). Prefer a
    # precomputed full-image slice when available.
    if precomputed_struct is not None:
        required = {"structural_mask", "near_structural"}
        missing = required - set(precomputed_struct)
        if missing:
            raise ValueError(
                f"precomputed_struct missing keys: {sorted(missing)}"
            )
        for key in ("structural_mask", "near_structural"):
            if precomputed_struct[key].shape != (H, W):
                raise ValueError(
                    f"precomputed_struct[{key!r}] has shape "
                    f"{precomputed_struct[key].shape}, expected ({H}, {W})"
                )
        struct = precomputed_struct
    else:
        struct = make_structural_mask(patch, cfg)

    # 2. LoG blobs on pre + post
    pre_blobs = detect_blobs_log(patch[cfg.pre_channel], cfg)
    post_blobs = detect_blobs_log(patch[cfg.post_channel], cfg)

    # 3. optional z-score filter on each channel before rendering
    if cfg.use_zscore:
        pre_scored = score_blobs_zscore(patch[cfg.pre_channel], pre_blobs, cfg)
        post_scored = score_blobs_zscore(patch[cfg.post_channel], post_blobs, cfg)
        pre_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in pre_scored if s["kept"]]
        ).reshape(-1, 3)
        post_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in post_scored if s["kept"]]
        ).reshape(-1, 3)
    else:
        pre_scored, post_scored = [], []
        pre_kept, post_kept = pre_blobs, post_blobs

    pre_mask = blobs_to_mask(pre_kept, (H, W))
    post_mask = blobs_to_mask(post_kept, (H, W))

    # 4. co-localization
    coloc = colocalize_intersect(pre_mask, post_mask, cfg.coloc_dilation)

    # 5. shape priors
    shaped = filter_by_size_shape(coloc, cfg)

    # 6. restrict to the dilated structural mask
    label_mask = (shaped.astype(bool) & struct["near_structural"]).astype(np.uint8)

    intermediates = {
        **struct,
        "pre_blobs_raw": pre_blobs,
        "post_blobs_raw": post_blobs,
        "pre_blobs_kept": pre_kept,
        "post_blobs_kept": post_kept,
        "pre_scored": pre_scored,
        "post_scored": post_scored,
        "pre_mask": pre_mask,
        "post_mask": post_mask,
        "coloc_mask": coloc,
        "shaped_mask": shaped,
    }
    stats = {
        "n_pre_log": int(len(pre_blobs)),
        "n_post_log": int(len(post_blobs)),
        "n_pre_kept": int(len(pre_kept)),
        "n_post_kept": int(len(post_kept)),
        "px_coloc": int(coloc.sum()),
        "px_shaped": int(shaped.sum()),
        "px_label": int(label_mask.sum()),
        "frac_label": float(label_mask.mean()),
        "frac_dendrite": float(struct["dendrite_mask"].mean()),
        "frac_soma": float(struct["soma_mask"].mean()),
        "frac_near_structural": float(struct["near_structural"].mean()),
    }
    return label_mask, intermediates, stats


# ---------------------------------------------------------------------------
# 8. Full-image mode: structural mask on the full picture, LoG per patch
# ---------------------------------------------------------------------------

def compute_fullimage_structural_mask(
    full_image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> dict:
    """Run Meijering + soma detection on a full ``(C, H, W)`` image.

    Avoids per-patch border artifacts and fragmented cross-boundary somas.
    """
    return make_structural_mask(full_image, cfg)


def generate_pseudolabels_fullimage(
    full_image: np.ndarray,
    records: list[dict],
    cfg: BlobPseudoCfg,
    patch_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Per-patch LoG + co-localisation gated by a full-image structural mask.

    Return a dict mapping filename to ``(H, W)`` ``uint8`` mask.
    """
    if patch_size is None:
        patch_size = int(records[0]["patch_size"])

    # --- step 1: full-image structural mask ---
    struct = compute_fullimage_structural_mask(full_image, cfg)
    full_near = struct["near_structural"]  # (H_full, W_full) bool

    # --- step 2 + 3: per-patch LoG + gate with full structural ---
    result: dict[str, np.ndarray] = {}
    for rec in records:
        r = int(rec["grid_row"])
        c = int(rec["grid_col"])
        y0, x0 = r * patch_size, c * patch_size
        patch = full_image[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        near_patch = full_near[y0 : y0 + patch_size, x0 : x0 + patch_size]

        C, H, W = patch.shape

        # LoG on pre + post channels
        pre_blobs = detect_blobs_log(patch[cfg.pre_channel], cfg)
        post_blobs = detect_blobs_log(patch[cfg.post_channel], cfg)

        # optional z-score filter
        if cfg.use_zscore:
            pre_scored = score_blobs_zscore(patch[cfg.pre_channel], pre_blobs, cfg)
            post_scored = score_blobs_zscore(patch[cfg.post_channel], post_blobs, cfg)
            pre_kept = np.array(
                [[s["row"], s["col"], s["sigma"]] for s in pre_scored if s["kept"]]
            ).reshape(-1, 3)
            post_kept = np.array(
                [[s["row"], s["col"], s["sigma"]] for s in post_scored if s["kept"]]
            ).reshape(-1, 3)
        else:
            pre_kept, post_kept = pre_blobs, post_blobs

        pre_mask = blobs_to_mask(pre_kept, (H, W))
        post_mask = blobs_to_mask(post_kept, (H, W))

        # co-localization + shape filter
        coloc = colocalize_intersect(pre_mask, post_mask, cfg.coloc_dilation)
        shaped = filter_by_size_shape(coloc, cfg)

        # gate with full-image structural mask slice (the key difference)
        label_mask = (shaped.astype(bool) & near_patch).astype(np.uint8)
        result[rec["filename"]] = label_mask

    return result
