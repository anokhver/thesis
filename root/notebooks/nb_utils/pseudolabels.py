"""Blob pseudo-label generation for fluorescence-microscopy patches.

Operates on (C, H, W) patches with the following channel convention:
    channel 0 - presynaptic puncta marker (e.g. PSD-95-binding partner)
    channel 1 - postsynaptic puncta marker (e.g. PSD-95)
    channel 2 - structural/neurite marker (e.g. MAP2, beta-III-tubulin)

Pipeline (mirroring synaptic-puncta literature):

    pre channel  --LoG--> blobs_pre  ----+
                                          \
    post channel --LoG--> blobs_post  ----+--> intersect (with dilation)
                                          \
    structural channel --Meijering--> dendrite_mask --+
    structural channel --intensity --> soma_mask  ---+--> structural_mask
                                                              |
                            colocalised blobs --restrict-to--/
                                            |
                            shape/size filter
                                            |
                            (optional) per-blob z-score test
                                            |
                                       label_mask (binary)

References
----------
* Meijering E. et al. "Design and validation of a tool for neurite tracing and
  analysis in fluorescence microscopy images." Cytometry A, 2004.
  -- The neurite ridge filter used for the dendrite mask.
* Frangi A.F. et al. "Multiscale vessel enhancement filtering." MICCAI 1998.
  -- Family of Hessian-based ridge filters; Meijering is the
     fluorescence-tuned variant.
* Lindeberg T. "Feature detection with automatic scale selection." IJCV 1998.
  -- Scale-normalised Laplacian-of-Gaussian for blob detection.
* Wang Y., Wang C., et al. "SynQuant: an automatic tool to quantify synapses
  from fluorescence microscopy images." Bioinformatics 36(5):1599, 2020.
  -- Order-statistics z-score that filters LoG candidates against a local
     annular background. We use the closed-form Cramer-style approximation
     (z * sqrt(2 ln n)) suggested in the SynQuant supplement.
* Fantuzzo J.A. et al. "Intellicount: high-throughput quantification of
  fluorescent synaptic protein puncta by machine learning." eNeuro, 2017.
  -- Reference for puncta sizes and intensity ranges in cultured neurons.
* Xiao R. et al. "DDeep3M+: adaptive enhancement powered weakly supervised
  learning for neuron segmentation." Neurophotonics 10(3), 2023.
  -- Hessian/ridge-based pseudo-label generation, iterative refinement.

Patch-level operation (128x128) caveats
----------------------------------------
* `skimage.feature.blob_log` reflects the image at borders inside the
  Gaussian-Laplace convolution, but the reflected signal is symmetric and
  therefore yields an attenuated LoG response within ~3 sigma_max of the
  edge. Use `exclude_border` >= ceil(3 * sigma_max) to drop unreliable
  boundary detections. For sigma_max=1.8 px this is 5-6 px, i.e. about
  15% of a 128x128 patch is excluded from puncta detection.
* `skimage.filters.threshold_otsu` is a global, histogram-based statistic.
  Running it per-patch gives wildly inconsistent thresholds when some
  patches contain no neurites. Use `compute_global_meijering_threshold`
  to fit one threshold across a sample of patches, then reuse it.
* The annular z-score in `score_blob_zscore` operates inside a small local
  window (no full-image masks) so it stays fast even on dense patches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from skimage.draw import disk as draw_disk
from skimage.feature import blob_log
from skimage.filters import meijering, threshold_otsu
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import dilation, disk as morph_disk, remove_small_objects


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BlobPseudoCfg:
    """All knobs for the pseudo-label pipeline.

    Defaults are calibrated for 107 nm/px confocal of cultured neurons,
    puncta diameter ~2-5 px, dendrite diameter ~4-18 px.
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
    soma_min_area: int = 200  # px^2; ~2.3 um^2 at 107 nm/px

    # structural-mask post-processing (dendrites U somas, then dilate)
    structural_dilation: int = 4  # ~ 0.43 um -- "near-neurite" zone

    # shape priors on candidate puncta
    min_size: int = 3
    max_size: int = 40
    min_fill: float = 0.5
    max_wh_ratio: float = 4.0

    # per-blob SynQuant-lite z-score (set use_zscore=False to skip)
    use_zscore: bool = True
    zscore_inner_radius: int = 3
    zscore_outer_radius: int = 8
    zscore_threshold: float = 5.0  # SynQuant default = 10 (strict). 5-7 = friendlier.


# ---------------------------------------------------------------------------
# 1. LoG blob detection (per channel)
# ---------------------------------------------------------------------------

def detect_blobs_log(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """Run scale-space LoG on a (H, W) channel. Returns (N, 3) (row, col, sigma)."""
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
    """Render LoG blobs as a union of disks (radius = sqrt(2)*sigma)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col, sigma in blobs:
        radius = max(1, int(np.round(np.sqrt(2.0) * sigma)))
        rr, cc = draw_disk((int(row), int(col)), radius, shape=shape)
        mask[rr, cc] = 1
    return mask


# ---------------------------------------------------------------------------
# 2. Structural mask: dendrite (Meijering) U soma (intensity)
# ---------------------------------------------------------------------------

def meijering_response(
    image: np.ndarray,
    sigmas: Iterable[int],
) -> np.ndarray:
    """Multi-scale Meijering ridge response on a (H, W) structural channel.

    `black_ridges=False` because fluorescent neurites are bright on a dark
    background.
    """
    return meijering(image, sigmas=list(sigmas), black_ridges=False)


def compute_global_meijering_threshold(
    structural_patches: Sequence[np.ndarray],
    sigmas: Iterable[int],
    method: str = "otsu",
    percentile: float = 90.0,
) -> float:
    """Pool Meijering responses across a sample of patches and return one
    global threshold.

    Patch-level Otsu is unreliable because some patches contain no neurites
    (their Meijering histogram is unimodal and Otsu picks a meaningless
    cut). Pooling across patches stabilises the threshold.
    """
    pooled = []
    sigmas = list(sigmas)
    for ch_img in structural_patches:
        if ch_img.ndim != 2:
            raise ValueError(f"expected 2D patch, got {ch_img.shape}")
        r = meijering_response(ch_img, sigmas)
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
    """Returns (response, mask). Mask is `response > threshold`."""
    response = meijering_response(structural_image, sigmas)
    return response, response > threshold


def make_soma_mask(
    structural_image: np.ndarray,
    intensity_percentile: float = 99.0,
    min_area: int = 200,
) -> np.ndarray:
    """Soma mask: high-intensity connected components of size >= min_area.

    Meijering suppresses solid bright regions (it is a ridge filter), so
    somas are missed by the dendrite mask. We recover them with a simple
    intensity threshold + size filter.
    """
    if structural_image.ndim != 2:
        raise ValueError("make_soma_mask expects 2D")
    thr = float(np.percentile(structural_image, intensity_percentile))
    bright = structural_image > thr
    if not bright.any():
        return np.zeros_like(bright, dtype=bool)
    return remove_small_objects(bright, min_size=min_area)


def make_structural_mask(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> dict:
    """Build the dendrite U soma mask + a dilated 'near-neuron' zone.

    `image`: (C, H, W) patch. Uses `cfg.structural_channel`.
    """
    if cfg.dendrite_threshold is None:
        raise ValueError(
            "cfg.dendrite_threshold is None -- call "
            "compute_global_meijering_threshold first and assign the result."
        )
    struct = image[cfg.structural_channel]
    response, dendrite = make_dendrite_mask(
        struct, cfg.dendrite_sigmas, cfg.dendrite_threshold
    )
    soma = make_soma_mask(
        struct,
        intensity_percentile=cfg.soma_intensity_percentile,
        min_area=cfg.soma_min_area,
    )
    structural = dendrite | soma
    near = dilation(structural, morph_disk(cfg.structural_dilation))
    return {
        "meijering_response": response,
        "dendrite_mask": dendrite,
        "soma_mask": soma,
        "structural_mask": structural,
        "near_structural": near,
    }


# ---------------------------------------------------------------------------
# 3. Co-localization
# ---------------------------------------------------------------------------

def colocalize_intersect(
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    dilation_radius: int = 2,
) -> np.ndarray:
    """Pre dilated then intersected with post.

    Justification: at typical confocal lateral resolution (~250 nm) and
    the 107 nm/px sampling here, pre/post markers at a real synapse are
    co-localised within 1-2 px. Dilating pre by 2 px before intersecting
    with post tolerates this sub-pixel displacement.
    """
    if dilation_radius > 0:
        pre = dilation(pre_mask.astype(bool), morph_disk(dilation_radius))
    else:
        pre = pre_mask.astype(bool)
    return (pre & post_mask.astype(bool)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Per-blob z-score (SynQuant-lite, local-window implementation)
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
    """Annular-background z-score for one LoG candidate.

    Implementation note: a full-image annulus mask is unnecessarily
    expensive (the existing build_pseudolabels_C.py allocates two full-
    image bool arrays per blob). Here we crop a (2*r_out+1)^2 window
    around the blob and build masks inside that window only -- ~100x
    faster and identical outputs.
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
# 5. Shape / size filter
# ---------------------------------------------------------------------------

def filter_by_size_shape(
    mask: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """SynQuant-style shape priors. Drop CCs that violate min/max area,
    aspect ratio, or fill ratio."""
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
# 6. End-to-end orchestrator on a single (C, H, W) patch
# ---------------------------------------------------------------------------

def generate_blob_pseudolabel(
    patch: np.ndarray,
    cfg: BlobPseudoCfg,
) -> Tuple[np.ndarray, dict, dict]:
    """Run the full pipeline on a single 3-channel patch.

    Returns:
        label_mask: (H, W) uint8
        intermediates: per-step arrays + scored blobs
        stats: counts and fractions
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

    # 1. structural mask (dendrite U soma, dilated)
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
