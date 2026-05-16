"""Blob pseudo-label generation for fluorescence-microscopy patches.

Channel convention: 0=presynaptic, 1=postsynaptic, 2=structural/neurite.

Pipeline: scikit-image LoG blobs on pre+post → optional per-blob annular
z-score (SynQuant-inspired local-SNR test) → render as disks → union of
the two channels → restrict to dendrite ∪ soma structural mask
(Meijering ridge + intensity threshold) → shape filter.

NOTE: this pipeline does NOT enforce pre/post co-localisation. The label
therefore represents synaptic-MARKER puncta, not synapses (which by
definition require both pre and post markers).

References:
* Meijering et al. (Cytometry A 2004) — neurite ridge filter.
* Otsu (IEEE Trans SMC 1979) — per-image threshold on the Meijering response.
* Lindeberg (IJCV 1998) — scale-normalised LoG.
* Wang et al. (Bioinformatics 2020) — SynQuant: shape priors and the
  local foreground-vs-background SNR principle. NOTE: ``score_blob_zscore``
  uses a parametric Gaussian z-score against an annular background; SynQuant
  itself uses a Wilcoxon rank-sum statistic against tabulated null moments.
* Fantuzzo et al. (eNeuro 2017) — Intellicount: dilated-neurite-mask
  restriction of puncta detections.
* Pathak et al. (Front. Cell Dev. Biol. 2025, DOI 10.3389/fcell.2025.1631520,
  §2.5.2) — white top-hat with disk SE for neuronal mask preprocessing.

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

    # Meijering-based dendrite mask
    dendrite_sigmas: Sequence[int] = field(
        default_factory=lambda: list(range(2, 10))
    )
    # Meijering ridge threshold. None = per-image Otsu on the response
    # of the *full* image being processed (Otsu, IEEE Trans SMC 1979).
    # Set to a float to override with a fixed global value, e.g. from
    # `compute_global_meijering_threshold` (kept for diagnostics).
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

    # Shape priors on the co-localised mask (intersection of two
    # LoG-disk masks). Same form as SynQuant (paraP3D.java: minfill,
    # maxWHratio, area bounds). Default area bounds match the
    # LoG-disk intersection geometry at 107 nm/px, not the physical
    # synapse area; widen + bump log_max_sigma to target the latter.
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

    # per-blob z-score on an annular background (set use_zscore=False to skip).
    # Same local-SNR principle as SynQuant (Wang et al. 2020) but parametric:
    # z = (mu_in - mu_bg) / sigma_bg. Threshold ~ 2 picks puncta clearly above
    # local background; tighten to 3+ for stricter, loosen to 1.5 for friendlier.
    use_zscore: bool = True
    zscore_inner_radius: int = 3
    zscore_outer_radius: int = 8
    zscore_threshold: float = 2.0


# ---------------------------------------------------------------------------
# 1. Structural-channel smoothing
# ---------------------------------------------------------------------------

def smooth_structural_channel(
    image: np.ndarray,
    cfg: BlobPseudoCfg,
) -> np.ndarray:
    """Pre-process a 2D structural channel before ridge detection.

    Reduces pixelation / grid artifacts so Meijering captures dendrite
    ridges more cleanly. Applied to the structural channel before ridge
    detection but NOT before soma detection (percentile thresholds shift
    under blur).

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

    Smoothing runs on the structural channel before Meijering, NOT before
    soma detection (percentile thresholds shift under blur). The dendrite
    threshold is ``cfg.dendrite_threshold`` if set, else per-image Otsu on
    the non-zero Meijering response (Otsu, IEEE Trans SMC 1979). The
    actual threshold used is returned under ``dendrite_threshold``.
    """
    struct_raw = image[cfg.structural_channel]
    struct_smooth = smooth_structural_channel(struct_raw, cfg)
    response = meijering_response(struct_smooth, cfg.dendrite_sigmas)
    if cfg.dendrite_threshold is None:
        nz = response[response > 0]
        # Empty / degenerate response -> Otsu undefined; refuse to flag
        # any pixel as dendrite (np.inf > nothing).
        if nz.size > 1 and float(nz.max() - nz.min()) > 0:
            threshold = float(threshold_otsu(nz))
        else:
            threshold = float("inf")
    else:
        threshold = float(cfg.dendrite_threshold)
    dendrite = response > threshold
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
        "dendrite_threshold": threshold,
        "dendrite_mask": dendrite,
        "soma_mask": soma,
        "structural_mask": structural,
        "near_structural": near,
        "smoothed_structural": struct_smooth,
    }


# ---------------------------------------------------------------------------
# 4. Co-localization
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
    """Annular-background z-score for one LoG blob (local-SNR test).

    Same local-SNR principle as SynQuant (Wang et al. 2020) but parametric:
    ``z = (mu_in - mu_bg) / sigma_bg``. SynQuant itself uses a Wilcoxon
    rank-sum statistic against tabulated null moments; this implementation
    is a cheaper Gaussian approximation.
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
    mu_bg = float(bg_vals.mean())
    sigma_bg = float(bg_vals.std(ddof=1)) + 1e-8
    z = float((mu_in - mu_bg) / sigma_bg)
    return {
        "row": float(row), "col": float(col), "sigma": float(sigma),
        "radius": int(radius), "n_inside": int(inside.sum()),
        "mu_in": mu_in, "mu_bg": mu_bg, "sigma_bg": sigma_bg,
        "z": z, "kept": False,
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
    """Keep CCs passing area, bbox aspect ratio, and bbox fill bounds.

    Same form as SynQuant's shape priors (Wang et al. 2020,
    ``paraP3D.java``: ``minfill``, ``maxWHratio``, min/max area). Default
    area range targets the LoG-disk intersection geometry at 107 nm/px,
    not physical synapse area -- widen the bounds to match the latter.
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

    # 4. union of pre + post puncta (no co-localisation requirement;
    # surviving spots represent synaptic-marker puncta, not synapses)
    puncta_mask = (pre_mask.astype(bool) | post_mask.astype(bool)).astype(np.uint8)

    # 5. shape priors
    shaped = filter_by_size_shape(puncta_mask, cfg)

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
        "puncta_mask": puncta_mask,
        "shaped_mask": shaped,
    }
    stats = {
        "n_pre_log": int(len(pre_blobs)),
        "n_post_log": int(len(post_blobs)),
        "n_pre_kept": int(len(pre_kept)),
        "n_post_kept": int(len(post_kept)),
        "px_puncta": int(puncta_mask.sum()),
        "px_shaped": int(shaped.sum()),
        "px_label": int(label_mask.sum()),
        "frac_label": float(label_mask.mean()),
        # Defensive on dendrite/soma keys: a precomputed structural slice
        # may omit them (e.g. iterative refinement only carries the gates).
        "frac_dendrite": (
            float(struct["dendrite_mask"].mean())
            if "dendrite_mask" in struct else float("nan")
        ),
        "frac_soma": (
            float(struct["soma_mask"].mean())
            if "soma_mask" in struct else float("nan")
        ),
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
    """Per-patch pipeline gated by a full-image structural mask.

    Computes the structural mask once on the full image, then delegates
    per-patch work to ``generate_blob_pseudolabel`` with the appropriate
    structural slice. Returns a dict mapping filename to ``(H, W)``
    ``uint8`` mask.
    """
    if patch_size is None:
        patch_size = int(records[0]["patch_size"])

    # Single full-image structural mask (Meijering + soma + dilation).
    struct = compute_fullimage_structural_mask(full_image, cfg)
    H_full, W_full = full_image.shape[1:]

    # Slice every 2D ndarray of full-image shape; skip scalars (e.g.
    # ``dendrite_threshold``) and any unexpected entries.
    def _slice_struct(y0: int, x0: int) -> dict:
        sl = (slice(y0, y0 + patch_size), slice(x0, x0 + patch_size))
        return {
            key: arr[sl]
            for key, arr in struct.items()
            if isinstance(arr, np.ndarray)
            and arr.ndim == 2
            and arr.shape == (H_full, W_full)
        }

    result: dict[str, np.ndarray] = {}
    for rec in records:
        ps = int(rec.get("patch_size", patch_size))
        if ps != patch_size:
            raise ValueError(
                f"record {rec['filename']!r} has patch_size={ps}, "
                f"expected {patch_size}"
            )
        y0 = int(rec["grid_row"]) * patch_size
        x0 = int(rec["grid_col"]) * patch_size
        if y0 + patch_size > H_full or x0 + patch_size > W_full:
            raise ValueError(
                f"record {rec['filename']!r} grid position "
                f"({y0}, {x0}) + {patch_size} exceeds image "
                f"({H_full}, {W_full})"
            )
        patch = full_image[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        label, _, _ = generate_blob_pseudolabel(
            patch, cfg, precomputed_struct=_slice_struct(y0, x0),
        )
        result[rec["filename"]] = label

    return result
