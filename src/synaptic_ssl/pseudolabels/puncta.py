"""Puncta pseudo-labels for fluorescence-microscopy patches.

Channels: 0=pre, 1=post, 2=structural. Pipeline: LoG blob detection on pre+post
→ optional annular z-score → render disks → union → restrict to
dendrite (one of three detectors: Meijering ridge, density-skeleton, or
structure-tensor coherence) ∪ soma (intensity) mask → shape filter.
Does NOT enforce pre/post co-localisation; output represents synaptic
*marker* puncta, not synapses.

Annular z-score is a parametric Gaussian approximation of SynQuant's
Wilcoxon SNR test (Wang et al., Bioinformatics 2020). Other building
blocks: Meijering ridge (Meijering et al., Cytometry A 2004), structure
tensor (Bigun & Granlund, ICCV 1987; Weickert, "Anisotropic Diffusion
in Image Processing", 1998), Otsu threshold (IEEE Trans SMC 1979),
scale-normalised LoG (Lindeberg, IJCV 1998), neurite-mask gating
(Fantuzzo et al., eNeuro 2017), white top-hat (Pathak et al., Front
Cell Dev Biol 2025, §2.5.2).

Ref: https://github.com/yu-lab-vt/SynQuant
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.signal import convolve2d
from skimage.draw import disk as draw_disk
from skimage.feature import (
    blob_log,
    structure_tensor,
    structure_tensor_eigenvalues,
)
from skimage.filters import gaussian, meijering, threshold_otsu
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import (
    closing as morph_closing,
    dilation,
    disk as morph_disk,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
    white_tophat,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PunctaCfg:
    """Pseudo-label pipeline configuration.

    Defaults calibrated for 107 nm/px confocal: puncta 2-5 px,
    dendrites ~2-10 px (thin to thick).
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

    # Meijering-based dendrite mask. At 107 nm/px these scales cover
    # dendrite half-widths ~ 1-5 px (full width ~ 2-10 px). 5 scales is
    # enough; range(2, 10) attenuates the thinnest dendrites and pays
    # for 3 extra convolutions.
    dendrite_sigmas: Sequence[float] = field(
        default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0]
    )
    # Meijering ridge threshold. None = per-image Otsu on the response
    # of the *full* image being processed (Otsu, IEEE Trans SMC 1979).
    # Set to a float to override with a fixed global value, e.g. from
    # `compute_global_meijering_threshold`.
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
    # Solidity = area / area(convex_hull). Round/solid blobs ~ 1; branchy
    # dendritic arbors << 0.5 because the convex hull swallows the empty
    # space between branches. Drops CCs with solidity < threshold so a
    # bright dendritic tree that survives the intensity + area filter is
    # not labelled as a soma. Set to 0.0 to disable.
    soma_min_solidity: float = 0.85

    # structural-mask post-processing (dendrites U somas, then dilate)
    structural_dilation: int = 4  # ~ 0.43 um -- "near-neurite" zone

    # Shape priors on the puncta-union mask (pre OR post LoG disks).
    # Same shape-prior form as SynQuant (paraP3D.java: minfill,
    # maxWHratio, area bounds), but with our own values. Default area
    # bounds match the LoG-disk geometry at 107 nm/px (a single LoG
    # disk of max-sigma radius ~ 2.5 px has area ~ 20 px^2), not the
    # physical synapse area; widen + bump log_max_sigma to target the
    # latter.
    min_size: int = 3
    max_size: int = 40
    min_fill: float = 0.5
    max_wh_ratio: float = 4.0

    # per-blob z-score on an annular background (set use_zscore=False to skip).
    # Local-SNR test inspired by SynQuant (Wang et al. 2020): SynQuant uses a
    # Wilcoxon rank-sum statistic against tabulated null moments; this is a
    # parametric Gaussian approximation z = (mu_in - mu_bg) / sigma_bg.
    # Threshold ~ 2 picks puncta clearly above local background; tighten to
    # 3+ for stricter, loosen to 1.5 for friendlier.
    use_zscore: bool = True
    zscore_inner_radius: int = 3
    zscore_outer_radius: int = 8
    zscore_threshold: float = 2.0
    # Annulus statistic. mean+std collapses in crowded fields where
    # neighbouring puncta inflate both mu_bg and sigma_bg. A robust
    # low-percentile + MAD (SExtractor / SynQuant style) ignores bright
    # contaminants in the annulus and gives a real "background" estimate.
    zscore_bg_percentile: int = 0        # 0 = mean; e.g. 25 = use 25th-percentile
    zscore_bg_robust_scale: bool = False  # True = MAD*1.4826 instead of std
    # Absolute-intensity safety floors for the local-SNR test. In dark
    # regions sigma_bg collapses toward zero so any pixel noise gives a
    # huge z; require the raw intensity to clear an absolute floor too.
    # All default to 0.0 (off) for backward compatibility.
    zscore_sigma_bg_floor: float = 0.0   # clamp sigma_bg from below before dividing
    zscore_min_inner: float = 0.0        # require mu_in >= this absolute intensity
    zscore_min_contrast: float = 0.0     # require mu_in - mu_bg >= this absolute delta
    # White-tophat preprocessing radius (px). Flattens slow-varying
    # background so puncta in dense clusters keep contrast against their
    # local annulus -- without it the annulus picks up neighbouring
    # puncta and z-score collapses for obviously bright spots. Choose
    # disk radius >= sqrt(2) * log_max_sigma * 2 (i.e. ~2 puncta widths).
    # 0 = off, otherwise applied to the channel before LoG + z-score.
    intensity_tophat_radius: int = 0

    # "meijering" = Hessian ridge filter; "density" = puncta-density
    # pipeline (smooth + threshold + skeletonise) for structural channels
    # where the neurite signal is a chain of bright spots; "coherence" =
    # structure-tensor orientation-coherence test on the same density field
    # (rejects clusters whose puncta are not linearly arranged).
    dendrite_method: str = "meijering"

    # density-method parameters (only read when dendrite_method=="density")
    density_input: str = "tophat"  # "raw" | "tophat" | "puncta"; see density_response
    density_tophat_radius: int = 15
    # At 107 nm/px, dense puncta spacing is ~3-5 px; sigma~6 bridges them
    # into a continuous density blob.
    density_sigma: float = 6.0
    # None -> per-image Otsu on the smoothed response.
    density_global_threshold: float | None = None
    density_min_cc_area: int = 80
    density_use_skeleton: bool = True
    # Drop large round CCs before skeletonising so the skeleton step does
    # not produce pseudo-lines inside somas; soma_mask is unioned later.
    density_skeleton_soma_min_area: int = 2500
    density_skeleton_soma_max_eccentricity: float = 0.5
    # Attached leaf branches shorter than this are pruned by graph walk.
    # Isolated short components are dropped by density_min_cc_area.
    density_skeleton_prune: int = 10
    # Re-dilation radius for the pruned skeleton. structural_dilation is
    # applied on top by make_structural_mask -> near_structural; the
    # final acceptance radius is the sum of the two.
    density_skeleton_dilate: int = 2

    # Otsu separability guard: refuse the per-image Otsu threshold if
    # the response distribution does not actually separate (low eta) or
    # if "foreground" covers an implausible fraction of the image. Otsu
    # always returns *some* threshold; without this guard pure-noise /
    # uniform-background patches produce a plausible-looking false
    # dendrite mask.
    #   eta = sigma_between / sigma_total of Otsu's split.
    # Set eta_min<=0 and fg_frac_max>=1 to disable.
    density_min_otsu_separability: float = 0.7
    density_max_fg_fraction: float = 0.35

    # ---- coherence-method parameters (only read when dendrite_method=="coherence") ----
    # The density-field builder (density_input, density_tophat_radius)
    # is reused; coherence-specific knobs override only the smoothing
    # scale and add the structure-tensor integration scale + threshold.
    #
    # coherence_density_sigma: small Gaussian on the (raw|tophat|puncta)
    # field. Keep small (~ punctum radius) so per-spot gradients survive
    # for the structure tensor; the density branch's default 6.0 over-
    # smooths blobs into a single ridge and kills the discrimination.
    coherence_density_sigma: float = 2.0
    # coherence_integration_sigma: outer Gaussian inside
    # ``skimage.feature.structure_tensor``. Set to ~2x inter-spot pitch
    # so the tensor sees several consecutive puncta. At 107 nm/px with
    # ~6 px pitch, 10 px integrates across ~3 spots.
    coherence_integration_sigma: float = 10.0
    # Minimum orientation coherence c = (lam1 - lam2) / (lam1 + lam2).
    # c ~ 1 for a linear chain of puncta, c ~ 0 for an isotropic cluster.
    # 0.55 is a balanced default; raise to 0.7+ for stricter linearity.
    coherence_threshold: float = 0.55
    # Drop CCs smaller than this many pixels after the AND of coherence
    # and intensity gates (kills sub-spot noise the structure tensor
    # occasionally accepts at low coherence).
    coherence_min_cc_area: int = 30
    # Post-dilation in pixels applied to the thin coherence-ridge mask
    # before make_structural_mask adds the global structural_dilation.
    coherence_dilate: int = 2


# ---------------------------------------------------------------------------
# LoG blob detection (per channel)
# ---------------------------------------------------------------------------

def detect_puncta_log(
    image: np.ndarray,
    cfg: PunctaCfg,
) -> np.ndarray:
    """Scale-space LoG blob detection on a 2D channel.

    Return ``(N, 3)`` array of ``(row, col, sigma)``. Scale-normalised LoG
    (Lindeberg, IJCV 1998).
    """
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
# Structural mask: dendrite (Meijering) + soma (intensity)
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
) -> float:
    """Pool Meijering responses across patches and return a single threshold.

    Stabilises the threshold against per-patch Otsu failure on neurite-free
    patches.

    Raises ``ValueError`` if no patch produces a non-zero response.
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
        raise ValueError(
            "compute_global_meijering_threshold: no calibration patch "
            "produced a non-zero Meijering response. Returning 0.0 would "
            "flood every patch with false dendrite pixels."
        )
    pooled = np.concatenate(pooled)
    if method == "otsu":
        return float(threshold_otsu(pooled))
    if method == "percentile":
        return float(np.percentile(pooled, percentile))
    raise ValueError(f"unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Density-based dendrite detection
# ---------------------------------------------------------------------------
# For structural channels whose "dendrite" is a chain of dense bright
# puncta. Hessian ridge filters look for `lambda1 ~ 0, lambda2 << 0`
# (ridge), but a punctum has `lambda1 ~ lambda2 << 0` (blob), so they
# match the wrong signal. This pipeline aggregates puncta into a density
# field and reads the neurite shape off the resulting blob.


def _otsu_with_separability(
    smoothed: np.ndarray,
    eta_min: float,
    fg_frac_max: float,
) -> float:
    """Per-image Otsu on the smoothed density response, guarded.

    Returns ``inf`` if the response is degenerate, if Otsu's split has
    eta = sigma_between/sigma_total below ``eta_min`` (weak bimodality
    → unimodal noise), or if the foreground fraction exceeds
    ``fg_frac_max`` ("everything passes" → uniform background). The
    inf sentinel makes any downstream ``response > thr`` mask empty.
    """
    nz = smoothed[smoothed > 0]
    if nz.size <= 1 or float(nz.max() - nz.min()) <= 0:
        return float("inf")
    t = float(threshold_otsu(nz))
    fg = nz[nz > t]
    bg = nz[nz <= t]
    if fg.size == 0 or bg.size == 0:
        return float("inf")
    var_total = float(nz.var())
    if var_total <= 0:
        return float("inf")
    w_fg = fg.size / nz.size
    mean_all = float(nz.mean())
    var_between = (
        w_fg * (float(fg.mean()) - mean_all) ** 2
        + (1.0 - w_fg) * (float(bg.mean()) - mean_all) ** 2
    )
    eta = var_between / var_total
    if eta_min > 0.0 and eta < eta_min:
        return float("inf")
    if fg_frac_max < 1.0:
        fg_frac = float((smoothed > t).mean())
        if fg_frac > fg_frac_max:
            return float("inf")
    return t


def _build_structural_field(
    structural_image: np.ndarray,
    cfg: PunctaCfg,
) -> np.ndarray:
    """Unsmoothed input field shared by density and coherence branches.

    Returns ``raw`` (cast to float32), ``white_tophat`` of radius
    ``cfg.density_tophat_radius``, or a disk map of LoG-detected puncta,
    depending on ``cfg.density_input``.
    """
    if structural_image.ndim != 2:
        raise ValueError(
            f"_build_structural_field expects 2D, got {structural_image.shape}"
        )
    method = cfg.density_input
    img = structural_image.astype(np.float32, copy=False)
    if method == "raw":
        return img
    if method == "tophat":
        radius = max(1, int(cfg.density_tophat_radius))
        return white_tophat(img, morph_disk(radius)).astype(np.float32)
    if method == "puncta":
        blobs = blob_log(
            img,
            min_sigma=cfg.log_min_sigma,
            max_sigma=cfg.log_max_sigma,
            num_sigma=cfg.log_num_sigma,
            threshold=cfg.log_threshold,
            overlap=cfg.log_overlap,
            exclude_border=cfg.log_exclude_border,
        )
        spot_map = np.zeros_like(img, dtype=np.float32)
        for row, col, sigma in blobs:
            radius = max(1, int(round(np.sqrt(2.0) * sigma)))
            rr, cc = draw_disk((int(row), int(col)), radius, shape=img.shape)
            spot_map[rr, cc] = 1.0
        return spot_map
    raise ValueError(
        f"unknown density_input: {method!r}; "
        f"expected 'raw', 'tophat', or 'puncta'"
    )


def density_response(
    structural_image: np.ndarray,
    cfg: PunctaCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a smoothed density field from the structural channel.

    Returns ``(input_field, smoothed_response)``. ``input_field`` is the
    per-``cfg.density_input`` intermediate (raw / white top-hat / LoG-
    puncta disk map); ``smoothed_response`` is its Gaussian.
    """
    field = _build_structural_field(structural_image, cfg)
    sigma = max(1e-3, float(cfg.density_sigma))
    smoothed = gaussian(field, sigma=sigma, preserve_range=True).astype(
        np.float32
    )
    return field, smoothed


def _drop_soma_like_ccs(
    mask: np.ndarray,
    min_area: int,
    max_eccentricity: float,
) -> np.ndarray:
    """Drop CCs that are both large and round.

    Used before skeletonisation so the skeleton step does not produce
    pseudo-lines inside somas; the real soma_mask is unioned later.
    """
    if not mask.any():
        return mask.astype(bool, copy=True)
    lbl = cc_label(mask, connectivity=2)
    out = np.asarray(mask, dtype=bool).copy()
    for prop in regionprops(lbl):
        if prop.area >= min_area and prop.eccentricity <= max_eccentricity:
            out[lbl == prop.label] = False
    return out


# 3x3 neighbour-count kernel (centre excluded).
_NEIGHBOUR_KERNEL = np.array(
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
)


def _skel_neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Per-pixel skeleton-neighbour count (0 outside the skeleton)."""
    counts = convolve2d(
        skel.astype(np.uint8), _NEIGHBOUR_KERNEL, mode="same", boundary="fill"
    )
    out = np.zeros_like(counts, dtype=np.int8)
    out[skel] = counts[skel]
    return out


def _walk_branch(
    skel: np.ndarray,
    nbr: np.ndarray,
    start: Tuple[int, int],
    max_len: int,
) -> List[Tuple[int, int]]:
    """Walk a skeleton branch from an endpoint until junction / max_len.

    Returns the branch pixels (endpoint first, excludes the junction).
    Stops at ``max_len`` pixels: any longer doesn't change the prune
    decision.
    """
    H, W = skel.shape
    path: List[Tuple[int, int]] = [start]
    prev: Tuple[int, int] | None = None
    current = start
    while len(path) < max_len:
        r, c = current
        next_pixel: Tuple[int, int] | None = None
        n_neighbours = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if not skel[nr, nc]:
                    continue
                if prev is not None and (nr, nc) == prev:
                    continue
                n_neighbours += 1
                next_pixel = (nr, nc)
        if next_pixel is None or n_neighbours != 1:
            # End of chain, or junction reached (>=2 forward neighbours).
            break
        if nbr[next_pixel] >= 3:
            break  # next pixel is itself a junction
        prev = current
        current = next_pixel
        path.append(current)
    return path


def _prune_skeleton_branches(
    skel: np.ndarray,
    max_len: int,
) -> np.ndarray:
    """Iteratively remove leaf branches shorter than ``max_len`` px.

    For each endpoint (skeleton pixel with exactly one skeleton
    neighbour), walk to the first junction and drop the branch if its
    length is < ``max_len``. Iterates because removing one leaf can
    expose a new endpoint on the same tree.
    """
    if max_len <= 0 or not skel.any():
        return skel.astype(bool, copy=True)
    work = np.asarray(skel, dtype=bool).copy()
    while True:
        nbr = _skel_neighbour_count(work)
        endpoints = np.argwhere((nbr == 1))
        if endpoints.size == 0:
            break
        removed_any = False
        for r, c in endpoints:
            if not work[r, c]:
                continue  # already removed via another endpoint this pass
            path = _walk_branch(work, nbr, (int(r), int(c)), max_len)
            if len(path) < max_len:
                for pr, pc in path:
                    work[pr, pc] = False
                removed_any = True
            # else: branch is long enough, keep it
        if not removed_any:
            break
    return work


def make_density_dendrite_mask(
    structural_image: np.ndarray,
    cfg: PunctaCfg,
) -> dict:
    """Density-based dendrite mask for punctate structural channels.

    Threshold is ``cfg.density_global_threshold`` if set, else a guarded
    per-image Otsu (see ``_otsu_with_separability``). Returns a dict with
    keys ``density_input_field``, ``density_response``,
    ``dendrite_threshold``, ``raw_density_mask`` (post-threshold +
    small-CC filter, pre-skeleton), ``line_candidates`` (post-soma-drop),
    ``skeleton``, ``pruned_skeleton``, and ``dendrite_mask`` (final). All
    intermediate masks are present even when empty so callers can log
    where a collapse happens.
    """
    input_field, response = density_response(structural_image, cfg)

    if cfg.density_global_threshold is not None:
        threshold = float(cfg.density_global_threshold)
    else:
        threshold = _otsu_with_separability(
            response,
            eta_min=float(cfg.density_min_otsu_separability),
            fg_frac_max=float(cfg.density_max_fg_fraction),
        )

    raw_mask = response > threshold
    if raw_mask.any():
        # skimage >=0.26: max_size=N removes area <= N (was: min_size=N
        # removed area < N). -1 preserves the strict-less-than semantics.
        raw_mask = remove_small_objects(
            raw_mask, max_size=max(0, int(cfg.density_min_cc_area) - 1)
        )

    line_candidates: np.ndarray | None = None
    skel: np.ndarray | None = None
    pruned: np.ndarray | None = None
    if not cfg.density_use_skeleton:
        dendrite_mask = raw_mask
    elif not raw_mask.any():
        dendrite_mask = raw_mask
    else:
        line_candidates = _drop_soma_like_ccs(
            raw_mask,
            min_area=int(cfg.density_skeleton_soma_min_area),
            max_eccentricity=float(cfg.density_skeleton_soma_max_eccentricity),
        )
        if line_candidates.any():
            skel = skeletonize(line_candidates)
            pruned = _prune_skeleton_branches(
                skel, max_len=int(cfg.density_skeleton_prune)
            )
            dilate_r = max(0, int(cfg.density_skeleton_dilate))
            if dilate_r > 0 and pruned.any():
                dendrite_mask = dilation(pruned, morph_disk(dilate_r))
            else:
                dendrite_mask = pruned
        else:
            dendrite_mask = np.zeros_like(raw_mask, dtype=bool)

    zero = np.zeros_like(raw_mask, dtype=bool)
    return {
        "density_input_field": input_field,
        "density_response": response,
        "dendrite_threshold": threshold,
        "raw_density_mask": raw_mask.astype(bool, copy=False),
        "line_candidates": (line_candidates if line_candidates is not None else zero).astype(bool, copy=False),
        "skeleton": (skel if skel is not None else zero).astype(bool, copy=False),
        "pruned_skeleton": (pruned if pruned is not None else zero).astype(bool, copy=False),
        "dendrite_mask": dendrite_mask.astype(bool, copy=False),
    }


# ---------------------------------------------------------------------------
# Coherence-based dendrite detection
# ---------------------------------------------------------------------------
# Discriminates "puncta arranged along a line" from "puncta clumped in a
# blob" using the 2D structure tensor (Bigun & Granlund, ICCV 1987;
# Weickert, "Anisotropic Diffusion in Image Processing", 1998). Same
# input field as the density branch (raw / tophat / LoG-puncta disks),
# but instead of smoothing into a blob and skeletonising, we read off
# the local orientation coherence:
#
#     c(x) = (lam1 - lam2) / (lam1 + lam2)    in [0, 1]
#
# where lam1 >= lam2 are the eigenvalues of the Gaussian-windowed
# gradient outer-product (the structure tensor). A linear chain of
# puncta has a dominant gradient direction (perpendicular to the chain)
# so c -> 1; an isotropic cluster has gradients in every direction so
# c -> 0. The final mask is (c > coherence_threshold) AND (smoothed
# field > guarded-Otsu intensity threshold), with small-CC removal and
# a thin dilation. No skeletonisation: the coherence test already
# concentrates response along dendrite axes.


def coherence_response(
    structural_image: np.ndarray,
    cfg: PunctaCfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the smoothed field + per-pixel orientation coherence map.

    Returns ``(input_field, smoothed_field, coherence_map)``.

    The smoothed field uses ``cfg.coherence_density_sigma`` (small, so
    per-spot gradients survive) rather than ``cfg.density_sigma``. The
    structure tensor is integrated with ``cfg.coherence_integration_sigma``.
    """
    field = _build_structural_field(structural_image, cfg)
    sigma_in = max(1e-3, float(cfg.coherence_density_sigma))
    smoothed = gaussian(field, sigma=sigma_in, preserve_range=True).astype(
        np.float32
    )

    sigma_int = max(1e-3, float(cfg.coherence_integration_sigma))
    arr, arc, acc = structure_tensor(
        smoothed, sigma=sigma_int, mode="reflect", order="rc"
    )
    eigs = structure_tensor_eigenvalues([arr, arc, acc])
    lam1, lam2 = eigs[0], eigs[1]
    denom = lam1 + lam2
    # c is well-defined only where the gradient energy is non-trivial.
    # numpy.true_divide with where= avoids RuntimeWarning and yields 0
    # at degenerate pixels (which the intensity gate will reject anyway).
    coherence = np.zeros_like(denom, dtype=np.float32)
    np.divide(
        lam1 - lam2, denom + 1e-8, out=coherence,
        where=denom > 0,
    )
    return field, smoothed, coherence


def make_coherence_dendrite_mask(
    structural_image: np.ndarray,
    cfg: PunctaCfg,
) -> dict:
    """Orientation-coherence dendrite mask for punctate structural channels.

    Keeps pixels whose smoothed field exceeds a guarded-Otsu threshold
    AND whose structure-tensor coherence exceeds ``cfg.coherence_threshold``;
    drops sub-``cfg.coherence_min_cc_area`` components and optionally
    dilates by ``cfg.coherence_dilate`` pixels. Returns a dict with keys
    ``coherence_input_field``, ``coherence_smoothed``,
    ``coherence_map``, ``coherence_intensity_threshold``,
    ``coherence_intensity_mask``, ``raw_coherence_mask``,
    ``dendrite_threshold`` (= ``cfg.coherence_threshold``) and
    ``dendrite_mask`` (final). All intermediate masks are present even
    when empty so callers can log where a collapse happens.
    """
    field, smoothed, coherence = coherence_response(structural_image, cfg)

    intensity_thr = _otsu_with_separability(
        smoothed,
        eta_min=float(cfg.density_min_otsu_separability),
        fg_frac_max=float(cfg.density_max_fg_fraction),
    )
    intensity_mask = smoothed > intensity_thr

    raw_mask = intensity_mask & (coherence > float(cfg.coherence_threshold))
    if raw_mask.any():
        raw_mask = remove_small_objects(
            raw_mask, max_size=max(0, int(cfg.coherence_min_cc_area) - 1)
        )

    dilate_r = max(0, int(cfg.coherence_dilate))
    if dilate_r > 0 and raw_mask.any():
        dendrite_mask = dilation(raw_mask, morph_disk(dilate_r))
    else:
        dendrite_mask = raw_mask

    return {
        "coherence_input_field": field,
        "coherence_smoothed": smoothed,
        "coherence_map": coherence,
        "coherence_intensity_threshold": intensity_thr,
        "coherence_intensity_mask": intensity_mask.astype(bool, copy=False),
        "raw_coherence_mask": raw_mask.astype(bool, copy=False),
        "dendrite_threshold": float(cfg.coherence_threshold),
        "dendrite_mask": dendrite_mask.astype(bool, copy=False),
    }


def make_soma_mask(
    structural_image: np.ndarray,
    intensity_percentile: float = 99.0,
    min_area: int = 200,
    closing_radius: int = 0,
    fill_holes: bool = False,
    min_solidity: float = 0.0,
) -> np.ndarray:
    """Soma mask from high-intensity connected components in the structural channel.

    Recovers bright solid somas that Meijering misses. Keep CCs with
    area ≥ ``min_area`` after optional closing and hole-fill. When
    ``min_solidity > 0`` also drop CCs whose ``area / convex_hull_area``
    is below the threshold (branchy / dendritic arbors).
    """
    if structural_image.ndim != 2:
        raise ValueError("make_soma_mask expects 2D")
    thr = float(np.percentile(structural_image, intensity_percentile))
    bright = structural_image > thr
    if not bright.any():
        return np.zeros_like(bright, dtype=bool)
    if closing_radius > 0:
        bright = morph_closing(bright, morph_disk(closing_radius))
    bright = remove_small_objects(bright, max_size=max(0, int(min_area) - 1))
    if fill_holes and bright.any():
        bright = remove_small_holes(bright, max_size=max(0, int(min_area) - 1))
    if min_solidity > 0.0 and bright.any():
        lbl = cc_label(bright, connectivity=2)
        out = np.zeros_like(bright, dtype=bool)
        for prop in regionprops(lbl):
            if prop.solidity >= min_solidity:
                out[lbl == prop.label] = True
        bright = out
    return bright


def make_structural_mask(
    image: np.ndarray,
    cfg: PunctaCfg,
) -> dict:
    """Build dendrite ∪ soma mask plus dilated near-neuron zone from ``(C, H, W)``.

    Dispatches on ``cfg.dendrite_method`` (``"meijering"``, ``"density"``,
    or ``"coherence"``).

    The returned dict always contains ``dendrite_response``,
    ``dendrite_threshold``, ``dendrite_mask``, ``soma_mask``,
    ``structural_mask`` and ``near_structural``. The density branch also
    exposes ``density_input_field`` and ``raw_density_mask``; the
    coherence branch exposes ``coherence_input_field``,
    ``coherence_smoothed``, ``coherence_map``,
    ``coherence_intensity_threshold``, ``coherence_intensity_mask`` and
    ``raw_coherence_mask``.
    """
    struct_raw = image[cfg.structural_channel]
    method = cfg.dendrite_method

    if method == "meijering":
        response = meijering_response(struct_raw, cfg.dendrite_sigmas)
        if cfg.dendrite_threshold is None:
            nz = response[response > 0]
            # Empty / degenerate response -> Otsu undefined; refuse to
            # flag any pixel as dendrite (np.inf > nothing).
            if nz.size > 1 and float(nz.max() - nz.min()) > 0:
                threshold = float(threshold_otsu(nz))
            else:
                threshold = float("inf")
        else:
            threshold = float(cfg.dendrite_threshold)
        dendrite = response > threshold
        method_extras = {"dendrite_response": response}
    elif method == "density":
        dens = make_density_dendrite_mask(struct_raw, cfg)
        dendrite = dens["dendrite_mask"]
        threshold = dens["dendrite_threshold"]
        method_extras = {
            "density_input_field": dens["density_input_field"],
            "density_response": dens["density_response"],
            "raw_density_mask": dens["raw_density_mask"],
            "dendrite_response": dens["density_response"],
        }
    elif method == "coherence":
        coh = make_coherence_dendrite_mask(struct_raw, cfg)
        dendrite = coh["dendrite_mask"]
        threshold = coh["dendrite_threshold"]
        method_extras = {
            "coherence_input_field": coh["coherence_input_field"],
            "coherence_smoothed": coh["coherence_smoothed"],
            "coherence_map": coh["coherence_map"],
            "coherence_intensity_threshold": coh["coherence_intensity_threshold"],
            "coherence_intensity_mask": coh["coherence_intensity_mask"],
            "raw_coherence_mask": coh["raw_coherence_mask"],
            "dendrite_response": coh["coherence_map"],
        }
    else:
        raise ValueError(
            f"unknown dendrite_method: {method!r}; "
            f"expected 'meijering', 'density', or 'coherence'"
        )

    soma = make_soma_mask(
        struct_raw,
        intensity_percentile=cfg.soma_intensity_percentile,
        min_area=cfg.soma_min_area,
        closing_radius=cfg.soma_closing_radius,
        fill_holes=cfg.soma_fill_holes,
        min_solidity=cfg.soma_min_solidity,
    )
    structural = dendrite | soma
    near = dilation(structural, morph_disk(cfg.structural_dilation))
    return {
        "dendrite_threshold": threshold,
        "dendrite_mask": dendrite,
        "soma_mask": soma,
        "structural_mask": structural,
        "near_structural": near,
        **method_extras,
    }


# ---------------------------------------------------------------------------
# Per-blob z-score (annular-background local-SNR test)
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
    """Annular-background z-score for one LoG blob (local-SNR test).

    Same local-SNR principle as SynQuant (Wang et al. 2020) but parametric:
    ``z = (mu_in - mu_bg) / sigma_bg``. SynQuant itself uses a Wilcoxon
    rank-sum statistic against tabulated null moments.

    ``sigma_bg_floor`` clamps ``sigma_bg`` from below before dividing, so
    near-uniform dark regions can't blow z up by collapsing the denominator.

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
        "mu_in": mu_in, "mu_bg": mu_bg, "sigma_bg": sigma_bg,
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
    (``zscore_min_inner``, ``zscore_min_contrast``). The floors default to
    0.0 (off) and exist to stop the dark-void z-score blow-up where a
    tiny ``sigma_bg`` would otherwise let any pixel noise pass.
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
                and (rec["mu_in"] - rec["mu_bg"]) >= cfg.zscore_min_contrast
            )
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Shape / size filter
# ---------------------------------------------------------------------------

def filter_by_size_shape(
    mask: np.ndarray,
    cfg: PunctaCfg,
) -> np.ndarray:
    """Keep CCs passing area, bbox aspect ratio, and bbox fill bounds.

    Same shape-prior form as SynQuant (Wang et al. 2020, ``paraP3D.java``:
    ``minfill``, ``maxWHratio``, min/max area), but with our own values.
    Default area range targets the LoG-disk geometry at 107 nm/px (a
    single LoG disk of max-sigma radius ~ 2.5 px has area ~ 20 px^2),
    not physical synapse area -- widen the bounds + bump log_max_sigma
    to target the latter.
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
# End-to-end orchestrator on a single (C, H, W) patch
# ---------------------------------------------------------------------------

def generate_puncta_pseudolabel(
    patch: np.ndarray,
    cfg: PunctaCfg,
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

    # Structural mask: prefer a precomputed full-image slice to avoid
    # per-patch border artefacts (Meijering / LoG / annular window).
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

    pre_puncta = detect_puncta_log(patch[cfg.pre_channel], cfg)
    post_puncta = detect_puncta_log(patch[cfg.post_channel], cfg)

    if cfg.use_zscore:
        pre_scored = score_puncta_zscore(patch[cfg.pre_channel], pre_puncta, cfg)
        post_scored = score_puncta_zscore(patch[cfg.post_channel], post_puncta, cfg)
        pre_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in pre_scored if s["kept"]]
        ).reshape(-1, 3)
        post_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in post_scored if s["kept"]]
        ).reshape(-1, 3)
    else:
        pre_scored, post_scored = [], []
        pre_kept, post_kept = pre_puncta, post_puncta

    pre_mask = puncta_to_mask(pre_kept, (H, W))
    post_mask = puncta_to_mask(post_kept, (H, W))

    # Union without co-localisation: output represents synaptic-marker
    # puncta, not strictly co-localised synapses.
    puncta_mask = (pre_mask.astype(bool) | post_mask.astype(bool)).astype(np.uint8)

    shaped = filter_by_size_shape(puncta_mask, cfg)
    label_mask = (shaped.astype(bool) & struct["near_structural"]).astype(np.uint8)

    intermediates = {
        **struct,
        "pre_blobs_raw": pre_puncta,
        "post_blobs_raw": post_puncta,
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
        "n_pre_log": int(len(pre_puncta)),
        "n_post_log": int(len(post_puncta)),
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
# Full-image mode: structural mask on the full image, LoG per patch
# ---------------------------------------------------------------------------

def compute_fullimage_structural_mask(
    full_image: np.ndarray,
    cfg: PunctaCfg,
) -> dict:
    """Run the structural pipeline (cfg.dendrite_method) on a full ``(C, H, W)`` image.

    Avoids per-patch border artifacts and fragmented cross-boundary somas.
    """
    return make_structural_mask(full_image, cfg)


def generate_pseudolabels_fullimage(
    full_image: np.ndarray,
    records: list[dict],
    cfg: PunctaCfg,
    patch_size: int | None = None,
) -> Tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Full-image pipeline: LoG + z-score + render + shape + gate at full
    scale, then slice per patch.

    Per-patch LoG with ``exclude_border > 0`` would create a dead zone
    at every tile seam (~15 % of a 128² patch at ``exclude_border=5``);
    the z-score annular window would also get truncated at patch
    borders. Running everything at full image scale and slicing the
    final mask kills both biases.

    Returns ``(labels, stats)`` where:
      * ``labels`` maps filename -> ``(H, W) uint8`` patch mask.
      * ``stats`` maps filename -> per-patch metrics dict (counts of
        kept blobs whose centre falls inside the patch, label coverage,
        and structural fractions sliced from the full-image mask).
    """
    if not records:
        return {}, {}
    if patch_size is None:
        patch_size = int(records[0]["patch_size"])

    C, H_full, W_full = full_image.shape
    if C <= max(cfg.pre_channel, cfg.post_channel, cfg.structural_channel):
        raise ValueError(
            f"full_image has {C} channels but cfg references "
            f"pre={cfg.pre_channel}, post={cfg.post_channel}, "
            f"structural={cfg.structural_channel}"
        )

    struct = compute_fullimage_structural_mask(full_image, cfg)

    # Full-image LoG: exclude_border now applies only at real image
    # edges, not at every tile seam.
    pre_puncta = detect_puncta_log(full_image[cfg.pre_channel], cfg)
    post_puncta = detect_puncta_log(full_image[cfg.post_channel], cfg)

    # Full-image z-score: the annular window stays local to each blob
    # and is cropped by ``_local_window`` at real image edges.
    if cfg.use_zscore:
        pre_scored = score_puncta_zscore(full_image[cfg.pre_channel], pre_puncta, cfg)
        post_scored = score_puncta_zscore(full_image[cfg.post_channel], post_puncta, cfg)
        pre_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in pre_scored if s["kept"]]
        ).reshape(-1, 3)
        post_kept = np.array(
            [[s["row"], s["col"], s["sigma"]] for s in post_scored if s["kept"]]
        ).reshape(-1, 3)
    else:
        pre_kept, post_kept = pre_puncta, post_puncta

    # Render, union, shape-filter, structural gate -- all at full scale.
    pre_mask_full = puncta_to_mask(pre_kept, (H_full, W_full))
    post_mask_full = puncta_to_mask(post_kept, (H_full, W_full))
    puncta_full = (pre_mask_full.astype(bool) | post_mask_full.astype(bool)).astype(np.uint8)
    shaped_full = filter_by_size_shape(puncta_full, cfg)
    label_full = (shaped_full.astype(bool) & struct["near_structural"]).astype(np.uint8)

    def _puncta_in_patch(arr: np.ndarray, y0: int, x0: int, ps: int) -> int:
        if arr.shape[0] == 0:
            return 0
        r = arr[:, 0]; c = arr[:, 1]
        return int(((r >= y0) & (r < y0 + ps) & (c >= x0) & (c < x0 + ps)).sum())

    labels: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
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
        sl = (slice(y0, y0 + patch_size), slice(x0, x0 + patch_size))
        lbl = label_full[sl].copy()
        labels[rec["filename"]] = lbl
        stats[rec["filename"]] = {
            "n_pre_kept": _puncta_in_patch(pre_kept, y0, x0, patch_size),
            "n_post_kept": _puncta_in_patch(post_kept, y0, x0, patch_size),
            "px_puncta": int(puncta_full[sl].sum()),
            "px_shaped": int(shaped_full[sl].sum()),
            "px_label": int(lbl.sum()),
            "frac_label": float(lbl.mean()),
            "frac_dendrite": float(struct["dendrite_mask"][sl].mean()),
            "frac_soma": float(struct["soma_mask"][sl].mean()),
            "frac_near_structural": float(struct["near_structural"][sl].mean()),
        }

    return labels, stats


# ---------------------------------------------------------------------------
# Per-channel detection helpers for the live (puncta) pipeline.
# Visualisation lives in ``pseudolabels.viz``.
# ---------------------------------------------------------------------------


def derive_zscore_floors(detected: np.ndarray) -> Tuple[float, float, float]:
    """Per-image safety floors from the (already tophatted) image bg.

    Returns ``(sigma_bg_floor, min_inner, min_contrast)``. ``bg`` = pixels
    at or below the 30th percentile; ``sigma`` = ``1.4826 * MAD(bg)``;
    floors = ``(sigma, median(bg) + 4*sigma, 4*sigma)``. Returns
    ``(0, 0, 0)`` (floors disabled) on degenerate input.

    Wire into a ``PunctaCfg`` via ``dataclasses.replace(cfg,
    zscore_sigma_bg_floor=sg, zscore_min_inner=mi, zscore_min_contrast=mc)``.
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
      * ``scored`` — full ``score_puncta_zscore`` records (one per raw blob),
        each carrying ``z, mu_in, mu_bg, sigma_bg, kept, ...``. Empty
        list if z-scoring is off or no candidates.
      * ``kept`` — subset of ``raw`` that survived z + floor gating.
      * ``cfg_eff`` — cfg actually used (with per-image floors patched in
        when ``auto_floors=True``); pass this to the viz helpers so
        titles reflect the real thresholds.
    """
    import dataclasses

    r = int(cfg.intensity_tophat_radius)
    det = white_tophat(img2d, footprint=morph_disk(r)) if r > 0 else img2d
    if auto_floors:
        sg, mi, mc = derive_zscore_floors(det)
        cfg_eff = dataclasses.replace(
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