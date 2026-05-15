"""Iterative pseudo-label refinement (2D DDeep3M+ adaptation).

2D adaptation of Xiao et al., "DDeep3M+: adaptive enhancement powered
weakly supervised learning for neuron segmentation", Neurophotonics
10(3), 035003, 2023 (PMC10289179). This module covers steps 3 (region
growing) and 4 (image-probability fusion). Step 1 (blob detection)
lives in ``pseudolabels.blobs``; step 2 is the SwinUNETR training loop.

Deviations from upstream:
* 2D, not 3D. 5x5 ring (24 px) replaces the paper's 5x5x5 - centre
  124-voxel ring; 8-connected dilation replaces 26-neighbour growth.
* Channel-selective fusion. Only the synaptic pre/post channels are
  fused; the structural channel is left raw to avoid feedback into
  the anatomical prior.
* Fusion formula reading. We implement the paper's intent
  ``F = w_raw * 1[P>d] * I + w_prob * I_M * P`` with two named weights
  (the published Eq. 7 squares ``I`` due to an apparent typo in the
  ``Theta`` definition).
* Stopping criterion. Without ground truth, mean per-patch IoU between
  consecutive rounds replaces the paper's held-out F1 plateau.

Ref: https://github.com/cakuba/DDeep3m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class RefineCfg:
    """One DDeep3M+ refinement step's configuration.

    Region-growing knobs follow Eq. 5-6 adapted to 2D. Fusion knobs
    follow Eq. 7 with the reading documented in the module docstring.
    """

    # Region growing (DDeep3M+ step 3) ---------------------------------

    # Ring radius (in pixels) used to estimate the adaptive growth
    # threshold rho. paper: 5x5x5 - centre = 124 voxels. 2D analogue
    # with radius=2 gives a 5x5 - centre = 24-pixel ring.
    ring_radius: int = 2

    # 8-connected growth neighbourhood. Set to 4 for 4-connectivity.
    growth_connectivity: int = 8

    # Cap on the number of growth iterations inside a single call to
    # region_grow_from_prob. paper does not specify; 10 is plenty for
    # 128x128 patches.
    max_grow_iters: int = 10

    # Hard floor on the adaptive threshold rho. Without this, an empty
    # or near-empty seed mask can produce rho ~ 0 and the whole patch
    # is swallowed in a single step.
    rho_min: float = 0.30

    # Cap on rho. Past 0.95 the network is already very confident and
    # growth degenerates to its native prediction; let blob_new
    # introduce new seeds instead.
    rho_max: float = 0.95

    # Hard cap on grown area as a multiple of the seed area. Stops
    # runaway expansion when prob_map is noisy on a tiny seed.
    max_growth_ratio: float = 5.0

    # ANDing the grown mask with ``near_structural`` (i.e. the dilated
    # dendrite + soma mask) before returning. Strongly recommended for
    # biological data where synapses live near neurites; prevents
    # multi-iteration drift of the mask outside anatomical priors.
    gate_growth_by_structural: bool = True

    # Image-probability fusion (DDeep3M+ step 4) -----------------------

    # Probability threshold below which the raw-image term is gated to
    # zero. paper writes delta=2 on a [0,255] image; on normalised
    # [0,1] inputs the natural value is around 0.30 (i.e. only keep
    # raw intensity where the network is mildly confident).
    prob_threshold: float = 0.30

    # Weight on the raw-intensity term (DDeep3M+ Eq. 7's a/(a+b) factor
    # with a=0.8, b=0.2).
    weight_raw: float = 0.80

    # Weight on the probability-reinforcement term. paper uses
    # ``b/(a+b) * (1 - a)`` = 0.04. Keep this as a single combined
    # knob; default reproduces 0.20 (the b/(a+b) factor alone) which
    # is the natural reading when ``(1-a)`` is treated as a typo.
    weight_prob: float = 0.20

    # Channels to fuse (indices into the (C, H, W) patch). Defaults to
    # pre + post synaptic channels; structural/neurite channel index 2
    # is left raw to avoid a feedback loop into the anatomical prior.
    fuse_channels: tuple[int, ...] = (0, 1)

    # I_M cap for the probability-reinforcement term. None -> compute
    # from the image (max over fused channels). For normalised inputs
    # in [0, 1] you can pin this to 1.0 to make the fusion behaviour
    # patch-invariant.
    intensity_max: float | None = 1.0


# ---------------------------------------------------------------------------
# Step 3: region growing
# ---------------------------------------------------------------------------

def _ring_around(mask: np.ndarray, radius: int) -> np.ndarray:
    """Return the pixels within ``radius`` of ``mask`` but outside it."""
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    dilated = binary_dilation(mask, structure=struct)
    return dilated & ~mask


def _growth_kernel(connectivity: int) -> np.ndarray:
    """3x3 boolean kernel for 4- or 8-connected dilation."""
    if connectivity == 8:
        return np.ones((3, 3), dtype=bool)
    if connectivity == 4:
        k = np.zeros((3, 3), dtype=bool)
        k[1, :] = True
        k[:, 1] = True
        return k
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def region_grow_from_prob(
    seed_mask: np.ndarray,
    prob_map: np.ndarray,
    cfg: RefineCfg,
    structural_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grow ``seed_mask`` from ``prob_map`` (2D DDeep3M+ Eq. 5-6).

    Adaptive threshold ``rho`` is the mean ``prob_map`` value in the ring
    around the current grown region; new pixels in the
    ``growth_connectivity`` neighbourhood with ``prob > rho`` are added.
    Iterate until no new pixels, ``max_grow_iters``, or
    ``max_growth_ratio * seed_area`` is reached.

    Pass ``structural_mask`` (dilated dendrite + soma) when
    ``cfg.gate_growth_by_structural=True`` to constrain growth to
    anatomically plausible regions.

    Parameters
    ----------
    seed_mask : (H, W) bool or 0/1 uint8.
    prob_map : (H, W) in ``[0, 1]``.
    cfg : RefineCfg
    structural_mask : (H, W) array, optional. Required when
        ``cfg.gate_growth_by_structural=True``.

    Returns
    -------
    (H, W) uint8.
    """
    if seed_mask.shape != prob_map.shape:
        raise ValueError(
            f"shape mismatch: seed {seed_mask.shape} vs prob {prob_map.shape}"
        )
    if seed_mask.ndim != 2:
        raise ValueError(f"region_grow_from_prob expects 2D, got {seed_mask.shape}")

    seed = seed_mask.astype(bool)
    seed_area = int(seed.sum())

    # Empty-seed edge case: nothing to grow from. Step 1 (blob_new) will
    # introduce fresh seeds in the next iteration.
    if seed_area == 0:
        return np.zeros_like(seed_mask, dtype=np.uint8)

    # Pre-compute the structural gate once (binary AND each step).
    if cfg.gate_growth_by_structural:
        if structural_mask is None:
            raise ValueError(
                "gate_growth_by_structural=True requires structural_mask"
            )
        if structural_mask.shape != seed.shape:
            raise ValueError(
                f"structural_mask shape {structural_mask.shape} != "
                f"seed shape {seed.shape}"
            )
        struct_gate = structural_mask.astype(bool)
    else:
        struct_gate = None

    grown = seed.copy()
    kernel = _growth_kernel(cfg.growth_connectivity)

    for _ in range(cfg.max_grow_iters):
        ring = _ring_around(grown, cfg.ring_radius)
        if not ring.any():
            break
        rho = float(prob_map[ring].mean())
        rho = float(np.clip(rho, cfg.rho_min, cfg.rho_max))

        candidates = binary_dilation(grown, structure=kernel) & ~grown
        addable = candidates & (prob_map > rho)
        if struct_gate is not None:
            addable &= struct_gate
        if not addable.any():
            break

        grown |= addable

        if grown.sum() > cfg.max_growth_ratio * seed_area:
            break

    if struct_gate is not None:
        grown &= struct_gate

    return grown.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 4: image-probability fusion
# ---------------------------------------------------------------------------

def fuse_image_with_prob(
    image: np.ndarray,
    prob_map: np.ndarray,
    cfg: RefineCfg,
) -> np.ndarray:
    """Fuse raw image with probability map (2D DDeep3M+ Eq. 7).

    For each channel in ``cfg.fuse_channels``::

        F(x) = weight_raw * 1[P(x) > prob_threshold] * I(x)
             + weight_prob * I_M * P(x)

    Other channels pass through unchanged. Output is clipped to
    ``[0, max(image.max(), I_M)]``. ``image`` is not modified.

    Parameters
    ----------
    image : (C, H, W) float32, expected ``[0, 1]``.
    prob_map : (H, W) in ``[0, 1]``.
    cfg : RefineCfg

    Returns
    -------
    (C, H, W) float32.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be (C, H, W); got {image.shape}")
    if prob_map.ndim != 2 or prob_map.shape != image.shape[1:]:
        raise ValueError(
            f"prob_map shape {prob_map.shape} != image spatial {image.shape[1:]}"
        )

    fused = image.astype(np.float32, copy=True)
    if not cfg.fuse_channels:
        return fused

    gate = (prob_map > cfg.prob_threshold).astype(np.float32)
    intensity_max = (
        float(cfg.intensity_max)
        if cfg.intensity_max is not None
        else float(image[list(cfg.fuse_channels)].max() or 1.0)
    )

    reinforcement = cfg.weight_prob * intensity_max * prob_map.astype(np.float32)
    clip_high = max(float(image.max()), intensity_max)

    for c in cfg.fuse_channels:
        if c < 0 or c >= image.shape[0]:
            raise ValueError(
                f"fuse_channels index {c} out of range for image with "
                f"{image.shape[0]} channels"
            )
        raw_term = cfg.weight_raw * gate * image[c].astype(np.float32)
        fused[c] = np.clip(raw_term + reinforcement, 0.0, clip_high)

    return fused


# ---------------------------------------------------------------------------
# Convergence / diagnostic metrics
# ---------------------------------------------------------------------------

def compute_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU between two same-shape masks. Empty-vs-empty → 1.0."""
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    if union == 0:
        return 1.0
    return inter / union


def compute_positive_fraction(mask: np.ndarray) -> float:
    """Mean of a binary mask. Empty → 0.0."""
    if mask.size == 0:
        return 0.0
    return float(mask.astype(bool).mean())
