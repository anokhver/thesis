"""Iterative pseudo-label refinement (2D DDeep3M+).

2D adaptation of Xiao et al., "DDeep3M+: adaptive enhancement powered
weakly supervised learning for neuron segmentation", Neurophotonics
10(3), 035003, 2023 (PMC10289179). Covers steps 3 (region growing) and
4 (image-probability fusion); step 1 is in ``pseudolabels.puncta``,
step 2 is the SwinUNETR training loop.

Faithful behaviour
------------------

Step 3 seed source (§3.3 step 1): "According to the predicted
probability map, the algorithm with maximum probability classification
is used to generate the seed region of neurite." We implement this in
``seed_from_prob_map``: the seed is a hard threshold on ``prob_map`` (in
[0, 1]) at ``cfg.seed_prob_threshold`` (default 0.5, the natural
max-prob argmax of a binary sigmoid head). ``region_grow_from_prob``
derives ``seed`` from the prob map by default (paper-faithful);
passing an explicit ``seed_mask`` is an opt-in deviation.

Step 3 ring + grown neighbourhoods (§3.3 Eqs. 5-6): paper uses a
``N'(v_0)`` = 124-voxel ring (5x5x5 - centre, 3D) for ``rho`` and
``N(v_0)`` = 8-voxel grown neighbourhood (paper text). We collapse to
2D: ``ring_radius=2`` -> 5x5 - centre = 24 ring pixels, and
``growth_connectivity=8`` -> 3x3 - centre = 8 grown neighbours.
``rho`` is the mean of ``prob_map`` in that ring; pixels with
``prob > rho`` are added.

Step 4 fusion (§3.4 Eq. 7)::

    F(x) = a/(a+b) * Theta(I(x)-d) * I(x)
         + b/(a+b) * (1 - a) * I_M * P(x)

with a=0.8, b=0.2, d=2 on [0, 255] inputs. ``Theta(I(x)-d)`` is
defined in the paper as ``I(x)`` when ``P(x) > d`` and 0 otherwise
(despite the symbol's argument being I, the gate is on P; we follow
the paper's prose). On normalised [0, 1] inputs we use
``d = 2/255 ~= 0.00784``. The ``(1 - a)`` factor is explicit in the
paper, not a typo, so the natural ``weight_prob`` default is
``b/(a+b) * (1 - a) = 0.04``, not ``0.20``.

The paper writes the second term as ``floor((1-a) * I_M * P(x))``,
which only matters on [0, 255] integer pixels where the floor maps
the small ``0.04 * 255 * P`` reinforcement to an integer. On
normalised [0, 1] floats the floor would zero out almost every
pixel; we omit it so the reinforcement actually has effect, which
is the same behaviour the paper achieves on its 8-bit inputs.

Ambiguity (Theta * I = I**2 ?): with the paper's prose definition of
Theta, the first term reads ``Theta * I = I(x)**2 * gate``, which
looks like an unintended duplication (Theta already equals I when
the gate is on). We default to the **linear** reading
``gate * I``; set ``cfg.square_intensity=True`` to reproduce the
paper-literal ``gate * I**2``.

Deviations we intentionally keep
--------------------------------

- 2D, not 3D (patches are 128x128 MIPs of confocal stacks).
- Fusion limited to ``cfg.fuse_channels`` (default pre+post synaptic),
  leaving the structural channel raw so the anatomical prior used by
  step 1 stays stable across iterations.
- Structural-mask gating of grown regions
  (``cfg.gate_growth_by_structural``); the paper has no structural
  channel.
- Self-consistency stopping in the outer loop (paper §3.4 uses
  held-out F1 plateau; we have no ground truth at train time).
- ``rho_min``, ``rho_max``, ``max_growth_ratio``, ``max_grow_iters``
  are extra-paper safeguards against runaway growth on noisy probs;
  the paper is silent on these.

Ref: https://github.com/cakuba/DDeep3m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass
class RefineCfg:
    """One DDeep3M+ refinement step's configuration.

    Region-growing knobs follow §3.3 Eqs. 5-6 collapsed to 2D. Fusion
    knobs follow §3.4 Eq. 7 with the explicit (1-a) factor and the
    "gate on P, multiply I" reading. See module docstring for the
    Theta*I=I**2 ambiguity (``square_intensity``).
    """

    # Region-growing seed (DDeep3M+ §3.3 step 1) -----------------------

    # Hard threshold on the probability map used to derive the seed
    # region O_reg per §3.3: "maximum probability classification". On a
    # sigmoid head with classes {bg, fg}, the natural max-prob
    # threshold is 0.5.
    seed_prob_threshold: float = 0.5

    # Region growing (DDeep3M+ §3.3 Eqs. 5-6) --------------------------

    # Ring radius (in pixels) used to estimate the adaptive growth
    # threshold rho. Paper N'(v_0) = 5x5x5 - centre = 124 voxels. 2D
    # analogue with radius=2 -> 5x5 - centre = 24-pixel ring.
    ring_radius: int = 2

    # 8-connected growth neighbourhood. Paper N(v_0) = 8 voxels (text
    # in §3.3); 2D 8-connectivity is the natural collapse. Set to 4
    # for 4-connectivity.
    growth_connectivity: int = 8

    # Cap on the number of growth iterations inside a single call to
    # region_grow_from_prob. Paper does not specify; 10 is plenty for
    # 128x128 patches.
    max_grow_iters: int = 10

    # Hard floor on the adaptive threshold rho. Paper does not specify;
    # without this, an empty / near-empty seed region can produce
    # rho ~ 0 and swallow the whole patch in a single step.
    rho_min: float = 0.30

    # Cap on rho. Paper does not specify. Past 0.95 the network is
    # already very confident and growth degenerates to its native
    # prediction; let the next iteration's seed step introduce fresh
    # seeds instead.
    rho_max: float = 0.95

    # Hard cap on grown area as a multiple of the seed area. Paper
    # does not specify. Stops runaway expansion when prob_map is noisy
    # on a tiny seed.
    max_growth_ratio: float = 5.0

    # AND the grown mask with ``near_structural`` (dilated dendrite +
    # soma mask) before returning. Paper has no structural channel;
    # this is a thesis-specific anatomical prior.
    gate_growth_by_structural: bool = True

    # Image-probability fusion (DDeep3M+ §3.4 Eq. 7) -------------------

    # Probability threshold below which the raw-image term is gated to
    # zero. Paper writes d=2 on a [0,255] image; on normalised [0,1]
    # inputs the faithful value is 2/255 ~= 0.00784.
    prob_threshold: float = 2.0 / 255.0

    # Weight on the raw-intensity term. Paper §3.4: a/(a+b) with
    # a=0.8, b=0.2 -> 0.80.
    weight_raw: float = 0.80

    # Weight on the probability-reinforcement term. Paper §3.4:
    # b/(a+b) * (1 - a) = 0.2 * 0.2 = 0.04. The (1-a) factor is
    # explicit in Eq. 7, not a typo.
    weight_prob: float = 0.04

    # Square the raw-intensity term: Eq. 7 prose redefines
    # Theta(I-d) = I when P>d (else 0), so Theta*I = I^2 when the gate
    # is on. We default to the linear reading (gate * I) because the
    # squaring is most plausibly a duplication typo; set this to True
    # to reproduce the paper-literal gate * I^2.
    square_intensity: bool = False

    # Channels to fuse (indices into the (C, H, W) patch). Defaults to
    # pre + post synaptic channels; structural/neurite channel index 2
    # is left raw to avoid a feedback loop into the anatomical prior.
    fuse_channels: tuple[int, ...] = (0, 1)

    # I_M cap for the probability-reinforcement term. None -> compute
    # from the image (max over fused channels). For normalised inputs
    # in [0, 1] pin this to 1.0 to make the fusion behaviour
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


def seed_from_prob_map(
    prob_map: np.ndarray,
    cfg: RefineCfg,
    structural_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Seed region O_reg from the probability map (DDeep3M+ §3.3 step 1).

    Paper: "According to the predicted probability map, the algorithm
    with maximum probability classification is used to generate the
    seed region of neurite." For a sigmoid binary head, max-prob
    classification is a hard threshold at ``cfg.seed_prob_threshold``
    (default 0.5). When ``cfg.gate_growth_by_structural`` is True the
    seed is additionally AND-ed with ``structural_mask``; this is a
    thesis-only deviation (paper has no structural prior).
    """
    if prob_map.ndim != 2:
        raise ValueError(f"seed_from_prob_map expects 2D, got {prob_map.shape}")
    seed = prob_map > cfg.seed_prob_threshold
    if cfg.gate_growth_by_structural:
        if structural_mask is None:
            raise ValueError(
                "gate_growth_by_structural=True requires structural_mask"
            )
        if structural_mask.shape != seed.shape:
            raise ValueError(
                f"structural_mask shape {structural_mask.shape} != "
                f"prob_map shape {seed.shape}"
            )
        seed &= structural_mask.astype(bool)
    return seed.astype(np.uint8)


def region_grow_from_prob(
    prob_map: np.ndarray,
    cfg: RefineCfg,
    structural_mask: np.ndarray | None = None,
    *,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grow seed region from ``prob_map`` (2D DDeep3M+ §3.3 Eqs. 5-6).

    Paper-faithful path (default): ``seed_mask=None`` -> seed is derived
    from ``prob_map`` via ``seed_from_prob_map``. Backwards-compat path
    (off-paper): pass an explicit ``seed_mask`` (e.g. the previous
    iteration's pseudo-label).

    Adaptive threshold ``rho`` is the mean ``prob_map`` value in the ring
    around the current grown region; new pixels in the
    ``growth_connectivity`` neighbourhood with ``prob > rho`` are added.
    Iterate until no new pixels, ``max_grow_iters``, or
    ``max_growth_ratio * seed_area``. ``structural_mask`` required when
    ``cfg.gate_growth_by_structural=True``. All inputs 2D, same shape.
    Returns ``(H, W)`` uint8.
    """
    if prob_map.ndim != 2:
        raise ValueError(f"region_grow_from_prob expects 2D, got {prob_map.shape}")

    if seed_mask is None:
        seed_arr = seed_from_prob_map(prob_map, cfg, structural_mask)
    else:
        if seed_mask.shape != prob_map.shape:
            raise ValueError(
                f"shape mismatch: seed {seed_mask.shape} vs prob {prob_map.shape}"
            )
        seed_arr = seed_mask

    seed = seed_arr.astype(bool)
    seed_area = int(seed.sum())

    # Empty-seed edge case: the network predicted no foreground anywhere
    # above ``seed_prob_threshold``. Nothing to grow from; step 1 of the
    # next outer iteration (Hessian on the fused image) will introduce
    # fresh seeds.
    if seed_area == 0:
        return np.zeros(prob_map.shape, dtype=np.uint8)

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
    """Fuse raw image with probability map (2D DDeep3M+ §3.4 Eq. 7).

    For each channel in ``cfg.fuse_channels``::

        F(x) = weight_raw * 1[P(x) > prob_threshold] * I_term(x)
             + weight_prob * I_M * P(x)

    where ``I_term(x) = I(x)`` by default, or ``I(x)**2`` when
    ``cfg.square_intensity=True`` (literal Theta*I reading of Eq. 7;
    see module docstring). Channels not in ``cfg.fuse_channels`` pass
    through unchanged. Output clipped to
    ``[0, max(image.max(), I_M)]``. ``image`` not modified. ``image``
    is ``(C, H, W)`` float32 in ``[0, 1]``; ``prob_map`` is ``(H, W)``
    in ``[0, 1]``. Returns ``(C, H, W)`` float32.
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
        i_term = image[c].astype(np.float32)
        if cfg.square_intensity:
            i_term = i_term * i_term
        raw_term = cfg.weight_raw * gate * i_term
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
