"""Dendrite trace from Frangi on the pseudo-FDT field.

Companion to ``soma_fdt``. The same pseudo-FDT field used for soma
detection is fed to a Frangi vesselness filter (Frangi et al., MICCAI
1998); positive response above a per-image percentile threshold is the
dendrite trace, with the soma mask subtracted so the trace does not
bleed into cell bodies.

Why FDT rather than the raw structural channel: Hessian ridge filters
require a continuous bright line (lambda1 ~ 0, lambda2 << 0). Our raw
structural channel is *punctate* -- the dendrite signal is a chain of
spots, each of which has lambda1 ~ lambda2 << 0 (blob), the wrong
eigenvalue regime. The pseudo-FDT smooths puncta into continuous
bright regions whose ridges genuinely are lines, restoring the
assumption Frangi was designed for.

Defaults are calibrated empirically on the 20251219 dataset at
107 nm/px: see ``DEFAULT_DENDRITE_CFG``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.measure import label as cc_label
from skimage.morphology import (
    closing as morph_closing,
    dilation,
    disk,
    remove_small_objects,
)

from .soma_fdt import (
    DEFAULT_SOMA_CFG,
    compute_pseudo_fdt,
    extract_soma_mask,
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_DENDRITE_CFG: dict = dict(
    # ---- Frangi vesselness ---------------------------------------------
    # Scales cover dendrite half-widths 1-4 px (~0.1-0.4 um at 107 nm/px),
    # which is the range surviving the FDT smoothing.
    frangi_sigmas=(1.0, 1.5, 2.0, 3.0, 4.0),
    # Frangi alpha (line-vs-blob discriminator). Smaller -> stricter line
    # preference. 0.5 is the standard textbook default; on the FDT field
    # this already excludes most blob-like junctions.
    frangi_alpha=0.5,
    # Frangi beta (structureness threshold). 0.5 textbook default.
    frangi_beta=0.5,
    # FDT-derived ridges are bright on a dark FDT background.
    frangi_black_ridges=False,

    # ---- Threshold on the Frangi response ------------------------------
    # The positive response has a heavy tail: junctions / soma centres
    # are very bright, dendrite shafts are mid-bright, background is
    # near-zero. A high percentile (Otsu, p90) misses shafts. p50 of the
    # positive response keeps roughly the whole ridge network without
    # bleeding into noise. Lower -> noisier, higher -> sparser.
    threshold_pct=50.0,

    # Optional hysteresis: keep pixels >= low_pct that are connected to
    # at least one seed pixel >= high_pct. Useful when shafts are dim
    # but junctions/somas are clearly above noise. Off by default --
    # on this dataset the single p50 threshold is cleaner.
    use_hysteresis=False,
    hysteresis_high_pct=85.0,
    hysteresis_low_pct=50.0,

    # ---- Soma carving --------------------------------------------------
    # Soma mask (from ``soma_fdt``) is dilated by this many pixels and
    # subtracted from the dendrite candidate, so the trace does not
    # bleed into cell bodies. 5 px = ~0.5 um buffer at 107 nm/px.
    soma_buffer_dilate=5,

    # ---- Cleanup -------------------------------------------------------
    # Drop CCs strictly smaller than this; default 100 px = ~10 um chain
    # at 1-2 px width. Raise to filter more aggressively.
    min_cc_area=100,
    # Two-stage filter: small CCs (>= small_cc_area, < min_cc_area) are
    # KEPT only if they sit within small_cc_proximity_px of a big CC.
    # Rescues dim periphery dendrite branches that drop below min_cc_area
    # but are connected (visually) to the main dendrite tree, while
    # still dropping isolated noise specks of the same size.
    # 0 = off (pure min_cc_area filter).
    small_cc_area=0,
    small_cc_proximity_px=0,
    # Small closing to bridge 1-2 px pinholes inside the trace.
    closing_radius=2,
    # Optional final dilation -- 0 keeps the 1-3 px ridge thickness.
    dilate_r=0,

    # ---- Display -------------------------------------------------------
    overlay_dim=0.45,
    response_clip_pct=99.0,
    trace_color=(1.0, 0.10, 0.10),
    soma_outline_color=(0.2, 1.0, 1.0),
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def resolve_cfg(cfg: Optional[dict] = None) -> dict:
    """Merge user overrides into ``DEFAULT_DENDRITE_CFG``."""
    out = dict(DEFAULT_DENDRITE_CFG)
    if cfg:
        out.update(cfg)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def compute_frangi_response(
    fdt: np.ndarray,
    cfg: Optional[dict] = None,
) -> np.ndarray:
    """Frangi vesselness on a precomputed pseudo-FDT field."""
    cfg = resolve_cfg(cfg)
    return frangi(
        np.asarray(fdt, dtype=np.float32),
        sigmas=tuple(cfg["frangi_sigmas"]),
        alpha=float(cfg["frangi_alpha"]),
        beta=float(cfg["frangi_beta"]),
        black_ridges=bool(cfg["frangi_black_ridges"]),
    )


def extract_dendrite_mask(
    response: np.ndarray,
    soma_mask: np.ndarray,
    cfg: Optional[dict] = None,
) -> dict:
    """Threshold + carve + cleanup on a Frangi response.

    Returns a dict with keys ``raw_mask`` (pre-cleanup), ``trace``
    (after small-CC removal + closing + optional dilation),
    ``soma_buffer`` (dilated soma mask used for carving),
    ``threshold`` (the absolute response threshold actually used),
    and ``threshold_kind`` (``'percentile'`` or ``'hysteresis'``).
    """
    cfg = resolve_cfg(cfg)
    response = np.asarray(response, dtype=np.float32)
    nz = response[response > 0]

    if nz.size < 100:
        zero = np.zeros_like(response, dtype=bool)
        return dict(
            raw_mask=zero, trace=zero, soma_buffer=np.asarray(soma_mask, dtype=bool),
            threshold=float("inf"), threshold_kind="degenerate",
        )

    if cfg["use_hysteresis"]:
        high = float(np.percentile(nz, float(cfg["hysteresis_high_pct"])))
        low = float(np.percentile(nz, float(cfg["hysteresis_low_pct"])))
        raw_mask = apply_hysteresis_threshold(response, low=low, high=high)
        threshold = low
        kind = "hysteresis"
    else:
        threshold = float(np.percentile(nz, float(cfg["threshold_pct"])))
        raw_mask = response > threshold
        kind = "percentile"

    # Carve out somas (with buffer) so the trace doesn't bleed into them.
    soma_buffer = np.asarray(soma_mask, dtype=bool)
    if cfg["soma_buffer_dilate"] > 0 and soma_buffer.any():
        soma_buffer = dilation(soma_buffer, disk(int(cfg["soma_buffer_dilate"])))
    trace = raw_mask & ~soma_buffer

    # Cleanup
    if trace.any():
        big_floor = max(0, int(cfg["min_cc_area"]) - 1)
        small_floor = int(cfg.get("small_cc_area", 0))
        prox = int(cfg.get("small_cc_proximity_px", 0))
        if small_floor > 0 and prox > 0 and small_floor < cfg["min_cc_area"]:
            # Two-stage: keep big CCs, then rescue small CCs within
            # prox px of a big CC (dim periphery branches), drop the
            # rest (isolated noise).
            lab, n = cc_label(trace, return_num=True)
            if n > 0:
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                big_ids = np.where(sizes >= int(cfg["min_cc_area"]))[0]
                small_keep_ids = np.where(
                    (sizes >= small_floor) & (sizes < int(cfg["min_cc_area"]))
                )[0]
                big_mask = np.isin(lab, big_ids)
                neighborhood = dilation(big_mask, disk(prox)) if big_mask.any() else big_mask
                kept_small = np.zeros_like(trace)
                if small_keep_ids.size and neighborhood.any():
                    for cid in small_keep_ids:
                        cc = lab == cid
                        if (cc & neighborhood).any():
                            kept_small |= cc
                trace = big_mask | kept_small
        else:
            trace = remove_small_objects(trace, max_size=big_floor)
    if cfg["closing_radius"] > 0 and trace.any():
        trace = morph_closing(trace, disk(int(cfg["closing_radius"])))
    if cfg["dilate_r"] > 0 and trace.any():
        trace = dilation(trace, disk(int(cfg["dilate_r"])))

    return dict(
        raw_mask=raw_mask.astype(bool, copy=False),
        trace=trace.astype(bool, copy=False),
        soma_buffer=soma_buffer.astype(bool, copy=False),
        threshold=threshold,
        threshold_kind=kind,
    )


def run_dendrite_on_image(
    image_path,
    cfg: Optional[dict] = None,
    *,
    soma_cfg: Optional[dict] = None,
    dend_fdt_cfg: Optional[dict] = None,
    structural_channel: int = 2,
) -> tuple[np.ndarray, dict, dict, dict]:
    """End-to-end on one full-image MIP file.

    Computes the pseudo-FDT, runs the SOMA pipeline (so the trace can
    be carved), then the Frangi-on-FDT dendrite pipeline.

    ``soma_cfg`` overrides the soma-side FDT/threshold knobs.
    ``dend_fdt_cfg`` overrides the dendrite-side FDT knobs ONLY (e.g.
    ``support_close``, ``support_min_area``, ``bg_sigma_k`` etc.); if
    provided, a second FDT is computed with ``soma_cfg | dend_fdt_cfg``
    and fed to Frangi while soma extraction keeps the unmodified FDT.
    If ``dend_fdt_cfg`` is ``None`` (default), the same FDT is shared.

    Returns ``(structural, fdt_info, soma_info, dendrite_info)``.
    The returned ``fdt_info`` is the dendrite-side FDT (what Frangi
    saw), so the visualisation matches the trace; the soma-side FDT is
    available as ``dendrite_info['soma_fdt_info']``.
    """
    full = np.load(Path(image_path))
    structural = full[structural_channel].astype(np.float32)

    soma_fdt_info = compute_pseudo_fdt(structural, soma_cfg)
    soma_info = extract_soma_mask(soma_fdt_info["fdt"], soma_cfg)

    if dend_fdt_cfg:
        merged = dict(soma_cfg or {})
        merged.update(dend_fdt_cfg)
        dend_fdt_info = compute_pseudo_fdt(structural, merged)
    else:
        dend_fdt_info = soma_fdt_info

    response = compute_frangi_response(dend_fdt_info["fdt"], cfg)
    dendrite_info = extract_dendrite_mask(response, soma_info["mask"], cfg)
    dendrite_info["response"] = response
    dendrite_info["n_trace_cc"] = int(cc_label(dendrite_info["trace"]).max())
    dendrite_info["soma_fdt_info"] = soma_fdt_info

    return structural, dend_fdt_info, soma_info, dendrite_info


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    scale = max(hi - lo, np.finfo(np.float32).eps)
    return np.clip((image - lo) / scale, 0.0, 1.0)


def visualise_dendrite_mask(
    structural: np.ndarray,
    fdt_info: dict,
    soma_info: dict,
    dendrite_info: dict,
    cfg: Optional[dict] = None,
    *,
    axes=None,
    title_prefix: str = "",
):
    """4-panel: structural / FDT field / Frangi response / trace overlay.

    Pass a ``(4,)`` array of axes to embed (e.g. one row in a batch
    grid), or omit ``axes`` for a standalone ``(1, 4)`` figure.
    """
    import matplotlib.pyplot as plt

    cfg = resolve_cfg(cfg)
    fdt = np.asarray(fdt_info["fdt"], dtype=np.float32)
    response = dendrite_info["response"]
    trace = dendrite_info["trace"]
    soma_buffer = dendrite_info["soma_buffer"]

    fdt_pos = fdt[fdt > 0]
    fdt_vmax = (
        float(np.percentile(fdt_pos, cfg["response_clip_pct"]))
        if fdt_pos.size else 1.0
    )
    nz = response[response > 0]
    resp_vmax = (
        float(np.percentile(nz, cfg["response_clip_pct"]))
        if nz.size else 1.0
    )

    base = _normalize_for_display(structural) * cfg["overlay_dim"]
    overlay = np.dstack([base, base, base])
    trace_color = np.array(cfg["trace_color"], dtype=np.float32)
    soma_color = np.array(cfg["soma_outline_color"], dtype=np.float32)
    if trace.any():
        halo = dilation(trace, disk(2))
        overlay[halo] = 0.45 * overlay[halo] + 0.55 * trace_color
        overlay[dilation(trace, disk(1))] = trace_color
    if soma_buffer.any():
        edges = dilation(soma_buffer, disk(1)) & ~soma_buffer
        overlay[edges] = soma_color
    overlay = np.clip(overlay, 0.0, 1.0)

    if axes is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig = axes[0].figure
    if len(axes) < 4:
        raise ValueError(
            f"visualise_dendrite_mask now needs 4 axes (got {len(axes)}); "
            f"use plt.subplots(1, 4) or plt.subplots(N, 4)."
        )

    axes[0].imshow(_normalize_for_display(structural), cmap="gray")
    axes[0].set_title(f"{title_prefix}structural".strip())
    axes[1].imshow(fdt, cmap="magma", vmin=0, vmax=fdt_vmax)
    axes[1].set_title(f"pseudo-FDT  (vmax={fdt_vmax:.2f})")
    axes[2].imshow(response, cmap="magma", vmin=0, vmax=resp_vmax)
    axes[2].set_title(
        f"Frangi(FDT) response  "
        f"thr[{dendrite_info['threshold_kind']}]={dendrite_info['threshold']:.4f}"
    )
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"DENDRITE trace "
        f"({dendrite_info.get('n_trace_cc', '?')} CCs, "
        f"cov={trace.mean():.2%})"
    )
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return fig, axes


__all__ = [
    "DEFAULT_DENDRITE_CFG",
    "resolve_cfg",
    "compute_frangi_response",
    "extract_dendrite_mask",
    "run_dendrite_on_image",
    "visualise_dendrite_mask",
]
