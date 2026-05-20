"""Soma-only mask from a Saha-style pseudo-FDT field.

Full-image (2D MIP) alternative to ``pseudolabels.puncta.make_soma_mask``.
The pipeline:

1. Robust background + linear fuzzy membership (Zadeh 1965) on the
   structural channel.
2. ``EDT * mu`` Gaussian-smoothed -> the pseudo-FDT field. Documented
   shortcut (Saha 2002, *CVIU* 86(3):171-190) -- not the path-integrated
   FDT, agrees only when mu is binary.
3. Per-image high-percentile threshold on the FDT (or Otsu /
   multi-Otsu) keeps only thick / round regions.
4. Size + solidity filters reject noise and branchy CCs.
5. Per-CC shape regularisation (closing / convex hull / ellipse / circle)
   gives a clean output.

The default ``DEFAULT_SOMA_CFG`` is the calibration the human selected
empirically on the 20251219 dataset at 107 nm/px.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.draw import disk as draw_disk, ellipse as draw_ellipse
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.measure import label as cc_label, regionprops
from skimage.morphology import (
    closing as morph_closing,
    convex_hull_image,
    dilation,
    disk,
    opening as morph_opening,
    remove_small_holes,
    remove_small_objects,
)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
#
# All knobs in one place. Every entry has a why-comment so the call site
# does not need to know the pipeline internals to choose values.
#
# Size knobs left as None are derived per call from pixel_size_nm + the
# biology priors (see ``derive_size_params``); set them to an int to
# force a manual override.

DEFAULT_SOMA_CFG: dict = dict(
    # ---- Hardware --------------------------------------------------------
    # 60x confocal default in this thesis; change once per microscope.
    pixel_size_nm=107.0,

    # ---- Biology priors -------------------------------------------------
    # Smallest soma to accept. 3 um is a small cultured neuron; below
    # that we cannot distinguish from a bright dendrite junction.
    min_soma_diameter_um=3.0,
    # Largest dark interior hole filled before EDT (nucleus-style).
    # 2.5 um is small on purpose: filling bigger holes inflates the
    # central junction tangle and loses round-soma discrimination.
    max_hole_diameter_um=2.5,
    # Morphological opening kernel radius (~ a punctate-noise neck).
    # 0.3 um at 107 nm/px = 1 px, which is the floor.
    neck_width_um=0.3,

    # ---- Pseudo-FDT pipeline -------------------------------------------
    # Light Gaussian on the raw structural channel before thresholding.
    # 3 px (~0.32 um) suppresses per-pixel noise without smearing somas.
    smooth_sigma=3.0,
    # Robust background: t_low = median + bg_sigma_k * 1.4826 * MAD.
    # 3 sigma is the standard ~99.7 % rejection convention.
    bg_sigma_k=3.0,
    # t_high taken as this percentile of pixels above t_low. 80 %
    # leaves enough headroom for the linear membership ramp without
    # saturating on the brightest few pixels.
    t_high_fg_pct=80.0,
    # Post-EDT Gaussian. 1 px barely smooths the FDT; keep small so the
    # high-percentile threshold still resolves individual somas.
    fdt_sigma=1.0,

    # ---- FDT-field enhancement (default: off, current behaviour) -------
    # These three knobs reshape the FDT field. Defaults keep the current
    # behaviour. Set them to brighten dim dendrites relative to somas,
    # which matters for the Frangi-on-FDT dendrite tracer (otherwise
    # somas dominate the dynamic range and dendrites disappear).
    #
    # mu_gamma: exponent applied to the linear membership (mu -> mu**gamma)
    # BEFORE multiplying by the EDT. gamma < 1 boosts dim pixels (dendrite
    # cytoplasm) without saturating bright pixels (somas). 1.0 = off.
    mu_gamma=1.0,
    # support_dilate: dilate the binary support by N pixels before the
    # EDT. Thin 1-2 px dendrites get a larger inside-distance ->
    # brighter FDT response. mu is still 0 outside the original
    # support, so dilation never manufactures signal where the channel
    # is dark. 0 = off.
    support_dilate=0,
    # support_min_area: drop CCs strictly smaller than this from the
    # support (and from mu) BEFORE the closing/EDT steps. Removes
    # isolated noise puncta that would otherwise be bridged into the
    # dendrite trace by support_close. 0 = off.
    support_min_area=0,
    # support_close: morphological closing radius (px) applied to both
    # the binary support AND mu (grayscale closing). Bridges adjacent
    # puncta along dendrites so the FDT becomes a continuous bright
    # line instead of a chain of dots. Should be ~ inter-punctum
    # spacing (3-5 px at 107 nm/px). 0 = off. Combine with
    # support_min_area to avoid bridging noise.
    support_close=0,
    # fdt_compress: post-EDT dynamic-range compression. Somas have FDT ~
    # radius (30-100 px), dendrites have FDT ~ half-width (1-3 px), so
    # linear FDT is dominated by somas. 'sqrt' or 'log' shrinks this
    # 30-100x ratio to ~5-10x, making dendrites visible to thresholders
    # and ridge filters. Choices: 'linear' | 'sqrt' | 'log'.
    fdt_compress="linear",
    # fdt_mode: which "thickness" primitive to use.
    #   'edt'      -- classical EDT(support) * mu. Good for somas; gives
    #                  tiny values inside thin connectors so chain-of-
    #                  puncta dendrites stay dotty.
    #   'gaussian' -- gaussian(mu, sigma=fuzzy_sigma). Continuous bright
    #                  field wherever any pixel within sigma is in the
    #                  support; bridges puncta into ridges that Frangi
    #                  can trace. Loses the "thickness" semantics so
    #                  soma vs dendrite ratio collapses; safe to use on
    #                  the dendrite-side FDT via dend_fdt_cfg only.
    fdt_mode="edt",
    # fuzzy_sigma: only used when fdt_mode='gaussian'. Roughly the
    # max gap between puncta to be bridged (px). At 107 nm/px, 10 px ~
    # 1.07 um, the typical inter-punctum spacing.
    fuzzy_sigma=10.0,

    # ---- Blob threshold on the FDT field --------------------------------
    # 'percentile' (default) adapts per image and is the most permissive
    # on dimmer somas. 'otsu' is stricter. 'multi_otsu' (top class)
    # keeps only the very brightest cores.
    blob_threshold_method="percentile",
    # p95 was the human-selected value (was p90 default). At p90, raising
    # solidity gave too many branchy false positives; p95 lets us drop
    # the solidity floor without flooding.
    blob_fdt_pct=95.0,
    # K for multi_otsu; we keep the upper threshold (ts[-1]).
    blob_fdt_multi_classes=3,

    # ---- Cleanup -------------------------------------------------------
    # None -> derived from biology + pixel_size_nm.
    blob_min_area=None,
    hole_max_area=None,
    open_radius=None,
    # area / convex_hull. 0.4 is loose on purpose: at p95 most surviving
    # CCs are already round; a low floor recovers somas that have one
    # bright dendrite stuck to them. Tighten to 0.6+ if peripheral
    # dendrite junctions sneak through.
    min_solidity=0.4,
    # Final dilation in px (5 px = ~0.5 um). Recovers the cytoplasmic
    # boundary that the high-percentile threshold underestimates.
    dilate_r=5,

    # ---- Shape regularisation per CC -----------------------------------
    # 'closing' gives the most natural-looking somas: it smooths the
    # jagged FDT contour while keeping the real outline. Alternatives:
    # 'raw' (no smoothing), 'convex' (convex hull), 'ellipse' / 'circle'
    # (perfect synthetic shapes from regionprops).
    shape_mode="closing",
    # Only used iff shape_mode='closing'. 8 px (~0.85 um) matches the
    # typical FDT-contour roughness scale.
    closing_radius=8,
    # > 0 bridges fragmented CCs into one blob BEFORE the per-CC shape
    # draw. Default off; raise to ~10 if a single soma fragments into
    # several touching detections.
    pre_merge_radius=0,

    # ---- Display -------------------------------------------------------
    # Dimming factor applied to the structural channel under the green
    # mask overlay. 0.6 keeps the dendrite context visible.
    overlay_dim=0.6,
    # Upper percentile for the FDT colour scale in panel 2.
    fdt_clip_pct=99.0,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def derive_size_params(cfg: dict) -> dict:
    """Pixel-unit blob knobs from ``pixel_size_nm`` + biology priors."""
    px_um = cfg["pixel_size_nm"] / 1000.0
    return dict(
        blob_min_area=int(round(np.pi * (cfg["min_soma_diameter_um"] / 2.0 / px_um) ** 2)),
        hole_max_area=int(round(np.pi * (cfg["max_hole_diameter_um"] / 2.0 / px_um) ** 2)),
        open_radius=max(1, int(round(cfg["neck_width_um"] / 2.0 / px_um))),
    )


def resolve_cfg(cfg: Optional[dict] = None) -> dict:
    """Merge user overrides into ``DEFAULT_SOMA_CFG`` and fill derived sizes.

    ``cfg`` may be ``None`` (use all defaults) or a partial dict; only the
    keys present override the default. Any size knob still ``None`` after
    the merge is filled via ``derive_size_params``.
    """
    out = dict(DEFAULT_SOMA_CFG)
    if cfg:
        out.update(cfg)
    derived = derive_size_params(out)
    for k, v in derived.items():
        if out.get(k) is None:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Pseudo-FDT
# ---------------------------------------------------------------------------

def compute_pseudo_fdt(
    structural: np.ndarray,
    cfg: Optional[dict] = None,
) -> dict:
    """Robust-bg + fuzzy membership + EDT*mu + Gaussian.

    Five optional enhancement knobs in ``cfg`` reshape the FDT field
    (all default to a no-op that preserves the classical behaviour):

    - ``mu_gamma`` (default 1.0): ``mu -> mu**gamma`` before EDT*mu.
      Values <1 boost dim membership.
    - ``support_min_area`` (default 0): drop CCs strictly smaller than
      this from support+mu before closing/EDT. Removes isolated noise
      puncta so ``support_close`` does not bridge them into the trace.
    - ``support_close`` (default 0): morphological closing radius (px)
      applied to both support (binary) and mu (grayscale). Bridges
      adjacent puncta along a dendrite into a continuous bright FDT
      line. Combine with ``support_min_area`` to reject noise.
    - ``support_dilate`` (default 0): dilate the (closed) support by N
      pixels before the EDT, so thin dendrites get a larger inside-
      distance. ``mu`` is still zero outside the closed support.
    - ``fdt_compress`` (default ``'linear'``): post-EDT compression of
      the dynamic range. ``'sqrt'`` or ``'log'`` shrink the soma:dendrite
      brightness ratio so ridge filters see both at comparable scale.

    Returns a dict with keys ``smoothed``, ``mu``, ``support``, ``fdt``,
    ``bg_med``, ``bg_sigma``, ``t_low``, ``t_high``.
    """
    cfg = resolve_cfg(cfg)
    structural = np.asarray(structural, dtype=np.float32)
    smoothed = gaussian_filter(structural, sigma=cfg["smooth_sigma"])

    bg = smoothed[smoothed <= np.percentile(smoothed, 50)]
    bg_med = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    bg_sigma_robust = 1.4826 * bg_mad

    t_low = bg_med + cfg["bg_sigma_k"] * bg_sigma_robust
    fg = smoothed[smoothed > t_low]
    t_high = float(np.percentile(fg, cfg["t_high_fg_pct"])) if fg.size else t_low * 2.0
    scale = max(t_high - t_low, np.finfo(np.float32).eps)

    mu = np.clip((smoothed - t_low) / scale, 0.0, 1.0).astype(np.float32)
    mu_gamma = float(cfg.get("mu_gamma", 1.0))
    if mu_gamma != 1.0:
        mu = np.power(mu, mu_gamma).astype(np.float32)

    support = mu > 0
    smin = int(cfg.get("support_min_area", 0))
    if smin > 0 and support.any():
        keep = remove_small_objects(support, max_size=max(0, smin - 1))
        support = keep
        mu = np.where(keep, mu, 0.0).astype(np.float32)
    sclose = int(cfg.get("support_close", 0))
    if sclose > 0 and support.any():
        kern = disk(sclose)
        support = morph_closing(support, kern)
        # Grayscale closing on mu fills the bridged gaps with the local
        # max value, so the FDT is non-zero along the whole closed shaft.
        mu = morph_closing(mu, kern).astype(np.float32)
        mu = np.where(support, mu, 0.0).astype(np.float32)
    edt_support = support
    sd = int(cfg.get("support_dilate", 0))
    if sd > 0 and support.any():
        # Dilate ONLY the support used for the EDT; mu stays 0 outside
        # the original support so the dilated band carries no signal.
        # ``dilation`` (vs the deprecated ``binary_dilation``) is the
        # unified skimage >=0.26 entry point.
        edt_support = dilation(support, disk(sd))

    mode = str(cfg.get("fdt_mode", "edt")).lower()
    if mode == "edt":
        fdt = distance_transform_edt(edt_support).astype(np.float32) * mu
    elif mode == "gaussian":
        # Replace EDT*mu with a Gaussian fuzzy distance on mu. Bridges
        # chain-of-puncta dendrites into continuous bright ridges that
        # Frangi can trace (EDT*mu collapses to zero in thin connectors;
        # this primitive does not).
        fsig = float(cfg.get("fuzzy_sigma", 10.0))
        fdt = gaussian_filter(mu, sigma=fsig).astype(np.float32)
    else:
        raise ValueError(
            f"unknown fdt_mode: {mode!r}; expected 'edt' or 'gaussian'"
        )
    fdt = gaussian_filter(fdt, sigma=cfg["fdt_sigma"])

    compress = cfg.get("fdt_compress", "linear")
    if compress == "sqrt":
        fdt = np.sqrt(np.clip(fdt, 0.0, None)).astype(np.float32)
    elif compress == "log":
        fdt = np.log1p(np.clip(fdt, 0.0, None)).astype(np.float32)
    elif compress != "linear":
        raise ValueError(
            f"unknown fdt_compress: {compress!r}; "
            f"expected 'linear', 'sqrt', or 'log'"
        )

    return dict(
        smoothed=smoothed, mu=mu, support=support, fdt=fdt,
        bg_med=bg_med, bg_sigma=bg_sigma_robust,
        t_low=t_low, t_high=t_high,
    )


# ---------------------------------------------------------------------------
# Soma mask
# ---------------------------------------------------------------------------

def _blob_threshold(fdt_pos: np.ndarray, cfg: dict) -> tuple[float, str]:
    """Per-image FDT threshold dispatched on ``cfg['blob_threshold_method']``."""
    if fdt_pos.size < 100:
        return float("inf"), "degenerate"
    method = cfg.get("blob_threshold_method", "percentile")
    if method == "percentile":
        return float(np.percentile(fdt_pos, cfg["blob_fdt_pct"])), method
    if method == "otsu":
        return float(threshold_otsu(fdt_pos)), method
    if method == "multi_otsu":
        ts = threshold_multiotsu(
            fdt_pos, classes=int(cfg.get("blob_fdt_multi_classes", 3))
        )
        return float(ts[-1]), method
    raise ValueError(
        f"unknown blob_threshold_method: {method!r}; "
        f"expected 'percentile', 'otsu', or 'multi_otsu'"
    )


def _round_blobs(blob_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Per-CC shape regularisation. Dispatches on ``cfg['shape_mode']``.

    If ``cfg['pre_merge_radius'] > 0``, runs a closing first to merge
    nearby fragmented CCs before the per-CC shape draw.
    """
    if not blob_mask.any():
        return blob_mask
    pre_r = int(cfg.get("pre_merge_radius", 0))
    if pre_r > 0:
        blob_mask = morph_closing(blob_mask, disk(pre_r))
    mode = cfg.get("shape_mode", "raw")
    if mode == "raw":
        return blob_mask
    if mode == "closing":
        return morph_closing(blob_mask, disk(int(cfg.get("closing_radius", 8))))
    lbl = cc_label(blob_mask, connectivity=2)
    out = np.zeros_like(blob_mask, dtype=bool)
    for prop in regionprops(lbl):
        y0, x0, y1, x1 = prop.bbox
        if mode == "convex":
            sub = (lbl[y0:y1, x0:x1] == prop.label)
            out[y0:y1, x0:x1] |= convex_hull_image(sub)
        elif mode == "ellipse":
            ry = max(prop.major_axis_length / 2.0, 1.0)
            rx = max(prop.minor_axis_length / 2.0, 1.0)
            cy, cx = prop.centroid
            rr, cc = draw_ellipse(
                cy, cx, ry, rx,
                rotation=-prop.orientation,
                shape=blob_mask.shape,
            )
            out[rr, cc] = True
        elif mode == "circle":
            r = max(prop.equivalent_diameter_area / 2.0, 1.0)
            cy, cx = prop.centroid
            rr, cc = draw_disk((cy, cx), r, shape=blob_mask.shape)
            out[rr, cc] = True
        else:
            raise ValueError(
                f"unknown shape_mode: {mode!r}; expected "
                f"'raw' | 'closing' | 'convex' | 'ellipse' | 'circle'"
            )
    return out


def extract_soma_mask(
    fdt: np.ndarray,
    cfg: Optional[dict] = None,
) -> dict:
    """Threshold + size + solidity + shape + dilation on a pseudo-FDT field.

    Returns a dict with ``blob`` (pre-dilation, post-shape),
    ``mask`` (final), ``threshold``, ``threshold_method``, ``n_raw``,
    ``n_kept``, ``n_blobs``.
    """
    cfg = resolve_cfg(cfg)
    fdt_pos = fdt[fdt > 0]
    threshold, method = _blob_threshold(fdt_pos, cfg)

    blob = fdt > threshold
    # skimage >=0.26: max_size=N removes area <= N. -1 preserves the
    # strict-less-than semantics of the deprecated min_size argument.
    blob = remove_small_objects(blob, max_size=max(0, int(cfg["blob_min_area"]) - 1))
    if cfg["hole_max_area"] > 0 and blob.any():
        blob = remove_small_holes(blob, max_size=max(0, int(cfg["hole_max_area"]) - 1))
    if cfg["open_radius"] > 0 and blob.any():
        blob = morph_opening(blob, disk(int(cfg["open_radius"])))

    n_raw = int(cc_label(blob).max())
    if cfg["min_solidity"] > 0.0 and blob.any():
        lbl = cc_label(blob, connectivity=2)
        kept = np.zeros_like(blob)
        for prop in regionprops(lbl):
            if (prop.solidity >= cfg["min_solidity"]
                    and prop.area >= int(cfg["blob_min_area"])):
                kept[lbl == prop.label] = True
        blob = kept
    n_kept = int(cc_label(blob).max())

    blob = _round_blobs(blob, cfg)

    if cfg["dilate_r"] > 0 and blob.any():
        mask = dilation(blob, disk(int(cfg["dilate_r"])))
    else:
        mask = blob.copy()

    return dict(
        blob=blob, mask=mask,
        threshold=threshold, threshold_method=method,
        n_raw=n_raw, n_kept=n_kept,
        n_blobs=int(cc_label(mask).max()),
    )


def run_soma_on_image(
    image_path,
    cfg: Optional[dict] = None,
    *,
    structural_channel: int = 2,
) -> tuple[np.ndarray, dict, dict]:
    """End-to-end on a single full-image MIP file.

    ``image_path`` must point to a ``.npy`` of shape ``(C, H, W)``.
    Returns ``(structural, fdt_info, blob_info)`` -- the structural
    slice, the ``compute_pseudo_fdt`` dict, and the ``extract_soma_mask``
    dict.
    """
    full = np.load(Path(image_path))
    structural = full[structural_channel].astype(np.float32)
    fdt_info = compute_pseudo_fdt(structural, cfg)
    blob_info = extract_soma_mask(fdt_info["fdt"], cfg)
    return structural, fdt_info, blob_info


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1, 99])
    scale = max(hi - lo, np.finfo(np.float32).eps)
    return np.clip((image - lo) / scale, 0.0, 1.0)


def visualise_soma_mask(
    structural: np.ndarray,
    fdt_info: dict,
    blob_info: dict,
    cfg: Optional[dict] = None,
    *,
    axes=None,
    title_prefix: str = "",
):
    """3-panel diagnostic: structural / FDT with mask outline / mask overlay.

    Pass a ``(3,)`` array of matplotlib axes to embed (e.g. one row in a
    batch grid), or omit ``axes`` to get a standalone ``(1, 3)`` figure.
    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    cfg = resolve_cfg(cfg)
    fdt = fdt_info["fdt"]
    fdt_pos = fdt[fdt > 0]
    fdt_vmax = (
        float(np.percentile(fdt_pos, cfg["fdt_clip_pct"]))
        if fdt_pos.size else 1.0
    )
    mask = blob_info["mask"]

    base = _normalize_for_display(structural) * cfg["overlay_dim"]
    overlay = np.dstack([base, base, base])
    blob_color = np.array([0.0, 1.0, 0.4], dtype=np.float32)
    if mask.any():
        overlay[mask] = 0.5 * overlay[mask] + 0.5 * blob_color
        overlay[dilation(mask, disk(1)) & ~mask] = blob_color
    overlay = np.clip(overlay, 0.0, 1.0)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig = axes[0].figure

    axes[0].imshow(_normalize_for_display(structural), cmap="gray")
    axes[0].set_title(f"{title_prefix}structural".strip())
    axes[1].imshow(fdt, cmap="magma", vmin=0, vmax=fdt_vmax)
    axes[1].contour(mask, levels=[0.5], colors="lime", linewidths=0.5)
    axes[1].set_title(
        f"pseudo-FDT  thr[{blob_info['threshold_method']}]={blob_info['threshold']:.2f}"
    )
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"SOMA mask [{cfg['shape_mode']}] "
        f"({blob_info['n_blobs']} blobs, cov={mask.mean():.2%})"
    )
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    return fig, axes


__all__ = [
    "DEFAULT_SOMA_CFG",
    "derive_size_params",
    "resolve_cfg",
    "compute_pseudo_fdt",
    "extract_soma_mask",
    "run_soma_on_image",
    "visualise_soma_mask",
]
