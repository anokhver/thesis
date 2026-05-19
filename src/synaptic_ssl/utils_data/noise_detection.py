"""Image-level noise-floor detection (complements per-patch ``damage_detection``).

Per-patch ``damage_detection`` looks for *within-image* outlier patches and so
cannot detect images where **every** patch is equally noisy. That situation
arises naturally with this pipeline's :func:`normalize_percentile` stretch:
a channel that has no real signal still contains sensor shot noise, and the
per-channel ``[p1, p99.8]`` linear stretch in
``preprocess_training.py`` maps that narrow noise band to the full ``[0, 1]``
dynamic range. The result is a uniform, high-frequency speckle that looks
"pixelated as hell" on display.

This module exposes two scale-invariant frequency-domain metrics per channel
that distinguish such noise-dominated channels from real signal:

* ``hp_var_ratio``  — ``Var(image - Gaussian-blur(image)) / Var(image)``.
  White noise -> ~1.0; real images concentrate energy at low frequencies
  and score < 0.2.
* ``hf_energy_frac`` — fraction of FFT energy outside a centred disk of
  radius ``low_radius_frac * r_max``. Also ~1.0 for white noise, < 0.2 for
  real images.

An image is flagged when **any** non-dead channel exceeds **both** thresholds
simultaneously. Defaults were calibrated on the project's microscopy patches
(see commit message / notebook) to flag the obviously noise-dominated source
images while keeping the clean ``KONTROLA`` controls below threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter


# --- Calibrated thresholds --------------------------------------------------
# Empirical reference on date 20251208 (62 source images):
#   * clean KONTROLA: worst-channel hp_var_ratio ~ 0.13, hf_energy_frac ~ 0.15
#   * complaint target (5_PSI_01uM_48hod_10_..._260): hp ~ 0.63, hf ~ 0.65
#   * worst PSI_10uM_48hod_*: hp 0.73 - 0.80, hf 0.74 - 0.81
# Thresholds 0.55 / 0.50 sit comfortably between control and the complaint
# target; the user can tighten or loosen via the CLI / function args.
DEFAULT_HP_VAR_RATIO  = 0.55
DEFAULT_HF_ENERGY_FRAC = 0.50

# Spatial low-frequency definition.
DEFAULT_BLUR_SIGMA     = 1.5
# Radial frequency definition: anything outside this fraction of r_max is "high".
DEFAULT_LOW_RADIUS_FRAC = 0.25
# Channels with std below this are treated as "dead" (constant) and skipped.
DEFAULT_DEAD_STD       = 1e-4


# ---------------------------------------------------------------------------
# Per-channel and per-image metrics
# ---------------------------------------------------------------------------
def channel_noise_stats(
    channel: np.ndarray,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    low_radius_frac: float = DEFAULT_LOW_RADIUS_FRAC,
    dead_std: float = DEFAULT_DEAD_STD,
) -> dict:
    """Compute noise-floor statistics for a single 2-D channel.

    Returns a dict with ``mean, std, hp_var_ratio, hf_energy_frac, p01, p50,
    p99, dyn, dead`` (``dyn = p99 - p01``).
    """
    ch = np.asarray(channel, dtype=np.float32)
    if ch.ndim != 2:
        raise ValueError(f"channel must be 2-D, got shape {ch.shape}")

    mean = float(ch.mean())
    std  = float(ch.std())
    p01, p50, p99 = (float(v) for v in np.percentile(ch, [1.0, 50.0, 99.0]))
    dyn = p99 - p01
    dead = std < dead_std

    if dead:
        # Constant channel: no noise to measure, no risk of false flag.
        return {
            "mean": mean, "std": std,
            "hp_var_ratio": 0.0, "hf_energy_frac": 0.0,
            "p01": p01, "p50": p50, "p99": p99, "dyn": dyn,
            "dead": True,
        }

    # --- spatial high-pass: image minus its Gaussian-blurred version --------
    low = gaussian_filter(ch, sigma=float(blur_sigma), mode="reflect")
    hp  = ch - low
    var_total = float(ch.var())
    var_hp    = float(hp.var())
    hp_var_ratio = var_hp / max(var_total, 1e-12)

    # --- FFT outer-disk energy fraction ------------------------------------
    # Use unshifted FFT + fftfreq so the math is identical to fftshift but
    # avoids the (slow) shift step.
    spec = np.fft.fft2(ch - ch.mean())
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
    H, W = ch.shape
    fy = np.fft.fftfreq(H)  # cycles per pixel, in [-0.5, 0.5)
    fx = np.fft.fftfreq(W)
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    r_max = float(rr.max())
    low_disk = rr <= (low_radius_frac * r_max)
    total_e  = float(power.sum())
    if total_e <= 0.0:
        hf_energy_frac = 0.0
    else:
        hf_energy_frac = float(power[~low_disk].sum() / total_e)

    return {
        "mean": mean, "std": std,
        "hp_var_ratio": float(hp_var_ratio),
        "hf_energy_frac": float(hf_energy_frac),
        "p01": p01, "p50": p50, "p99": p99, "dyn": dyn,
        "dead": False,
    }


def image_noise_stats(
    full_image: np.ndarray,
    channel_names: Sequence[str] | None = None,
    hp_var_thresh: float = DEFAULT_HP_VAR_RATIO,
    hf_energy_thresh: float = DEFAULT_HF_ENERGY_FRAC,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    low_radius_frac: float = DEFAULT_LOW_RADIUS_FRAC,
    dead_std: float = DEFAULT_DEAD_STD,
) -> dict:
    """Score a ``(C, H, W)`` image.

    An image is *flagged* when at least one non-dead channel exceeds **both**
    ``hp_var_thresh`` and ``hf_energy_thresh``. The worst non-dead channel's
    metrics are also reported for ranking.
    """
    full_image = np.asarray(full_image)
    if full_image.ndim != 3:
        raise ValueError(
            f"full_image must be (C, H, W), got shape {full_image.shape}"
        )

    C = full_image.shape[0]
    names = list(channel_names) if channel_names else [f"ch{c}" for c in range(C)]
    if len(names) != C:
        names = (names + [f"ch{c}" for c in range(C)])[:C]

    per_channel: list[dict] = []
    noisy: list[str] = []
    worst_idx, worst_hp, worst_hf = -1, -1.0, -1.0

    for c in range(C):
        stats = channel_noise_stats(
            full_image[c],
            blur_sigma=blur_sigma,
            low_radius_frac=low_radius_frac,
            dead_std=dead_std,
        )
        stats["channel"] = names[c]
        per_channel.append(stats)
        if stats["dead"]:
            continue
        # Track the worst (most noise-like) live channel for ranking.
        if stats["hp_var_ratio"] > worst_hp:
            worst_idx, worst_hp, worst_hf = c, stats["hp_var_ratio"], stats["hf_energy_frac"]
        if (stats["hp_var_ratio"] >= hp_var_thresh
                and stats["hf_energy_frac"] >= hf_energy_thresh):
            noisy.append(names[c])

    return {
        "per_channel":         per_channel,
        "worst_channel":       names[worst_idx] if worst_idx >= 0 else "",
        "worst_hp_var_ratio":  float(worst_hp) if worst_idx >= 0 else 0.0,
        "worst_hf_energy_frac": float(worst_hf) if worst_idx >= 0 else 0.0,
        "noisy_channels":      ";".join(noisy),
        "flagged":             bool(noisy),
    }


# ---------------------------------------------------------------------------
# Bulk scoring over a patches root
# ---------------------------------------------------------------------------
def _maybe_tqdm(iterable, *, desc: str, enable: bool):
    if not enable:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)


def score_image_set(
    patch_root: str | Path,
    exclude_patterns: Sequence[str] | None = ("KONTROLA",),
    channel_names: Sequence[str] | None = None,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    low_radius_frac: float = DEFAULT_LOW_RADIUS_FRAC,
    hp_var_thresh: float = DEFAULT_HP_VAR_RATIO,
    hf_energy_thresh: float = DEFAULT_HF_ENERGY_FRAC,
    dead_std: float = DEFAULT_DEAD_STD,
    progress: bool = True,
) -> list[dict]:
    """Reassemble every source image under *patch_root* and score it.

    Returns a list of records (one per ``image_index``); the index/source
    fields make the result easy to join with the patch CSV. Records also
    carry the full ``per_channel`` list so callers can render their own
    summaries.
    """
    # Lazy import: reassemble pulls in numpy only, but keeping the import
    # local avoids a circular import via ``utils_data/__init__.py``.
    from .reassemble import _load_index, reassemble_image

    patch_root = Path(patch_root)
    by_image = _load_index(patch_root)
    pats = [p.upper() for p in (exclude_patterns or [])]

    indices = sorted(by_image.keys())
    records: list[dict] = []

    for idx in _maybe_tqdm(indices, desc="score_image_set", enable=progress):
        recs = by_image[idx]
        head = recs[0]
        src_image = str(head.get("source_image", ""))
        src_npy   = str(head.get("source_npy",   ""))
        src_path  = str(head.get("source_path",  ""))
        haystack = " ".join((src_image, src_npy, src_path)).upper()
        if pats and any(p in haystack for p in pats):
            continue

        try:
            full, _ = reassemble_image(patch_root, idx)
            stats = image_noise_stats(
                full,
                channel_names=channel_names,
                hp_var_thresh=hp_var_thresh,
                hf_energy_thresh=hf_energy_thresh,
                blur_sigma=blur_sigma,
                low_radius_frac=low_radius_frac,
                dead_std=dead_std,
            )
        except Exception as e:  # pragma: no cover -- robustness over batches
            records.append({
                "image_index":   int(idx),
                "source_image":  src_image,
                "source_npy":    src_npy,
                "source_path":   src_path,
                "n_patches":     len(recs),
                "error":         f"{type(e).__name__}: {e}",
                "flagged":       False,
            })
            continue

        rec = {
            "image_index":           int(idx),
            "source_image":          src_image,
            "source_npy":            src_npy,
            "source_path":           src_path,
            "n_patches":             len(recs),
            "worst_channel":         stats["worst_channel"],
            "worst_hp_var_ratio":    stats["worst_hp_var_ratio"],
            "worst_hf_energy_frac":  stats["worst_hf_energy_frac"],
            "noisy_channels":        stats["noisy_channels"],
            "flagged":               stats["flagged"],
            "per_channel":           stats["per_channel"],
        }
        for ch in stats["per_channel"]:
            name = ch["channel"]
            rec[f"hp_{name}"]  = ch["hp_var_ratio"]
            rec[f"hf_{name}"]  = ch["hf_energy_frac"]
            rec[f"std_{name}"] = ch["std"]
            rec[f"dyn_{name}"] = ch["dyn"]
        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def summarize_scores(records: list[dict], top_n: int = 10) -> str:
    """Return a human-readable summary of the worst-N images by ``hp_var_ratio``."""
    scored = [r for r in records if "error" not in r]
    errs   = [r for r in records if "error" in r]
    if not scored:
        return f"(no images scored; {len(errs)} errors)"

    scored.sort(key=lambda r: r["worst_hp_var_ratio"], reverse=True)
    n_flagged = sum(1 for r in scored if r["flagged"])
    hp = np.array([r["worst_hp_var_ratio"] for r in scored])
    hf = np.array([r["worst_hf_energy_frac"] for r in scored])
    lines = [
        f"scored {len(scored)} images  ({n_flagged} flagged, {len(errs)} errors)",
        f"  hp_var_ratio   min={hp.min():.3f}  p50={np.percentile(hp, 50):.3f}  "
        f"p90={np.percentile(hp, 90):.3f}  p95={np.percentile(hp, 95):.3f}  max={hp.max():.3f}",
        f"  hf_energy_frac min={hf.min():.3f}  p50={np.percentile(hf, 50):.3f}  "
        f"p90={np.percentile(hf, 90):.3f}  p95={np.percentile(hf, 95):.3f}  max={hf.max():.3f}",
        "",
        f"worst {top_n} images by hp_var_ratio:",
        f"  {'idx':>5s}  {'hp':>6s}  {'hf':>6s}  flag  src",
    ]
    for r in scored[:top_n]:
        src = r["source_npy"] or r["source_image"] or r["source_path"]
        lines.append(
            f"  {r['image_index']:>5d}  "
            f"{r['worst_hp_var_ratio']:.3f}  {r['worst_hf_energy_frac']:.3f}  "
            f"{'YES ' if r['flagged'] else '    '}  {src}"
        )
    if errs:
        lines.append("")
        lines.append(f"{len(errs)} errored images:")
        for r in errs[:5]:
            lines.append(f"  idx={r['image_index']}  err={r.get('error','?')}")
    return "\n".join(lines)


def flagged_source_names(records: list[dict]) -> list[str]:
    """Return unique ``source_npy`` names of flagged records (sorted)."""
    names: set[str] = set()
    for r in records:
        if r.get("flagged") and r.get("source_npy"):
            names.add(str(r["source_npy"]))
    return sorted(names)
