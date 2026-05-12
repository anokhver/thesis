"""Detect damaged patches by per-source-image intensity outlier analysis.

Flag patches if base or mean intensity is a MAD outlier, mean is saturated,
or max is near zero. Return records ready for ``index.csv`` with ``mean``, ``std``,
``min``, ``max``, ``low_p5``, ``high_p95``, ``damaged``, and ``damage_reasons``.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Default thresholds. Tuned conservatively so only obviously-bad patches are
# flagged when there's no a-priori knowledge.
DEFAULT_MAD_K           = 5.0       # robust z-score equivalent of ~5 sigma
DEFAULT_SATURATION_MEAN = 0.97      # patch is essentially white
DEFAULT_DEAD_MAX        = 1e-4      # patch is essentially black
DEFAULT_LOW_PERCENTILE  = 5.0       # "base intensity" = 5th percentile
DEFAULT_HIGH_PERCENTILE = 95.0
MAD_FLOOR               = 1e-4      # avoid divide-by-zero on uniform groups


# ---------------------------------------------------------------------------
# Per-patch statistics
# ---------------------------------------------------------------------------

def patch_stats(
    patch: np.ndarray,
    low_p: float = DEFAULT_LOW_PERCENTILE,
    high_p: float = DEFAULT_HIGH_PERCENTILE,
) -> dict[str, float]:
    """Compute scalar intensity stats over all channels and pixels.

    Return ``mean``, ``std``, ``min``, ``max``, ``low_p{n}``, and ``high_p{n}``.
    """
    arr = np.asarray(patch, dtype=np.float64)
    return {
        "mean":            float(arr.mean()),
        "std":             float(arr.std()),
        "min":             float(arr.min()),
        "max":             float(arr.max()),
        f"low_p{int(low_p)}":   float(np.percentile(arr, low_p)),
        f"high_p{int(high_p)}": float(np.percentile(arr, high_p)),
    }


# ---------------------------------------------------------------------------
# Robust outlier detection
# ---------------------------------------------------------------------------

def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    """Return (median, MAD) of a 1-D iterable. MAD is floored to MAD_FLOOR
    to avoid divide-by-zero."""
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, max(mad, MAD_FLOOR)


def flag_damaged(
    records: list[dict],
    *,
    group_key:        str   = "source_image",
    mad_k:            float = DEFAULT_MAD_K,
    saturation_mean:  float = DEFAULT_SATURATION_MEAN,
    dead_max:         float = DEFAULT_DEAD_MAX,
    low_p:            float = DEFAULT_LOW_PERCENTILE,
) -> list[dict]:
    """Annotate each record with ``damaged`` and ``damage_reasons``.

    Mutate and return the input list. Records must already contain ``patch_stats`` keys.
    Run MAD outlier checks within each ``group_key`` group. Skip them for single-patch groups.
    """
    low_key = f"low_p{int(low_p)}"

    # Build per-image medians + MADs
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[group_key]].append(r)

    group_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for gkey, recs in groups.items():
        if len(recs) < 2:
            group_stats[gkey] = {}  # skip MAD checks for singleton groups
            continue
        means    = [r["mean"]   for r in recs]
        lows     = [r[low_key]  for r in recs]
        group_stats[gkey] = {
            "mean":  _median_mad(means),
            low_key: _median_mad(lows),
        }

    n_flagged = 0
    for r in records:
        reasons: list[str] = []

        # 1. saturation (whole patch nearly white -- light leak / overexposure)
        if r["mean"] >= saturation_mean:
            reasons.append(f"saturated(mean={r['mean']:.3f}>={saturation_mean})")

        # 2. dead (whole patch nearly black -- sensor dropout / acquisition gap)
        if r["max"] <= dead_max:
            reasons.append(f"dead(max={r['max']:.2e}<={dead_max})")

        gst = group_stats.get(r[group_key], {})

        # 3. mean outlier vs same-image peers
        if "mean" in gst:
            med, mad = gst["mean"]
            z = (r["mean"] - med) / mad
            if abs(z) > mad_k:
                reasons.append(
                    f"mean_outlier(z={z:+.2f},img_med={med:.3f},mad={mad:.3f})"
                )

        # 4. base-intensity outlier vs same-image peers
        if low_key in gst:
            med, mad = gst[low_key]
            z = (r[low_key] - med) / mad
            if abs(z) > mad_k:
                reasons.append(
                    f"base_outlier(z={z:+.2f},img_med={med:.3f},mad={mad:.3f})"
                )

        r["damaged"]        = len(reasons) > 0
        r["damage_reasons"] = ";".join(reasons)
        if r["damaged"]:
            n_flagged += 1

    return records


# ---------------------------------------------------------------------------
# Convenience: load patches from disk + index, compute stats, run flag.
# ---------------------------------------------------------------------------

def attach_stats(
    records: Iterable[dict],
    patches_root: Path,
    *,
    low_p:  float = DEFAULT_LOW_PERCENTILE,
    high_p: float = DEFAULT_HIGH_PERCENTILE,
) -> list[dict]:
    """Load each patch from disk and merge ``patch_stats`` into its record.

    Require a ``filename`` field on each record. Return records with stat keys added.
    """
    out: list[dict] = []
    for r in records:
        arr = np.load(Path(patches_root) / r["filename"], allow_pickle=False)
        s = patch_stats(arr, low_p=low_p, high_p=high_p)
        merged = dict(r); merged.update(s)
        out.append(merged)
    return out


def coerce_record_floats(records: list[dict], keys: Iterable[str]) -> None:
    """When records are loaded from CSV every value is a string; cast the
    listed numeric keys to float in-place."""
    for r in records:
        for k in keys:
            if k in r and isinstance(r[k], str) and r[k] != "":
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def damage_summary(records: list[dict]) -> str:
    """Return a multi-line text summary of `flag_damaged` output."""
    n = len(records)
    if n == 0:
        return "no records"
    flagged = [r for r in records if r.get("damaged")]
    by_image: dict[str, int] = defaultdict(int)
    by_reason: dict[str, int] = defaultdict(int)
    for r in flagged:
        by_image[r.get("source_image", "?")] += 1
        for reason in (r.get("damage_reasons") or "").split(";"):
            if not reason:
                continue
            tag = reason.split("(", 1)[0]
            by_reason[tag] += 1

    lines = [
        f"Damage summary: {len(flagged)}/{n} patches "
        f"({100*len(flagged)/n:.2f}%) flagged",
    ]
    if by_reason:
        lines.append("  by reason:")
        for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:20s} {v}")
    if by_image:
        top = sorted(by_image.items(), key=lambda kv: -kv[1])[:10]
        lines.append("  top source images by flag count:")
        for k, v in top:
            lines.append(f"    {v:4d}  {k}")
    return "\n".join(lines)
