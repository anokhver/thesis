#!/usr/bin/env python3
"""Score every source image under a patches root for noise-floor pathology.

Output:

* ``--out-csv``   (default ``data/noise_scores.csv``) — full per-image table.
* ``--out-json``  (default ``data/flagged_sources.json``) — denylist of
  ``source_npy`` strings ready to pass into
  :class:`synaptic_ssl.utils_data.PatchDataset(exclude_sources=...)`.

Usage::

    python scripts/score_image_noise.py \\
        --patch-root data/patches_128_from_zip \\
        --exclude-patterns KONTROLA \\
        --hp-thresh 0.55 --hf-thresh 0.50

See :mod:`synaptic_ssl.utils_data.noise_detection` for the metric definitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Make ``src/`` importable when running from a fresh checkout.
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from synaptic_ssl.utils_data.noise_detection import (  # noqa: E402
    DEFAULT_HF_ENERGY_FRAC, DEFAULT_HP_VAR_RATIO,
    flagged_source_names, score_image_set, summarize_scores,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score reassembled source images for percentile-stretched-noise "
                    "pathology and emit a denylist for PatchDataset.",
    )
    p.add_argument("--patch-root", required=True, type=Path,
                   help="Root of patches_128_from_zip (flat or nested layout).")
    p.add_argument("--out-csv", type=Path, default=Path("data/noise_scores.csv"))
    p.add_argument("--out-json", type=Path, default=Path("data/flagged_sources.json"))
    p.add_argument("--exclude-patterns", nargs="*", default=["KONTROLA"],
                   help="Source-name substrings to skip entirely.")
    p.add_argument("--channel-names", nargs="*", default=None,
                   help="Optional channel names for nicer per-channel columns.")
    p.add_argument("--hp-thresh", type=float, default=DEFAULT_HP_VAR_RATIO,
                   help="High-pass / total variance threshold (default %(default).2f).")
    p.add_argument("--hf-thresh", type=float, default=DEFAULT_HF_ENERGY_FRAC,
                   help="FFT outer-disk energy-fraction threshold (default %(default).2f).")
    p.add_argument("--top-percentile", type=float, default=None,
                   help="If set (e.g. 15), additionally flag the worst N%% of "
                        "images by hp_var_ratio. Combined OR-wise with the "
                        "absolute thresholds. Useful when image quality varies "
                        "between acquisition dates.")
    p.add_argument("--blur-sigma", type=float, default=1.5,
                   help="Gaussian sigma (px) defining 'low-frequency'.")
    p.add_argument("--low-radius-frac", type=float, default=0.25,
                   help="FFT inner-disk radius as fraction of r_max.")
    p.add_argument("--min-channels", type=int, default=2,
                   help="Flag any image with fewer than this many channels "
                        "(default %(default)d, e.g. drops 1-channel acquisitions "
                        "in a 3-channel dataset). Pass 0 to disable.")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of worst images to print in the summary.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("score_image_noise")

    args = _parse_args()
    log.info(f"patch_root  = {args.patch_root}")
    log.info(f"out_csv     = {args.out_csv}")
    log.info(f"out_json    = {args.out_json}")
    log.info(f"thresholds  hp>={args.hp_thresh:.3f}  hf>={args.hf_thresh:.3f}")
    if args.top_percentile is not None:
        log.info(f"top-percentile fallback: worst {args.top_percentile:g}%")

    records = score_image_set(
        args.patch_root,
        exclude_patterns=args.exclude_patterns or None,
        channel_names=args.channel_names,
        blur_sigma=args.blur_sigma,
        low_radius_frac=args.low_radius_frac,
        hp_var_thresh=args.hp_thresh,
        hf_energy_thresh=args.hf_thresh,
        min_channels=args.min_channels if args.min_channels > 0 else None,
        progress=True,
    )

    # Optional: OR-wise flag the worst N% of images by hp_var_ratio.
    if args.top_percentile is not None:
        import numpy as np
        scored = [r for r in records if "error" not in r]
        if scored:
            hp = np.array([r["worst_hp_var_ratio"] for r in scored])
            cutoff = float(np.percentile(hp, 100.0 - args.top_percentile))
            n_added = 0
            for r in scored:
                if r["worst_hp_var_ratio"] >= cutoff and not r["flagged"]:
                    r["flagged"] = True
                    extra = f"top{args.top_percentile:g}pct(hp>={cutoff:.3f})"
                    r["noisy_channels"] = (
                        r["noisy_channels"] + ";" + extra
                        if r["noisy_channels"] else extra
                    )
                    n_added += 1
            log.info(
                f"--top-percentile {args.top_percentile}: cutoff hp>={cutoff:.3f}, "
                f"added {n_added} additional flags"
            )

    # --- CSV --------------------------------------------------------------
    # Stable column order: scalar fields first, then sorted per-channel cols.
    scalar_fields = [
        "image_index", "source_image", "source_npy", "source_path", "n_patches",
        "n_channels", "worst_channel", "worst_hp_var_ratio", "worst_hf_energy_frac",
        "noisy_channels", "flag_reason", "flagged", "error",
    ]
    per_channel_fields: set[str] = set()
    for r in records:
        for k in r:
            if k.startswith(("hp_", "hf_", "std_", "dyn_")):
                per_channel_fields.add(k)
    fields = scalar_fields + sorted(per_channel_fields)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fields}
            w.writerow(row)
    log.info(f"wrote CSV: {args.out_csv}  ({len(records)} rows)")

    # --- summary ---------------------------------------------------------
    print()
    print(summarize_scores(records, top_n=args.top_n))
    print()

    # --- JSON denylist ---------------------------------------------------
    denylist = flagged_source_names(records)
    payload = {
        "patch_root":        str(args.patch_root),
        "hp_var_thresh":     args.hp_thresh,
        "hf_energy_thresh":  args.hf_thresh,
        "blur_sigma":        args.blur_sigma,
        "low_radius_frac":   args.low_radius_frac,
        "top_percentile":    args.top_percentile,
        "min_channels":      args.min_channels,
        "exclude_patterns":  list(args.exclude_patterns or []),
        "n_images":          len(records),
        "n_flagged":         len(denylist),
        "flagged_sources":   denylist,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    log.info(f"wrote denylist: {args.out_json}  ({len(denylist)} sources)")


if __name__ == "__main__":
    main()
