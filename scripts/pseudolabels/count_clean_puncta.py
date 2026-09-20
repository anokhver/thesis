#!/usr/bin/env python3
"""Count final PRE/POST/overlap puncta for clean source images.

The counts are computed from the saved binary PRE and POST masks, not from
the detector candidate counts in ``puncta_index.csv``. Missing overlap masks
are rendered from those two masks and saved beside them.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from synaptic_ssl.pseudolabels.puncta_common import (
    DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    mask_overlap,
)


FIELDS = (
    "session",
    "source_npy",
    "pre_mask_filename",
    "post_mask_filename",
    "overlap_mask_filename",
    "height",
    "width",
    "n_pre_puncta",
    "n_post_puncta",
    "n_overlapping_puncta",
    "n_pre_pixels",
    "n_post_pixels",
    "n_overlap_pixels",
    "overlap_rule",
    "overlap_max_distance_px",
)


def _load_flagged(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {Path(name).name for name in data.get("flagged_sources", [])}


def _iter_index_rows(mask_root: Path):
    for index_path in sorted(mask_root.glob("*/puncta_index.csv")):
        with index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield index_path.parent.name, row


def build_counts(
    mask_root: Path,
    output_csv: Path,
    *,
    flagged_json: Path | None = None,
    overlap_max_distance_px: float = DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    save_overlap_masks: bool = True,
) -> int:
    """Write one count row per unflagged PRE/POST mask pair.

    Overlap uses fixed-distance, one-to-one centroid pairing. The resulting
    masks and CSV document the physical cutoff used for the pairing.
    """
    flagged = _load_flagged(flagged_json) if flagged_json else set()
    rows = []
    seen: set[tuple[str, str]] = set()

    for session, source in _iter_index_rows(mask_root):
        source_name = Path(source["source_npy"]).name
        key = (session, source_name)
        if key in seen or source_name in flagged:
            continue
        seen.add(key)

        session_dir = mask_root / session
        pre_path = session_dir / source["pre_mask_filename"]
        post_path = session_dir / source["post_mask_filename"]
        if not pre_path.exists() or not post_path.exists():
            continue
        pre_mask = np.load(pre_path).astype(bool)
        post_mask = np.load(post_path).astype(bool)
        overlap_mask, n_pre, n_post, n_overlap = mask_overlap(
            pre_mask, post_mask, max_distance_px=overlap_max_distance_px
        )

        overlap_path = session_dir / f"{Path(source_name).stem}_overlap.npy"
        if save_overlap_masks:
            np.save(overlap_path, overlap_mask.astype(np.uint8))

        rows.append({
            "session": session,
            "source_npy": source_name,
            "pre_mask_filename": pre_path.name,
            "post_mask_filename": post_path.name,
            "overlap_mask_filename": overlap_path.name,
            "height": int(pre_mask.shape[0]),
            "width": int(pre_mask.shape[1]),
            "n_pre_puncta": n_pre,
            "n_post_puncta": n_post,
            "n_overlapping_puncta": n_overlap,
            "n_pre_pixels": int(pre_mask.sum()),
            "n_post_pixels": int(post_mask.sum()),
            "n_overlap_pixels": int(overlap_mask.sum()),
            "overlap_rule": "fixed_distance_one_to_one",
            "overlap_max_distance_px": overlap_max_distance_px,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["session"], row["source_npy"])))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask_root", type=Path, default=Path("data/pseudolabels/puncta"))
    parser.add_argument(
        "--flagged_json", type=Path,
        default=Path("data/data_analysis/flagged_sources.json"),
        help="JSON denylist; listed source images are excluded as non-clean.",
    )
    parser.add_argument(
        "--output_csv", type=Path,
        default=Path("data/pseudolabels/puncta/puncta_counts_clean.csv"),
    )
    parser.add_argument(
        "--overlap_max_distance_px", type=float,
        default=DEFAULT_OVERLAP_MAX_DISTANCE_PX,
        help=(
            "Maximum PRE/POST centroid distance for one-to-one pairing "
            f"(default: {DEFAULT_OVERLAP_MAX_DISTANCE_PX:.3f} px = 252 nm at 107 nm/px)."
        ),
    )
    parser.add_argument("--no_save_overlap_masks", action="store_true")
    args = parser.parse_args()
    count = build_counts(
        args.mask_root,
        args.output_csv,
        flagged_json=args.flagged_json,
        overlap_max_distance_px=args.overlap_max_distance_px,
        save_overlap_masks=not args.no_save_overlap_masks,
    )
    print(f"wrote {count} clean images to {args.output_csv}")


if __name__ == "__main__":
    main()