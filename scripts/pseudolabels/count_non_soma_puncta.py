#!/usr/bin/env python3
"""Remove saved puncta that touch somas and count the remaining components.

This is post-processing only: it never reruns puncta detection. It reads each
``puncta_index.csv`` under ``--mask_root``, loads its PRE and POST masks, and
loads matching soma and dendrite masks from their roots (stitching per-patch
tiles where needed). It writes cleaned masks under ``--output_root``.
``*_combined.npy`` encodes background=0, PRE-only=1, POST-only=2, both=3.

Each output CSV row carries every column from the source ``puncta_index.csv``
row (session, original mask filenames, detector counts, etc.) plus:
``n_somas`` (neuron count), soma and dendrite area/fraction, and non-soma
PRE/POST/overlap mask filenames, component counts, and pixel counts.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from skimage.measure import label

from synaptic_ssl.pseudolabels.puncta_common import (
    DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    mask_overlap,
)


def _stitch_tiled_mask(mask_dir: Path, stem: str, kind: str,
                       full_shape: tuple[int, int]) -> np.ndarray:
    """Stitch ``{stem}_r##_c##_{kind}.npy`` tiles into one (H, W) bool mask."""
    H, W = full_shape
    pattern = f"{stem}_r??_c??_{kind}.npy"
    files = sorted(mask_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no patches matching {pattern} in {mask_dir}")
    first = np.load(files[0])
    ps = first.shape[0]
    nr, nc = H // ps, W // ps
    if len(files) != nr * nc:
        raise ValueError(
            f"expected {nr*nc} tiles ({nr}x{nc} at ps={ps}) for {stem}, "
            f"got {len(files)} in {mask_dir}"
        )
    rx = re.compile(r"_r(\d{2})_c(\d{2})_[a-z]+\.npy$")
    out = np.zeros((H, W), dtype=bool)
    for fp in files:
        m = rx.search(fp.name)
        if not m:
            raise ValueError(f"bad tile filename: {fp.name}")
        rr, cc = int(m.group(1)), int(m.group(2))
        out[rr*ps:(rr+1)*ps, cc*ps:(cc+1)*ps] = np.load(fp).astype(bool)
    return out


def _load_structural_mask(
    mask_dir: Path,
    stem: str,
    kind: str,
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Load a full structural mask or stitch its per-patch tiles."""
    full = mask_dir / f"{stem}_{kind}.npy"
    if full.exists():
        return np.load(full).astype(bool)
    return _stitch_tiled_mask(mask_dir, stem, kind, full_shape)


def remove_soma_components(puncta_mask: np.ndarray, soma_mask: np.ndarray) -> np.ndarray:
    """Keep connected puncta components that do not overlap a soma pixel."""
    puncta = np.asarray(puncta_mask, dtype=bool)
    soma = np.asarray(soma_mask, dtype=bool)
    if puncta.shape != soma.shape:
        raise ValueError(f"puncta/soma shape mismatch: {puncta.shape} vs {soma.shape}")
    labels = label(puncta, connectivity=2)
    soma_labels = np.unique(labels[soma])
    return (labels > 0) & ~np.isin(labels, soma_labels[soma_labels > 0])


def count_somas(soma_mask: np.ndarray) -> int:
    """Number of connected soma components (neurons) in the mask."""
    return int(label(np.asarray(soma_mask, dtype=bool), connectivity=2).max())


def _iter_index_rows(mask_root: Path):
    for index_path in sorted(mask_root.glob("*/puncta_index.csv")):
        with index_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                yield index_path.parent.name, row


def build_non_soma_counts(
    mask_root: Path,
    soma_root: Path,
    dendrite_root: Path,
    output_root: Path,
    output_csv: Path,
    *,
    overlap_max_distance_px: float = DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    limit: int | None = None,
    save_masks: bool = True,
) -> int:
    """Write soma-free masks and one count row for each saved puncta image.

    Each row keeps every field from the source ``puncta_index.csv`` row plus
    soma/dendrite area and the neuron and non-soma puncta counts. ``limit``
    caps processed images (e.g. for a quick example run).
    """
    rows = []
    seen: set[tuple[str, str]] = set()
    fieldnames: list[str] = []
    for session, source in _iter_index_rows(mask_root):
        if limit is not None and len(seen) >= limit:
            break
        source_name = Path(source["source_npy"]).name
        key = (session, source_name)
        if key in seen:
            continue
        seen.add(key)

        source_dir = mask_root / session
        pre_path = source_dir / source["pre_mask_filename"]
        post_path = source_dir / source["post_mask_filename"]
        if not pre_path.exists() or not post_path.exists():
            continue

        pre_mask = np.load(pre_path).astype(bool)
        post_mask = np.load(post_path).astype(bool)
        stem = Path(source_name).stem
        try:
            soma_mask = _load_structural_mask(
                soma_root / session, stem, "soma", pre_mask.shape
            )
            dendrite_mask = _load_structural_mask(
                dendrite_root / session, stem, "dend", pre_mask.shape
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"skip {session}/{source_name}: {error}")
            continue
        pre_clean = remove_soma_components(pre_mask, soma_mask)
        post_clean = remove_soma_components(post_mask, soma_mask)
        overlap_mask, n_pre, n_post, n_overlap = mask_overlap(
            pre_clean, post_clean, max_distance_px=overlap_max_distance_px
        )
        combined_mask = pre_clean.astype(np.uint8) + 2 * post_clean.astype(np.uint8)
        n_somas = count_somas(soma_mask)
        image_area = int(pre_mask.size)
        soma_area = int(soma_mask.sum())
        dendrite_area = int(dendrite_mask.sum())

        output_dir = output_root / session
        pre_output = output_dir / f"{stem}_pre_non_soma.npy"
        post_output = output_dir / f"{stem}_post_non_soma.npy"
        overlap_output = output_dir / f"{stem}_overlap_non_soma.npy"
        combined_output = output_dir / f"{stem}_combined_non_soma.npy"
        if save_masks:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(pre_output, pre_clean.astype(np.uint8))
            np.save(post_output, post_clean.astype(np.uint8))
            np.save(overlap_output, overlap_mask.astype(np.uint8))
            np.save(combined_output, combined_mask)

        row = {"session": session, **source}
        row.update({
            "n_somas": n_somas,
            "image_area_pixels": image_area,
            "soma_area_pixels": soma_area,
            "soma_area_fraction": soma_area / image_area,
            "dendrite_area_pixels": dendrite_area,
            "dendrite_area_fraction": dendrite_area / image_area,
            "pre_non_soma_mask_filename": pre_output.name if save_masks else "",
            "post_non_soma_mask_filename": post_output.name if save_masks else "",
            "overlap_non_soma_mask_filename": overlap_output.name if save_masks else "",
            "combined_non_soma_mask_filename": combined_output.name if save_masks else "",
            "n_pre_non_soma": n_pre,
            "n_post_non_soma": n_post,
            "n_overlap_non_soma": n_overlap,
            "n_pre_non_soma_pixels": int(pre_clean.sum()),
            "n_post_non_soma_pixels": int(post_clean.sum()),
            "n_overlap_non_soma_pixels": int(overlap_mask.sum()),
            "overlap_max_distance_px": overlap_max_distance_px,
        })
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
        rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["session"], row["source_npy"])))
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask_root", type=Path, default=Path("data/pseudolabels/puncta"))
    parser.add_argument("--soma_root", type=Path, default=Path("data/pseudolabels/soma"))
    parser.add_argument("--dendrite_root", type=Path, default=Path("data/pseudolabels/dend"))
    parser.add_argument(
        "--output_root", type=Path,
        default=Path("data/pseudolabels/puncta_non_soma"),
    )
    parser.add_argument(
        "--output_csv", type=Path,
        default=Path("data/pseudolabels/puncta_non_soma/puncta_counts.csv"),
    )
    parser.add_argument(
        "--overlap_max_distance_px", type=float,
        default=DEFAULT_OVERLAP_MAX_DISTANCE_PX,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N source images (e.g. for a quick example run).",
    )
    parser.add_argument(
        "--counts_only", action="store_true",
        help="Write counts/coverage CSV without saving duplicate cleaned mask files.",
    )
    args = parser.parse_args()
    count = build_non_soma_counts(
        args.mask_root,
        args.soma_root,
        args.dendrite_root,
        args.output_root,
        args.output_csv,
        overlap_max_distance_px=args.overlap_max_distance_px,
        limit=args.limit,
        save_masks=not args.counts_only,
    )
    print(f"wrote {count} non-soma puncta records to {args.output_csv}")


if __name__ == "__main__":
    main()