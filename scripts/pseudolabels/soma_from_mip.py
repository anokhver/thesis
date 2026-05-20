#!/usr/bin/env python3
r"""Run the pseudo-FDT soma pipeline on full-MIP .npy files or on a
tiled-patches session (stitch → soma → slice back per patch).

Two input modes (mirroring ``scripts/tile_from_mip.py``):

  1. **MIP mode** -- inputs are full ``(C, H, W)`` ``.npy`` files (the
     output of the no-tile preprocessing pipeline). One
     ``<stem>_soma.npy`` (bool, ``(H, W)``) is written per source image
     plus a per-folder ``soma_index.csv``.

  2. **Patches mode** -- inputs are tiled patch folders containing
     ``index.csv`` + ``<stem>_r##_c##.npy``. The structural channel of
     all patches sharing a ``source_npy`` is stitched into one full
     image, soma is run once at full scale (avoids per-patch border
     artefacts), then the resulting mask is sliced back so each patch
     gets a ``<patch_stem>_soma.npy`` of shape ``(ps, ps)``. A
     ``soma_index.csv`` records per-patch diagnostics.

Usage::

    # MIP mode, single folder:
    python scripts/soma_from_mip.py \
        --input_dir  Microscopy_no_patch/SessionName \
        --output_dir Microscopy_soma/SessionName

    # MIP mode, all session subfolders under a root:
    python scripts/soma_from_mip.py \
        --input_root  Microscopy_no_patch \
        --output_root Microscopy_soma

    # Patches mode, single session:
    python scripts/soma_from_mip.py \
        --input_patches  Microscopy/SessionName \
        --output_dir     Microscopy_soma/SessionName

    # Patches mode, all session subfolders under a root:
    python scripts/soma_from_mip.py \
        --input_patch_root  Microscopy \
        --output_root       Microscopy_soma

    # Override soma cfg (any DEFAULT_SOMA_CFG key, JSON):
    python scripts/soma_from_mip.py --input_patch_root ... --output_root ... \
        --soma_cfg '{"pixel_size_nm": 65.0, "blob_fdt_pct": 97.0}'
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "src"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.pseudolabels.soma_fdt import (  # noqa: E402
    compute_pseudo_fdt,
    extract_soma_mask,
    run_soma_on_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SOMA_INDEX_FIELDS: tuple[str, ...] = (
    "source_npy",          # basename of the full-MIP .npy that was processed
    "mask_filename",       # basename of the (H, W) bool .npy mask written
    "height",
    "width",
    "structural_channel",
    "n_blobs",             # connected components in the final mask
    "threshold",           # absolute FDT threshold used
    "threshold_method",    # 'percentile' | 'otsu' | 'multi_otsu' | 'degenerate'
    "frac_mask",           # mean of the bool mask
)


PATCH_SOMA_INDEX_FIELDS: tuple[str, ...] = (
    "filename",            # basename of the patch (matches the tiler's index.csv)
    "mask_filename",       # basename of the per-patch (ps, ps) bool mask
    "source_npy",          # basename of the full-MIP this patch came from
    "image_index",
    "grid_row",
    "grid_col",
    "patch_size",
    "n_blobs_full",        # CCs in the FULL stitched mask (per source)
    "threshold",           # FDT threshold from the full-image extract
    "threshold_method",
    "frac_mask_patch",     # mean of the bool mask sliced for this patch
)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def soma_one(
    npy_path: Path,
    output_dir: Path,
    soma_cfg: dict | None,
    structural_channel: int,
    overwrite: bool,
) -> dict | None:
    """Run soma on one full-MIP .npy. Save ``<stem>_soma.npy``. Return record."""
    mask_path = output_dir / f"{npy_path.stem}_soma.npy"
    if mask_path.exists() and not overwrite:
        logger.info(f"  {npy_path.name}: mask exists, skipping (use --overwrite to redo)")
        try:
            existing = np.load(mask_path, mmap_mode="r")
            H, W = existing.shape
            frac = float(existing.mean())
        except Exception:
            H = W = 0
            frac = float("nan")
        return {
            "source_npy": npy_path.name,
            "mask_filename": mask_path.name,
            "height": H,
            "width": W,
            "structural_channel": structural_channel,
            "n_blobs": -1,
            "threshold": float("nan"),
            "threshold_method": "skipped",
            "frac_mask": frac,
        }

    try:
        structural, _fdt_info, blob_info = run_soma_on_image(
            npy_path, soma_cfg, structural_channel=structural_channel
        )
    except Exception as e:
        logger.error(f"  {npy_path.name}: FAILED ({e})")
        return None

    mask = blob_info["mask"].astype(bool)
    np.save(mask_path, mask)
    H, W = mask.shape
    logger.info(
        f"  {npy_path.name}: {H}×{W} → {blob_info['n_blobs']} soma(s), "
        f"thr={blob_info['threshold']:.3g} ({blob_info['threshold_method']})"
    )

    rec = {
        "source_npy": npy_path.name,
        "mask_filename": mask_path.name,
        "height": H,
        "width": W,
        "structural_channel": structural_channel,
        "n_blobs": int(blob_info["n_blobs"]),
        "threshold": float(blob_info["threshold"]),
        "threshold_method": str(blob_info["threshold_method"]),
        "frac_mask": float(mask.mean()),
    }

    del structural, mask, blob_info
    gc.collect()
    return rec


# ---------------------------------------------------------------------------
# Single-folder processing
# ---------------------------------------------------------------------------

def _soma_one_worker(args: tuple) -> dict | None:
    npy_path, output_dir, soma_cfg, structural_channel, overwrite = args
    return soma_one(
        Path(npy_path), Path(output_dir), soma_cfg, structural_channel, overwrite
    )


def process_dir(
    input_dir: Path,
    output_dir: Path,
    soma_cfg: dict | None,
    structural_channel: int,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run soma on all full-MIP .npy in ``input_dir``. Write masks + soma_index.csv."""
    npy_files = sorted(p for p in input_dir.glob("*.npy") if p.name != "index.csv")
    if not npy_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(p), str(output_dir), soma_cfg, structural_channel, overwrite)
        for p in npy_files
    ]

    records: list[dict] = []
    if workers > 1 and len(npy_files) > 1:
        n_workers = min(workers, len(npy_files))
        logger.info(f"  Running {len(npy_files)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_soma_one_worker, item): item[0] for item in work_items}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    logger.error(f"  Worker crashed on {futures[fut]}: {e}")
                    continue
                if rec is not None:
                    records.append(rec)
    else:
        for item in work_items:
            rec = _soma_one_worker(item)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r["source_npy"])

    csv_path = output_dir / "soma_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(SOMA_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  soma_index.csv → {csv_path}  ({len(records)} images)")
    else:
        logger.warning(f"  No masks produced for {input_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Patches mode: stitch → soma → slice back
# ---------------------------------------------------------------------------

def _stitch_structural(
    patch_dir: Path,
    rows: list[dict],
    structural_channel: int,
) -> tuple[np.ndarray, int, int, int]:
    """Stitch the structural channel of all patches for one source image.

    Returns ``(structural, n_rows, n_cols, patch_size)``. Only the
    structural channel is materialised; the other channels are skipped
    because soma extraction does not need them.
    """
    ps_set = {int(r["patch_size"]) for r in rows}
    if len(ps_set) != 1:
        raise ValueError(f"mixed patch_size in {rows[0]['source_npy']!r}: {ps_set}")
    ps = ps_set.pop()
    n_rows = max(int(r["grid_row"]) for r in rows) + 1
    n_cols = max(int(r["grid_col"]) for r in rows) + 1

    structural = np.zeros((n_rows * ps, n_cols * ps), dtype=np.float32)
    seen = np.zeros((n_rows, n_cols), dtype=bool)
    for r in rows:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        patch = np.load(patch_dir / r["filename"])
        if patch.ndim != 3:
            raise ValueError(
                f"patch {r['filename']} has shape {patch.shape}, expected (C, ps, ps)"
            )
        if structural_channel >= patch.shape[0]:
            raise ValueError(
                f"patch {r['filename']} has {patch.shape[0]} channels but "
                f"structural_channel={structural_channel}"
            )
        if patch.shape[1] != ps or patch.shape[2] != ps:
            raise ValueError(
                f"patch {r['filename']} spatial shape {patch.shape[1:]} != ({ps}, {ps})"
            )
        y0, x0 = gr * ps, gcol * ps
        structural[y0:y0 + ps, x0:x0 + ps] = patch[structural_channel].astype(np.float32)
        seen[gr, gcol] = True

    if not seen.all():
        missing = [(int(r), int(c)) for r, c in zip(*np.where(~seen))]
        raise ValueError(
            f"source {rows[0]['source_npy']!r}: missing patches at "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"({len(missing)} of {n_rows * n_cols} grid cells)"
        )
    return structural, n_rows, n_cols, ps


def soma_for_source(
    patch_dir: Path,
    output_dir: Path,
    source_npy: str,
    rows: list[dict],
    soma_cfg: dict | None,
    structural_channel: int,
    overwrite: bool,
) -> list[dict]:
    """Stitch one source's patches, run soma, write per-patch masks."""
    expected = [
        (r, output_dir / f"{Path(r['filename']).stem}_soma.npy") for r in rows
    ]
    if not overwrite and all(mp.exists() for _, mp in expected):
        logger.info(f"  {source_npy}: all {len(rows)} patch masks exist, skipping")
        # Still emit index rows so the per-folder soma_index.csv is complete.
        out: list[dict] = []
        for r, mp in expected:
            try:
                m = np.load(mp, mmap_mode="r")
                frac = float(m.mean())
            except Exception:
                frac = float("nan")
            out.append({
                "filename": r["filename"],
                "mask_filename": mp.name,
                "source_npy": source_npy,
                "image_index": int(r.get("image_index", -1) or -1),
                "grid_row": int(r["grid_row"]),
                "grid_col": int(r["grid_col"]),
                "patch_size": int(r["patch_size"]),
                "n_blobs_full": -1,
                "threshold": float("nan"),
                "threshold_method": "skipped",
                "frac_mask_patch": frac,
            })
        return out

    structural, n_rows, n_cols, ps = _stitch_structural(
        patch_dir, rows, structural_channel
    )

    fdt_info = compute_pseudo_fdt(structural, soma_cfg)
    blob_info = extract_soma_mask(fdt_info["fdt"], soma_cfg)
    full_mask = blob_info["mask"].astype(bool)

    logger.info(
        f"  {source_npy}: {n_rows}×{n_cols} grid ({n_rows * ps}×{n_cols * ps}) → "
        f"{blob_info['n_blobs']} soma(s), thr={blob_info['threshold']:.3g} "
        f"({blob_info['threshold_method']})"
    )

    records: list[dict] = []
    for r, mp in expected:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        y0, x0 = gr * ps, gcol * ps
        sl = full_mask[y0:y0 + ps, x0:x0 + ps].copy()
        np.save(mp, sl)
        records.append({
            "filename": r["filename"],
            "mask_filename": mp.name,
            "source_npy": source_npy,
            "image_index": int(r.get("image_index", -1) or -1),
            "grid_row": gr,
            "grid_col": gcol,
            "patch_size": ps,
            "n_blobs_full": int(blob_info["n_blobs"]),
            "threshold": float(blob_info["threshold"]),
            "threshold_method": str(blob_info["threshold_method"]),
            "frac_mask_patch": float(sl.mean()),
        })

    del structural, fdt_info, blob_info, full_mask
    gc.collect()
    return records


def _soma_for_source_worker(args: tuple) -> list[dict]:
    patch_dir, output_dir, source_npy, rows, soma_cfg, sc, overwrite = args
    return soma_for_source(
        Path(patch_dir), Path(output_dir), source_npy, rows,
        soma_cfg, sc, overwrite,
    )


def process_patches_dir(
    patch_dir: Path,
    output_dir: Path,
    soma_cfg: dict | None,
    structural_channel: int,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run soma per source image in a tiled-patches session folder."""
    index_csv = patch_dir / "index.csv"
    if not index_csv.exists():
        logger.error(f"  {patch_dir}: index.csv not found, cannot stitch")
        return []

    with open(index_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        logger.warning(f"  {patch_dir}: index.csv is empty")
        return []

    # Group by source image. Prefer source_npy; fall back to image_index.
    groups: dict[str, list[dict]] = {}
    for r in all_rows:
        key = r.get("source_npy") or f"img{int(r.get('image_index', 0)):04d}"
        groups.setdefault(key, []).append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(patch_dir), str(output_dir), key, rows,
         soma_cfg, structural_channel, overwrite)
        for key, rows in sorted(groups.items())
    ]

    records: list[dict] = []
    if workers > 1 and len(work_items) > 1:
        n_workers = min(workers, len(work_items))
        logger.info(
            f"  Stitching + soma on {len(work_items)} source image(s) "
            f"with {n_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_soma_for_source_worker, item): item[2]
                for item in work_items
            }
            for fut in as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception as e:
                    logger.error(f"  Worker crashed on {futures[fut]}: {e}")
    else:
        for item in work_items:
            try:
                records.extend(_soma_for_source_worker(item))
            except Exception as e:
                logger.error(f"  FAILED on {item[2]}: {e}")

    records.sort(
        key=lambda r: (r["source_npy"], r["grid_row"], r["grid_col"])
    )

    csv_path = output_dir / "soma_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(PATCH_SOMA_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  soma_index.csv → {csv_path}  ({len(records)} patches)")
    else:
        logger.warning(f"  No masks produced for {patch_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_soma_cfg(s: str | None) -> dict | None:
    if not s:
        return None
    p = Path(s)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pseudo-FDT soma pipeline on full-MIP .npy files."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dir", type=Path,
        help="MIP mode: single folder of full-MIP .npy files.",
    )
    group.add_argument(
        "--input_root", type=Path,
        help="MIP mode: root containing session subfolders of full-MIP .npy files.",
    )
    group.add_argument(
        "--input_patches", type=Path,
        help="Patches mode: single tiled-patch folder (must contain index.csv).",
    )
    group.add_argument(
        "--input_patch_root", type=Path,
        help="Patches mode: root containing session subfolders of tiled patches.",
    )

    parser.add_argument(
        "--output_dir", type=Path,
        help="Output folder for soma masks (required with --input_dir).",
    )
    parser.add_argument(
        "--output_root", type=Path,
        help="Output root; session subfolders are mirrored (required with --input_root).",
    )
    parser.add_argument(
        "--structural_channel", type=int, default=2,
        help="Index of the structural channel in (C, H, W) (default: 2).",
    )
    parser.add_argument(
        "--soma_cfg", type=str, default=None,
        help="JSON dict or path to .json with DEFAULT_SOMA_CFG overrides.",
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many session folders (0 = all, --input_root only).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Number of parallel workers (0 = auto = max(1, cpu-2), 1 = sequential).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite existing mask files.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without running soma.",
    )

    args = parser.parse_args()

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir")
    if args.input_root and not args.output_root:
        parser.error("--output_root is required when using --input_root")
    if args.input_patches and not args.output_dir:
        parser.error("--output_dir is required when using --input_patches")
    if args.input_patch_root and not args.output_root:
        parser.error("--output_root is required when using --input_patch_root")

    try:
        soma_cfg = _parse_soma_cfg(args.soma_cfg)
    except (json.JSONDecodeError, OSError) as e:
        parser.error(f"--soma_cfg: {e}")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 2)

    # ---- MIP, single folder ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info(f"Mode:       MIP (single folder)")
        logger.info(f"Input:      {args.input_dir}")
        logger.info(f"Output:     {args.output_dir}")
        logger.info(f"Channel:    {args.structural_channel}")
        logger.info(f"Workers:    {workers}")
        logger.info(f"Soma cfg:   {soma_cfg or '(defaults)'}")
        if args.dry_run:
            n = len([p for p in args.input_dir.glob("*.npy") if p.name != "index.csv"])
            logger.info(f"[DRY RUN] Would run soma on {n} .npy file(s).")
            return
        process_dir(
            args.input_dir, args.output_dir, soma_cfg,
            args.structural_channel, workers, args.overwrite,
        )
        return

    # ---- Patches, single folder --------------------------------------------
    if args.input_patches:
        if not args.input_patches.is_dir():
            logger.error(f"Input patches dir does not exist: {args.input_patches}")
            sys.exit(1)
        if not (args.input_patches / "index.csv").exists():
            logger.error(f"index.csv not found in {args.input_patches}")
            sys.exit(1)
        logger.info(f"Mode:       Patches (single folder, stitch → soma → slice)")
        logger.info(f"Input:      {args.input_patches}")
        logger.info(f"Output:     {args.output_dir}")
        logger.info(f"Channel:    {args.structural_channel}")
        logger.info(f"Workers:    {workers}")
        logger.info(f"Soma cfg:   {soma_cfg or '(defaults)'}")
        if args.dry_run:
            with open(args.input_patches / "index.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            sources = {r.get("source_npy") for r in rows}
            logger.info(
                f"[DRY RUN] Would stitch {len(sources)} source image(s) from "
                f"{len(rows)} patches."
            )
            return
        process_patches_dir(
            args.input_patches, args.output_dir, soma_cfg,
            args.structural_channel, workers, args.overwrite,
        )
        return

    # ---- Batch (root) mode -------------------------------------------------
    if args.input_root:
        input_root = args.input_root
        mode_name = "MIP (root)"
        select = lambda d: any(p.name != "index.csv" for p in d.glob("*.npy"))  # noqa: E731
        per_folder = process_dir
    else:
        input_root = args.input_patch_root
        mode_name = "Patches (root, stitch → soma → slice)"
        select = lambda d: (d / "index.csv").exists()  # noqa: E731
        per_folder = process_patches_dir

    if not input_root.is_dir():
        logger.error(f"Input root does not exist: {input_root}")
        sys.exit(1)

    folders = sorted(d for d in input_root.iterdir() if d.is_dir() and select(d))
    if not folders:
        logger.error(
            f"No eligible session subfolders found in {input_root} "
            f"({'index.csv required' if args.input_patch_root else '.npy files required'})"
        )
        sys.exit(1)

    if args.max_folders and args.max_folders > 0:
        folders = folders[:args.max_folders]

    logger.info(f"\n{'='*70}")
    logger.info("Soma Configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Mode:        {mode_name}")
    logger.info(f"Input root:  {input_root}")
    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Channel:     {args.structural_channel}")
    logger.info(f"Workers:     {workers}")
    logger.info(f"Folders:     {len(folders)}")
    logger.info(f"Soma cfg:    {soma_cfg or '(defaults)'}")

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for d in folders:
            if args.input_patch_root:
                with open(d / "index.csv", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                sources = {r.get("source_npy") for r in rows}
                logger.info(
                    f"  - {d.name}  ({len(sources)} source image(s), {len(rows)} patches)"
                )
            else:
                n = len([p for p in d.glob("*.npy") if p.name != "index.csv"])
                logger.info(f"  - {d.name}  ({n} .npy file(s))")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
        recs = per_folder(
            folder, args.output_root / folder.name, soma_cfg,
            args.structural_channel, workers, args.overwrite,
        )
        results[folder.name] = len(recs)

    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    for name, n in results.items():
        status = f"{n} masks" if n > 0 else "FAILED / empty"
        logger.info(f"  {name}: {status}")
    logger.info(f"Total: {sum(results.values())} masks across {len(folders)} folder(s)")


if __name__ == "__main__":
    main()
