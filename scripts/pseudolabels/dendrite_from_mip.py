#!/usr/bin/env python3
r"""Run the Frangi-on-FDT dendrite pipeline on full-MIP .npy files or on a
tiled-patches session, using pre-computed soma masks for carving.

The soma masks are NOT recomputed here. They are loaded from a separate
directory (the output of ``scripts/soma_from_mip.py``). For a MIP input
``<stem>.npy`` the script expects ``<stem>_soma.npy`` (bool, ``(H, W)``)
under ``--soma_dir`` / ``--soma_root``.

Two input modes (mirroring ``scripts/soma_from_mip.py``):

  1. **MIP mode** -- inputs are full ``(C, H, W)`` ``.npy`` files. One
     ``<stem>_dend.npy`` (bool, ``(H, W)``) is written per source image
     plus a per-folder ``dend_index.csv``.

  2. **Patches mode** -- inputs are tiled patch folders containing
     ``index.csv`` + ``<stem>_r##_c##.npy``. The structural channel
     and the per-patch soma masks are stitched into one full image and
     one full soma mask, the dendrite pipeline runs once at full scale
     (avoiding per-patch border artefacts), then the resulting trace is
     sliced back per patch.

Defaults bake in the calibration from
``notebooks/pseudolabels/dendrite_methods/02c_dendrite_frangi.ipynb``
(20251219 dataset, 107 nm/px). Override any of them via JSON:

  --dend_cfg      keys of DEFAULT_DENDRITE_CFG (Frangi, threshold, cleanup)
  --soma_cfg      keys of DEFAULT_SOMA_CFG (FDT enhancement; base for FDT)
  --dend_fdt_cfg  dendrite-side FDT overrides merged on top of --soma_cfg

Usage::

    # MIP mode, single folder:
    python scripts/dendrite_from_mip.py \
        --input_dir  Microscopy_no_patch/SessionName \
        --soma_dir   Microscopy_soma/SessionName \
        --output_dir Microscopy_dend/SessionName

    # MIP mode, all session subfolders under a root:
    python scripts/dendrite_from_mip.py \
        --input_root  Microscopy_no_patch \
        --soma_root   Microscopy_soma \
        --output_root Microscopy_dend

    # Patches mode, single session:
    python scripts/dendrite_from_mip.py \
        --input_patches  Microscopy/SessionName \
        --soma_dir       Microscopy_soma/SessionName \
        --output_dir     Microscopy_dend/SessionName

    # Patches mode, all session subfolders under a root:
    python scripts/dendrite_from_mip.py \
        --input_patch_root Microscopy \
        --soma_root        Microscopy_soma \
        --output_root      Microscopy_dend

    # Override a knob (JSON):
    python scripts/dendrite_from_mip.py --input_dir ... --soma_dir ... \
        --output_dir ... --dend_cfg '{"hysteresis_low_pct": 45.0}'
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# Use 'spawn' so worker processes do NOT inherit the parent's open file
# descriptors / mmap'd pages / BLAS thread state. On FUSE mounts (ntfs-3g)
# inherited FDs from a fork can trip EIO on the child's first I/O. Spawn
# starts each worker from a clean interpreter.
_MP_CTX = mp.get_context("spawn")


def _np_load(path, *, mmap_mode=None):
    """Read a .npy via one big sequential read.

    NTFS-3G (FUSE) occasionally returns EIO on np.load's multi-chunk
    read pattern but tolerates one large read fine. Reading the whole
    file into a bytes buffer first sidesteps that. ``mmap_mode`` is
    accepted only for the existence-check call site that wants a cheap
    header read; on flaky mounts we fall back to a full read.
    """
    path = Path(path)
    if mmap_mode is not None:
        try:
            return np.load(path, mmap_mode=mmap_mode)
        except OSError:
            pass
    try:
        with open(path, "rb", buffering=0) as fh:
            data = fh.read()
        return np.load(io.BytesIO(data))
    except OSError as e:
        # Re-raise with filename so callers / pool logs show which file failed.
        raise OSError(f"{e} while reading {path}") from e

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "src"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.pseudolabels.dendrite_frangi import (  # noqa: E402
    compute_frangi_response,
    extract_dendrite_mask,
)
from synaptic_ssl.pseudolabels.soma_fdt import compute_pseudo_fdt  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibrated defaults from notebooks/.../02c_dendrite_frangi.ipynb
# (cell 4 base + cell 8 overrides, after the fuzzy_sigma bubble).
# ---------------------------------------------------------------------------

DEFAULT_SOMA_FDT_CFG: dict = dict(
    fdt_compress="log",
    mu_gamma=0.3,
    bg_sigma_k=5.0,
    smooth_sigma=3.0,
    fdt_sigma=5.0,
    support_dilate=0,
)

DEFAULT_DEND_FDT_CFG: dict = dict(
    fdt_mode="gaussian",
    fuzzy_sigma=14.0,
    bg_sigma_k=4.0,
)

DEFAULT_DEND_CFG: dict = dict(
    frangi_sigmas=(2.0, 3.0, 4.5, 7.0),
    frangi_alpha=0.5,
    use_hysteresis=True,
    hysteresis_high_pct=88.0,
    hysteresis_low_pct=50.0,
    min_cc_area=2500,
    small_cc_area=300,
    small_cc_proximity_px=30,
    border_rescue_min_area=400,
    border_rescue_min_axis=30,
    border_pad=21,
    spur_prune_px=40,
    closing_radius=4,
)


DEND_INDEX_FIELDS: tuple[str, ...] = (
    "source_npy",
    "mask_filename",
    "soma_mask_filename",
    "height",
    "width",
    "structural_channel",
    "n_trace_cc",
    "trace_cov",
    "threshold",
    "threshold_kind",
)


PATCH_DEND_INDEX_FIELDS: tuple[str, ...] = (
    "filename",
    "mask_filename",
    "soma_mask_filename",
    "source_npy",
    "image_index",
    "grid_row",
    "grid_col",
    "patch_size",
    "n_trace_cc_full",
    "threshold",
    "threshold_kind",
    "trace_cov_patch",
)


# ---------------------------------------------------------------------------
# Core pipeline (no soma extraction; loads soma mask from disk)
# ---------------------------------------------------------------------------

def _run_dendrite(
    structural: np.ndarray,
    soma_mask: np.ndarray,
    soma_cfg: dict,
    dend_fdt_cfg: dict,
    dend_cfg: dict,
) -> dict:
    """FDT (merged soma+dend cfg) -> Frangi -> threshold/carve/cleanup.

    Returns the dendrite_info dict from ``extract_dendrite_mask`` augmented
    with ``response`` and ``n_trace_cc``.
    """
    if structural.shape != soma_mask.shape:
        raise ValueError(
            f"shape mismatch: structural {structural.shape} vs "
            f"soma {soma_mask.shape}"
        )
    merged_fdt = dict(soma_cfg)
    merged_fdt.update(dend_fdt_cfg)
    fdt_info = compute_pseudo_fdt(structural, merged_fdt)
    response = compute_frangi_response(fdt_info["fdt"], dend_cfg)
    dend_info = extract_dendrite_mask(response, soma_mask, dend_cfg)
    dend_info["response"] = response
    # n_trace_cc via simple CC count
    from skimage.measure import label as cc_label
    dend_info["n_trace_cc"] = int(cc_label(dend_info["trace"]).max())
    return dend_info


# ---------------------------------------------------------------------------
# MIP mode -- per file
# ---------------------------------------------------------------------------

def _soma_mask_path(soma_dir: Path, stem: str) -> Path:
    return soma_dir / f"{stem}_soma.npy"


def dend_one(
    npy_path: Path,
    output_dir: Path,
    soma_dir: Path,
    soma_cfg: dict,
    dend_fdt_cfg: dict,
    dend_cfg: dict,
    structural_channel: int,
    overwrite: bool,
) -> dict | None:
    """Run dendrite on one MIP .npy. Save ``<stem>_dend.npy``. Return record."""
    mask_path = output_dir / f"{npy_path.stem}_dend.npy"
    soma_path = _soma_mask_path(soma_dir, npy_path.stem)

    if not soma_path.exists():
        logger.error(f"  {npy_path.name}: soma mask missing at {soma_path}")
        return None

    if mask_path.exists() and not overwrite:
        logger.info(f"  {npy_path.name}: mask exists, skipping (use --overwrite to redo)")
        try:
            existing = _np_load(mask_path, mmap_mode="r")
            H, W = existing.shape
            cov = float(existing.mean())
        except Exception:
            H = W = 0
            cov = float("nan")
        return {
            "source_npy": npy_path.name,
            "mask_filename": mask_path.name,
            "soma_mask_filename": soma_path.name,
            "height": H,
            "width": W,
            "structural_channel": structural_channel,
            "n_trace_cc": -1,
            "trace_cov": cov,
            "threshold": float("nan"),
            "threshold_kind": "skipped",
        }

    try:
        full = _np_load(npy_path)
        if structural_channel >= full.shape[0]:
            raise ValueError(
                f"{npy_path.name}: {full.shape[0]} channels but "
                f"structural_channel={structural_channel}"
            )
        structural = full[structural_channel].astype(np.float32)
        soma_mask = _np_load(soma_path).astype(bool)
        dend_info = _run_dendrite(
            structural, soma_mask, soma_cfg, dend_fdt_cfg, dend_cfg,
        )
    except Exception as e:
        logger.error(f"  {npy_path.name}: FAILED ({e})")
        return None

    mask = dend_info["trace"].astype(bool)
    np.save(mask_path, mask)
    H, W = mask.shape
    logger.info(
        f"  {npy_path.name}: {H}×{W} -> {dend_info['n_trace_cc']} CC(s), "
        f"cov={mask.mean():.2%}, thr={dend_info['threshold']:.4g} "
        f"({dend_info['threshold_kind']})"
    )

    rec = {
        "source_npy": npy_path.name,
        "mask_filename": mask_path.name,
        "soma_mask_filename": soma_path.name,
        "height": H,
        "width": W,
        "structural_channel": structural_channel,
        "n_trace_cc": int(dend_info["n_trace_cc"]),
        "trace_cov": float(mask.mean()),
        "threshold": float(dend_info["threshold"]),
        "threshold_kind": str(dend_info["threshold_kind"]),
    }

    del full, structural, soma_mask, mask, dend_info
    gc.collect()
    return rec


def _dend_one_worker(args: tuple) -> dict | None:
    (npy_path, output_dir, soma_dir, soma_cfg, dend_fdt_cfg, dend_cfg,
     structural_channel, overwrite) = args
    return dend_one(
        Path(npy_path), Path(output_dir), Path(soma_dir),
        soma_cfg, dend_fdt_cfg, dend_cfg, structural_channel, overwrite,
    )


def process_dir(
    input_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    soma_cfg: dict,
    dend_fdt_cfg: dict,
    dend_cfg: dict,
    structural_channel: int,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run dendrite on all MIP .npy in ``input_dir``. Write masks + dend_index.csv."""
    npy_files = sorted(p for p in input_dir.glob("*.npy") if p.name != "index.csv")
    if not npy_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return []
    if not soma_dir.is_dir():
        logger.error(f"  soma dir does not exist: {soma_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(p), str(output_dir), str(soma_dir),
         soma_cfg, dend_fdt_cfg, dend_cfg, structural_channel, overwrite)
        for p in npy_files
    ]

    records: list[dict] = []
    if workers > 1 and len(npy_files) > 1:
        n_workers = min(workers, len(npy_files))
        logger.info(f"  Running {len(npy_files)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CTX) as ex:
            futures = {ex.submit(_dend_one_worker, item): item[0] for item in work_items}
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
            rec = _dend_one_worker(item)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r["source_npy"])

    csv_path = output_dir / "dend_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(DEND_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  dend_index.csv -> {csv_path}  ({len(records)} images)")
    else:
        logger.warning(f"  No masks produced for {input_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Patches mode: stitch structural + soma -> dendrite -> slice back
# ---------------------------------------------------------------------------

def _stitch_structural_and_soma(
    patch_dir: Path,
    soma_dir: Path,
    rows: list[dict],
    structural_channel: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Stitch structural (from MIP patches) and soma mask (from per-patch
    soma .npy under ``soma_dir``). Returns
    ``(structural, soma_mask, n_rows, n_cols, patch_size)``.
    """
    ps_set = {int(r["patch_size"]) for r in rows}
    if len(ps_set) != 1:
        raise ValueError(f"mixed patch_size in {rows[0]['source_npy']!r}: {ps_set}")
    ps = ps_set.pop()
    n_rows = max(int(r["grid_row"]) for r in rows) + 1
    n_cols = max(int(r["grid_col"]) for r in rows) + 1

    structural = np.zeros((n_rows * ps, n_cols * ps), dtype=np.float32)
    soma = np.zeros((n_rows * ps, n_cols * ps), dtype=bool)
    seen = np.zeros((n_rows, n_cols), dtype=bool)
    for r in rows:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        patch = _np_load(patch_dir / r["filename"])
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
        soma_path = _soma_mask_path(soma_dir, Path(r["filename"]).stem)
        if not soma_path.exists():
            raise FileNotFoundError(f"missing patch soma mask: {soma_path}")
        soma_patch = _np_load(soma_path).astype(bool)
        if soma_patch.shape != (ps, ps):
            raise ValueError(
                f"soma patch {soma_path.name} shape {soma_patch.shape} != ({ps}, {ps})"
            )
        y0, x0 = gr * ps, gcol * ps
        structural[y0:y0 + ps, x0:x0 + ps] = patch[structural_channel].astype(np.float32)
        soma[y0:y0 + ps, x0:x0 + ps] = soma_patch
        seen[gr, gcol] = True

    if not seen.all():
        missing = [(int(r), int(c)) for r, c in zip(*np.where(~seen))]
        raise ValueError(
            f"source {rows[0]['source_npy']!r}: missing patches at "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"({len(missing)} of {n_rows * n_cols} grid cells)"
        )
    return structural, soma, n_rows, n_cols, ps


def dend_for_source(
    patch_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    source_npy: str,
    rows: list[dict],
    soma_cfg: dict,
    dend_fdt_cfg: dict,
    dend_cfg: dict,
    structural_channel: int,
    overwrite: bool,
) -> list[dict]:
    """Stitch one source's patches + soma masks, run dendrite, write per-patch masks."""
    expected = [
        (r,
         output_dir / f"{Path(r['filename']).stem}_dend.npy",
         _soma_mask_path(soma_dir, Path(r['filename']).stem))
        for r in rows
    ]
    if not overwrite and all(mp.exists() for _, mp, _ in expected):
        logger.info(f"  {source_npy}: all {len(rows)} patch masks exist, skipping")
        out: list[dict] = []
        for r, mp, sp in expected:
            try:
                m = _np_load(mp, mmap_mode="r")
                cov = float(m.mean())
            except Exception:
                cov = float("nan")
            out.append({
                "filename": r["filename"],
                "mask_filename": mp.name,
                "soma_mask_filename": sp.name,
                "source_npy": source_npy,
                "image_index": int(r.get("image_index", -1) or -1),
                "grid_row": int(r["grid_row"]),
                "grid_col": int(r["grid_col"]),
                "patch_size": int(r["patch_size"]),
                "n_trace_cc_full": -1,
                "threshold": float("nan"),
                "threshold_kind": "skipped",
                "trace_cov_patch": cov,
            })
        return out

    structural, soma_full, n_rows, n_cols, ps = _stitch_structural_and_soma(
        patch_dir, soma_dir, rows, structural_channel
    )

    dend_info = _run_dendrite(
        structural, soma_full, soma_cfg, dend_fdt_cfg, dend_cfg,
    )
    full_mask = dend_info["trace"].astype(bool)

    logger.info(
        f"  {source_npy}: {n_rows}×{n_cols} grid ({n_rows * ps}×{n_cols * ps}) -> "
        f"{dend_info['n_trace_cc']} CC(s), cov={full_mask.mean():.2%}, "
        f"thr={dend_info['threshold']:.4g} ({dend_info['threshold_kind']})"
    )

    records: list[dict] = []
    for r, mp, sp in expected:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        y0, x0 = gr * ps, gcol * ps
        sl = full_mask[y0:y0 + ps, x0:x0 + ps].copy()
        np.save(mp, sl)
        records.append({
            "filename": r["filename"],
            "mask_filename": mp.name,
            "soma_mask_filename": sp.name,
            "source_npy": source_npy,
            "image_index": int(r.get("image_index", -1) or -1),
            "grid_row": gr,
            "grid_col": gcol,
            "patch_size": ps,
            "n_trace_cc_full": int(dend_info["n_trace_cc"]),
            "threshold": float(dend_info["threshold"]),
            "threshold_kind": str(dend_info["threshold_kind"]),
            "trace_cov_patch": float(sl.mean()),
        })

    del structural, soma_full, dend_info, full_mask
    gc.collect()
    return records


def _dend_for_source_worker(args: tuple) -> list[dict]:
    (patch_dir, output_dir, soma_dir, source_npy, rows,
     soma_cfg, dend_fdt_cfg, dend_cfg, sc, overwrite) = args
    return dend_for_source(
        Path(patch_dir), Path(output_dir), Path(soma_dir),
        source_npy, rows, soma_cfg, dend_fdt_cfg, dend_cfg, sc, overwrite,
    )


def process_patches_dir(
    patch_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    soma_cfg: dict,
    dend_fdt_cfg: dict,
    dend_cfg: dict,
    structural_channel: int,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run dendrite per source image in a tiled-patches session folder."""
    index_csv = patch_dir / "index.csv"
    if not index_csv.exists():
        logger.error(f"  {patch_dir}: index.csv not found, cannot stitch")
        return []
    if not soma_dir.is_dir():
        logger.error(f"  soma dir does not exist: {soma_dir}")
        return []

    with open(index_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        logger.warning(f"  {patch_dir}: index.csv is empty")
        return []

    groups: dict[str, list[dict]] = {}
    for r in all_rows:
        key = r.get("source_npy") or f"img{int(r.get('image_index', 0)):04d}"
        groups.setdefault(key, []).append(r)

    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(patch_dir), str(output_dir), str(soma_dir), key, rows,
         soma_cfg, dend_fdt_cfg, dend_cfg, structural_channel, overwrite)
        for key, rows in sorted(groups.items())
    ]

    records: list[dict] = []
    if workers > 1 and len(work_items) > 1:
        n_workers = min(workers, len(work_items))
        logger.info(
            f"  Stitching + dendrite on {len(work_items)} source image(s) "
            f"with {n_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CTX) as ex:
            futures = {
                ex.submit(_dend_for_source_worker, item): item[3]
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
                records.extend(_dend_for_source_worker(item))
            except Exception as e:
                logger.error(f"  FAILED on {item[3]}: {e}")

    records.sort(
        key=lambda r: (r["source_npy"], r["grid_row"], r["grid_col"])
    )

    csv_path = output_dir / "dend_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(PATCH_DEND_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  dend_index.csv -> {csv_path}  ({len(records)} patches)")
    else:
        logger.warning(f"  No masks produced for {patch_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_json_cfg(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    out.update(override or {})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Frangi-on-FDT dendrite pipeline on full-MIP .npy files "
            "or tiled patches, using pre-computed soma masks."
        )
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
        "--soma_dir", type=Path,
        help=(
            "Folder containing pre-computed soma masks (<stem>_soma.npy). "
            "Required with --input_dir / --input_patches."
        ),
    )
    parser.add_argument(
        "--soma_root", type=Path,
        help=(
            "Root containing soma-mask session subfolders mirrored from "
            "the input root. Required with --input_root / --input_patch_root."
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        help="Output folder for dendrite masks (required with --input_dir / --input_patches).",
    )
    parser.add_argument(
        "--output_root", type=Path,
        help=(
            "Output root; session subfolders are mirrored "
            "(required with --input_root / --input_patch_root)."
        ),
    )
    parser.add_argument(
        "--structural_channel", type=int, default=2,
        help="Index of the structural channel in (C, H, W) (default: 2).",
    )
    parser.add_argument(
        "--dend_cfg", type=str, default=None,
        help="JSON dict or path to .json with DEFAULT_DEND_CFG overrides.",
    )
    parser.add_argument(
        "--soma_cfg", type=str, default=None,
        help=(
            "JSON dict or path to .json with FDT-enhancement overrides "
            "(keys of DEFAULT_SOMA_CFG). Used as the BASE for FDT computation."
        ),
    )
    parser.add_argument(
        "--dend_fdt_cfg", type=str, default=None,
        help=(
            "JSON dict or path to .json with dendrite-side FDT overrides "
            "(e.g. fdt_mode, fuzzy_sigma, bg_sigma_k); merged on top of --soma_cfg."
        ),
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many session folders (0 = all; root modes only).",
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
        help="Print what would be done without running the pipeline.",
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

    if (args.input_dir or args.input_patches) and not args.soma_dir:
        parser.error("--soma_dir is required with --input_dir / --input_patches")
    if (args.input_root or args.input_patch_root) and not args.soma_root:
        parser.error("--soma_root is required with --input_root / --input_patch_root")

    try:
        soma_cfg = _merge(DEFAULT_SOMA_FDT_CFG, _parse_json_cfg(args.soma_cfg))
        dend_fdt_cfg = _merge(DEFAULT_DEND_FDT_CFG, _parse_json_cfg(args.dend_fdt_cfg))
        dend_cfg = _merge(DEFAULT_DEND_CFG, _parse_json_cfg(args.dend_cfg))
    except (json.JSONDecodeError, OSError) as e:
        parser.error(f"cfg parse error: {e}")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 2)

    def _log_cfg() -> None:
        logger.info(f"Channel:        {args.structural_channel}")
        logger.info(f"Workers:        {workers}")
        logger.info(f"Soma-FDT cfg:   {soma_cfg}")
        logger.info(f"Dend-FDT cfg:   {dend_fdt_cfg}")
        logger.info(f"Dend cfg:       {dend_cfg}")

    # ---- MIP, single folder ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info("Mode:           MIP (single folder)")
        logger.info(f"Input:          {args.input_dir}")
        logger.info(f"Soma masks:     {args.soma_dir}")
        logger.info(f"Output:         {args.output_dir}")
        _log_cfg()
        if args.dry_run:
            n = len([p for p in args.input_dir.glob("*.npy") if p.name != "index.csv"])
            logger.info(f"[DRY RUN] Would run dendrite on {n} .npy file(s).")
            return
        process_dir(
            args.input_dir, args.output_dir, args.soma_dir,
            soma_cfg, dend_fdt_cfg, dend_cfg,
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
        logger.info("Mode:           Patches (single folder, stitch -> dendrite -> slice)")
        logger.info(f"Input:          {args.input_patches}")
        logger.info(f"Soma masks:     {args.soma_dir}")
        logger.info(f"Output:         {args.output_dir}")
        _log_cfg()
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
            args.input_patches, args.output_dir, args.soma_dir,
            soma_cfg, dend_fdt_cfg, dend_cfg,
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
        mode_name = "Patches (root, stitch -> dendrite -> slice)"
        select = lambda d: (d / "index.csv").exists()  # noqa: E731
        per_folder = process_patches_dir

    if not input_root.is_dir():
        logger.error(f"Input root does not exist: {input_root}")
        sys.exit(1)
    if not args.soma_root.is_dir():
        logger.error(f"Soma root does not exist: {args.soma_root}")
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
    logger.info("Dendrite configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Mode:           {mode_name}")
    logger.info(f"Input root:     {input_root}")
    logger.info(f"Soma root:      {args.soma_root}")
    logger.info(f"Output root:    {args.output_root}")
    logger.info(f"Folders:        {len(folders)}")
    _log_cfg()

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for d in folders:
            soma_sub = args.soma_root / d.name
            missing = "" if soma_sub.is_dir() else "  [SOMA DIR MISSING]"
            if args.input_patch_root:
                with open(d / "index.csv", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                sources = {r.get("source_npy") for r in rows}
                logger.info(
                    f"  - {d.name}  ({len(sources)} source image(s), "
                    f"{len(rows)} patches){missing}"
                )
            else:
                n = len([p for p in d.glob("*.npy") if p.name != "index.csv"])
                logger.info(f"  - {d.name}  ({n} .npy file(s)){missing}")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}

    for i, folder in enumerate(folders, start=1):
        logger.info(f"\n[{i}/{len(folders)}] {folder.name}")
        soma_sub = args.soma_root / folder.name
        if not soma_sub.is_dir():
            logger.error(f"  soma subfolder missing: {soma_sub}; skipping")
            results[folder.name] = 0
            continue
        recs = per_folder(
            folder, args.output_root / folder.name, soma_sub,
            soma_cfg, dend_fdt_cfg, dend_cfg,
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
