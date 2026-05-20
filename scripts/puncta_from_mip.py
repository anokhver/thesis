#!/usr/bin/env python3
r"""Run the pre/post synaptic-puncta detection pipeline on full-MIP .npy
files or a tiled-patches session, using pre-computed soma + dendrite
masks for the structural near-gate.

Mirrors ``scripts/soma_from_mip.py`` and ``scripts/dendrite_from_mip.py``.
The soma and dendrite masks are NOT recomputed here -- they are loaded
from separate directories (the outputs of those two scripts). For a MIP
input ``<stem>.npy`` the script expects:

  - ``<stem>_soma.npy`` under ``--soma_dir`` / ``--soma_root``
  - ``<stem>_dend.npy`` under ``--dendrite_dir`` / ``--dendrite_root``

Two input modes (mirroring the soma/dendrite scripts):

  1. **MIP mode** -- inputs are full ``(C, H, W)`` ``.npy`` files. Two
     masks ``<stem>_pre.npy`` and ``<stem>_post.npy`` (bool, ``(H, W)``)
     are written per source image plus a per-folder
     ``puncta_index.csv``.

  2. **Patches mode** -- inputs are tiled patch folders containing
     ``index.csv`` + ``<stem>_r##_c##.npy``. The pre + post channels and
     the per-patch soma + dendrite masks are stitched into one full
     image / mask per source, detection runs once at full scale
     (avoiding per-patch border artefacts), then the per-channel mask
     is sliced back so each patch gets a ``<patch_stem>_pre.npy`` /
     ``<patch_stem>_post.npy`` of shape ``(ps, ps)``.

Defaults bake in the calibration from
``notebooks/pseudolabels/puncta_detection.ipynb`` (20251219 dataset,
107 nm/px). Override any of them via JSON:

  --cfg_pre   keys of DEFAULT_CFG_PRE  (LoG + z-score + tophat for pre)
  --cfg_post  keys of DEFAULT_CFG_POST (same, for post)

Per-image absolute-intensity floors (sigma_bg, min_inner, min_contrast)
are auto-derived from each image's tophatted background by default.
Disable with ``--no_auto_floors`` to use the literal cfg values.

Usage::

    # MIP mode, single folder:
    python scripts/puncta_from_mip.py \
        --input_dir     Microscopy_no_patch/SessionName \
        --soma_dir      Microscopy_soma/SessionName \
        --dendrite_dir  Microscopy_dend/SessionName \
        --output_dir    Microscopy_puncta/SessionName

    # MIP mode, all session subfolders under a root:
    python scripts/puncta_from_mip.py \
        --input_root     Microscopy_no_patch \
        --soma_root      Microscopy_soma \
        --dendrite_root  Microscopy_dend \
        --output_root    Microscopy_puncta

    # Patches mode, single session:
    python scripts/puncta_from_mip.py \
        --input_patches Microscopy/SessionName \
        --soma_dir      Microscopy_soma/SessionName \
        --dendrite_dir  Microscopy_dend/SessionName \
        --output_dir    Microscopy_puncta/SessionName

    # Override a knob (JSON):
    python scripts/puncta_from_mip.py --input_dir ... --soma_dir ... \
        --dendrite_dir ... --output_dir ... \
        --cfg_pre '{"zscore_threshold": 3.0, "intensity_tophat_radius": 10}'
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
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
from skimage.morphology import disk as morph_disk, dilation, white_tophat


_MP_CTX = mp.get_context("spawn")


def _np_load(path, *, mmap_mode=None):
    """Read a .npy via one big sequential read.

    NTFS-3G (FUSE) occasionally returns EIO on np.load's multi-chunk
    read pattern but tolerates one large read fine. Mirrors the helper
    in ``scripts/dendrite_from_mip.py``.
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
        raise OSError(f"{e} while reading {path}") from e


_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "src"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.pseudolabels.blobs import (  # noqa: E402
    BlobPseudoCfg,
    blobs_to_mask,
    detect_blobs_log,
    filter_by_size_shape,
    score_blobs_zscore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibrated defaults from notebooks/pseudolabels/puncta_detection.ipynb
# (cell 4, after the robust-annulus tuning of 2026-05-20).
# Channel-role fields (pre/post/structural_channel) are intentionally
# left at BlobPseudoCfg defaults: detect_blobs_log / score_blobs_zscore
# take a 2D array directly so the cfg field is unused at the detector
# level. The script's --pre_channel / --post_channel flags drive the
# actual channel selection.
# ---------------------------------------------------------------------------

DEFAULT_CFG_PRE: dict = dict(
    log_min_sigma=1.4,
    log_max_sigma=2.4,
    log_num_sigma=4,
    log_threshold=0.006,
    log_overlap=0.5,
    log_exclude_border=8,
    use_zscore=True,
    zscore_inner_radius=5,
    zscore_outer_radius=12,
    zscore_threshold=2.5,
    zscore_bg_percentile=25,
    zscore_bg_robust_scale=True,
    intensity_tophat_radius=8,
    min_size=13,
    max_size=60,
    min_fill=0.5,
    max_wh_ratio=4.0,
)

DEFAULT_CFG_POST: dict = dict(
    log_min_sigma=1.3,
    log_max_sigma=1.8,
    log_num_sigma=3,
    log_threshold=0.007,
    log_overlap=0.5,
    log_exclude_border=6,
    use_zscore=True,
    zscore_inner_radius=3,
    zscore_outer_radius=8,
    zscore_threshold=3.0,
    zscore_bg_percentile=25,
    zscore_bg_robust_scale=True,
    intensity_tophat_radius=6,
    min_size=11,
    max_size=30,
    min_fill=0.5,
    max_wh_ratio=3.0,
)

DEFAULT_NEAR_DILATE_PX: int = 4


PUNCTA_INDEX_FIELDS: tuple[str, ...] = (
    "source_npy",
    "pre_mask_filename",
    "post_mask_filename",
    "soma_mask_filename",
    "dend_mask_filename",
    "height",
    "width",
    "pre_channel",
    "post_channel",
    "near_dilate_px",
    "frac_near",
    "n_pre_raw",         # kept after z-score + floors (before near-gate)
    "n_post_raw",
    "n_pre_on_near",     # kept after centre-on-near gate
    "n_post_on_near",
    "pre_tophat_r",
    "pre_floor_sigma_bg",
    "pre_floor_min_inner",
    "pre_floor_min_contrast",
    "post_tophat_r",
    "post_floor_sigma_bg",
    "post_floor_min_inner",
    "post_floor_min_contrast",
)


PATCH_PUNCTA_INDEX_FIELDS: tuple[str, ...] = (
    "filename",
    "pre_mask_filename",
    "post_mask_filename",
    "soma_mask_filename",
    "dend_mask_filename",
    "source_npy",
    "image_index",
    "grid_row",
    "grid_col",
    "patch_size",
    "pre_channel",
    "post_channel",
    "near_dilate_px",
    "frac_near_patch",
    "n_pre_full_on_near",
    "n_post_full_on_near",
    "n_pre_pixels_patch",
    "n_post_pixels_patch",
    "pre_tophat_r",
    "pre_floor_sigma_bg",
    "pre_floor_min_inner",
    "pre_floor_min_contrast",
    "post_tophat_r",
    "post_floor_sigma_bg",
    "post_floor_min_inner",
    "post_floor_min_contrast",
)


# ---------------------------------------------------------------------------
# Cfg helpers
# ---------------------------------------------------------------------------

def _merge_cfg(base: dict, override: dict | None) -> dict:
    out = dict(base)
    out.update(override or {})
    return out


def _build_blob_cfg(merged: dict) -> BlobPseudoCfg:
    """Construct a BlobPseudoCfg from a merged dict.

    Filters to fields BlobPseudoCfg actually declares so a user-supplied
    JSON with extra keys produces a clean error rather than a silent miss.
    """
    valid = {f.name for f in dataclasses.fields(BlobPseudoCfg)}
    unknown = set(merged) - valid
    if unknown:
        raise ValueError(
            f"unknown BlobPseudoCfg field(s) in cfg JSON: {sorted(unknown)}"
        )
    return BlobPseudoCfg(**merged)


def _parse_json_cfg(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


# ---------------------------------------------------------------------------
# Detection on one 2D channel
# ---------------------------------------------------------------------------

def _derive_floors(detected: np.ndarray) -> tuple[float, float, float]:
    """Per-image safety floors from bg stats of the (already tophatted)
    array the detector will see.

    Returns ``(sigma_bg_floor, min_inner, min_contrast)``. Falls back to
    ``(0.0, 0.0, 0.0)`` (floors disabled) on degenerate / non-finite
    inputs. Mirrors ``set_bg_floors`` in the puncta_detection notebook.
    """
    finite = detected[np.isfinite(detected)]
    if finite.size == 0:
        return 0.0, 0.0, 0.0
    p30 = np.percentile(finite, 30)
    bg = finite[finite <= p30]
    if bg.size == 0:
        return 0.0, 0.0, 0.0
    bg_med = float(np.median(bg))
    bg_mad = float(np.median(np.abs(bg - bg_med)))
    bg_sigma = 1.4826 * bg_mad
    if not (np.isfinite(bg_med) and np.isfinite(bg_sigma)):
        return 0.0, 0.0, 0.0
    return bg_sigma, bg_med + 4.0 * bg_sigma, 4.0 * bg_sigma


def _detect_channel(
    img2d: np.ndarray,
    cfg: BlobPseudoCfg,
    *,
    auto_floors: bool,
) -> tuple[np.ndarray, BlobPseudoCfg, tuple[float, float, float]]:
    """Tophat -> (optionally derive floors) -> LoG -> z-score + floor gate.

    Returns ``(kept_blobs[N,3], effective_cfg, (sigma_bg_floor,
    min_inner, min_contrast))``. ``effective_cfg`` is the cfg actually
    used for scoring (after the per-image floor overwrite if
    auto_floors is on), so callers can log it.
    """
    r = int(cfg.intensity_tophat_radius)
    det = white_tophat(img2d, footprint=morph_disk(r)) if r > 0 else img2d

    if auto_floors:
        sg_floor, in_floor, dlt_floor = _derive_floors(det)
        cfg_eff = dataclasses.replace(
            cfg,
            zscore_sigma_bg_floor=sg_floor,
            zscore_min_inner=in_floor,
            zscore_min_contrast=dlt_floor,
        )
    else:
        sg_floor = cfg.zscore_sigma_bg_floor
        in_floor = cfg.zscore_min_inner
        dlt_floor = cfg.zscore_min_contrast
        cfg_eff = cfg

    raw = detect_blobs_log(det, cfg_eff)
    if not cfg_eff.use_zscore or raw.shape[0] == 0:
        kept = raw
    else:
        scored = score_blobs_zscore(det, raw, cfg_eff)
        if scored:
            kept = np.array(
                [[s["row"], s["col"], s["sigma"]] for s in scored if s["kept"]]
            ).reshape(-1, 3)
        else:
            kept = np.empty((0, 3), dtype=np.float64)
    return kept, cfg_eff, (sg_floor, in_floor, dlt_floor)


def _restrict_to_near(blobs: np.ndarray, near_mask: np.ndarray) -> np.ndarray:
    """Keep blobs whose rounded centre lies on near_mask."""
    if blobs.shape[0] == 0:
        return blobs
    H, W = near_mask.shape
    rr = np.clip(np.round(blobs[:, 0]).astype(int), 0, H - 1)
    cc = np.clip(np.round(blobs[:, 1]).astype(int), 0, W - 1)
    return blobs[near_mask[rr, cc].astype(bool)]


def _run_puncta(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    soma_mask: np.ndarray,
    dend_mask: np.ndarray,
    cfg_pre: BlobPseudoCfg,
    cfg_post: BlobPseudoCfg,
    *,
    near_dilate_px: int,
    auto_floors: bool,
    apply_shape_filter: bool,
) -> dict:
    """Full pipeline on already-loaded full-shape arrays.

    Returns a dict with the final ``pre_mask`` / ``post_mask`` (bool
    H x W), the ``near_mask``, per-channel counts and the per-image
    floors actually used.
    """
    if pre_img.shape != post_img.shape:
        raise ValueError(
            f"pre/post shape mismatch: {pre_img.shape} vs {post_img.shape}"
        )
    if pre_img.shape != soma_mask.shape:
        raise ValueError(
            f"image vs soma shape mismatch: {pre_img.shape} vs {soma_mask.shape}"
        )
    if pre_img.shape != dend_mask.shape:
        raise ValueError(
            f"image vs dend shape mismatch: {pre_img.shape} vs {dend_mask.shape}"
        )

    structural = soma_mask | dend_mask
    if near_dilate_px > 0:
        near_mask = dilation(structural, morph_disk(int(near_dilate_px)))
    else:
        near_mask = structural

    kept_pre, _, fp = _detect_channel(pre_img, cfg_pre, auto_floors=auto_floors)
    kept_post, _, fq = _detect_channel(post_img, cfg_post, auto_floors=auto_floors)

    on_pre = _restrict_to_near(kept_pre, near_mask)
    on_post = _restrict_to_near(kept_post, near_mask)

    H, W = pre_img.shape
    pre_mask = blobs_to_mask(on_pre, (H, W)).astype(bool)
    post_mask = blobs_to_mask(on_post, (H, W)).astype(bool)

    if apply_shape_filter:
        pre_mask = filter_by_size_shape(pre_mask, cfg_pre).astype(bool)
        post_mask = filter_by_size_shape(post_mask, cfg_post).astype(bool)

    return dict(
        pre_mask=pre_mask,
        post_mask=post_mask,
        near_mask=near_mask,
        n_pre_raw=int(kept_pre.shape[0]),
        n_post_raw=int(kept_post.shape[0]),
        n_pre_on_near=int(on_pre.shape[0]),
        n_post_on_near=int(on_post.shape[0]),
        pre_floors=fp,
        post_floors=fq,
    )


# ---------------------------------------------------------------------------
# MIP mode -- per file
# ---------------------------------------------------------------------------

def _soma_mask_path(soma_dir: Path, stem: str) -> Path:
    return soma_dir / f"{stem}_soma.npy"


def _dend_mask_path(dend_dir: Path, stem: str) -> Path:
    return dend_dir / f"{stem}_dend.npy"


def _pre_mask_path(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}_pre.npy"


def _post_mask_path(out_dir: Path, stem: str) -> Path:
    return out_dir / f"{stem}_post.npy"


def puncta_one(
    npy_path: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: BlobPseudoCfg,
    cfg_post: BlobPseudoCfg,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floors: bool,
    apply_shape_filter: bool,
    overwrite: bool,
) -> dict | None:
    """Detect puncta on one MIP .npy. Save ``<stem>_pre.npy`` and
    ``<stem>_post.npy``. Return one CSV record.
    """
    pre_path = _pre_mask_path(output_dir, npy_path.stem)
    post_path = _post_mask_path(output_dir, npy_path.stem)
    soma_path = _soma_mask_path(soma_dir, npy_path.stem)
    dend_path = _dend_mask_path(dend_dir, npy_path.stem)

    if not soma_path.exists():
        logger.error(f"  {npy_path.name}: soma mask missing at {soma_path}")
        return None
    if not dend_path.exists():
        logger.error(f"  {npy_path.name}: dend mask missing at {dend_path}")
        return None

    if pre_path.exists() and post_path.exists() and not overwrite:
        logger.info(
            f"  {npy_path.name}: pre+post masks exist, skipping "
            f"(use --overwrite to redo)"
        )
        try:
            pm = _np_load(pre_path, mmap_mode="r")
            qm = _np_load(post_path, mmap_mode="r")
            H, W = pm.shape
            n_pre_pix = int(pm.sum())
            n_post_pix = int(qm.sum())
        except Exception:
            H = W = 0
            n_pre_pix = n_post_pix = -1
        return {
            "source_npy": npy_path.name,
            "pre_mask_filename": pre_path.name,
            "post_mask_filename": post_path.name,
            "soma_mask_filename": soma_path.name,
            "dend_mask_filename": dend_path.name,
            "height": H,
            "width": W,
            "pre_channel": pre_channel,
            "post_channel": post_channel,
            "near_dilate_px": near_dilate_px,
            "frac_near": float("nan"),
            "n_pre_raw": -1,
            "n_post_raw": -1,
            "n_pre_on_near": n_pre_pix,   # px count as a stale proxy
            "n_post_on_near": n_post_pix,
            "pre_tophat_r": cfg_pre.intensity_tophat_radius,
            "pre_floor_sigma_bg": float("nan"),
            "pre_floor_min_inner": float("nan"),
            "pre_floor_min_contrast": float("nan"),
            "post_tophat_r": cfg_post.intensity_tophat_radius,
            "post_floor_sigma_bg": float("nan"),
            "post_floor_min_inner": float("nan"),
            "post_floor_min_contrast": float("nan"),
        }

    try:
        full = _np_load(npy_path)
        if full.ndim != 3:
            raise ValueError(
                f"{npy_path.name} has shape {full.shape}, expected (C, H, W)"
            )
        if pre_channel >= full.shape[0] or post_channel >= full.shape[0]:
            raise ValueError(
                f"{npy_path.name}: {full.shape[0]} channels but pre={pre_channel} / "
                f"post={post_channel}"
            )
        pre_img = full[pre_channel].astype(np.float32)
        post_img = full[post_channel].astype(np.float32)
        soma_mask = _np_load(soma_path).astype(bool)
        dend_mask = _np_load(dend_path).astype(bool)
        result = _run_puncta(
            pre_img, post_img, soma_mask, dend_mask,
            cfg_pre, cfg_post,
            near_dilate_px=near_dilate_px,
            auto_floors=auto_floors,
            apply_shape_filter=apply_shape_filter,
        )
    except Exception as e:
        logger.error(f"  {npy_path.name}: FAILED ({e})")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(pre_path, result["pre_mask"].astype(np.uint8))
    np.save(post_path, result["post_mask"].astype(np.uint8))
    H, W = result["pre_mask"].shape
    fp = result["pre_floors"]
    fq = result["post_floors"]
    logger.info(
        f"  {npy_path.name}: {H}x{W} -> "
        f"pre {result['n_pre_raw']}->{result['n_pre_on_near']}  "
        f"post {result['n_post_raw']}->{result['n_post_on_near']}  "
        f"near={result['near_mask'].mean():.2%}"
    )

    rec = {
        "source_npy": npy_path.name,
        "pre_mask_filename": pre_path.name,
        "post_mask_filename": post_path.name,
        "soma_mask_filename": soma_path.name,
        "dend_mask_filename": dend_path.name,
        "height": H,
        "width": W,
        "pre_channel": pre_channel,
        "post_channel": post_channel,
        "near_dilate_px": near_dilate_px,
        "frac_near": float(result["near_mask"].mean()),
        "n_pre_raw": result["n_pre_raw"],
        "n_post_raw": result["n_post_raw"],
        "n_pre_on_near": result["n_pre_on_near"],
        "n_post_on_near": result["n_post_on_near"],
        "pre_tophat_r": cfg_pre.intensity_tophat_radius,
        "pre_floor_sigma_bg": float(fp[0]),
        "pre_floor_min_inner": float(fp[1]),
        "pre_floor_min_contrast": float(fp[2]),
        "post_tophat_r": cfg_post.intensity_tophat_radius,
        "post_floor_sigma_bg": float(fq[0]),
        "post_floor_min_inner": float(fq[1]),
        "post_floor_min_contrast": float(fq[2]),
    }

    del full, pre_img, post_img, soma_mask, dend_mask, result
    gc.collect()
    return rec


def _puncta_one_worker(args: tuple) -> dict | None:
    (npy_path, output_dir, soma_dir, dend_dir,
     cfg_pre, cfg_post, pre_ch, post_ch, near_dilate_px,
     auto_floors, apply_shape_filter, overwrite) = args
    return puncta_one(
        Path(npy_path), Path(output_dir), Path(soma_dir), Path(dend_dir),
        cfg_pre, cfg_post, pre_ch, post_ch, near_dilate_px,
        auto_floors, apply_shape_filter, overwrite,
    )


def process_dir(
    input_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: BlobPseudoCfg,
    cfg_post: BlobPseudoCfg,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floors: bool,
    apply_shape_filter: bool,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run puncta on all MIP .npy in ``input_dir``. Write masks + puncta_index.csv."""
    npy_files = sorted(p for p in input_dir.glob("*.npy") if p.name != "index.csv")
    if not npy_files:
        logger.warning(f"No .npy files found in {input_dir}")
        return []
    if not soma_dir.is_dir():
        logger.error(f"  soma dir does not exist: {soma_dir}")
        return []
    if not dend_dir.is_dir():
        logger.error(f"  dend dir does not exist: {dend_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    work_items = [
        (str(p), str(output_dir), str(soma_dir), str(dend_dir),
         cfg_pre, cfg_post, pre_channel, post_channel, near_dilate_px,
         auto_floors, apply_shape_filter, overwrite)
        for p in npy_files
    ]

    records: list[dict] = []
    if workers > 1 and len(npy_files) > 1:
        n_workers = min(workers, len(npy_files))
        logger.info(f"  Running {len(npy_files)} files with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CTX) as ex:
            futures = {ex.submit(_puncta_one_worker, it): it[0] for it in work_items}
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    logger.error(f"  Worker crashed on {futures[fut]}: {e}")
                    continue
                if rec is not None:
                    records.append(rec)
    else:
        for it in work_items:
            rec = _puncta_one_worker(it)
            if rec is not None:
                records.append(rec)

    records.sort(key=lambda r: r["source_npy"])

    csv_path = output_dir / "puncta_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(PUNCTA_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  puncta_index.csv -> {csv_path}  ({len(records)} images)")
    else:
        logger.warning(f"  No masks produced for {input_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Patches mode: stitch pre + post + soma + dend -> puncta -> slice back
# ---------------------------------------------------------------------------

def _stitch_for_source(
    patch_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    rows: list[dict],
    pre_channel: int,
    post_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Stitch pre + post channels (from MIP patches) and soma + dend
    masks (from per-patch .npy under ``soma_dir`` / ``dend_dir``).

    Returns ``(pre, post, soma, dend, n_rows, n_cols, patch_size)``.
    """
    ps_set = {int(r["patch_size"]) for r in rows}
    if len(ps_set) != 1:
        raise ValueError(f"mixed patch_size in {rows[0]['source_npy']!r}: {ps_set}")
    ps = ps_set.pop()
    n_rows = max(int(r["grid_row"]) for r in rows) + 1
    n_cols = max(int(r["grid_col"]) for r in rows) + 1

    pre = np.zeros((n_rows * ps, n_cols * ps), dtype=np.float32)
    post = np.zeros((n_rows * ps, n_cols * ps), dtype=np.float32)
    soma = np.zeros((n_rows * ps, n_cols * ps), dtype=bool)
    dend = np.zeros((n_rows * ps, n_cols * ps), dtype=bool)
    seen = np.zeros((n_rows, n_cols), dtype=bool)
    for r in rows:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        if seen[gr, gcol]:
            raise ValueError(
                f"duplicate patch at (grid_row={gr}, grid_col={gcol}) "
                f"for source {rows[0]['source_npy']!r}"
            )
        patch = _np_load(patch_dir / r["filename"])
        if patch.ndim != 3:
            raise ValueError(
                f"patch {r['filename']} has shape {patch.shape}, expected (C, ps, ps)"
            )
        if max(pre_channel, post_channel) >= patch.shape[0]:
            raise ValueError(
                f"patch {r['filename']} has {patch.shape[0]} channels but "
                f"pre={pre_channel} / post={post_channel}"
            )
        if patch.shape[1] != ps or patch.shape[2] != ps:
            raise ValueError(
                f"patch {r['filename']} spatial shape {patch.shape[1:]} != ({ps}, {ps})"
            )
        soma_path = _soma_mask_path(soma_dir, Path(r["filename"]).stem)
        dend_path = _dend_mask_path(dend_dir, Path(r["filename"]).stem)
        if not soma_path.exists():
            raise FileNotFoundError(f"missing patch soma mask: {soma_path}")
        if not dend_path.exists():
            raise FileNotFoundError(f"missing patch dend mask: {dend_path}")
        soma_patch = _np_load(soma_path).astype(bool)
        dend_patch = _np_load(dend_path).astype(bool)
        if soma_patch.shape != (ps, ps):
            raise ValueError(
                f"soma patch {soma_path.name} shape {soma_patch.shape} != ({ps}, {ps})"
            )
        if dend_patch.shape != (ps, ps):
            raise ValueError(
                f"dend patch {dend_path.name} shape {dend_patch.shape} != ({ps}, {ps})"
            )
        y0, x0 = gr * ps, gcol * ps
        pre[y0:y0 + ps, x0:x0 + ps] = patch[pre_channel].astype(np.float32)
        post[y0:y0 + ps, x0:x0 + ps] = patch[post_channel].astype(np.float32)
        soma[y0:y0 + ps, x0:x0 + ps] = soma_patch
        dend[y0:y0 + ps, x0:x0 + ps] = dend_patch
        seen[gr, gcol] = True

    if not seen.all():
        missing = [(int(r), int(c)) for r, c in zip(*np.where(~seen))]
        raise ValueError(
            f"source {rows[0]['source_npy']!r}: missing patches at "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"({len(missing)} of {n_rows * n_cols} grid cells)"
        )
    return pre, post, soma, dend, n_rows, n_cols, ps


def puncta_for_source(
    patch_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    source_npy: str,
    rows: list[dict],
    cfg_pre: BlobPseudoCfg,
    cfg_post: BlobPseudoCfg,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floors: bool,
    apply_shape_filter: bool,
    overwrite: bool,
) -> list[dict]:
    """Stitch one source's patches + masks, run puncta, write per-patch masks."""
    expected = [
        (
            r,
            _pre_mask_path(output_dir, Path(r["filename"]).stem),
            _post_mask_path(output_dir, Path(r["filename"]).stem),
            _soma_mask_path(soma_dir, Path(r["filename"]).stem),
            _dend_mask_path(dend_dir, Path(r["filename"]).stem),
        )
        for r in rows
    ]

    if not overwrite and all(pp.exists() and qp.exists() for _, pp, qp, _, _ in expected):
        logger.info(
            f"  {source_npy}: all {len(rows)} pre+post patch masks exist, skipping"
        )
        out: list[dict] = []
        for r, pp, qp, sp, dp in expected:
            try:
                pm = _np_load(pp, mmap_mode="r")
                qm = _np_load(qp, mmap_mode="r")
                n_pre_pix = int(pm.sum())
                n_post_pix = int(qm.sum())
            except Exception:
                n_pre_pix = n_post_pix = -1
            out.append({
                "filename": r["filename"],
                "pre_mask_filename": pp.name,
                "post_mask_filename": qp.name,
                "soma_mask_filename": sp.name,
                "dend_mask_filename": dp.name,
                "source_npy": source_npy,
                "image_index": int(r.get("image_index", -1) or -1),
                "grid_row": int(r["grid_row"]),
                "grid_col": int(r["grid_col"]),
                "patch_size": int(r["patch_size"]),
                "pre_channel": pre_channel,
                "post_channel": post_channel,
                "near_dilate_px": near_dilate_px,
                "frac_near_patch": float("nan"),
                "n_pre_full_on_near": -1,
                "n_post_full_on_near": -1,
                "n_pre_pixels_patch": n_pre_pix,
                "n_post_pixels_patch": n_post_pix,
                "pre_tophat_r": cfg_pre.intensity_tophat_radius,
                "pre_floor_sigma_bg": float("nan"),
                "pre_floor_min_inner": float("nan"),
                "pre_floor_min_contrast": float("nan"),
                "post_tophat_r": cfg_post.intensity_tophat_radius,
                "post_floor_sigma_bg": float("nan"),
                "post_floor_min_inner": float("nan"),
                "post_floor_min_contrast": float("nan"),
            })
        return out

    pre, post, soma, dend, n_rows, n_cols, ps = _stitch_for_source(
        patch_dir, soma_dir, dend_dir, rows, pre_channel, post_channel
    )

    result = _run_puncta(
        pre, post, soma, dend, cfg_pre, cfg_post,
        near_dilate_px=near_dilate_px,
        auto_floors=auto_floors,
        apply_shape_filter=apply_shape_filter,
    )
    full_pre = result["pre_mask"]
    full_post = result["post_mask"]
    full_near = result["near_mask"]
    fp = result["pre_floors"]
    fq = result["post_floors"]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"  {source_npy}: {n_rows}x{n_cols} grid ({n_rows * ps}x{n_cols * ps}) -> "
        f"pre {result['n_pre_raw']}->{result['n_pre_on_near']}  "
        f"post {result['n_post_raw']}->{result['n_post_on_near']}  "
        f"near={full_near.mean():.2%}"
    )

    records: list[dict] = []
    for r, pp, qp, sp, dp in expected:
        gr, gcol = int(r["grid_row"]), int(r["grid_col"])
        y0, x0 = gr * ps, gcol * ps
        pre_tile = full_pre[y0:y0 + ps, x0:x0 + ps].copy()
        post_tile = full_post[y0:y0 + ps, x0:x0 + ps].copy()
        near_tile = full_near[y0:y0 + ps, x0:x0 + ps]
        np.save(pp, pre_tile.astype(np.uint8))
        np.save(qp, post_tile.astype(np.uint8))
        records.append({
            "filename": r["filename"],
            "pre_mask_filename": pp.name,
            "post_mask_filename": qp.name,
            "soma_mask_filename": sp.name,
            "dend_mask_filename": dp.name,
            "source_npy": source_npy,
            "image_index": int(r.get("image_index", -1) or -1),
            "grid_row": gr,
            "grid_col": gcol,
            "patch_size": ps,
            "pre_channel": pre_channel,
            "post_channel": post_channel,
            "near_dilate_px": near_dilate_px,
            "frac_near_patch": float(near_tile.mean()),
            "n_pre_full_on_near": int(result["n_pre_on_near"]),
            "n_post_full_on_near": int(result["n_post_on_near"]),
            "n_pre_pixels_patch": int(pre_tile.sum()),
            "n_post_pixels_patch": int(post_tile.sum()),
            "pre_tophat_r": cfg_pre.intensity_tophat_radius,
            "pre_floor_sigma_bg": float(fp[0]),
            "pre_floor_min_inner": float(fp[1]),
            "pre_floor_min_contrast": float(fp[2]),
            "post_tophat_r": cfg_post.intensity_tophat_radius,
            "post_floor_sigma_bg": float(fq[0]),
            "post_floor_min_inner": float(fq[1]),
            "post_floor_min_contrast": float(fq[2]),
        })

    del pre, post, soma, dend, full_pre, full_post, full_near, result
    gc.collect()
    return records


def _puncta_for_source_worker(args: tuple) -> list[dict]:
    (patch_dir, output_dir, soma_dir, dend_dir, source_npy, rows,
     cfg_pre, cfg_post, pre_ch, post_ch, near_dilate_px,
     auto_floors, apply_shape_filter, overwrite) = args
    return puncta_for_source(
        Path(patch_dir), Path(output_dir), Path(soma_dir), Path(dend_dir),
        source_npy, rows, cfg_pre, cfg_post,
        pre_ch, post_ch, near_dilate_px,
        auto_floors, apply_shape_filter, overwrite,
    )


def process_patches_dir(
    patch_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: BlobPseudoCfg,
    cfg_post: BlobPseudoCfg,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floors: bool,
    apply_shape_filter: bool,
    workers: int,
    overwrite: bool,
) -> list[dict]:
    """Run puncta per source image in a tiled-patches session folder."""
    index_csv = patch_dir / "index.csv"
    if not index_csv.exists():
        logger.error(f"  {patch_dir}: index.csv not found, cannot stitch")
        return []
    if not soma_dir.is_dir():
        logger.error(f"  soma dir does not exist: {soma_dir}")
        return []
    if not dend_dir.is_dir():
        logger.error(f"  dend dir does not exist: {dend_dir}")
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
        (str(patch_dir), str(output_dir), str(soma_dir), str(dend_dir),
         key, rows, cfg_pre, cfg_post,
         pre_channel, post_channel, near_dilate_px,
         auto_floors, apply_shape_filter, overwrite)
        for key, rows in sorted(groups.items())
    ]

    records: list[dict] = []
    if workers > 1 and len(work_items) > 1:
        n_workers = min(workers, len(work_items))
        logger.info(
            f"  Stitching + puncta on {len(work_items)} source image(s) "
            f"with {n_workers} workers"
        )
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=_MP_CTX) as ex:
            futures = {
                ex.submit(_puncta_for_source_worker, it): it[4]
                for it in work_items
            }
            for fut in as_completed(futures):
                try:
                    records.extend(fut.result())
                except Exception as e:
                    logger.error(f"  Worker crashed on {futures[fut]}: {e}")
    else:
        for it in work_items:
            try:
                records.extend(_puncta_for_source_worker(it))
            except Exception as e:
                logger.error(f"  FAILED on {it[4]}: {e}")

    records.sort(
        key=lambda r: (r["source_npy"], r["grid_row"], r["grid_col"])
    )

    csv_path = output_dir / "puncta_index.csv"
    if records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=list(PATCH_PUNCTA_INDEX_FIELDS), extrasaction="ignore"
            )
            w.writeheader()
            w.writerows(records)
        logger.info(f"  puncta_index.csv -> {csv_path}  ({len(records)} patches)")
    else:
        logger.warning(f"  No masks produced for {patch_dir.name}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre/post synaptic-puncta detection pipeline on full-MIP "
            ".npy files or tiled patches, using pre-computed soma + dendrite masks."
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
            "Folder of pre-computed soma masks (<stem>_soma.npy). "
            "Required with --input_dir / --input_patches."
        ),
    )
    parser.add_argument(
        "--soma_root", type=Path,
        help=(
            "Root of soma-mask session subfolders mirrored from the input root. "
            "Required with --input_root / --input_patch_root."
        ),
    )
    parser.add_argument(
        "--dendrite_dir", type=Path,
        help=(
            "Folder of pre-computed dendrite masks (<stem>_dend.npy). "
            "Required with --input_dir / --input_patches."
        ),
    )
    parser.add_argument(
        "--dendrite_root", type=Path,
        help=(
            "Root of dendrite-mask session subfolders mirrored from the input root. "
            "Required with --input_root / --input_patch_root."
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        help="Output folder for pre/post puncta masks (with --input_dir / --input_patches).",
    )
    parser.add_argument(
        "--output_root", type=Path,
        help=(
            "Output root; session subfolders are mirrored "
            "(with --input_root / --input_patch_root)."
        ),
    )
    parser.add_argument(
        "--pre_channel", type=int, default=0,
        help="Index of the pre-synaptic channel in (C, H, W) (default: 0).",
    )
    parser.add_argument(
        "--post_channel", type=int, default=1,
        help="Index of the post-synaptic channel in (C, H, W) (default: 1).",
    )
    parser.add_argument(
        "--near_dilate_px", type=int, default=DEFAULT_NEAR_DILATE_PX,
        help=(
            f"Extra dilation (px) applied to soma U dendrite to form the "
            f"'near-neuron' acceptance mask (default: {DEFAULT_NEAR_DILATE_PX}; "
            f"~0.43 um at 107 nm/px)."
        ),
    )
    parser.add_argument(
        "--cfg_pre", type=str, default=None,
        help="JSON dict or path to .json with DEFAULT_CFG_PRE overrides.",
    )
    parser.add_argument(
        "--cfg_post", type=str, default=None,
        help="JSON dict or path to .json with DEFAULT_CFG_POST overrides.",
    )
    parser.add_argument(
        "--no_auto_floors", action="store_true",
        help=(
            "Disable per-image auto-derivation of absolute-intensity floors. "
            "When off (default), each image's bg median + MAD-sigma in tophat "
            "space sets zscore_{sigma_bg_floor, min_inner, min_contrast}; "
            "when on, the literal cfg values are used."
        ),
    )
    parser.add_argument(
        "--apply_shape_filter", action="store_true",
        help=(
            "Apply filter_by_size_shape to the rendered per-channel mask "
            "after the centre-on-near gate. Off by default: whole LoG disks "
            "of any blob whose centre lies on near_mask are kept."
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
        help="Recompute and overwrite existing pre+post mask files.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without running the pipeline.",
    )

    args = parser.parse_args()

    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required with --input_dir")
    if args.input_root and not args.output_root:
        parser.error("--output_root is required with --input_root")
    if args.input_patches and not args.output_dir:
        parser.error("--output_dir is required with --input_patches")
    if args.input_patch_root and not args.output_root:
        parser.error("--output_root is required with --input_patch_root")

    if (args.input_dir or args.input_patches) and not (args.soma_dir and args.dendrite_dir):
        parser.error(
            "--soma_dir AND --dendrite_dir are required with --input_dir / --input_patches"
        )
    if (args.input_root or args.input_patch_root) and not (args.soma_root and args.dendrite_root):
        parser.error(
            "--soma_root AND --dendrite_root are required with --input_root / --input_patch_root"
        )

    try:
        cfg_pre = _build_blob_cfg(_merge_cfg(DEFAULT_CFG_PRE, _parse_json_cfg(args.cfg_pre)))
        cfg_post = _build_blob_cfg(_merge_cfg(DEFAULT_CFG_POST, _parse_json_cfg(args.cfg_post)))
    except (json.JSONDecodeError, OSError, ValueError) as e:
        parser.error(f"cfg parse error: {e}")

    if args.pre_channel == args.post_channel:
        parser.error(
            f"--pre_channel and --post_channel must differ (both {args.pre_channel})"
        )

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 2)
    auto_floors = not args.no_auto_floors

    def _log_cfg() -> None:
        logger.info(f"Pre-channel:    {args.pre_channel}")
        logger.info(f"Post-channel:   {args.post_channel}")
        logger.info(f"Near dilate:    {args.near_dilate_px} px")
        logger.info(f"Auto floors:    {auto_floors}")
        logger.info(f"Shape filter:   {args.apply_shape_filter}")
        logger.info(f"Workers:        {workers}")
        logger.info(
            f"PRE  cfg: sigma=[{cfg_pre.log_min_sigma}, {cfg_pre.log_max_sigma}] "
            f"x{cfg_pre.log_num_sigma}  log_thr={cfg_pre.log_threshold}  "
            f"tophat_r={cfg_pre.intensity_tophat_radius}  "
            f"z>={cfg_pre.zscore_threshold} "
            f"(ann {cfg_pre.zscore_inner_radius}/{cfg_pre.zscore_outer_radius}, "
            f"mu_bg="
            f"{'p' + str(cfg_pre.zscore_bg_percentile) if cfg_pre.zscore_bg_percentile else 'mean'}, "
            f"sg={'MAD' if cfg_pre.zscore_bg_robust_scale else 'std'})  "
            f"size=[{cfg_pre.min_size}, {cfg_pre.max_size}]"
        )
        logger.info(
            f"POST cfg: sigma=[{cfg_post.log_min_sigma}, {cfg_post.log_max_sigma}] "
            f"x{cfg_post.log_num_sigma}  log_thr={cfg_post.log_threshold}  "
            f"tophat_r={cfg_post.intensity_tophat_radius}  "
            f"z>={cfg_post.zscore_threshold} "
            f"(ann {cfg_post.zscore_inner_radius}/{cfg_post.zscore_outer_radius}, "
            f"mu_bg="
            f"{'p' + str(cfg_post.zscore_bg_percentile) if cfg_post.zscore_bg_percentile else 'mean'}, "
            f"sg={'MAD' if cfg_post.zscore_bg_robust_scale else 'std'})  "
            f"size=[{cfg_post.min_size}, {cfg_post.max_size}]"
        )

    # ---- MIP, single folder ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info("Mode:           MIP (single folder)")
        logger.info(f"Input:          {args.input_dir}")
        logger.info(f"Soma masks:     {args.soma_dir}")
        logger.info(f"Dend masks:     {args.dendrite_dir}")
        logger.info(f"Output:         {args.output_dir}")
        _log_cfg()
        if args.dry_run:
            n = len([p for p in args.input_dir.glob("*.npy") if p.name != "index.csv"])
            logger.info(f"[DRY RUN] Would run puncta on {n} .npy file(s).")
            return
        process_dir(
            args.input_dir, args.output_dir, args.soma_dir, args.dendrite_dir,
            cfg_pre, cfg_post,
            args.pre_channel, args.post_channel, args.near_dilate_px,
            auto_floors, args.apply_shape_filter,
            workers, args.overwrite,
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
        logger.info("Mode:           Patches (single folder, stitch -> puncta -> slice)")
        logger.info(f"Input:          {args.input_patches}")
        logger.info(f"Soma masks:     {args.soma_dir}")
        logger.info(f"Dend masks:     {args.dendrite_dir}")
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
            args.input_patches, args.output_dir,
            args.soma_dir, args.dendrite_dir,
            cfg_pre, cfg_post,
            args.pre_channel, args.post_channel, args.near_dilate_px,
            auto_floors, args.apply_shape_filter,
            workers, args.overwrite,
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
        mode_name = "Patches (root, stitch -> puncta -> slice)"
        select = lambda d: (d / "index.csv").exists()  # noqa: E731
        per_folder = process_patches_dir

    if not input_root.is_dir():
        logger.error(f"Input root does not exist: {input_root}")
        sys.exit(1)
    if not args.soma_root.is_dir():
        logger.error(f"Soma root does not exist: {args.soma_root}")
        sys.exit(1)
    if not args.dendrite_root.is_dir():
        logger.error(f"Dend root does not exist: {args.dendrite_root}")
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
    logger.info("Puncta configuration")
    logger.info(f"{'='*70}")
    logger.info(f"Mode:           {mode_name}")
    logger.info(f"Input root:     {input_root}")
    logger.info(f"Soma root:      {args.soma_root}")
    logger.info(f"Dend root:      {args.dendrite_root}")
    logger.info(f"Output root:    {args.output_root}")
    logger.info(f"Folders:        {len(folders)}")
    _log_cfg()

    if args.dry_run:
        logger.info("[DRY RUN] Would process:")
        for d in folders:
            soma_sub = args.soma_root / d.name
            dend_sub = args.dendrite_root / d.name
            missing_parts = []
            if not soma_sub.is_dir():
                missing_parts.append("SOMA")
            if not dend_sub.is_dir():
                missing_parts.append("DEND")
            missing = f"  [{'+'.join(missing_parts)} DIR MISSING]" if missing_parts else ""
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
        dend_sub = args.dendrite_root / folder.name
        if not soma_sub.is_dir():
            logger.error(f"  soma subfolder missing: {soma_sub}; skipping")
            results[folder.name] = 0
            continue
        if not dend_sub.is_dir():
            logger.error(f"  dend subfolder missing: {dend_sub}; skipping")
            results[folder.name] = 0
            continue
        recs = per_folder(
            folder, args.output_root / folder.name, soma_sub, dend_sub,
            cfg_pre, cfg_post,
            args.pre_channel, args.post_channel, args.near_dilate_px,
            auto_floors, args.apply_shape_filter,
            workers, args.overwrite,
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
