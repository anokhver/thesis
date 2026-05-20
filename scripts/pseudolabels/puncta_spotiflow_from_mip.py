#!/usr/bin/env python3
r"""Run the pre/post synaptic-puncta detection pipeline on full-MIP .npy
files or a tiled-patches session, using a pretrained Spotiflow model
(Dominguez Mantes et al., Nat. Methods 2025) plus pre-computed soma +
dendrite masks for the structural near-gate.

Parallel to ``scripts/pseudolabels/puncta_from_mip.py``: same CLI
surface for inputs / structural masks / outputs, same on-disk mask
convention (``<stem>_pre.npy`` / ``<stem>_post.npy`` plus a per-folder
``puncta_index.csv``), but the detector is Spotiflow instead of LoG.

Single-process execution: Spotiflow holds one model per process on the
GPU, so we load it once and loop over images sequentially (no
``--workers``). Spotiflow does its own percentile normalisation
internally, so there is no white-tophat step.

Two input modes (mirroring the LoG script):

  1. **MIP mode** -- inputs are full ``(C, H, W)`` ``.npy`` files. Two
     masks ``<stem>_pre.npy`` and ``<stem>_post.npy`` (uint8, ``(H, W)``)
     are written per source image plus a per-folder
     ``puncta_index.csv``.

  2. **Patches mode** -- inputs are tiled patch folders containing
     ``index.csv`` + ``<stem>_r##_c##.npy``. The pre + post channels
     and per-patch soma + dendrite masks are stitched into one full
     image / mask per source, Spotiflow runs once at full scale
     (avoiding per-patch border artefacts), then the per-channel mask
     is sliced back so each patch gets a ``<patch_stem>_pre.npy`` /
     ``<patch_stem>_post.npy`` of shape ``(ps, ps)``.

Defaults bake in the per-channel render radii from
``notebooks/pseudolabels/puncta_spotiflow.ipynb`` (PRE=4 px, POST=2 px
at 107 nm/px). Override any field of ``SpotiflowPunctaCfg`` via JSON:

  --cfg_pre   keys of DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW
  --cfg_post  keys of DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW

Per-image absolute-intensity floors (analogue of LoG zscore_min_inner)
are auto-derived from each channel's raw background by default. Disable
with ``--no_auto_floor`` to use the literal cfg value.

Usage::

    # MIP mode, single folder:
    python scripts/pseudolabels/puncta_spotiflow_from_mip.py \
        --input_dir     Microscopy_no_patch/SessionName \
        --soma_dir      Microscopy_soma/SessionName \
        --dendrite_dir  Microscopy_dend/SessionName \
        --output_dir    Microscopy_puncta_spotiflow/SessionName

    # MIP mode, all session subfolders under a root:
    python scripts/pseudolabels/puncta_spotiflow_from_mip.py \
        --input_root     Microscopy_no_patch \
        --soma_root      Microscopy_soma \
        --dendrite_root  Microscopy_dend \
        --output_root    Microscopy_puncta_spotiflow

    # Patches mode, single session:
    python scripts/pseudolabels/puncta_spotiflow_from_mip.py \
        --input_patches Microscopy/SessionName \
        --soma_dir      Microscopy_soma/SessionName \
        --dendrite_dir  Microscopy_dend/SessionName \
        --output_dir    Microscopy_puncta_spotiflow/SessionName

    # Override a knob (JSON):
    python scripts/pseudolabels/puncta_spotiflow_from_mip.py --input_dir ... \
        --soma_dir ... --dendrite_dir ... --output_dir ... \
        --cfg_pre '{"render_radius_px": 5.0, "prob_thresh": 0.4}'
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
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage.morphology import disk as morph_disk, dilation


_MP_CTX = mp.get_context("spawn")


def _np_load(path, *, mmap_mode=None):
    """Read a .npy via one big sequential read.

    NTFS-3G (FUSE) occasionally returns EIO on np.load's multi-chunk
    read pattern but tolerates one large read fine. Mirrors the helper
    in ``scripts/pseudolabels/puncta_from_mip.py``.
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

from synaptic_ssl.pseudolabels.puncta_spotiflow import (  # noqa: E402
    SpotiflowPunctaCfg,
    DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW,
    DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW,
    resolve_cfg as resolve_spotiflow_cfg,
    detect_puncta_channel,
    load_model,
)
from synaptic_ssl.pseudolabels.puncta_common import (  # noqa: E402
    puncta_to_mask,
    restrict_puncta_to_near,
)
from synaptic_ssl.pseudolabels.puncta_log import DEFAULT_NEAR_DILATE_PX  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    "n_pre_raw",         # kept after intensity floor (before near-gate)
    "n_post_raw",
    "n_pre_on_near",
    "n_post_on_near",
    "model",
    "pre_prob_thresh",
    "pre_min_distance",
    "pre_render_radius",
    "pre_intensity_floor",
    "post_prob_thresh",
    "post_min_distance",
    "post_render_radius",
    "post_intensity_floor",
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
    "model",
    "pre_prob_thresh",
    "pre_min_distance",
    "pre_render_radius",
    "pre_intensity_floor",
    "post_prob_thresh",
    "post_min_distance",
    "post_render_radius",
    "post_intensity_floor",
)


# ---------------------------------------------------------------------------
# Cfg helpers
# ---------------------------------------------------------------------------

def _parse_json_cfg(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _parse_n_tiles(s: str | None) -> tuple[int, int] | None:
    if s is None or s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--n_tiles expects 'NY,NX', got {s!r}"
        )
    try:
        ny, nx = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--n_tiles values must be integers, got {s!r}"
        ) from e
    if ny <= 0 or nx <= 0:
        raise argparse.ArgumentTypeError(
            f"--n_tiles values must be positive, got {s!r}"
        )
    return (ny, nx)


def _scalar_or_nan(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


# ---------------------------------------------------------------------------
# Detection pipeline on full-shape arrays
# ---------------------------------------------------------------------------

def _run_puncta(
    pre_img: np.ndarray,
    post_img: np.ndarray,
    soma_mask: np.ndarray,
    dend_mask: np.ndarray,
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    *,
    model,
    near_dilate_px: int,
    auto_floor: bool,
) -> dict:
    """Full Spotiflow pipeline on already-loaded full-shape arrays."""
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

    _, _, kept_pre, cfg_eff_pre = detect_puncta_channel(
        pre_img, cfg_pre, model=model, auto_floor=auto_floor,
    )
    _, _, kept_post, cfg_eff_post = detect_puncta_channel(
        post_img, cfg_post, model=model, auto_floor=auto_floor,
    )

    on_pre = restrict_puncta_to_near(kept_pre, near_mask)
    on_post = restrict_puncta_to_near(kept_post, near_mask)

    H, W = pre_img.shape
    pre_mask = puncta_to_mask(on_pre, (H, W)).astype(bool)
    post_mask = puncta_to_mask(on_post, (H, W)).astype(bool)

    return dict(
        pre_mask=pre_mask,
        post_mask=post_mask,
        near_mask=near_mask,
        n_pre_raw=int(kept_pre.shape[0]),
        n_post_raw=int(kept_post.shape[0]),
        n_pre_on_near=int(on_pre.shape[0]),
        n_post_on_near=int(on_post.shape[0]),
        cfg_eff_pre=cfg_eff_pre,
        cfg_eff_post=cfg_eff_post,
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
    if first.shape != (ps, ps):
        raise ValueError(f"non-square tile {first.shape} in {files[0].name}")
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


def _load_struct_mask(mask_dir: Path, stem: str, kind: str,
                      full_shape: tuple[int, int]) -> np.ndarray:
    """Load ``{stem}_{kind}.npy`` (full image) or stitch tiled patches."""
    full = mask_dir / f"{stem}_{kind}.npy"
    if full.exists():
        return _np_load(full).astype(bool)
    return _stitch_tiled_mask(mask_dir, stem, kind, full_shape)


def _record_from_cfgs(
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    cfg_eff_pre: SpotiflowPunctaCfg | None,
    cfg_eff_post: SpotiflowPunctaCfg | None,
) -> dict:
    """Fill the cfg-derived fields shared by MIP + patches records."""
    cep = cfg_eff_pre if cfg_eff_pre is not None else cfg_pre
    ceq = cfg_eff_post if cfg_eff_post is not None else cfg_post
    return dict(
        model=cfg_pre.pretrained_model,
        pre_prob_thresh=_scalar_or_nan(cep.prob_thresh),
        pre_min_distance=int(cep.min_distance),
        pre_render_radius=float(cep.render_radius_px),
        pre_intensity_floor=float(cep.intensity_floor),
        post_prob_thresh=_scalar_or_nan(ceq.prob_thresh),
        post_min_distance=int(ceq.min_distance),
        post_render_radius=float(ceq.render_radius_px),
        post_intensity_floor=float(ceq.intensity_floor),
    )


# ---------------------------------------------------------------------------
# Multi-worker plumbing (CPU only). Each worker process loads its own
# Spotiflow model in `_worker_init` and stashes it in a module global;
# torch is clamped to 1 thread per worker so N workers x all-cores doesn't
# thrash. Refuse non-CPU devices upstream in main().
# ---------------------------------------------------------------------------

_WORKER_MODEL = None  # set in each spawned process by _worker_init


def _clamp_torch_threads(n: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass


def _worker_init(cfg_pre: SpotiflowPunctaCfg, threads_per_worker: int) -> None:
    _clamp_torch_threads(threads_per_worker)
    global _WORKER_MODEL
    _WORKER_MODEL = load_model(cfg_pre)


def _puncta_one_worker(work: tuple) -> dict | None:
    (npy_path_str, output_dir_str, soma_dir_str, dend_dir_str,
     cfg_pre, cfg_post, pre_channel, post_channel,
     near_dilate_px, auto_floor, overwrite) = work
    return puncta_one(
        Path(npy_path_str), Path(output_dir_str),
        Path(soma_dir_str), Path(dend_dir_str),
        cfg_pre, cfg_post,
        model=_WORKER_MODEL,
        pre_channel=pre_channel,
        post_channel=post_channel,
        near_dilate_px=near_dilate_px,
        auto_floor=auto_floor,
        overwrite=overwrite,
    )


def _puncta_for_source_worker(work: tuple) -> list[dict]:
    (patch_dir_str, output_dir_str, soma_dir_str, dend_dir_str,
     source_npy, rows, cfg_pre, cfg_post,
     pre_channel, post_channel, near_dilate_px,
     auto_floor, overwrite) = work
    return puncta_for_source(
        Path(patch_dir_str), Path(output_dir_str),
        Path(soma_dir_str), Path(dend_dir_str),
        source_npy, rows, cfg_pre, cfg_post,
        model=_WORKER_MODEL,
        pre_channel=pre_channel,
        post_channel=post_channel,
        near_dilate_px=near_dilate_px,
        auto_floor=auto_floor,
        overwrite=overwrite,
    )


def puncta_one(
    npy_path: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    *,
    model,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floor: bool,
    overwrite: bool,
) -> dict | None:
    """Detect puncta on one MIP .npy with Spotiflow. Save masks. Return CSV record."""
    pre_path = _pre_mask_path(output_dir, npy_path.stem)
    post_path = _post_mask_path(output_dir, npy_path.stem)
    soma_full = _soma_mask_path(soma_dir, npy_path.stem)
    dend_full = _dend_mask_path(dend_dir, npy_path.stem)
    soma_tiles = sorted(soma_dir.glob(f"{npy_path.stem}_r??_c??_soma.npy"))
    dend_tiles = sorted(dend_dir.glob(f"{npy_path.stem}_r??_c??_dend.npy"))

    if not soma_full.exists() and not soma_tiles:
        logger.error(
            f"  {npy_path.name}: soma mask missing "
            f"(no {soma_full.name} or tiled patches in {soma_dir})"
        )
        return None
    if not dend_full.exists() and not dend_tiles:
        logger.error(
            f"  {npy_path.name}: dend mask missing "
            f"(no {dend_full.name} or tiled patches in {dend_dir})"
        )
        return None
    soma_path = soma_full if soma_full.exists() else soma_tiles[0]
    dend_path = dend_full if dend_full.exists() else dend_tiles[0]

    cfg_record = _record_from_cfgs(cfg_pre, cfg_post, None, None)

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
            "n_pre_on_near": n_pre_pix,
            "n_post_on_near": n_post_pix,
            **cfg_record,
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
        H, W = pre_img.shape
        soma_mask = _load_struct_mask(soma_dir, npy_path.stem, "soma", (H, W))
        dend_mask = _load_struct_mask(dend_dir, npy_path.stem, "dend", (H, W))
        result = _run_puncta(
            pre_img, post_img, soma_mask, dend_mask,
            cfg_pre, cfg_post,
            model=model,
            near_dilate_px=near_dilate_px,
            auto_floor=auto_floor,
        )
    except Exception as e:
        logger.error(f"  {npy_path.name}: FAILED ({e})")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(pre_path, result["pre_mask"].astype(np.uint8))
    np.save(post_path, result["post_mask"].astype(np.uint8))
    H, W = result["pre_mask"].shape
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
        **_record_from_cfgs(
            cfg_pre, cfg_post,
            result["cfg_eff_pre"], result["cfg_eff_post"],
        ),
    }

    del full, pre_img, post_img, soma_mask, dend_mask, result
    gc.collect()
    return rec


def process_dir(
    input_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    *,
    model,
    workers: int,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floor: bool,
    overwrite: bool,
) -> list[dict]:
    """Run Spotiflow puncta on all MIP .npy in ``input_dir``. Write masks + CSV.

    `workers == 1` runs sequentially with the already-loaded `model`.
    `workers > 1` spawns a process pool; each worker loads its own model
    in `_worker_init` so we don't pickle a torch model across processes.
    """
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

    records: list[dict] = []
    if workers > 1 and len(npy_files) > 1:
        n_workers = min(workers, len(npy_files))
        logger.info(
            f"  Running {len(npy_files)} files with {n_workers} workers "
            f"(each loads its own Spotiflow model)"
        )
        work_items = [
            (str(p), str(output_dir), str(soma_dir), str(dend_dir),
             cfg_pre, cfg_post, pre_channel, post_channel,
             near_dilate_px, auto_floor, overwrite)
            for p in npy_files
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CTX,
            initializer=_worker_init,
            initargs=(cfg_pre, 1),
        ) as ex:
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
        logger.info(f"  Running {len(npy_files)} files (sequential; single model)")
        for p in npy_files:
            rec = puncta_one(
                p, output_dir, soma_dir, dend_dir,
                cfg_pre, cfg_post,
                model=model,
                pre_channel=pre_channel,
                post_channel=post_channel,
                near_dilate_px=near_dilate_px,
                auto_floor=auto_floor,
                overwrite=overwrite,
            )
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
# Patches mode: stitch pre + post + soma + dend -> spotiflow -> slice back
# ---------------------------------------------------------------------------

def _stitch_for_source(
    patch_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    rows: list[dict],
    pre_channel: int,
    post_channel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Stitch pre + post channels and soma + dend masks for one source.

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
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    *,
    model,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floor: bool,
    overwrite: bool,
) -> list[dict]:
    """Stitch one source's patches + masks, run Spotiflow, write per-patch masks."""
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

    cfg_record = _record_from_cfgs(cfg_pre, cfg_post, None, None)

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
                **cfg_record,
            })
        return out

    pre, post, soma, dend, n_rows, n_cols, ps = _stitch_for_source(
        patch_dir, soma_dir, dend_dir, rows, pre_channel, post_channel
    )

    result = _run_puncta(
        pre, post, soma, dend, cfg_pre, cfg_post,
        model=model,
        near_dilate_px=near_dilate_px,
        auto_floor=auto_floor,
    )
    full_pre = result["pre_mask"]
    full_post = result["post_mask"]
    full_near = result["near_mask"]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"  {source_npy}: {n_rows}x{n_cols} grid ({n_rows * ps}x{n_cols * ps}) -> "
        f"pre {result['n_pre_raw']}->{result['n_pre_on_near']}  "
        f"post {result['n_post_raw']}->{result['n_post_on_near']}  "
        f"near={full_near.mean():.2%}"
    )

    cfg_record_eff = _record_from_cfgs(
        cfg_pre, cfg_post,
        result["cfg_eff_pre"], result["cfg_eff_post"],
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
            **cfg_record_eff,
        })

    del pre, post, soma, dend, full_pre, full_post, full_near, result
    gc.collect()
    return records


def process_patches_dir(
    patch_dir: Path,
    output_dir: Path,
    soma_dir: Path,
    dend_dir: Path,
    cfg_pre: SpotiflowPunctaCfg,
    cfg_post: SpotiflowPunctaCfg,
    *,
    model,
    workers: int,
    pre_channel: int,
    post_channel: int,
    near_dilate_px: int,
    auto_floor: bool,
    overwrite: bool,
) -> list[dict]:
    """Run Spotiflow puncta per source image in a tiled-patches session folder.

    `workers > 1` uses a process pool where each worker loads its own
    Spotiflow model in `_worker_init` (no torch-model pickling).
    """
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

    records: list[dict] = []
    if workers > 1 and len(groups) > 1:
        n_workers = min(workers, len(groups))
        logger.info(
            f"  Stitching + spotiflow on {len(groups)} source image(s) "
            f"with {n_workers} workers"
        )
        work_items = [
            (str(patch_dir), str(output_dir), str(soma_dir), str(dend_dir),
             key, rows, cfg_pre, cfg_post,
             pre_channel, post_channel, near_dilate_px,
             auto_floor, overwrite)
            for key, rows in sorted(groups.items())
        ]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=_MP_CTX,
            initializer=_worker_init,
            initargs=(cfg_pre, 1),
        ) as ex:
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
        logger.info(
            f"  Stitching + spotiflow on {len(groups)} source image(s) (sequential)"
        )
        for key, rows in sorted(groups.items()):
            try:
                records.extend(puncta_for_source(
                    patch_dir, output_dir, soma_dir, dend_dir,
                    key, rows, cfg_pre, cfg_post,
                    model=model,
                    pre_channel=pre_channel,
                    post_channel=post_channel,
                    near_dilate_px=near_dilate_px,
                    auto_floor=auto_floor,
                    overwrite=overwrite,
                ))
            except Exception as e:
                logger.error(f"  FAILED on {key}: {e}")

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
            ".npy files or tiled patches using a pretrained Spotiflow model."
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
        help=(
            "JSON dict or path to .json with DEFAULT_PUNCTA_CFG_PRE_SPOTIFLOW "
            "overrides (any field of SpotiflowPunctaCfg)."
        ),
    )
    parser.add_argument(
        "--cfg_post", type=str, default=None,
        help=(
            "JSON dict or path to .json with DEFAULT_PUNCTA_CFG_POST_SPOTIFLOW "
            "overrides (any field of SpotiflowPunctaCfg)."
        ),
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Pretrained Spotiflow model name (overrides cfg). Defaults to the "
            "value in the per-channel cfg ('general' out of the box)."
        ),
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Spotiflow device override: 'auto' | 'cuda' | 'cpu' | 'mps'.",
    )
    parser.add_argument(
        "--n_tiles", type=_parse_n_tiles, default=None,
        help=(
            "Spotiflow tiling override 'NY,NX' (e.g. '3,3' for a 2304x2304 "
            "MIP on a small GPU). Empty = let spotiflow infer."
        ),
    )
    parser.add_argument(
        "--no_auto_floor", action="store_true",
        help=(
            "Disable per-image auto-derivation of the absolute-intensity "
            "floor. When off (default), each channel's bg median + "
            "k*MAD-sigma sets the floor; when on, the literal cfg value is "
            "used."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel CPU worker processes (default: 1 = sequential). "
            "Each worker loads its own Spotiflow model and is clamped to a "
            "single torch thread. REQUIRES --device cpu (GPU + multiple "
            "processes would OOM each other)."
        ),
    )
    parser.add_argument(
        "--max_folders", type=int, default=0,
        help="Process at most this many session folders (0 = all; root modes only).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite existing pre+post mask files.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without loading the model or running the pipeline.",
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

    # CLI overrides land in the cfg before resolving, so --model / --device /
    # --n_tiles win over any value in --cfg_pre / --cfg_post.
    cli_extra: dict = {}
    if args.model is not None:
        cli_extra["pretrained_model"] = args.model
    if args.device is not None:
        cli_extra["device"] = args.device
    if args.n_tiles is not None:
        cli_extra["n_tiles"] = args.n_tiles

    try:
        cfg_pre = resolve_spotiflow_cfg(
            {**_parse_json_cfg(args.cfg_pre), **cli_extra}, channel="pre",
        )
        cfg_post = resolve_spotiflow_cfg(
            {**_parse_json_cfg(args.cfg_post), **cli_extra}, channel="post",
        )
    except (json.JSONDecodeError, OSError, ValueError) as e:
        parser.error(f"cfg parse error: {e}")

    if cfg_pre.pretrained_model != cfg_post.pretrained_model:
        parser.error(
            "pre and post must use the same pretrained_model "
            f"(got {cfg_pre.pretrained_model!r} vs {cfg_post.pretrained_model!r}); "
            "one model is loaded per run."
        )

    if args.pre_channel == args.post_channel:
        parser.error(
            f"--pre_channel and --post_channel must differ (both {args.pre_channel})"
        )

    if args.workers < 1:
        parser.error(f"--workers must be >= 1, got {args.workers}")
    if args.workers > 1 and cfg_pre.device.lower() not in ("cpu",):
        parser.error(
            f"--workers > 1 requires --device cpu (got device={cfg_pre.device!r}); "
            "multiple processes sharing a GPU will OOM each other."
        )

    auto_floor = not args.no_auto_floor

    def _log_cfg() -> None:
        logger.info(f"Pre-channel:    {args.pre_channel}")
        logger.info(f"Post-channel:   {args.post_channel}")
        logger.info(f"Near dilate:    {args.near_dilate_px} px")
        logger.info(f"Auto floor:     {auto_floor}")
        logger.info(f"Workers:        {args.workers}")
        logger.info(f"Model:          {cfg_pre.pretrained_model}  (device={cfg_pre.device})")
        logger.info(
            f"PRE  cfg: prob_thresh={cfg_pre.prob_thresh}  "
            f"min_dist={cfg_pre.min_distance}  "
            f"render_r={cfg_pre.render_radius_px}px  "
            f"intensity_floor_k={cfg_pre.intensity_floor_k} "
            f"(inner_r={cfg_pre.intensity_inner_radius})  "
            f"n_tiles={cfg_pre.n_tiles}"
        )
        logger.info(
            f"POST cfg: prob_thresh={cfg_post.prob_thresh}  "
            f"min_dist={cfg_post.min_distance}  "
            f"render_r={cfg_post.render_radius_px}px  "
            f"intensity_floor_k={cfg_post.intensity_floor_k} "
            f"(inner_r={cfg_post.intensity_inner_radius})  "
            f"n_tiles={cfg_post.n_tiles}"
        )

    # ---- MIP, single folder ------------------------------------------------
    if args.input_dir:
        if not args.input_dir.is_dir():
            logger.error(f"Input dir does not exist: {args.input_dir}")
            sys.exit(1)
        logger.info("Mode:           MIP (single folder, Spotiflow)")
        logger.info(f"Input:          {args.input_dir}")
        logger.info(f"Soma masks:     {args.soma_dir}")
        logger.info(f"Dend masks:     {args.dendrite_dir}")
        logger.info(f"Output:         {args.output_dir}")
        _log_cfg()
        if args.dry_run:
            n = len([p for p in args.input_dir.glob("*.npy") if p.name != "index.csv"])
            logger.info(f"[DRY RUN] Would run Spotiflow on {n} .npy file(s).")
            return
        model = None if args.workers > 1 else load_model(cfg_pre)
        process_dir(
            args.input_dir, args.output_dir, args.soma_dir, args.dendrite_dir,
            cfg_pre, cfg_post,
            model=model,
            workers=args.workers,
            pre_channel=args.pre_channel,
            post_channel=args.post_channel,
            near_dilate_px=args.near_dilate_px,
            auto_floor=auto_floor,
            overwrite=args.overwrite,
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
        logger.info("Mode:           Patches (single folder, stitch -> spotiflow -> slice)")
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
        model = None if args.workers > 1 else load_model(cfg_pre)
        process_patches_dir(
            args.input_patches, args.output_dir,
            args.soma_dir, args.dendrite_dir,
            cfg_pre, cfg_post,
            model=model,
            workers=args.workers,
            pre_channel=args.pre_channel,
            post_channel=args.post_channel,
            near_dilate_px=args.near_dilate_px,
            auto_floor=auto_floor,
            overwrite=args.overwrite,
        )
        return

    # ---- Batch (root) mode -------------------------------------------------
    if args.input_root:
        input_root = args.input_root
        mode_name = "MIP (root, Spotiflow)"
        select = lambda d: any(p.name != "index.csv" for p in d.glob("*.npy"))  # noqa: E731
        per_folder = process_dir
    else:
        input_root = args.input_patch_root
        mode_name = "Patches (root, stitch -> spotiflow -> slice)"
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
    logger.info("Spotiflow puncta configuration")
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

    model = None if args.workers > 1 else load_model(cfg_pre)
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
            model=model,
            workers=args.workers,
            pre_channel=args.pre_channel,
            post_channel=args.post_channel,
            near_dilate_px=args.near_dilate_px,
            auto_floor=auto_floor,
            overwrite=args.overwrite,
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
