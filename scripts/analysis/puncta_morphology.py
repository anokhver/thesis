#!/usr/bin/env python3
r"""Per-punctum morphology CSV from full-MIP ``.npy`` files + dendrite masks.

Thin CLI wrapper around ``synaptic_ssl.analysis.puncta_morphology.analyse_image``.
Mirrors the config/CLI conventions of ``scripts/pseudolabels/puncta_from_mip.py``
but writes ONE row per detected punctum (pre + post) instead of boolean masks,
so every histogram / density / group comparison downstream can be computed from
the CSV alone (see ``notebooks/analysis/puncta_histograms.ipynb``).

For each ``<stem>.npy`` under ``--input_dir`` (shape ``(C, H, W)``) the
dendrite mask is loaded from ``--dendrite_dir`` as either:

  - ``<stem>_dend.npy`` (full-image bool mask), or
  - tiled ``<stem>_r##_c##_dend.npy`` patches (stitched to the image shape).

Images without a dendrite mask are skipped with a logged error (not a crash).

Usage::

    python scripts/analysis/puncta_morphology.py \
        --input_dir data/mips/SessionName \
        --dendrite_dir data/pseudolabels/dend/SessionName \
        --output_csv outs/puncta_morphology.csv \
        --group_patterns configs/clustering/group_patterns.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import dilation, disk

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
for _p in (_SRC, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.analysis.puncta_morphology import (  # noqa: E402
    PUNCTA_CSV_COLUMNS,
    analyse_image,
)
from synaptic_ssl.clustering.groups import (  # noqa: E402
    classify_image_by_patterns,
    load_group_rules,
)
from synaptic_ssl.pseudolabels.puncta_log import (  # noqa: E402
    DEFAULT_NEAR_DILATE_PX,
    PunctaCfg,
    resolve_cfg,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

#: Default pixel size (um/px), 107 nm/px, as used throughout the repo.
DEFAULT_PX_SIZE_UM = 0.10685428060522417

#: Per-image summary CSV columns (one row per file_name x role).
IMAGES_CSV_COLUMNS: tuple[str, ...] = (
    "file_name",
    "group_name",
    "role",
    "channel_index",
    "n_puncta",
    "dendrite_area_um2",
)

_TILE_RE = re.compile(r"_r(\d{2})_c(\d{2})_[a-z]+\.npy$")


def _parse_json_cfg(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(s)


def _stitch_tiled_dend(dend_dir: Path, stem: str, full_shape: tuple[int, int]) -> np.ndarray:
    """Stitch ``{stem}_r##_c##_dend.npy`` tiles into one (H, W) bool mask."""
    H, W = full_shape
    files = sorted(dend_dir.glob(f"{stem}_r??_c??_dend.npy"))
    if not files:
        raise FileNotFoundError(f"no tiled dend patches matching {stem} in {dend_dir}")
    first = np.load(files[0])
    ps = first.shape[0]
    if first.shape != (ps, ps):
        raise ValueError(f"non-square tile {first.shape} in {files[0].name}")
    nr, nc = H // ps, W // ps
    if len(files) != nr * nc:
        raise ValueError(
            f"expected {nr * nc} tiles ({nr}x{nc} at ps={ps}) for {stem}, "
            f"got {len(files)} in {dend_dir}"
        )
    out = np.zeros((H, W), dtype=bool)
    for fp in files:
        m = _TILE_RE.search(fp.name)
        if not m:
            raise ValueError(f"bad tile filename: {fp.name}")
        rr, cc = int(m.group(1)), int(m.group(2))
        out[rr * ps:(rr + 1) * ps, cc * ps:(cc + 1) * ps] = np.load(fp).astype(bool)
    return out


def _load_dend_mask(dend_dir: Path, stem: str, full_shape: tuple[int, int]) -> np.ndarray:
    full = dend_dir / f"{stem}_dend.npy"
    if full.exists():
        return np.load(full).astype(bool)
    return _stitch_tiled_dend(dend_dir, stem, full_shape)


_SKIP_SUFFIXES = ("_pre", "_post", "_soma", "_dend", "_overlap")


def _is_skippable(stem: str) -> bool:
    return any(stem.endswith(suffix) for suffix in _SKIP_SUFFIXES)


def _analyse_one(
    npy_path: Path,
    dendrite_dir: Path,
    pre_channel: int,
    post_channel: int,
    dendrite_channel: int,
    cfg_pre: PunctaCfg,
    cfg_post: PunctaCfg,
    near_dilate_px: int,
    px_size_um: float,
    auto_floors: bool,
    k: float,
    group_rules: dict | None,
) -> tuple[pd.DataFrame, dict] | None:
    """Returns ``(df, meta)`` where ``meta`` carries per-image fields that
    survive even when a role has zero kept puncta (so the images-CSV can
    still record ``n_puncta=0`` rows with the correct dendrite area).
    """
    try:
        full_shape = np.load(npy_path, mmap_mode="r").shape[1:]
        dend_mask = _load_dend_mask(dendrite_dir, npy_path.stem, full_shape)
    except Exception as e:
        logger.error("  %s: dendrite mask missing/invalid (%s), skipping", npy_path.name, e)
        return None
    try:
        near = (
            dilation(dend_mask, disk(int(near_dilate_px)))
            if near_dilate_px > 0
            else dend_mask
        )
        dendrite_area_um2 = float(near.sum()) * px_size_um ** 2
        group_name = (
            classify_image_by_patterns(npy_path.name, **group_rules) or ""
            if group_rules is not None
            else ""
        )
        df = analyse_image(
            npy_path,
            dend_mask,
            pre_channel=pre_channel,
            post_channel=post_channel,
            cfg_pre=cfg_pre,
            cfg_post=cfg_post,
            near_dilate_px=near_dilate_px,
            px_size_um=px_size_um,
            auto_floors=auto_floors,
            k=k,
            group_rules=group_rules,
            dendrite_channel=dendrite_channel,
        )
        meta = dict(
            file_name=npy_path.name,
            group_name=group_name,
            dendrite_area_um2=dendrite_area_um2,
        )
        return df, meta
    except Exception as e:
        logger.error("  %s: FAILED (%s)", npy_path.name, e)
        return None


def _analyse_one_worker(args: tuple) -> tuple[pd.DataFrame, dict] | None:
    (npy_path, dendrite_dir, pre_channel, post_channel, dendrite_channel,
     cfg_pre, cfg_post, near_dilate_px, px_size_um, auto_floors, k,
     group_rules) = args
    return _analyse_one(
        Path(npy_path), Path(dendrite_dir), pre_channel, post_channel,
        dendrite_channel, cfg_pre, cfg_post, near_dilate_px, px_size_um,
        auto_floors, k, group_rules,
    )


def _stitch_mip_from_patches(patch_dir: Path, rows: list[dict]) -> np.ndarray:
    """Stitch one source image's tiled patches into one (C, H, W) float32 MIP.

    Patches are un-renormalized crops of the already-normalized full MIP
    (see ``preprocess_training.process_single_image``), so this is
    pixel-identical to re-running the raw-image pipeline -- verified by
    diffing against ``load_image`` -> ``maximum_intensity_projection`` ->
    ``normalize_percentile`` on the matching raw ``.vsi`` file.
    """
    ps = int(rows[0]["patch_size"])
    nch = int(rows[0]["channels"])
    n_rows = max(int(r["grid_row"]) for r in rows) + 1
    n_cols = max(int(r["grid_col"]) for r in rows) + 1
    full = np.zeros((nch, n_rows * ps, n_cols * ps), dtype=np.float32)
    seen = np.zeros((n_rows, n_cols), dtype=bool)
    for r in rows:
        gr, gc = int(r["grid_row"]), int(r["grid_col"])
        patch = np.load(patch_dir / r["filename"])
        full[:, gr * ps:(gr + 1) * ps, gc * ps:(gc + 1) * ps] = patch
        seen[gr, gc] = True
    if not seen.all():
        n_missing = int((~seen).sum())
        raise ValueError(f"missing {n_missing}/{n_rows * n_cols} patches")
    return full


def _analyse_one_from_patches(
    source_npy: str,
    rows: list[dict],
    patch_dir: Path,
    dendrite_dir: Path,
    pre_channel: int,
    post_channel: int,
    dendrite_channel: int,
    cfg_pre: PunctaCfg,
    cfg_post: PunctaCfg,
    near_dilate_px: int,
    px_size_um: float,
    auto_floors: bool,
    k: float,
    group_rules: dict | None,
) -> tuple[pd.DataFrame, dict] | None:
    """Patches-mode equivalent of ``_analyse_one``: stitches the tiled
    patches for one source image in memory, hands the full array to
    ``analyse_image`` via a scratch temp file (deleted immediately after),
    and never persists a full-MIP ``.npy`` on disk.
    """
    stem = Path(source_npy).stem
    try:
        full = _stitch_mip_from_patches(patch_dir, rows)
    except Exception as e:
        logger.error("  %s: FAILED to stitch patches (%s)", source_npy, e)
        return None
    try:
        dend_mask = _load_dend_mask(dendrite_dir, stem, full.shape[1:])
    except Exception as e:
        logger.error("  %s: dendrite mask missing/invalid (%s), skipping", source_npy, e)
        return None
    try:
        near = (
            dilation(dend_mask, disk(int(near_dilate_px)))
            if near_dilate_px > 0
            else dend_mask
        )
        dendrite_area_um2 = float(near.sum()) * px_size_um ** 2
        group_name = (
            classify_image_by_patterns(source_npy, **group_rules) or ""
            if group_rules is not None
            else ""
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / source_npy
            np.save(tmp_path, full)
            df = analyse_image(
                tmp_path,
                dend_mask,
                pre_channel=pre_channel,
                post_channel=post_channel,
                cfg_pre=cfg_pre,
                cfg_post=cfg_post,
                near_dilate_px=near_dilate_px,
                px_size_um=px_size_um,
                auto_floors=auto_floors,
                k=k,
                group_rules=group_rules,
                dendrite_channel=dendrite_channel,
            )
        meta = dict(
            file_name=source_npy,
            group_name=group_name,
            dendrite_area_um2=dendrite_area_um2,
        )
        return df, meta
    except Exception as e:
        logger.error("  %s: FAILED (%s)", source_npy, e)
        return None


def _analyse_one_from_patches_worker(args: tuple) -> tuple[pd.DataFrame, dict] | None:
    (source_npy, rows, patch_dir, dendrite_dir, pre_channel, post_channel,
     dendrite_channel, cfg_pre, cfg_post, near_dilate_px, px_size_um,
     auto_floors, k, group_rules) = args
    return _analyse_one_from_patches(
        source_npy, rows, Path(patch_dir), Path(dendrite_dir), pre_channel,
        post_channel, dendrite_channel, cfg_pre, cfg_post, near_dilate_px,
        px_size_um, auto_floors, k, group_rules,
    )


def _load_patch_groups(
    patch_dir: Path, time_filter: str | None
) -> dict[str, list[dict]]:
    """Group ``index.csv`` rows by ``source_npy``, optionally keeping only
    sources whose name matches ``time_filter`` (case-insensitive regex).
    """
    index_csv = patch_dir / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found in {patch_dir}")
    with open(index_csv, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    pattern = re.compile(time_filter, re.IGNORECASE) if time_filter else None
    groups: dict[str, list[dict]] = {}
    for r in all_rows:
        key = r["source_npy"]
        if pattern is not None and not pattern.search(key):
            continue
        groups.setdefault(key, []).append(r)
    return groups


def _images_summary(
    df: pd.DataFrame,
    metas: list[dict],
    pre_channel: int,
    post_channel: int,
) -> pd.DataFrame:
    """Build the per-(file_name, role) summary, including roles with 0 puncta."""
    counts: dict[tuple[str, str], int] = {}
    if not df.empty:
        for (file_name, role), sub in df.groupby(["file_name", "role"]):
            counts[(file_name, role)] = len(sub)

    rows: list[dict] = []
    for meta in metas:
        for role, channel_index in (("pre", pre_channel), ("post", post_channel)):
            rows.append(dict(
                file_name=meta["file_name"],
                group_name=meta["group_name"],
                role=role,
                channel_index=channel_index,
                n_puncta=counts.get((meta["file_name"], role), 0),
                dendrite_area_um2=meta["dendrite_area_um2"],
            ))
    out = pd.DataFrame(rows, columns=list(IMAGES_CSV_COLUMNS))
    return out.sort_values(["file_name", "role"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute one CSV row per detected punctum (pre + post), with "
            "ellipse morphology + dendrite-area density inputs, for every "
            "full-MIP .npy in --input_dir (or every source image in a tiled "
            "--input_patches session, stitched in memory)."
        )
    )
    parser.add_argument("--input_dir", type=Path, default=None,
                        help="Folder of full-MIP (C, H, W) .npy files. "
                             "Mutually exclusive with --input_patches.")
    parser.add_argument("--input_patches", type=Path, default=None,
                        help="Folder of tiled patches + index.csv (e.g. "
                             "data/patches_128_from_zip/<date>). Each source image's "
                             "patches are stitched in memory -- no full-MIP .npy is "
                             "ever written to disk. Mutually exclusive with --input_dir.")
    parser.add_argument("--time_filter", type=str, default=None,
                        help="--input_patches only: case-insensitive regex; only source "
                             "images whose name matches are processed (e.g. '24[_ ]?hod').")
    parser.add_argument("--dendrite_dir", type=Path, required=True,
                        help="Folder of dendrite masks (<stem>_dend.npy or tiled patches).")
    parser.add_argument("--output_csv", type=Path, required=True,
                        help="Output CSV path (per-punctum rows). An "
                             "'<stem>_images.csv' is also written alongside it.")
    parser.add_argument("--pre_channel", type=int, default=0,
                        help="Index of the pre-synaptic channel (default: 0).")
    parser.add_argument("--post_channel", type=int, default=1,
                        help="Index of the post-synaptic channel (default: 1).")
    parser.add_argument("--dendrite_channel", type=int, default=2,
                        help="Channel index recorded as dendrite_channel_index (default: 2).")
    parser.add_argument("--near_dilate_px", type=int, default=DEFAULT_NEAR_DILATE_PX,
                        help=f"Dilation (px) applied to the dendrite mask to form the "
                             f"acceptance area (default: {DEFAULT_NEAR_DILATE_PX}).")
    parser.add_argument("--px_size_um", type=float, default=DEFAULT_PX_SIZE_UM,
                        help=f"Pixel size in microns (default: {DEFAULT_PX_SIZE_UM}).")
    parser.add_argument("--k", type=float, default=3.0,
                        help="Ellipse-fit window half-size multiplier on sigma (default: 3.0).")
    parser.add_argument("--cfg_pre", type=str, default=None,
                        help="JSON dict or path to .json with DEFAULT_PUNCTA_CFG_PRE overrides.")
    parser.add_argument("--cfg_post", type=str, default=None,
                        help="JSON dict or path to .json with DEFAULT_PUNCTA_CFG_POST overrides.")
    parser.add_argument("--no_auto_floors", action="store_true",
                        help="Disable per-image auto-derivation of absolute-intensity floors.")
    parser.add_argument("--group_patterns", type=Path, default=None,
                        help="Path to configs/clustering/group_patterns.json for group_name "
                             "classification. Omit to leave group_name empty.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Process-pool workers (0 or 1 = sequential, default: 0).")
    args = parser.parse_args()

    if (args.input_dir is None) == (args.input_patches is None):
        logger.error("specify exactly one of --input_dir or --input_patches")
        raise SystemExit(1)
    if args.input_dir is not None and not args.input_dir.is_dir():
        logger.error("input_dir does not exist: %s", args.input_dir)
        raise SystemExit(1)
    if args.input_patches is not None and not args.input_patches.is_dir():
        logger.error("input_patches does not exist: %s", args.input_patches)
        raise SystemExit(1)
    if not args.dendrite_dir.is_dir():
        logger.error("dendrite_dir does not exist: %s", args.dendrite_dir)
        raise SystemExit(1)

    cfg_pre = resolve_cfg(_parse_json_cfg(args.cfg_pre), channel="pre")
    cfg_post = resolve_cfg(_parse_json_cfg(args.cfg_post), channel="post")
    group_rules = load_group_rules(args.group_patterns) if args.group_patterns else None

    frames: list[pd.DataFrame] = []
    metas: list[dict] = []

    if args.input_patches is not None:
        groups = _load_patch_groups(args.input_patches, args.time_filter)
        if not groups:
            logger.warning("No source images matched in %s (time_filter=%r)",
                            args.input_patches, args.time_filter)
            return
        logger.info("Matched %d source image(s) in %s", len(groups), args.input_patches)
        work_items = [
            (source_npy, rows, str(args.input_patches), str(args.dendrite_dir),
             args.pre_channel, args.post_channel, args.dendrite_channel,
             cfg_pre, cfg_post, args.near_dilate_px, args.px_size_um,
             not args.no_auto_floors, args.k, group_rules)
            for source_npy, rows in sorted(groups.items())
        ]
        if args.workers > 1 and len(work_items) > 1:
            n_workers = min(args.workers, len(work_items))
            logger.info("Running %d source images with %d workers", len(work_items), n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(_analyse_one_from_patches_worker, it): it[0] for it in work_items}
                for fut in as_completed(futures):
                    try:
                        out = fut.result()
                    except Exception as e:
                        logger.error("Worker crashed on %s: %s", futures[fut], e)
                        continue
                    if out is not None:
                        frames.append(out[0])
                        metas.append(out[1])
        else:
            for it in work_items:
                out = _analyse_one_from_patches_worker(it)
                if out is not None:
                    frames.append(out[0])
                    metas.append(out[1])
    else:
        npy_files = sorted(
            p for p in args.input_dir.glob("*.npy")
            if p.name != "index.csv" and not _is_skippable(p.stem)
        )
        if not npy_files:
            logger.warning("No .npy files found in %s", args.input_dir)
            return

        work_items = [
            (str(p), str(args.dendrite_dir), args.pre_channel, args.post_channel,
             args.dendrite_channel, cfg_pre, cfg_post, args.near_dilate_px,
             args.px_size_um, not args.no_auto_floors, args.k, group_rules)
            for p in npy_files
        ]

        if args.workers > 1 and len(work_items) > 1:
            n_workers = min(args.workers, len(work_items))
            logger.info("Running %d files with %d workers", len(npy_files), n_workers)
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(_analyse_one_worker, it): it[0] for it in work_items}
                for fut in as_completed(futures):
                    try:
                        out = fut.result()
                    except Exception as e:
                        logger.error("Worker crashed on %s: %s", futures[fut], e)
                        continue
                    if out is not None:
                        frames.append(out[0])
                        metas.append(out[1])
        else:
            for it in work_items:
                out = _analyse_one_worker(it)
                if out is not None:
                    frames.append(out[0])
                    metas.append(out[1])

    if frames:
        result = pd.concat(frames, ignore_index=True)[list(PUNCTA_CSV_COLUMNS)]
    else:
        result = pd.DataFrame(columns=list(PUNCTA_CSV_COLUMNS))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, sep=";", index=False, encoding="utf-8")
    logger.info("wrote %s (%d puncta rows)", args.output_csv, len(result))

    images_csv = args.output_csv.with_name(f"{args.output_csv.stem}_images.csv")
    images_df = _images_summary(result, metas, args.pre_channel, args.post_channel)
    images_df.to_csv(images_csv, sep=";", index=False, encoding="utf-8")
    logger.info("wrote %s (%d file x role rows)", images_csv, len(images_df))


if __name__ == "__main__":
    main()
