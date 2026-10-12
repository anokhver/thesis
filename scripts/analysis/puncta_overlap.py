#!/usr/bin/env python3
r"""PRE/POST puncta overlap from the per-punctum morphology CSV.

Reads the CSV written by ``scripts/analysis/puncta_morphology.py`` (one row
per punctum: ``file_name``, ``role``, ``x``, ``y``, ``radius_um``, ...) and,
for every image, pairs PRE with POST puncta using ONE overlap method chosen
with ``--method``. No images are loaded; only the coordinates in the CSV.

Writes two files into ``--output_dir`` (``<name>`` = ``--name`` or the
method name):

  - ``<name>_pairs.csv``  one row per PRE-POST pair:
    ``file_name;group_name;method;pre_row;post_row;pre_x;pre_y;post_x;post_y;
    distance_nm`` + any extra columns the method returns.
    ``pre_row`` / ``post_row`` are 0-based row numbers in the input CSV, so a
    pair can be joined back to the full morphology of both puncta (and the
    unpaired puncta are the input rows not listed here).
  - ``<name>_images.csv`` one row per image:
    ``file_name;group_name;method;n_pre;n_post;n_pairs;frac_pre_paired;
    frac_post_paired;dendrite_area_um2;pairs_per_um2``.
    Counts come from the input's ``_images.csv``, so images with 0 puncta
    in a role are still listed.

Run several methods (or one method with different parameters) on the same
CSV by calling the script again with another ``--method`` / ``--params`` /
``--name``; every run writes its own files.

Usage::

    python scripts/analysis/puncta_overlap.py \
        --input_csv outs/puncta_morphology.csv \
        --output_dir outs/puncta_overlap \
        --method distance --params '{"max_dist_nm": 250}'

    python scripts/analysis/puncta_overlap.py \
        --input_csv outs/puncta_morphology.csv \
        --output_dir outs/puncta_overlap \
        --method circles --params '{"scale": 1.0}' --name circles_1x

    python scripts/analysis/puncta_overlap.py --list_methods

Adding a method
---------------
Write a function decorated with ``@overlap_method("my_name")`` below. It
gets the PRE and POST puncta of ONE image as DataFrames (all CSV columns,
plus ``x_nm`` / ``y_nm``; the index is the row number in the input CSV)
and its keyword parameters from ``--params``. It returns a DataFrame with
at least the columns ``pre_row`` and ``post_row`` (one row per pair); any
extra columns are copied into ``<name>_pairs.csv``. Use
``one_to_one_closest`` if each punctum may be paired at most once.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PAIRS_BASE_COLUMNS: tuple[str, ...] = (
    "file_name", "group_name", "method", "pre_row", "post_row",
    "pre_x", "pre_y", "post_x", "post_y", "distance_nm",
)
IMAGES_COLUMNS: tuple[str, ...] = (
    "file_name", "group_name", "method", "n_pre", "n_post", "n_pairs",
    "frac_pre_paired", "frac_post_paired", "dendrite_area_um2", "pairs_per_um2",
)

OverlapFn = Callable[..., pd.DataFrame]
METHODS: dict[str, OverlapFn] = {}


def overlap_method(name: str) -> Callable[[OverlapFn], OverlapFn]:
    """Register ``fn`` under ``name`` so ``--method name`` can select it."""
    def register(fn: OverlapFn) -> OverlapFn:
        if name in METHODS:
            raise ValueError(f"overlap method {name!r} registered twice")
        METHODS[name] = fn
        return fn
    return register


# ---------------------------------------------------------------------------
# Shared helpers for methods
# ---------------------------------------------------------------------------

def candidate_pairs(
    pre: pd.DataFrame, post: pd.DataFrame, max_dist_nm: float
) -> pd.DataFrame:
    """All PRE-POST pairs whose centres are at most ``max_dist_nm`` apart.

    Returns ``pre_row``, ``post_row``, ``distance_nm`` (not one-to-one).
    """
    empty = pd.DataFrame({"pre_row": [], "post_row": [], "distance_nm": []})
    if pre.empty or post.empty or max_dist_nm <= 0:
        return empty
    pre_xy = pre[["x_nm", "y_nm"]].to_numpy()
    post_xy = post[["x_nm", "y_nm"]].to_numpy()
    sparse = cKDTree(pre_xy).sparse_distance_matrix(
        cKDTree(post_xy), max_dist_nm, output_type="coo_matrix"
    )
    if sparse.nnz == 0:
        return empty
    return pd.DataFrame({
        "pre_row": pre.index.to_numpy()[sparse.row],
        "post_row": post.index.to_numpy()[sparse.col],
        "distance_nm": sparse.data,
    })


def one_to_one_closest(pairs: pd.DataFrame, sort_by: str = "distance_nm") -> pd.DataFrame:
    """Keep each PRE and each POST in at most one pair, closest pairs first
    (globally sorted by ``sort_by``, ascending; same rule as
    ``pseudolabels.puncta_common.mask_overlap``)."""
    if pairs.empty:
        return pairs
    pairs = pairs.sort_values(sort_by, kind="stable")
    keep, used_pre, used_post = [], set(), set()
    for i, pre_row, post_row in zip(pairs.index, pairs["pre_row"], pairs["post_row"]):
        if pre_row in used_pre or post_row in used_post:
            continue
        used_pre.add(pre_row)
        used_post.add(post_row)
        keep.append(i)
    return pairs.loc[keep]


# ---------------------------------------------------------------------------
# Overlap methods: add new ones here
# ---------------------------------------------------------------------------

@overlap_method("distance")
def distance_method(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    *,
    max_dist_nm: float = 250.0,
    one_to_one: bool = True,
) -> pd.DataFrame:
    """Pair if the centres are at most ``max_dist_nm`` apart (default
    250 nm, about the optical resolution). Size of the puncta is ignored, so
    big puncta do not get a bigger chance of pairing."""
    pairs = candidate_pairs(pre, post, max_dist_nm)
    return one_to_one_closest(pairs) if one_to_one else pairs


@overlap_method("circles")
def circles_method(
    pre: pd.DataFrame,
    post: pd.DataFrame,
    *,
    scale: float = 1.0,
    one_to_one: bool = True,
) -> pd.DataFrame:
    """Pair if the two fitted circles overlap: centre distance at most
    ``scale * (radius_pre + radius_post)`` (``radius_um`` column). Extra
    column ``overlap_nm`` = how deep the circles overlap (> 0)."""
    if pre.empty or post.empty:
        return candidate_pairs(pre, post, 0.0)
    r_pre = pre["radius_um"] * 1000.0
    r_post = post["radius_um"] * 1000.0
    reach = scale * (r_pre.max() + r_post.max())
    pairs = candidate_pairs(pre, post, reach)
    if pairs.empty:
        return pairs
    limit = scale * (r_pre.loc[pairs["pre_row"]].to_numpy() + r_post.loc[pairs["post_row"]].to_numpy())
    pairs["overlap_nm"] = limit - pairs["distance_nm"].to_numpy()
    pairs = pairs[pairs["overlap_nm"] >= 0]
    return one_to_one_closest(pairs) if one_to_one else pairs


# ---------------------------------------------------------------------------
# Infrastructure (no need to touch when adding a method)
# ---------------------------------------------------------------------------

def run_method(
    puncta: pd.DataFrame,
    images: pd.DataFrame,
    method: str,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``method`` on every image; return ``(pairs_df, images_df)``."""
    fn = METHODS[method]
    puncta = puncta.copy()
    puncta["x_nm"] = puncta["x"] * puncta["px_size_um"] * 1000.0
    puncta["y_nm"] = puncta["y"] * puncta["px_size_um"] * 1000.0
    by_image = dict(tuple(puncta.groupby("file_name", sort=False)))

    per_image = images.pivot_table(
        index="file_name", columns="role", values="n_puncta", aggfunc="first"
    ).reindex(columns=["pre", "post"]).fillna(0).astype(int)
    meta = images.groupby("file_name")[["group_name", "dendrite_area_um2"]].first()

    pair_frames, image_rows = [], []
    for file_name in sorted(meta.index):
        rows = by_image.get(file_name, puncta.iloc[0:0])
        pre = rows[rows["role"] == "pre"]
        post = rows[rows["role"] == "post"]
        pairs = fn(pre, post, **params)
        missing = {"pre_row", "post_row"} - set(pairs.columns)
        if missing:
            raise ValueError(f"method {method!r} returned no column(s) {sorted(missing)}")
        if pairs["pre_row"].duplicated().any() or pairs["post_row"].duplicated().any():
            logger.debug("%s: %s pairs are not one-to-one", file_name, method)

        n_pairs = len(pairs)
        if n_pairs:
            pairs = pairs.reset_index(drop=True)
            pre_pts = puncta.loc[pairs["pre_row"], ["x", "y", "x_nm", "y_nm"]].to_numpy()
            post_pts = puncta.loc[pairs["post_row"], ["x", "y", "x_nm", "y_nm"]].to_numpy()
            pairs["pre_x"], pairs["pre_y"] = pre_pts[:, 0], pre_pts[:, 1]
            pairs["post_x"], pairs["post_y"] = post_pts[:, 0], post_pts[:, 1]
            if "distance_nm" not in pairs.columns:
                pairs["distance_nm"] = np.hypot(*(pre_pts[:, 2:] - post_pts[:, 2:]).T)
            pairs["file_name"] = file_name
            pairs["group_name"] = meta.at[file_name, "group_name"]
            pairs["method"] = method
            pair_frames.append(pairs)

        n_pre = int(per_image.at[file_name, "pre"]) if file_name in per_image.index else 0
        n_post = int(per_image.at[file_name, "post"]) if file_name in per_image.index else 0
        area = float(meta.at[file_name, "dendrite_area_um2"])
        n_pre_paired = int(pairs["pre_row"].nunique()) if n_pairs else 0
        n_post_paired = int(pairs["post_row"].nunique()) if n_pairs else 0
        image_rows.append(dict(
            file_name=file_name,
            group_name=meta.at[file_name, "group_name"],
            method=method,
            n_pre=n_pre,
            n_post=n_post,
            n_pairs=n_pairs,
            frac_pre_paired=n_pre_paired / n_pre if n_pre else np.nan,
            frac_post_paired=n_post_paired / n_post if n_post else np.nan,
            dendrite_area_um2=area,
            pairs_per_um2=n_pairs / area if area > 0 else np.nan,
        ))

    if pair_frames:
        pairs_df = pd.concat(pair_frames, ignore_index=True)
        extra = [c for c in pairs_df.columns if c not in PAIRS_BASE_COLUMNS]
        pairs_df = pairs_df[list(PAIRS_BASE_COLUMNS) + extra]
    else:
        pairs_df = pd.DataFrame(columns=list(PAIRS_BASE_COLUMNS))
    return pairs_df, pd.DataFrame(image_rows, columns=list(IMAGES_COLUMNS))


def _method_help() -> str:
    lines = []
    for name, fn in METHODS.items():
        params = ", ".join(
            f"{p.name}={p.default!r}"
            for p in inspect.signature(fn).parameters.values()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        )
        doc = " ".join((inspect.getdoc(fn) or "").split())
        lines.append(f"  {name}({params})\n      {doc}")
    return "\n".join(lines)


def _parse_params(s: str | None) -> dict:
    if not s:
        return {}
    p = Path(s)
    return json.loads(p.read_text(encoding="utf-8") if p.is_file() else s)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair PRE and POST puncta per image from the morphology CSV.",
        epilog="Available methods:\n" + _method_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_csv", type=Path,
                        help="Per-punctum CSV from scripts/analysis/puncta_morphology.py.")
    parser.add_argument("--images_csv", type=Path, default=None,
                        help="Per-image CSV (default: <input_csv stem>_images.csv next to it).")
    parser.add_argument("--output_dir", type=Path, default=Path("outs/puncta_overlap"),
                        help="Folder for <name>_pairs.csv and <name>_images.csv.")
    parser.add_argument("--method", choices=sorted(METHODS), default="distance",
                        help="Overlap method (default: distance).")
    parser.add_argument("--params", type=str, default=None,
                        help="JSON dict (or path to .json) with the method's keyword parameters.")
    parser.add_argument("--name", type=str, default=None,
                        help="Output file stem (default: the method name). Use it to keep "
                             "runs with different parameters apart, e.g. distance_300nm.")
    parser.add_argument("--list_methods", action="store_true",
                        help="Print the methods and their parameters, then exit.")
    args = parser.parse_args()

    if args.list_methods:
        print(_method_help())
        return
    if args.input_csv is None:
        parser.error("--input_csv is required")

    params = _parse_params(args.params)
    accepted = {
        p.name for p in inspect.signature(METHODS[args.method]).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    unknown = set(params) - accepted
    if unknown:
        parser.error(f"unknown parameter(s) for {args.method!r}: {sorted(unknown)}; "
                     f"accepted: {sorted(accepted)}")

    images_csv = args.images_csv or args.input_csv.with_name(f"{args.input_csv.stem}_images.csv")
    puncta = pd.read_csv(args.input_csv, sep=";", keep_default_na=False, na_values=[""])
    images = pd.read_csv(images_csv, sep=";", keep_default_na=False, na_values=[""])
    puncta["group_name"] = puncta["group_name"].fillna("")
    images["group_name"] = images["group_name"].fillna("")
    logger.info("read %d puncta, %d images from %s", len(puncta),
                images["file_name"].nunique(), args.input_csv)

    pairs_df, images_df = run_method(puncta, images, args.method, params)

    name = args.name or args.method
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / f"{name}_pairs.csv"
    images_path = args.output_dir / f"{name}_images.csv"
    pairs_df.to_csv(pairs_path, sep=";", index=False, encoding="utf-8")
    images_df.to_csv(images_path, sep=";", index=False, encoding="utf-8")
    (args.output_dir / f"{name}_params.json").write_text(
        json.dumps({"method": args.method, "params": params, "input_csv": str(args.input_csv)},
                   indent=2) + "\n", encoding="utf-8")
    logger.info("method=%s params=%s: %d pairs in %d images", args.method, params,
                len(pairs_df), len(images_df))
    logger.info("wrote %s and %s", pairs_path, images_path)


if __name__ == "__main__":
    main()
