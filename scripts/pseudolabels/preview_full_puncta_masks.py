#!/usr/bin/env python3
"""Render a full-field PRE/POST puncta-mask diagnostic from tiled MIPs.

The output has two rows of three panels:

* PRE + PRE mask, POST + POST mask, and PRE + POST + overlap masks on signal.
* PRE mask, POST mask, and overlap mask alone on black.

Example::

    python scripts/pseudolabels/preview_full_puncta_masks.py \
        --session 20251017 \
        --source "2.1.hrmin1_Multichannel Z-Stack_20251017_01.npy"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Session/date folder, e.g. 20251017.")
    parser.add_argument(
        "--source",
        required=True,
        help="Full-MIP filename or stem recorded in the patch index.",
    )
    parser.add_argument(
        "--patch_root",
        type=Path,
        default=REPO_ROOT / "data/patches_128_from_zip",
        help="Root containing <session>/index.csv and patch files.",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=REPO_ROOT / "data/pseudolabels/puncta",
        help="Root containing <session>/{pre,post,overlap} masks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: data/imaging_outputs/puncta_full_previews/).",
    )
    parser.add_argument(
        "--mask_dilate_px",
        type=int,
        default=2,
        help="Extra radius used only to make full-field masks visible (default: 2).",
    )
    return parser.parse_args()


def source_stem(source: str) -> str:
    return Path(source).stem


def read_patch_rows(index_path: Path, source: str) -> list[dict[str, str]]:
    expected_name = Path(source).name
    expected_stem = source_stem(source)
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if Path(row["source_npy"]).name == expected_name
            or Path(row["source_npy"]).stem == expected_stem
        ]
    if not rows:
        raise FileNotFoundError(f"No tiles for {source!r} in {index_path}")
    return rows


def stitch_mip(patch_dir: Path, rows: list[dict[str, str]]) -> np.ndarray:
    first_patch = np.load(patch_dir / rows[0]["filename"])
    if first_patch.ndim != 3:
        raise ValueError(f"Expected (C, H, W) patch, got {first_patch.shape}")
    channels, patch_height, patch_width = first_patch.shape
    max_row = max(int(row["grid_row"]) for row in rows)
    max_col = max(int(row["grid_col"]) for row in rows)
    mip = np.zeros(
        (channels, (max_row + 1) * patch_height, (max_col + 1) * patch_width),
        dtype=first_patch.dtype,
    )
    for row in rows:
        grid_row, grid_col = int(row["grid_row"]), int(row["grid_col"])
        patch = np.load(patch_dir / row["filename"])
        y0, x0 = grid_row * patch_height, grid_col * patch_width
        mip[:, y0:y0 + patch_height, x0:x0 + patch_width] = patch
    return mip


def display_signal(image: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    low, high = np.percentile(image, (1.0, 99.8))
    normalized = np.clip((image - low) / max(high - low, 1e-8), 0.0, 1.0)
    return normalized[..., None] * np.asarray(color)


def visible_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    size = 2 * radius + 1
    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0


def add_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float],
) -> np.ndarray:
    result = rgb.copy()
    result[mask] = color
    return result


def combined_masks(
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    overlap_mask: np.ndarray,
) -> np.ndarray:
    result = np.zeros((*pre_mask.shape, 3), dtype=np.float32)
    result[pre_mask] = (0.0, 1.0, 0.0)
    result[post_mask] = (1.0, 0.0, 1.0)
    result[overlap_mask] = (1.0, 1.0, 0.0)
    return result


def panel_image(panel: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    image = Image.fromarray((np.clip(panel, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    title_height = 42
    result = Image.new("RGB", (image.width, image.height + title_height), "black")
    result.paste(image, (0, title_height))
    ImageDraw.Draw(result).text((12, 8), title, fill="white", font=font)
    return result


def main() -> None:
    args = parse_args()
    patch_dir = args.patch_root / args.session
    mask_dir = args.mask_root / args.session
    rows = read_patch_rows(patch_dir / "index.csv", args.source)
    mip = stitch_mip(patch_dir, rows)
    if mip.shape[0] < 2:
        raise ValueError(f"{args.source} has {mip.shape[0]} channels; PRE and POST require two.")

    stem = source_stem(rows[0]["source_npy"])
    pre_mask = np.load(mask_dir / f"{stem}_pre.npy").astype(bool)
    post_mask = np.load(mask_dir / f"{stem}_post.npy").astype(bool)
    overlap_mask = np.load(mask_dir / f"{stem}_overlap.npy").astype(bool)
    if pre_mask.shape != mip.shape[1:]:
        raise ValueError(f"Mask shape {pre_mask.shape} does not match MIP shape {mip.shape[1:]}")

    pre_visible = visible_mask(pre_mask, args.mask_dilate_px)
    post_visible = visible_mask(post_mask, args.mask_dilate_px)
    overlap_visible = visible_mask(overlap_mask, args.mask_dilate_px)
    pre_signal = display_signal(mip[0], (0.0, 1.0, 0.0))
    post_signal = display_signal(mip[1], (1.0, 0.0, 1.0))
    paired_signal = np.maximum(pre_signal, post_signal)

    panels = (
        (add_mask_overlay(pre_signal, pre_visible, (1.0, 1.0, 1.0)), "PRE signal + PRE mask"),
        (add_mask_overlay(post_signal, post_visible, (1.0, 1.0, 1.0)), "POST signal + POST mask"),
        (add_mask_overlay(add_mask_overlay(add_mask_overlay(paired_signal, pre_visible, (0.0, 1.0, 0.0)), post_visible, (1.0, 0.0, 1.0)), overlap_visible, (1.0, 1.0, 0.0)), "PRE + POST + overlap masks"),
        (combined_masks(pre_visible, np.zeros_like(post_visible), np.zeros_like(overlap_visible)), "PRE mask"),
        (combined_masks(np.zeros_like(pre_visible), post_visible, np.zeros_like(overlap_visible)), "POST mask"),
        (combined_masks(np.zeros_like(pre_visible), np.zeros_like(post_visible), overlap_visible), "Overlap mask"),
    )

    output = args.output
    if output is None:
        output = REPO_ROOT / "data/imaging_outputs/puncta_full_previews" / f"{stem}_two_row_masks.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rendered_panels = [panel_image(panel, title, font) for panel, title in panels]
    panel_width, panel_height = rendered_panels[0].size
    figure = Image.new("RGB", (panel_width * 3, panel_height * 2), "black")
    for index, panel in enumerate(rendered_panels):
        figure.paste(panel, ((index % 3) * panel_width, (index // 3) * panel_height))
    figure.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)