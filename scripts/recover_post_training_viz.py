#!/usr/bin/env python
"""Regenerate post-training visualizations from completed/partial output folders.

This script scans run folders under an output root (or explicit run dirs), loads
model/config/checkpoint from each run, and regenerates the standard post-training
artifacts:
- training_curves.png (from metrics.csv)
- post_recon_train.png
- post_recon_val.png
- optional post_recon_fullimage.{png,npz}

Usage:
    python scripts/recover_post_training_viz.py \
        --output-root /auto/brno2/home/anokhver/thesis/data/training_outputs

    python scripts/recover_post_training_viz.py \
        --run-dirs /auto/.../training_outputs/run_a /auto/.../training_outputs/run_b

    python scripts/recover_post_training_viz.py \
        --output-root /auto/.../training_outputs --run-full-image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "root"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.models.swin import build_simmim_vicreg_heads, build_swin_encoder
from synaptic_ssl.ssl_training import (
    build_run_label,
    load_config,
    plot_post_training_curves,
    post_training_reconstruction,
    run_full_image_recon,
)
from synaptic_ssl.training.augment import MicroscopyTwoViewTransform, ValSingleViewTransform
from synaptic_ssl.training.checkpoints import find_latest_checkpoint, load_checkpoint
from synaptic_ssl.training.data import compute_channel_stats
from synaptic_ssl.training.logging import setup_logger
from synaptic_ssl.utils_data.patch_dataset import PatchDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate post-training visualizations.")
    p.add_argument(
        "--output-root",
        type=str,
        default="/auto/brno2/home/anokhver/thesis/data/training_outputs",
        help="Root folder containing experiment output subfolders.",
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        default=None,
        help="Optional explicit run directories. If provided, output-root scan is skipped.",
    )
    p.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_model.pt",
        help="Preferred checkpoint filename inside each run folder.",
    )
    p.add_argument(
        "--no-fallback-latest",
        action="store_true",
        help="Do not fallback to latest *.pt when preferred checkpoint is missing.",
    )
    p.add_argument(
        "--run-full-image",
        action="store_true",
        help="Also run full-image reconstruction when full_image.enabled=true in config.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference.",
    )
    return p.parse_args()


def resolve_run_dirs(output_root: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        out = []
        for d in explicit:
            p = Path(d).resolve()
            if p.is_dir():
                out.append(p)
        return sorted(out)

    if not output_root.exists():
        return []

    return sorted(
        p for p in output_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def load_channel_stats(run_dir: Path, train_subset, in_channels: int, max_samples: int, logger):
    stats_path = run_dir / "channel_stats.json"
    if stats_path.exists():
        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            mean = torch.tensor(payload["mean"], dtype=torch.float32)
            std = torch.tensor(payload["std"], dtype=torch.float32)
            if mean.numel() == in_channels and std.numel() == in_channels:
                return mean, std
            logger.warning(
                "channel_stats.json shape mismatch; recomputing stats from train subset"
            )
        except Exception as e:
            logger.warning(f"failed reading channel_stats.json ({e}); recomputing stats")

    mean, std = compute_channel_stats(
        train_subset,
        in_channels=in_channels,
        max_samples=max_samples,
    )
    payload = {"mean": mean.tolist(), "std": std.tolist()}
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mean, std


def pick_checkpoint(run_dir: Path, preferred_name: str, allow_fallback_latest: bool) -> Path | None:
    pref = run_dir / preferred_name
    if pref.exists():
        return pref
    if allow_fallback_latest:
        return find_latest_checkpoint(run_dir, pattern="*.pt", prefer_filename="last.pt")
    return None


def process_run(run_dir: Path, device: torch.device, args: argparse.Namespace) -> bool:
    logger = setup_logger("recover_post_training_viz")
    logger.info(f"=== run: {run_dir}")

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        logger.warning("missing config.json; skipping")
        return False

    cfg = load_config(cfg_path, root_dir=_ROOT)
    base_cfg = cfg["base_cfg"]
    data_cfg = cfg["data_cfg"]
    model_cfg = cfg["model_cfg"]
    ssl_cfg = cfg["ssl_cfg"]
    full_image_cfg = cfg["full_image"]

    encoder = build_swin_encoder(model_cfg).to(device)
    heads = build_simmim_vicreg_heads(model_cfg, ssl_cfg)
    heads = {k: v.to(device) for k, v in heads.items()}

    ckpt_path = pick_checkpoint(
        run_dir,
        preferred_name=args.checkpoint_name,
        allow_fallback_latest=not args.no_fallback_latest,
    )
    if ckpt_path is None:
        logger.warning("no checkpoint found; cannot create recon plots for this run")
        # Still try curves if metrics.csv exists.
        _ = plot_post_training_curves(run_dir, run_label=None, logger=logger, show=False)
        plt.close("all")
        return False

    ckpt = load_checkpoint(ckpt_path, encoder=encoder, heads=heads, map_location=device)
    logger.info(f"loaded checkpoint: {ckpt_path.name}")

    run_label = build_run_label(
        base_cfg,
        epoch=ckpt.get("epoch"),
        val_metric=ckpt.get("val_metric"),
    )

    # Curves are independent of data availability.
    _ = plot_post_training_curves(run_dir, run_label=run_label, logger=logger, show=False)

    data_root = Path(data_cfg.data_root)
    if not data_root.exists():
        logger.warning(f"data_root missing: {data_root}; skipping recon images")
        plt.close("all")
        return True

    raw_dataset = PatchDataset(root=data_cfg.data_root, exclude_patterns=data_cfg.exclude_patterns)
    if len(raw_dataset) < 2:
        logger.warning("dataset too small; skipping recon images")
        plt.close("all")
        return True

    g = torch.Generator().manual_seed(base_cfg.seed)
    n_val = int(len(raw_dataset) * data_cfg.val_split)
    n_train = len(raw_dataset) - n_val
    if n_train <= 0:
        logger.warning("empty train split; skipping recon images")
        plt.close("all")
        return True

    train_subset, val_subset = random_split(raw_dataset, [n_train, n_val], generator=g)
    ch_mean, ch_std = load_channel_stats(
        run_dir,
        train_subset,
        in_channels=model_cfg.in_channels,
        max_samples=data_cfg.channel_stats_max_samples,
        logger=logger,
    )

    train_transform = MicroscopyTwoViewTransform(ch_mean, ch_std)
    val_transform = ValSingleViewTransform(ch_mean, ch_std)

    _ = post_training_reconstruction(
        encoder,
        heads,
        train_subset,
        train_transform,
        ssl_cfg,
        ch_mean=ch_mean,
        ch_std=ch_std,
        save_dir=run_dir,
        base_seed=base_cfg.seed,
        view="train",
        run_label=run_label,
        channel_names=data_cfg.channel_names,
        device=device,
        show=False,
    )

    if len(val_subset) > 0:
        _ = post_training_reconstruction(
            encoder,
            heads,
            val_subset,
            val_transform,
            ssl_cfg,
            ch_mean=ch_mean,
            ch_std=ch_std,
            save_dir=run_dir,
            base_seed=base_cfg.seed,
            view="val",
            run_label=run_label,
            channel_names=data_cfg.channel_names,
            device=device,
            show=False,
        )

    if args.run_full_image and bool(full_image_cfg.get("enabled", False)):
        try:
            run_full_image_recon(
                encoder,
                heads,
                data_cfg,
                model_cfg,
                ssl_cfg,
                base_cfg,
                ch_mean,
                ch_std,
                run_dir,
                run_label,
                device,
                full_image_cfg,
                logger,
            )
        except Exception as e:
            logger.warning(f"full-image reconstruction failed: {e}")

    plt.close("all")
    logger.info("generated post-training artifacts")
    return True


def main() -> int:
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    output_root = Path(args.output_root).resolve()
    runs = resolve_run_dirs(output_root, args.run_dirs)
    if not runs:
        print("No run directories found.")
        return 1

    ok = 0
    for run_dir in runs:
        try:
            if process_run(run_dir, device, args):
                ok += 1
        except Exception as e:
            print(f"[ERROR] {run_dir}: {e}")

    print(f"Done. Processed {ok}/{len(runs)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
