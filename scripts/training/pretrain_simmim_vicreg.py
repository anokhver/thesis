#!/usr/bin/env python
"""SimMIM + VICReg pretraining script (headless equivalent of the notebook).

Usage:
    python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/default.json
    python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/default.json --dry-run
    python scripts/training/pretrain_simmim_vicreg.py --config configs/pretrain_moby/default.json --resume ../data/training_outputs/my_run/last.pt

Relative paths in the config are resolved against ``root/``. An existing
run's ``config.json`` can be reused as a starting point; missing keys
fall back to defaults.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: must precede any pyplot import

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_ROOT = _REPO / "src"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.training.config import dump_config
from synaptic_ssl.training.logging import setup_logger
from synaptic_ssl.training.seeding import seed_everything
from synaptic_ssl.ssl_training import (
    load_config,
    build_model,
    setup_data,
    setup_optimizer,
    run_sanity_checks,
    run_overfit_check,
    training_loop,
    post_training_flow,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SimMIM + VICReg self-supervised pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config file.",
    )
    p.add_argument(
        "--output-root", type=str, default=None,
        help="Override output_root from config.",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=None,
        help="Skip the full training loop.",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="Override resume_path from config.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ── Load config ──
    cfg = load_config(args.config, root_dir=_ROOT)
    base_cfg = cfg["base_cfg"]
    data_cfg = cfg["data_cfg"]
    model_cfg = cfg["model_cfg"]
    train_cfg = cfg["train_cfg"]
    ssl_cfg = cfg["ssl_cfg"]
    run_sanity = cfg["run_sanity"]
    run_overfit = cfg["run_overfit"]
    unfreeze_schedule = cfg["unfreeze_schedule"]
    full_image_cfg = cfg["full_image"]

    # ── CLI overrides ──
    if args.output_root is not None:
        base_cfg.output_root = str(Path(args.output_root).resolve())
    if args.dry_run is not None:
        base_cfg.dry_run = args.dry_run
    if args.resume is not None:
        base_cfg.resume_path = str(Path(args.resume).resolve())

    # ── Resource cleanup ──
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # ── Output directory and logger ──
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(base_cfg.output_root) / f"{base_cfg.experiment_name}_{base_cfg.tag}_{run_ts}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save_dir = {save_dir}")

    logger = setup_logger("pretrain", save_dir / "run.log")
    logger.info(f"experiment = {base_cfg.experiment_name}")
    logger.info(f"tag        = {base_cfg.tag}")
    logger.info(f"method     = {base_cfg.method_name}")
    logger.info(f"init_source= {base_cfg.init_source}")
    logger.info(f"save_dir   = {save_dir}")
    logger.info(f"config     = {args.config}")

    dump_config(
        save_dir / "config.json",
        base=base_cfg, data=data_cfg, model=model_cfg, train=train_cfg, ssl=ssl_cfg,
    )
    logger.info("config.json written")

    # ── Seed and device ──
    generator = seed_everything(base_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"seed   = {base_cfg.seed}")
    logger.info(f"device = {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"gpu    = {name}  ({mem:.1f} GB)")

    # ── Data ──
    data = setup_data(
        data_cfg, model_cfg, ssl_cfg,
        seed_generator=generator, logger=logger,
    )
    stats = {
        "channel_names": list(data_cfg.channel_names),
        "mean": data.ch_mean.tolist(),
        "std": data.ch_std.tolist(),
    }
    (save_dir / "channel_stats.json").write_text(json.dumps(stats, indent=2))

    # ── Sanity / overfit checks (on a disposable model) ──
    if run_sanity or run_overfit:
        sanity_model = build_model(base_cfg, model_cfg, ssl_cfg, device, logger=logger)
        if run_sanity:
            run_sanity_checks(
                sanity_model, data, ssl_cfg, model_cfg, data_cfg, train_cfg,
                device, save_dir, logger=logger,
            )
        if run_overfit:
            run_overfit_check(
                sanity_model, data, base_cfg, ssl_cfg, train_cfg, data_cfg,
                device, save_dir, logger=logger,
            )
        del sanity_model
    elif not run_sanity:
        logger.info("sanity/overfit checks skipped")

    # ══════════════════════════════════════════════════════════════════════
    # Full training
    # ══════════════════════════════════════════════════════════════════════
    generator = seed_everything(base_cfg.seed)

    model = build_model(base_cfg, model_cfg, ssl_cfg, device, logger=logger)
    logger.info(f"[reset] encoder params = {model.load_summary['n_target_params'] / 1e6:.2f} M")

    optim = setup_optimizer(
        model, train_cfg, device,
        resume_path=base_cfg.resume_path, logger=logger,
    )

    if base_cfg.dry_run:
        logger.info("dry_run=True -- skipping full training loop")
    else:
        training_loop(
            model, optim, data,
            base_cfg, data_cfg, ssl_cfg, train_cfg, unfreeze_schedule,
            device, save_dir, logger=logger,
        )

    # ── Post-training ──
    if not base_cfg.dry_run:
        post_training_flow(
            model, data,
            base_cfg, data_cfg, model_cfg, ssl_cfg, train_cfg, full_image_cfg,
            device, save_dir, logger=logger,
        )

    logger.info("done.")


if __name__ == "__main__":
    main()
