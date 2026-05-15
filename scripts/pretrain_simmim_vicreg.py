#!/usr/bin/env python
"""SimMIM + VICReg pretraining script (headless equivalent of the notebook).

Usage:
    python scripts/pretrain_simmim_vicreg.py --config configs/pretrain_moby.json
    python scripts/pretrain_simmim_vicreg.py --config configs/pretrain_moby.json --dry-run
    python scripts/pretrain_simmim_vicreg.py --config configs/pretrain_moby.json --resume outputs/my_run/last.pt

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
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split

_REPO = Path(__file__).resolve().parents[1]
_ROOT = _REPO / "root"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.training.config import BaseCfg, DataCfg, ModelCfg, TrainCfg, SSLCfg, dump_config
from synaptic_ssl.training.seeding import seed_everything
from synaptic_ssl.training.logging import setup_logger, CSVMetricLogger
from synaptic_ssl.training.data import TransformedSubset, compute_channel_stats
from synaptic_ssl.training.augment import MicroscopyTwoViewTransform, ValSingleViewTransform
from synaptic_ssl.training.masking import random_block_mask
from synaptic_ssl.training.losses import compute_simmim_vicreg_loss
from synaptic_ssl.training.lr_schedule import param_groups_layer_decay, make_warmup_cosine
from synaptic_ssl.training.checkpoints import save_checkpoint, load_checkpoint, find_latest_checkpoint
from synaptic_ssl.training.sanity_batch import SANITY_TRAIN_INDICES, fixed_two_view_batch, overfit_on_batch
from synaptic_ssl.training.viz import (
    plot_two_views, plot_channel_histograms, plot_recon_panel,
    plot_overfit_curves,
)
from synaptic_ssl.models.swin import build_swin_encoder, build_simmim_vicreg_heads, count_params
from synaptic_ssl.models.weight_loading import load_pretrained_into_encoder
from synaptic_ssl.ssl_training import (
    apply_unfreeze_schedule,
    build_run_label,
    eval_recon_batch,
    heads_iter_lrs,
    load_config,
    make_phase_tag,
    plot_post_training_curves,
    post_training_reconstruction,
    reload_best_checkpoint,
    run_full_image_recon,
    train_one_epoch,
    validate_one_epoch,
)
from synaptic_ssl.utils_data.patch_dataset import PatchDataset


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
    base_cfg:  BaseCfg  = cfg["base_cfg"]
    data_cfg:  DataCfg  = cfg["data_cfg"]
    model_cfg: ModelCfg = cfg["model_cfg"]
    train_cfg: TrainCfg = cfg["train_cfg"]
    ssl_cfg:   SSLCfg   = cfg["ssl_cfg"]
    run_sanity:  bool   = cfg["run_sanity"]
    run_overfit: bool   = cfg["run_overfit"]
    unfreeze_schedule   = cfg["unfreeze_schedule"]
    full_image_cfg      = cfg["full_image"]

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
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"seed   = {base_cfg.seed}")
    logger.info(f"device = {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"gpu    = {name}  ({mem:.1f} GB)")

    # ── Data ──
    raw_dataset = PatchDataset(
        root=data_cfg.data_root,
        exclude_patterns=data_cfg.exclude_patterns,
    )
    logger.info(f"raw patches = {len(raw_dataset)}")
    sample = raw_dataset[0]
    logger.info(f"sample shape = {tuple(sample.shape)}  dtype = {sample.dtype}")
    assert sample.ndim == 3 and sample.shape[0] == model_cfg.in_channels
    assert sample.shape[-1] == model_cfg.img_size

    n_val   = int(len(raw_dataset) * data_cfg.val_split)
    n_train = len(raw_dataset) - n_val
    train_subset, val_subset = random_split(
        raw_dataset, [n_train, n_val], generator=generator,
    )
    logger.info(f"split: train={n_train}  val={n_val}")

    ch_mean, ch_std = compute_channel_stats(
        train_subset, in_channels=model_cfg.in_channels,
        max_samples=data_cfg.channel_stats_max_samples,
    )
    for ch_name, m, s in zip(data_cfg.channel_names, ch_mean.tolist(), ch_std.tolist()):
        logger.info(f"  ch[{ch_name:>14s}] mean={m:.5f}  std={s:.5f}")
    stats = {
        "channel_names": list(data_cfg.channel_names),
        "mean": ch_mean.tolist(),
        "std": ch_std.tolist(),
    }
    (save_dir / "channel_stats.json").write_text(json.dumps(stats, indent=2))

    # Augmentation pipelines
    train_transform = MicroscopyTwoViewTransform(ch_mean, ch_std)
    val_transform   = ValSingleViewTransform(ch_mean, ch_std)

    # DataLoaders
    train_ds = TransformedSubset(train_subset, train_transform)
    val_ds   = TransformedSubset(val_subset,   val_transform)
    common = dict(
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.num_workers > 0,
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size, shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.batch_size, shuffle=False, **common,
    )
    logger.info(f"train batches = {len(train_loader)}  val batches = {len(val_loader)}")

    # ── Sanity / overfit checks ──
    if run_sanity or run_overfit:
        encoder = build_swin_encoder(model_cfg).to(device)
        logger.info(f"encoder params = {count_params(encoder) / 1e6:.2f} M")

        load_summary = load_pretrained_into_encoder(
            encoder, base_cfg, model_cfg, logger=logger,
        )
        n_loaded = load_summary["n_loaded"]
        n_target = load_summary["n_target_params"]
        pct = 100.0 * n_loaded / max(n_target, 1)
        logger.info(f"coverage: {n_loaded}/{n_target} params loaded ({pct:.1f}%)")
        if load_summary.get("n_shape_mismatch", 0) > 0:
            logger.warning(f"  shape mismatches: {load_summary['shape_mismatch']}")
        # The 80% guard only makes sense when we actually expect pretrained
        # weights. ``init_source="scratch"`` legitimately loads zero params.
        if base_cfg.init_source != "scratch" and n_loaded < 0.8 * n_target:
            raise RuntimeError(
                f"Only {pct:.1f}% of encoder loaded — expected >=80%. "
                f"Check the checkpoint is correct for init_source={base_cfg.init_source!r}."
            )

        random_keys = load_summary.get("random_init", [])
        # For scratch every encoder param is "random"; logging the full list
        # would just dump the entire state-dict.
        if random_keys and base_cfg.init_source != "scratch":
            logger.info(f"  random-init keys ({len(random_keys)}):")
            for k in random_keys:
                logger.info(f"    - {k}")
        elif random_keys:
            logger.info(f"  random-init keys: ALL ({len(random_keys)})")

        heads = build_simmim_vicreg_heads(model_cfg, ssl_cfg)
        heads = {k: v.to(device) for k, v in heads.items()}
        logger.info(f"head params = {count_params(heads) / 1e6:.2f} M")

    if run_sanity:
        # Dataloader sanity
        batch = next(iter(train_loader))
        assert isinstance(batch, (list, tuple)) and len(batch) == 2, batch
        v1, v2 = batch
        assert v1.shape == v2.shape
        assert v1.shape[1:] == (model_cfg.in_channels, model_cfg.img_size, model_cfg.img_size)
        assert v1.dtype == torch.float32
        assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
        diff = (v1 - v2).abs().mean().item()
        assert diff > 1e-3, f"view1 == view2  (mean abs diff = {diff:.2e})"
        logger.info(f"[ok] views shape={tuple(v1.shape)}  v1-v2 diff={diff:.4f}")

        # Two-view visualisation
        _ = plot_two_views(
            raw_subset=train_subset,
            transform=train_transform,
            indices=SANITY_TRAIN_INDICES,
            channel_names=data_cfg.channel_names,
            suptitle="Sanity batch -- two-view augmentation",
            save_to=save_dir / "sanity_two_views.png",
        )
        plt.close("all")

        # Channel histograms
        _ = plot_channel_histograms(
            v1, channel_names=data_cfg.channel_names,
            save_to=save_dir / "sanity_histograms.png",
        )
        plt.close("all")

        # Encoder forward + per-stage shapes
        encoder.eval()
        with torch.no_grad():
            feats = encoder(v1[:2].to(device).contiguous())
        assert isinstance(feats, list) and len(feats) == len(model_cfg.depths) + 1
        n_stages = len(feats)
        head_idx = (
            ssl_cfg.head_stage_index
            if ssl_cfg.head_stage_index >= 0
            else n_stages + ssl_cfg.head_stage_index
        )
        for i, f in enumerate(feats):
            tag = "  <- heads" if i == head_idx else ""
            logger.info(f"  stage {i}: {tuple(f.shape)}{tag}")
        encoder.train()

        # Gradient flow check
        encoder.train()
        for h in heads.values():
            h.train()
        _opt = torch.optim.AdamW(
            list(encoder.parameters())
            + [p for h in heads.values() for p in h.parameters()],
            lr=1e-4,
        )
        _v1 = v1.to(device)
        _v2 = v2.to(device)
        _opt.zero_grad(set_to_none=True)
        _loss, _m = compute_simmim_vicreg_loss(encoder, heads, _v1, _v2, ssl_cfg)
        _loss.backward()
        n_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
        n_tot  = sum(1 for _ in encoder.parameters())
        _opt.step()
        del _opt, _loss, _m, _v1, _v2
        logger.info(f"[ok] grad flow: {n_grad}/{n_tot} encoder params got gradients")

    # ── Overfit check ──
    overfit_result = None
    if run_overfit:
        n_overfit_steps = 1000
        overfit_lr      = 3e-4

        encoder_groups = param_groups_layer_decay(
            encoder,
            base_lr=train_cfg.base_lr,
            weight_decay=train_cfg.weight_decay,
            layer_decay=train_cfg.layer_decay,
        )
        head_params = [p for h in heads.values() for p in h.parameters()]
        head_group  = {
            "params": head_params,
            "lr": train_cfg.head_lr,
            "weight_decay": train_cfg.weight_decay,
        }
        optimizer = torch.optim.AdamW(encoder_groups + [head_group], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, make_warmup_cosine(train_cfg.warmup_epochs, train_cfg.epochs),
        )
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

        x1_fixed, x2_fixed, sanity_idx = fixed_two_view_batch(
            train_subset, SANITY_TRAIN_INDICES, train_transform,
            seed=base_cfg.seed + 17,
        )
        x1_fixed = x1_fixed.to(device)
        x2_fixed = x2_fixed.to(device)
        logger.info(f"sanity batch indices = {sanity_idx}  shape = {tuple(x1_fixed.shape)}")

        torch.manual_seed(base_cfg.seed + 18)
        mask_fixed = random_block_mask(
            x1_fixed, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio,
        ).clone()
        logger.info(f"fixed mask coverage = {mask_fixed.mean().item():.3f}")

        overfit_result = overfit_on_batch(
            encoder, heads, x1_fixed, x2_fixed, ssl_cfg,
            n_steps=n_overfit_steps, lr=overfit_lr,
            grad_clip=train_cfg.grad_clip_norm, fixed_mask=mask_fixed, logger=logger,
        )

        # Overfit plots
        _ = plot_overfit_curves(
            overfit_result["history"],
            suptitle=f"Sanity overfit on indices {sanity_idx}",
            save_to=save_dir / "sanity_curves_overfit.png",
            run_label=build_run_label(base_cfg, extra="overfit"),
        )
        plt.close("all")

        rec, v_masked = eval_recon_batch(encoder, heads, x1_fixed, mask_fixed)
        _ = plot_recon_panel(
            x1_fixed, v_masked, rec, mask_fixed, sanity_idx,
            ch_mean=ch_mean, ch_std=ch_std,
            suptitle="Sanity overfit -- reconstruction on canonical batch",
            save_to=save_dir / "sanity_recon_overfit.png",
            run_label=build_run_label(base_cfg, extra="overfit"),
            channel_names=data_cfg.channel_names,
        )
        plt.close("all")
        encoder.train()
        for h in heads.values():
            h.train()
    elif not run_sanity:
        logger.info("sanity/overfit checks skipped")

    # ══════════════════════════════════════════════════════════════════════
    # Full training
    # ══════════════════════════════════════════════════════════════════════

    # ── Reset for full training ──
    generator = seed_everything(base_cfg.seed)

    encoder = build_swin_encoder(model_cfg).to(device)
    load_summary = load_pretrained_into_encoder(
        encoder, base_cfg, model_cfg, logger=logger,
    )
    logger.info(f"[reset] encoder params = {count_params(encoder) / 1e6:.2f} M")

    heads = build_simmim_vicreg_heads(model_cfg, ssl_cfg)
    heads = {k: v.to(device) for k, v in heads.items()}
    logger.info(f"[reset] head params    = {count_params(heads) / 1e6:.2f} M")

    encoder_groups = param_groups_layer_decay(
        encoder,
        base_lr=train_cfg.base_lr,
        weight_decay=train_cfg.weight_decay,
        layer_decay=train_cfg.layer_decay,
    )
    head_params = [p for h in heads.values() for p in h.parameters()]
    head_group  = {
        "params": head_params,
        "lr": train_cfg.head_lr,
        "weight_decay": train_cfg.weight_decay,
    }
    optimizer = torch.optim.AdamW(encoder_groups + [head_group], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_warmup_cosine(train_cfg.warmup_epochs, train_cfg.epochs),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    # ── Resume ──
    start_epoch     = 1
    best_val_metric = float("inf") if train_cfg.val_metric_direction == "min" else float("-inf")
    best_epoch      = 0

    _resume = base_cfg.resume_path
    if _resume is not None:
        p = Path(_resume)
        if p.is_dir():
            p = find_latest_checkpoint(p)
        if p is None or not Path(p).exists():
            logger.warning(f"resume path {_resume!r} not found -- starting fresh")
        else:
            ckpt = load_checkpoint(
                p, encoder=encoder, heads=heads,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                map_location=device,
            )
            start_epoch     = int(ckpt.get("epoch", 0)) + 1
            best_val_metric = ckpt.get("val_metric", best_val_metric) or best_val_metric
            best_epoch      = int(ckpt.get("epoch", 0))
            logger.info(f"resumed from {p} (epoch {ckpt.get('epoch')})")
    logger.info(f"start_epoch = {start_epoch}  best_val_metric = {best_val_metric}")

    # ── CSV metric logger ──
    csv_fields = [
        "epoch", "phase",
        "train_loss", "val_metric",
        "lr_encoder", "lr_head",
        "epoch_time_s", "train_time_s", "val_time_s",
        "best_val_metric", "best_epoch",
        "grad_norm_mean", "grad_norm_max",
        "train_recon", "train_fourier", "train_sim", "train_std", "train_cov", "train_vicreg",
        "val_recon", "val_fourier", "val_std", "val_cov",
    ]
    csv_logger = CSVMetricLogger(save_dir / "metrics.csv", csv_fields)

    # ── Unfreeze schedule phase tags ──
    phase_tag = make_phase_tag(unfreeze_schedule)

    # ── Training loop ──
    _better = (
        (lambda new, best: new < best)
        if train_cfg.val_metric_direction == "min"
        else (lambda new, best: new > best)
    )

    train_losses: list[float] = []
    val_history:  list[dict]  = []
    lr_history:   list[tuple] = []
    total_t0 = time.time()

    if base_cfg.dry_run:
        logger.info("dry_run=True -- skipping full training loop")
    else:
        random_init_names = set(load_summary.get("random_init", []))
        for epoch in range(start_epoch, train_cfg.epochs + 1):
            active = apply_unfreeze_schedule(
                encoder, epoch, unfreeze_schedule, random_init_names,
            )
            phase_idx = max(
                (
                    i
                    for i, (start, _) in enumerate(unfreeze_schedule, 1)
                    if epoch >= start
                ),
                default=0,
            )
            phase = phase_tag.get(phase_idx, f"phase{phase_idx}")

            # ── Train ──
            train_loss, train_components, grads, train_time = train_one_epoch(
                encoder, heads, train_loader, optimizer, scaler,
                ssl_cfg, train_cfg, device,
                epoch=epoch, total_epochs=train_cfg.epochs, phase=phase,
            )
            scheduler.step()

            # ── Validate ──
            val_accum, val_metric, val_time = validate_one_epoch(
                encoder, heads, val_loader, ssl_cfg, train_cfg, device,
            )

            # ── Metrics & checkpointing ──
            enc_lr, head_lr = heads_iter_lrs(optimizer)
            epoch_time = train_time + val_time
            gmean = sum(grads) / max(1, len(grads))
            gmax  = max(grads) if grads else 0.0

            improved = ""
            if _better(val_metric, best_val_metric):
                best_val_metric = val_metric
                best_epoch      = epoch
                improved        = " *best*"
                save_checkpoint(
                    save_dir / "best_model.pt",
                    encoder=encoder, heads=heads,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, val_metric=val_metric, train_loss=train_loss,
                    extra={
                        "all_val_metrics": val_accum,
                        "channel_mean": ch_mean.tolist(),
                        "channel_std":  ch_std.tolist(),
                    },
                )
            save_checkpoint(
                save_dir / "last.pt",
                encoder=encoder, heads=heads,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, val_metric=val_metric, train_loss=train_loss,
                extra={
                    "channel_mean": ch_mean.tolist(),
                    "channel_std":  ch_std.tolist(),
                },
            )
            if train_cfg.save_every_n_epochs and epoch % train_cfg.save_every_n_epochs == 0:
                save_checkpoint(
                    save_dir / f"epoch_{epoch:04d}.pt",
                    encoder=encoder, heads=heads,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    epoch=epoch, val_metric=val_metric, train_loss=train_loss,
                    extra={
                        "channel_mean": ch_mean.tolist(),
                        "channel_std":  ch_std.tolist(),
                    },
                )

            # ── History & logging ──
            train_losses.append(train_loss)
            val_history.append(val_accum)
            lr_history.append((enc_lr, head_lr))
            csv_logger.log(dict(
                epoch=epoch, phase=phase,
                train_loss=train_loss, val_metric=val_metric,
                lr_encoder=enc_lr, lr_head=head_lr,
                epoch_time_s=epoch_time, train_time_s=train_time, val_time_s=val_time,
                best_val_metric=best_val_metric, best_epoch=best_epoch,
                grad_norm_mean=gmean, grad_norm_max=gmax,
                train_recon=train_components.get("recon"),
                train_fourier=train_components.get("fourier"),
                train_sim=train_components.get("sim"),
                train_std=train_components.get("std"),
                train_cov=train_components.get("cov"),
                train_vicreg=train_components.get("vicreg"),
                val_recon=val_accum.get("recon"),
                val_fourier=val_accum.get("fourier"),
                val_std=val_accum.get("std"),
                val_cov=val_accum.get("cov"),
            ))
            logger.info(
                f"ep {epoch:3d}/{train_cfg.epochs} [{phase}]  "
                f"train={train_loss:.5f}  val[{train_cfg.val_metric_key}]={val_metric:.5f}  "
                f"recon(t/v)={train_components.get('recon', 0):.4f}/{val_accum.get('recon', 0):.4f}  "
                f"std(t/v)={train_components.get('std', 0):.4f}/{val_accum.get('std', 0):.4f}  "
                f"lr(enc/head)={enc_lr:.2e}/{head_lr:.2e}  t={epoch_time:.1f}s  gn={gmean:.2f}{improved}"
            )

        csv_logger.close()
        logger.info(f"total training time = {(time.time() - total_t0) / 60:.1f} min")

    # ══════════════════════════════════════════════════════════════════════
    # Post-training
    # ══════════════════════════════════════════════════════════════════════
    if not base_cfg.dry_run:
        # Reload best checkpoint
        ckpt = reload_best_checkpoint(save_dir, encoder, heads, device, logger=logger)
        run_label = build_run_label(
            base_cfg,
            epoch=(ckpt or {}).get("epoch"),
            val_metric=(ckpt or {}).get("val_metric"),
        )

        # Reconstruction on canonical TRAIN sanity batch
        _ = post_training_reconstruction(
            encoder, heads, train_subset, train_transform, ssl_cfg,
            ch_mean=ch_mean, ch_std=ch_std, save_dir=save_dir,
            base_seed=base_cfg.seed, view="train",
            run_label=run_label, channel_names=data_cfg.channel_names,
            device=device,
        )

        # Reconstruction on canonical VAL sanity patches
        _ = post_training_reconstruction(
            encoder, heads, val_subset, val_transform, ssl_cfg,
            ch_mean=ch_mean, ch_std=ch_std, save_dir=save_dir,
            base_seed=base_cfg.seed, view="val",
            run_label=run_label, channel_names=data_cfg.channel_names,
            device=device,
        )

        # Training curves
        _ = plot_post_training_curves(save_dir, run_label=run_label, logger=logger)
        plt.close("all")

        # ── Save encoder-only checkpoint ──
        best_path = save_dir / "best_model.pt"
        if best_path.exists():
            src = torch.load(best_path, map_location=device, weights_only=False)
            enc_path = save_dir / train_cfg.encoder_save_name
            torch.save(
                {
                    "encoder_state_dict": src["encoder_state_dict"],
                    "channel_mean":       src.get("channel_mean", ch_mean.tolist()),
                    "channel_std":        src.get("channel_std",  ch_std.tolist()),
                    "epoch":              src.get("epoch"),
                    "val_metric":         src.get("val_metric"),
                    "init_source":        base_cfg.init_source,
                    "method_name":        base_cfg.method_name,
                },
                enc_path,
            )
            logger.info(f"encoder saved to {enc_path}")
        else:
            logger.warning("no best_model.pt -- nothing to export")

        # ── Full-image reconstruction (optional) ──
        if full_image_cfg.get("enabled", False):
            try:
                run_full_image_recon(
                    encoder, heads, data_cfg, model_cfg, ssl_cfg, base_cfg,
                    ch_mean, ch_std, save_dir, run_label, device,
                    full_image_cfg, logger,
                )
            except Exception as e:
                logger.warning(f"full-image reconstruction failed: {e}")

    logger.info("done.")


if __name__ == "__main__":
    main()
