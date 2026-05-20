"""Reusable helpers for assembling a SimMIM+VICReg pretraining run.

Keeps the script (`scripts/training/pretrain_simmim_vicreg.py`) thin
while making the same flow callable from notebooks or tests.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from ..models.swin import build_simmim_vicreg_heads, build_swin_encoder, count_params
from ..models.weight_loading import load_pretrained_into_encoder
from ..training.augment import (
    MicroscopyTwoViewTransform,
    SeededTwoViewTransform,
    ValSingleViewTransform,
)
from ..training.checkpoints import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..training.config import BaseCfg, DataCfg, ModelCfg, SSLCfg, TrainCfg
from ..training.data import TransformedSubset, compute_channel_stats
from ..training.logging import CSVMetricLogger
from ..training.losses import compute_simmim_vicreg_loss
from ..training.lr_schedule import make_warmup_cosine, param_groups_layer_decay
from ..training.masking import random_block_mask
from ..training.sanity_batch import (
    SANITY_TRAIN_INDICES,
    fixed_two_view_batch,
    overfit_on_batch,
)
from ..training.viz import (
    plot_channel_histograms,
    plot_overfit_curves,
    plot_recon_panel,
    plot_two_views,
)
from ..utils_data.patch_dataset import PatchDataset
from .post_training import (
    build_run_label,
    eval_recon_batch,
    plot_post_training_curves,
    post_training_reconstruction,
    reload_best_checkpoint,
)
from .full_image_recon import run_full_image_recon
from .train_loop import (
    heads_iter_lrs,
    train_one_epoch,
    validate_one_epoch,
)
from .unfreeze import (
    apply_unfreeze_schedule,
    make_phase_tag,
)


# ── Lightweight state containers ──────────────────────────────────────────

@dataclasses.dataclass
class DataState:
    """Everything produced by :func:`setup_data`."""
    raw_dataset: PatchDataset
    train_subset: Subset
    val_subset: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    ch_mean: torch.Tensor
    ch_std: torch.Tensor


@dataclasses.dataclass
class ModelState:
    """Encoder + heads + weight-loading metadata."""
    encoder: nn.Module
    heads: dict[str, nn.Module]
    load_summary: dict[str, Any]


@dataclasses.dataclass
class OptimState:
    """Optimizer, scheduler, scaler and epoch bookkeeping."""
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    start_epoch: int
    best_val_metric: float
    best_epoch: int


# ── Setup helpers ─────────────────────────────────────────────────────────

def setup_data(
    data_cfg: DataCfg,
    model_cfg: ModelCfg,
    ssl_cfg: SSLCfg,
    *,
    seed_generator: torch.Generator,
    logger=None,
) -> DataState:
    """Load dataset, split, compute channel stats, build DataLoaders."""
    log = logger.info if logger else (lambda *a: None)

    raw_dataset = PatchDataset(
        root=data_cfg.data_root,
        exclude_patterns=data_cfg.exclude_patterns,
    )
    log(f"raw patches = {len(raw_dataset)}")
    if len(raw_dataset) == 0:
        raise RuntimeError(
            f"No patches found under {data_cfg.data_root!r}. "
            "Check data_root/exclude_patterns in the config."
        )
    sample = raw_dataset[0]
    log(f"sample shape = {tuple(sample.shape)}  dtype = {sample.dtype}")
    assert sample.ndim == 3 and sample.shape[0] == model_cfg.in_channels
    assert sample.shape[-1] == model_cfg.img_size

    n_val = int(len(raw_dataset) * data_cfg.val_split)
    n_train = len(raw_dataset) - n_val
    if n_val == 0:
        raise RuntimeError(
            "Validation split produced zero samples. Increase data.val_split "
            "or use a larger dataset."
        )
    if n_train == 0:
        raise RuntimeError(
            "Training split produced zero samples. Decrease data.val_split."
        )
    train_subset, val_subset = random_split(
        raw_dataset, [n_train, n_val], generator=seed_generator,
    )
    log(f"split: train={n_train}  val={n_val}")

    ch_mean, ch_std = compute_channel_stats(
        train_subset, in_channels=model_cfg.in_channels,
        max_samples=data_cfg.channel_stats_max_samples,
    )
    for ch_name, m, s in zip(data_cfg.channel_names, ch_mean.tolist(), ch_std.tolist()):
        log(f"  ch[{ch_name:>14s}] mean={m:.5f}  std={s:.5f}")

    # Augmentation pipelines
    train_transform = MicroscopyTwoViewTransform(ch_mean, ch_std)
    if getattr(ssl_cfg, "two_view_validation", False):
        val_transform = SeededTwoViewTransform(
            ch_mean, ch_std,
            base_seed=int(getattr(ssl_cfg, "val_view_seed", 12345)),
        )
        log(f"validation: two-view mode (seeded, base_seed={ssl_cfg.val_view_seed})")
    else:
        val_transform = ValSingleViewTransform(ch_mean, ch_std)
        log("validation: single-view mode")

    train_ds = TransformedSubset(train_subset, train_transform)
    val_ds = TransformedSubset(val_subset, val_transform)
    common = dict(
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.num_workers > 0,
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size,
        shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.batch_size,
        shuffle=False, **common,
    )
    log(f"train batches = {len(train_loader)}  val batches = {len(val_loader)}")

    return DataState(
        raw_dataset=raw_dataset,
        train_subset=train_subset,
        val_subset=val_subset,
        train_loader=train_loader,
        val_loader=val_loader,
        ch_mean=ch_mean,
        ch_std=ch_std,
    )


def build_model(
    base_cfg: BaseCfg,
    model_cfg: ModelCfg,
    ssl_cfg: SSLCfg,
    device: torch.device,
    *,
    logger=None,
) -> ModelState:
    """Build encoder + heads and load pretrained weights."""
    log = logger.info if logger else (lambda *a: None)
    warn = logger.warning if logger else (lambda *a: None)

    encoder = build_swin_encoder(model_cfg).to(device)
    log(f"encoder params = {count_params(encoder) / 1e6:.2f} M")

    load_summary = load_pretrained_into_encoder(
        encoder, base_cfg, model_cfg, logger=logger,
    )
    n_loaded = load_summary["n_loaded"]
    n_target = load_summary["n_target_params"]
    pct = 100.0 * n_loaded / max(n_target, 1)
    log(f"coverage: {n_loaded}/{n_target} params loaded ({pct:.1f}%)")
    if load_summary.get("n_shape_mismatch", 0) > 0:
        warn(f"  shape mismatches: {load_summary['shape_mismatch']}")
    if base_cfg.init_source != "scratch" and n_loaded < 0.8 * n_target:
        raise RuntimeError(
            f"Only {pct:.1f}% of encoder loaded — expected >=80%. "
            f"Check the checkpoint is correct for init_source={base_cfg.init_source!r}."
        )

    random_keys = load_summary.get("random_init", [])
    if random_keys and base_cfg.init_source != "scratch":
        log(f"  random-init keys ({len(random_keys)}):")
        for k in random_keys:
            log(f"    - {k}")
    elif random_keys:
        log(f"  random-init keys: ALL ({len(random_keys)})")

    heads = build_simmim_vicreg_heads(model_cfg, ssl_cfg)
    heads = {k: v.to(device) for k, v in heads.items()}
    log(f"head params = {count_params(heads) / 1e6:.2f} M")

    return ModelState(encoder=encoder, heads=heads, load_summary=load_summary)


def setup_optimizer(
    model: ModelState,
    train_cfg: TrainCfg,
    device: torch.device,
    *,
    resume_path: str | None = None,
    logger=None,
) -> OptimState:
    """Build optimizer, scheduler, scaler; optionally resume from checkpoint."""
    log = logger.info if logger else (lambda *a: None)
    warn = logger.warning if logger else (lambda *a: None)

    encoder_groups = param_groups_layer_decay(
        model.encoder,
        base_lr=train_cfg.base_lr,
        weight_decay=train_cfg.weight_decay,
        layer_decay=train_cfg.layer_decay,
    )
    head_params = [p for h in model.heads.values() for p in h.parameters()]
    head_group = {
        "params": head_params,
        "lr": train_cfg.head_lr,
        "weight_decay": train_cfg.weight_decay,
    }
    optimizer = torch.optim.AdamW(encoder_groups + [head_group], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_warmup_cosine(train_cfg.warmup_epochs, train_cfg.epochs),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start_epoch = 1
    best_val_metric = (
        float("inf") if train_cfg.val_metric_direction == "min" else float("-inf")
    )
    best_epoch = 0

    if resume_path is not None:
        p = Path(resume_path)
        if p.is_dir():
            p = find_latest_checkpoint(p)
        if p is None or not Path(p).exists():
            warn(f"resume path {resume_path!r} not found -- starting fresh")
        else:
            ckpt = load_checkpoint(
                p, encoder=model.encoder, heads=model.heads,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                map_location=device,
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_metric = ckpt.get("val_metric", best_val_metric) or best_val_metric
            best_epoch = int(ckpt.get("epoch", 0))
            log(f"resumed from {p} (epoch {ckpt.get('epoch')})")

    log(f"start_epoch = {start_epoch}  best_val_metric = {best_val_metric}")

    return OptimState(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        start_epoch=start_epoch,
        best_val_metric=best_val_metric,
        best_epoch=best_epoch,
    )


# ── Sanity / overfit checks ──────────────────────────────────────────────

def run_sanity_checks(
    model: ModelState,
    data: DataState,
    ssl_cfg: SSLCfg,
    model_cfg: ModelCfg,
    data_cfg: DataCfg,
    train_cfg: TrainCfg,
    device: torch.device,
    save_dir: Path,
    *,
    logger=None,
) -> None:
    """Dataloader sanity, forward-pass shape check, gradient-flow check."""
    log = logger.info if logger else (lambda *a: None)
    encoder, heads = model.encoder, model.heads

    # Dataloader sanity
    batch = next(iter(data.train_loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, batch
    v1, v2 = batch
    assert v1.shape == v2.shape
    assert v1.shape[1:] == (model_cfg.in_channels, model_cfg.img_size, model_cfg.img_size)
    assert v1.dtype == torch.float32
    assert torch.isfinite(v1).all() and torch.isfinite(v2).all()
    diff = (v1 - v2).abs().mean().item()
    assert diff > 1e-3, f"view1 == view2  (mean abs diff = {diff:.2e})"
    log(f"[ok] views shape={tuple(v1.shape)}  v1-v2 diff={diff:.4f}")

    # Two-view visualisation
    train_transform = data.train_loader.dataset.transform
    _ = plot_two_views(
        raw_subset=data.train_subset,
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
        log(f"  stage {i}: {tuple(f.shape)}{tag}")
    encoder.train()

    # Gradient flow check (AMP-matched to training)
    encoder.train()
    for h in heads.values():
        h.train()
    _trainable = (
        list(encoder.parameters())
        + [p for h in heads.values() for p in h.parameters()]
    )
    _opt = torch.optim.AdamW(_trainable, lr=1e-4)
    _scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    _v1 = v1.to(device, non_blocking=True)
    _v2 = v2.to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        _loss, _m = compute_simmim_vicreg_loss(encoder, heads, _v1, _v2, ssl_cfg)
    _scaler.scale(_loss).backward()
    _scaler.unscale_(_opt)
    torch.nn.utils.clip_grad_norm_(_trainable, max_norm=train_cfg.grad_clip_norm)
    n_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
    n_tot = sum(1 for _ in encoder.parameters())
    _scaler.step(_opt)
    _scaler.update()
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        log(
            f"[ok] grad flow: {n_grad}/{n_tot} encoder params got gradients  "
            f"peak VRAM = {peak_gb:.2f} / {total_gb:.2f} GB "
            f"({100 * peak_gb / total_gb:.1f}%, AMP-matched to training)"
        )
    else:
        log(f"[ok] grad flow: {n_grad}/{n_tot} encoder params got gradients")
    del _opt, _scaler, _loss, _m, _v1, _v2, _trainable
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_overfit_check(
    model: ModelState,
    data: DataState,
    base_cfg: BaseCfg,
    ssl_cfg: SSLCfg,
    train_cfg: TrainCfg,
    data_cfg: DataCfg,
    device: torch.device,
    save_dir: Path,
    *,
    logger=None,
) -> dict | None:
    """Run the overfit-on-sanity-batch diagnostic. Returns overfit result dict."""
    log = logger.info if logger else (lambda *a: None)
    encoder, heads = model.encoder, model.heads
    train_transform = data.train_loader.dataset.transform

    encoder_groups = param_groups_layer_decay(
        encoder,
        base_lr=train_cfg.base_lr,
        weight_decay=train_cfg.weight_decay,
        layer_decay=train_cfg.layer_decay,
    )
    head_params = [p for h in heads.values() for p in h.parameters()]
    head_group = {
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
        data.train_subset, SANITY_TRAIN_INDICES, train_transform,
        seed=base_cfg.seed + 17,
    )
    x1_fixed = x1_fixed.to(device)
    x2_fixed = x2_fixed.to(device)
    log(f"sanity batch indices = {sanity_idx}  shape = {tuple(x1_fixed.shape)}")

    torch.manual_seed(base_cfg.seed + 18)
    mask_fixed = random_block_mask(
        x1_fixed, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio,
    ).clone()
    log(f"fixed mask coverage = {mask_fixed.mean().item():.3f}")

    n_overfit_steps = 1000
    overfit_lr = 3e-4
    overfit_result = overfit_on_batch(
        encoder, heads, x1_fixed, x2_fixed, ssl_cfg,
        n_steps=n_overfit_steps, lr=overfit_lr,
        grad_clip=train_cfg.grad_clip_norm, fixed_mask=mask_fixed, logger=logger,
    )

    _ = plot_overfit_curves(
        overfit_result["history"],
        suptitle=f"Sanity overfit on indices {sanity_idx}",
        save_to=save_dir / "sanity_curves_overfit.png",
        run_label=build_run_label(base_cfg, extra="overfit"),
    )
    plt.close("all")

    rec, v_masked = eval_recon_batch(
        encoder, heads, x1_fixed, mask_fixed, ssl_cfg.head_stage_index,
    )
    _ = plot_recon_panel(
        x1_fixed, v_masked, rec, mask_fixed, sanity_idx,
        ch_mean=data.ch_mean, ch_std=data.ch_std,
        suptitle="Sanity overfit -- reconstruction on canonical batch",
        save_to=save_dir / "sanity_recon_overfit.png",
        run_label=build_run_label(base_cfg, extra="overfit"),
        channel_names=data_cfg.channel_names,
    )
    plt.close("all")
    encoder.train()
    for h in heads.values():
        h.train()

    return overfit_result


# ── Training loop ─────────────────────────────────────────────────────────

def training_loop(
    model: ModelState,
    optim: OptimState,
    data: DataState,
    base_cfg: BaseCfg,
    data_cfg: DataCfg,
    ssl_cfg: SSLCfg,
    train_cfg: TrainCfg,
    unfreeze_schedule: list,
    device: torch.device,
    save_dir: Path,
    *,
    logger=None,
) -> None:
    """Run the full epoch loop with checkpointing and CSV logging."""
    log = logger.info if logger else (lambda *a: None)
    encoder, heads = model.encoder, model.heads
    ch_mean, ch_std = data.ch_mean, data.ch_std
    optimizer, scheduler, scaler = optim.optimizer, optim.scheduler, optim.scaler
    best_val_metric = optim.best_val_metric
    best_epoch = optim.best_epoch

    csv_fields = [
        "epoch", "phase",
        "train_loss", "val_metric",
        "lr_encoder", "lr_head",
        "epoch_time_s", "train_time_s", "val_time_s",
        "best_val_metric", "best_epoch",
        "grad_norm_mean", "grad_norm_max",
        "train_recon", "train_recon_fg",
        "train_fourier",
        "train_sim", "train_std", "train_cov", "train_vicreg",
        "val_recon", "val_recon_fg",
        "val_fourier",
        "val_std", "val_cov",
        "val_sim", "val_vicreg", "val_ssl_loss",
    ]
    csv_logger = CSVMetricLogger(save_dir / "metrics.csv", csv_fields)
    phase_tag = make_phase_tag(unfreeze_schedule)

    _better = (
        (lambda new, best: new < best)
        if train_cfg.val_metric_direction == "min"
        else (lambda new, best: new > best)
    )

    random_init_names = set(model.load_summary.get("random_init", []))
    total_t0 = time.time()

    for epoch in range(optim.start_epoch, train_cfg.epochs + 1):
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

        train_loss, train_components, grads, train_time = train_one_epoch(
            encoder, heads, data.train_loader, optimizer, scaler,
            ssl_cfg, train_cfg, device,
            epoch=epoch, total_epochs=train_cfg.epochs, phase=phase,
        )
        scheduler.step()

        val_accum, val_metric, val_time = validate_one_epoch(
            encoder, heads, data.val_loader, ssl_cfg, train_cfg, device,
        )

        enc_lr, head_lr = heads_iter_lrs(optimizer)
        epoch_time = train_time + val_time
        gmean = sum(grads) / max(1, len(grads))
        gmax = max(grads) if grads else 0.0

        improved = ""
        if _better(val_metric, best_val_metric):
            best_val_metric = val_metric
            best_epoch = epoch
            improved = " *best*"
            save_checkpoint(
                save_dir / "best_model.pt",
                encoder=encoder, heads=heads,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, val_metric=val_metric, train_loss=train_loss,
                extra={
                    "all_val_metrics": val_accum,
                    "channel_mean": ch_mean.tolist(),
                    "channel_std": ch_std.tolist(),
                },
            )
        save_checkpoint(
            save_dir / "last.pt",
            encoder=encoder, heads=heads,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, val_metric=val_metric, train_loss=train_loss,
            extra={
                "channel_mean": ch_mean.tolist(),
                "channel_std": ch_std.tolist(),
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
                    "channel_std": ch_std.tolist(),
                },
            )

        csv_logger.log(dict(
            epoch=epoch, phase=phase,
            train_loss=train_loss, val_metric=val_metric,
            lr_encoder=enc_lr, lr_head=head_lr,
            epoch_time_s=epoch_time, train_time_s=train_time, val_time_s=val_time,
            best_val_metric=best_val_metric, best_epoch=best_epoch,
            grad_norm_mean=gmean, grad_norm_max=gmax,
            train_recon=train_components.get("recon"),
            train_recon_fg=train_components.get("recon_fg"),
            train_fourier=train_components.get("fourier"),
            train_sim=train_components.get("sim"),
            train_std=train_components.get("std"),
            train_cov=train_components.get("cov"),
            train_vicreg=train_components.get("vicreg"),
            val_recon=val_accum.get("recon"),
            val_recon_fg=val_accum.get("recon_fg"),
            val_fourier=val_accum.get("fourier"),
            val_std=val_accum.get("std"),
            val_cov=val_accum.get("cov"),
            val_sim=val_accum.get("sim"),
            val_vicreg=val_accum.get("vicreg"),
            val_ssl_loss=val_accum.get("ssl_loss"),
        ))
        log(
            f"ep {epoch:3d}/{train_cfg.epochs} [{phase}]  "
            f"train={train_loss:.5f}  val[{train_cfg.val_metric_key}]={val_metric:.5f}  "
            + (
                f"val_ssl={val_accum.get('ssl_loss', 0):.4f}  "
                if val_accum.get("ssl_loss") is not None else ""
            )
            + f"recon(t/v)={train_components.get('recon', 0):.4f}/{val_accum.get('recon', 0):.4f}  "
            f"recon_fg(t/v)={train_components.get('recon_fg', 0):.4f}/{val_accum.get('recon_fg', 0):.4f}  "
            f"vicreg(t/v)={train_components.get('vicreg', 0):.4f}/{val_accum.get('vicreg', 0):.4f}  "
            f"sim(t/v)={train_components.get('sim', 0):.4f}/{val_accum.get('sim', 0):.4f}  "
            f"std(t/v)={train_components.get('std', 0):.4f}/{val_accum.get('std', 0):.4f}  "
            f"cov(t/v)={train_components.get('cov', 0):.4f}/{val_accum.get('cov', 0):.4f}  "
            f"lr(enc/head)={enc_lr:.2e}/{head_lr:.2e}  t={epoch_time:.1f}s  gn={gmean:.2f}{improved}"
        )

    csv_logger.close()
    log(f"total training time = {(time.time() - total_t0) / 60:.1f} min")


# ── Post-training ─────────────────────────────────────────────────────────

def post_training_flow(
    model: ModelState,
    data: DataState,
    base_cfg: BaseCfg,
    data_cfg: DataCfg,
    model_cfg: ModelCfg,
    ssl_cfg: SSLCfg,
    train_cfg: TrainCfg,
    full_image_cfg: dict,
    device: torch.device,
    save_dir: Path,
    *,
    logger=None,
) -> None:
    """Reload best checkpoint, generate plots/recon, export encoder."""
    log = logger.info if logger else (lambda *a: None)
    warn = logger.warning if logger else (lambda *a: None)
    encoder, heads = model.encoder, model.heads
    ch_mean, ch_std = data.ch_mean, data.ch_std

    ckpt = reload_best_checkpoint(save_dir, encoder, heads, device, logger=logger)
    run_label = build_run_label(
        base_cfg,
        epoch=(ckpt or {}).get("epoch"),
        val_metric=(ckpt or {}).get("val_metric"),
    )

    train_transform = data.train_loader.dataset.transform
    val_transform = data.val_loader.dataset.transform

    _ = post_training_reconstruction(
        encoder, heads, data.train_subset, train_transform, ssl_cfg,
        ch_mean=ch_mean, ch_std=ch_std, save_dir=save_dir,
        base_seed=base_cfg.seed, view="train",
        run_label=run_label, channel_names=data_cfg.channel_names,
        device=device,
    )

    _ = post_training_reconstruction(
        encoder, heads, data.val_subset, val_transform, ssl_cfg,
        ch_mean=ch_mean, ch_std=ch_std, save_dir=save_dir,
        base_seed=base_cfg.seed, view="val",
        run_label=run_label, channel_names=data_cfg.channel_names,
        device=device,
    )

    _ = plot_post_training_curves(save_dir, run_label=run_label, logger=logger)
    plt.close("all")

    # Export encoder-only checkpoint
    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        src = torch.load(best_path, map_location=device, weights_only=False)
        enc_path = save_dir / train_cfg.encoder_save_name
        torch.save(
            {
                "encoder_state_dict": src["encoder_state_dict"],
                "channel_mean": src.get("channel_mean", ch_mean.tolist()),
                "channel_std": src.get("channel_std", ch_std.tolist()),
                "epoch": src.get("epoch"),
                "val_metric": src.get("val_metric"),
                "init_source": base_cfg.init_source,
                "method_name": base_cfg.method_name,
            },
            enc_path,
        )
        log(f"encoder saved to {enc_path}")
    else:
        warn("no best_model.pt -- nothing to export")

    # Full-image reconstruction (optional)
    if full_image_cfg.get("enabled", False):
        try:
            run_full_image_recon(
                encoder, heads, data_cfg, model_cfg, ssl_cfg, base_cfg,
                ch_mean, ch_std, save_dir, run_label, device,
                full_image_cfg, logger,
            )
        except Exception as e:
            warn(f"full-image reconstruction failed: {e}")
