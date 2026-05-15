"""Per-epoch training and validation loops for SimMIM+VICReg pretraining."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from training.losses import compute_simmim_vicreg_loss, validation_simmim


def set_train(encoder: nn.Module, heads: dict[str, nn.Module], training: bool) -> None:
    """Toggle ``train``/``eval`` mode on encoder + every head."""
    encoder.train(training)
    for h in heads.values():
        h.train(training)


def heads_iter_lrs(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return ``(max_encoder_lr, head_lr)`` from the optimizer param groups.

    Encoder groups are identified by the ``stage`` key set by
    :func:`training.lr_schedule.param_groups_layer_decay`.
    """
    enc_lrs  = [g["lr"] for g in optimizer.param_groups if "stage" in g]
    head_lrs = [g["lr"] for g in optimizer.param_groups if "stage" not in g]
    return (
        max(enc_lrs) if enc_lrs else 0.0,
        head_lrs[0] if head_lrs else 0.0,
    )


def all_trainable_params(encoder: nn.Module, heads: dict[str, nn.Module]) -> list:
    """Collect every param with ``requires_grad`` from encoder + heads."""
    return (
        [p for p in encoder.parameters() if p.requires_grad]
        + [p for h in heads.values() for p in h.parameters() if p.requires_grad]
    )


def train_one_epoch(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    train_loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ssl_cfg,
    train_cfg,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    phase: str,
):
    """Run one training epoch. Returns ``(loss, components, grad_norms, time_s)``."""
    set_train(encoder, heads, True)
    running, grads = 0.0, []
    comp_accum: dict[str, float] = {}
    t0 = time.time()
    pbar = tqdm(train_loader, desc=f"ep {epoch}/{total_epochs} [{phase}]", leave=False)
    for v1_b, v2_b in pbar:
        v1_b = v1_b.to(device, non_blocking=True)
        v2_b = v2_b.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss, _m = compute_simmim_vicreg_loss(encoder, heads, v1_b, v2_b, ssl_cfg)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(
            all_trainable_params(encoder, heads),
            max_norm=train_cfg.grad_clip_norm,
        )
        grads.append(gn.item())
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        for _k, _v in _m.items():
            comp_accum[_k] = comp_accum.get(_k, 0.0) + float(_v)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    n_tb = max(1, len(train_loader))
    train_loss = running / n_tb
    train_components = {k: v / n_tb for k, v in comp_accum.items()}
    return train_loss, train_components, grads, time.time() - t0


def validate_one_epoch(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    val_loader,
    ssl_cfg,
    train_cfg,
    device: torch.device,
):
    """Run one validation pass. Returns ``(metrics, val_metric, time_s)``."""
    set_train(encoder, heads, False)
    val_accum: dict[str, float] = {}
    n_vb = 0
    t0 = time.time()
    with torch.no_grad():
        for v_b in val_loader:
            if isinstance(v_b, (list, tuple)):
                v_b = v_b[0]
            v_b = v_b.to(device, non_blocking=True)
            vm = validation_simmim(encoder, heads, v_b, ssl_cfg)
            for k, v in vm.items():
                val_accum[k] = val_accum.get(k, 0.0) + float(v)
            n_vb += 1
        for k in val_accum:
            val_accum[k] /= max(1, n_vb)
    val_metric = val_accum[train_cfg.val_metric_key]
    return val_accum, val_metric, time.time() - t0
