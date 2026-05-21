"""Step-based train/validate primitives for SwinUNETR self-training.

Atomic step (one forward/backward/update) and full validation pass.
Designed for an outer step-counting loop in :mod:`runner`. Mirrors the
style of :mod:`synaptic_ssl.ssl_training.train_loop` but iterates by
optimiser step, not by epoch.

Supports:
  * binary 1-channel targets ``(B, 1, H, W)`` -- legacy notebook path.
  * 2-channel PRE/POST targets ``(B, 2, H, W)`` -- recipe-v2 path.
    Per-channel Dice/BCE is recovered by reshaping ``(B, C, H, W)`` to
    ``(B*C, 1, H, W)`` before calling :class:`DiceBCELoss`, so each
    channel of each sample contributes one term to the per-sample mean
    inside the loss. The ``loss_mask`` (still ``(B, 1, H, W)``) is
    broadcast across the channel dim.

Batches may be either 2-tuples ``(image, target)`` or 3-tuples
``(image, target, loss_mask)``; both shapes are accepted.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import DiceBCELoss, compute_dice_metric


def cycle(loader: DataLoader) -> Iterator:
    """Yield batches forever, restarting the dataloader between epochs."""
    while True:
        for batch in loader:
            yield batch


def _unpack_batch(batch, device: torch.device):
    """Move a batch to ``device``; return ``(image, target, loss_mask | None)``."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        image, target = batch
        loss_mask = None
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        image, target, loss_mask = batch
    else:
        raise ValueError(
            f"batch must be a 2- or 3-tuple of tensors; got {type(batch)} "
            f"of length {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
        )
    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    if loss_mask is not None:
        loss_mask = loss_mask.to(device, non_blocking=True)
    return image, target, loss_mask


def _flatten_channels(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor | None,
):
    """Reshape ``(B, C, H, W)`` -> ``(B*C, 1, H, W)`` so DiceBCELoss averages per-channel.

    ``loss_mask`` (``(B, 1, H, W)``) is broadcast across the channel dim
    before being flattened, so the same anatomical gate applies to PRE
    and POST. When ``logits.size(1) == 1`` this is a no-op.
    """
    if logits.size(1) == 1:
        return logits, target, loss_mask
    B, C, H, W = logits.shape
    logits = logits.reshape(B * C, 1, H, W)
    target = target.reshape(B * C, 1, H, W)
    if loss_mask is not None:
        # broadcast (B,1,H,W) -> (B,C,H,W) -> (B*C,1,H,W)
        loss_mask = loss_mask.expand(B, C, H, W).reshape(B * C, 1, H, W).contiguous()
    return logits, target, loss_mask


def trainable_params(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def train_step(
    model: nn.Module,
    batch,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    loss_fn: DiceBCELoss,
    grad_clip_norm: float,
    device: torch.device,
) -> dict:
    """One AMP train step. Steps optimiser, scaler and scheduler.

    Returns a dict of plain Python floats (safe to log without keeping
    the autograd graph alive).
    """
    model.train()
    image, target, loss_mask = _unpack_batch(batch, device)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(image)
        l, t, lm = _flatten_channels(logits, target, loss_mask)
        losses = loss_fn(l, t, loss_mask=lm)

    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    gn = torch.nn.utils.clip_grad_norm_(
        trainable_params(model), max_norm=grad_clip_norm,
    )
    # GradScaler may skip optimizer.step() under inf/nan grads (signalled
    # by a downward scale change in scaler.update()). When that happens
    # PyTorch warns if we then call scheduler.step() -- and the LR would
    # silently desync from the step counter by one tick. Gate the
    # scheduler on whether the optimizer actually applied.
    pre_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() >= pre_scale:
        scheduler.step()

    with torch.no_grad():
        # hard Dice on the flat (B*C, 1, H, W) view -- same per-channel mean
        dice = compute_dice_metric(l, t, loss_mask=lm).item()

    return {
        "loss":      float(losses["loss"].item()),
        "dice_loss": float(losses["dice_loss"].item()),
        "bce_loss":  float(losses["bce_loss"].item()),
        "dice":      dice,
        "grad_norm": float(gn.item()) if torch.isfinite(gn) else float("nan"),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: Iterable,
    *,
    loss_fn: DiceBCELoss,
    device: torch.device,
) -> dict:
    """Full validation pass. Returns mean loss + hard Dice."""
    model.eval()
    n_b = 0
    sum_loss = 0.0
    sum_dice_l = 0.0
    sum_bce_l = 0.0
    sum_dice = 0.0
    for batch in val_loader:
        image, target, loss_mask = _unpack_batch(batch, device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(image)
            l, t, lm = _flatten_channels(logits, target, loss_mask)
            losses = loss_fn(l, t, loss_mask=lm)
        sum_loss += float(losses["loss"].item())
        sum_dice_l += float(losses["dice_loss"].item())
        sum_bce_l += float(losses["bce_loss"].item())
        sum_dice += float(compute_dice_metric(l, t, loss_mask=lm).item())
        n_b += 1
    n_b = max(1, n_b)
    return {
        "val_loss":      sum_loss / n_b,
        "val_dice_loss": sum_dice_l / n_b,
        "val_bce_loss":  sum_bce_l / n_b,
        "val_dice":      sum_dice / n_b,
    }


def freeze_swinvit(model: nn.Module, freeze: bool) -> None:
    """Set ``requires_grad`` on ``model.swinViT`` parameters."""
    for p in model.swinViT.parameters():
        p.requires_grad = not freeze


def encoder_decoder_lrs(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return ``(max encoder-stage LR, first decoder-stage LR)``.

    Param-group convention from :func:`runner.setup_seg_optimizer`:
    encoder groups carry a ``"stage"`` key (from
    :func:`param_groups_layer_decay`); the decoder group does not.
    """
    enc = [g["lr"] for g in optimizer.param_groups if g.get("stage") is not None]
    dec = [g["lr"] for g in optimizer.param_groups if g.get("stage") is None]
    return (max(enc) if enc else 0.0, dec[0] if dec else 0.0)
