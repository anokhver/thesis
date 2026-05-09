"""SimMIM + VICReg losses and the joint training/validation steps.

References
----------
SimMIM (masked-recon term) -- Xie, Zhang, Cao, Lin, Bao, Yao, Dai, Hu.
    "SimMIM: A Simple Framework for Masked Image Modeling." CVPR 2022.
    https://github.com/microsoft/SimMIM
VICReg (sim / std / cov terms) -- Bardes, Ponce, LeCun.
    "VICReg: Variance-Invariance-Covariance Regularization for
    Self-Supervised Learning." ICLR 2022.
    https://github.com/facebookresearch/vicreg

The math follows both papers; the code is a re-implementation, not a
verbatim copy. The default coefficients (lambda_sim, lambda_std,
lambda_cov) = (25, 25, 1) are the original VICReg defaults.
The joint SimMIM+VICReg objective combined here is original to this
project.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SSLCfg
from .masking import apply_mask, random_block_mask


def _encode_deepest(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run encoder and return the deepest feature map ``(B, C, S, S)``."""
    return encoder(x.contiguous())[-1]


def _simmim_recon_l1(pred, target, mask) -> torch.Tensor:
    # SimMIM Eq. (1) -- masked L1 averaged over (masked pixels * channels).
    err = (pred - target).abs() * mask
    denom = mask.sum() * pred.shape[1] + 1e-8
    return err.sum() / denom


def _simmim_recon_l1_l2(pred, target, mask) -> torch.Tensor:
    err1 = (pred - target).abs() * mask
    err2 = (pred - target).pow(2) * mask
    denom = mask.sum() * pred.shape[1] + 1e-8
    return 0.5 * (err1.sum() / denom) + 0.5 * (err2.sum() / denom)


def simmim_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "l1",
) -> torch.Tensor:
    """Masked-only reconstruction loss (``'l1'`` or ``'l1_l2_mix'``)."""
    if kind == "l1":
        return _simmim_recon_l1(pred, target, mask)
    if kind == "l1_l2_mix":
        return _simmim_recon_l1_l2(pred, target, mask)
    raise ValueError(f"Unknown loss_kind={kind!r}; expected 'l1' or 'l1_l2_mix'.")


def vicreg_terms(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4):
    """Compute VICReg's invariance / variance / covariance terms in fp32.

    Follows Bardes et al. 2022 (ICLR) eqs. (1)-(4). Implementation parallels
    ``main_vicreg.py`` in https://github.com/facebookresearch/vicreg but is
    written from scratch.
    """
    z1 = z1.float()
    z2 = z2.float()
    B, D = z1.shape
    L_sim = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
    L_std = 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    z2c = z2 - z2.mean(dim=0, keepdim=True)
    denom = max(B - 1, 1)
    cov1 = (z1c.T @ z1c) / denom
    cov2 = (z2c.T @ z2c) / denom
    off1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
    off2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
    L_cov = (off1 + off2) / (2.0 * D)
    return L_sim, L_std, L_cov


def compute_simmim_vicreg_loss(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    view1: torch.Tensor,
    view2: torch.Tensor,
    ssl_cfg: SSLCfg,
    *,
    fixed_mask: torch.Tensor | None = None,
):
    """Joint SimMIM (masked recon) + VICReg loss.

    If ``fixed_mask`` is None, a fresh random block mask is sampled per call;
    pass a tensor to reuse the same mask (e.g. for the overfit sanity step).

    The encoder + decoder forward run in whatever precision the caller set
    (typically fp16 under ``torch.amp.autocast``). The projector forward,
    VICReg variance/covariance terms and the masked recon loss are forced
    to fp32 because their math (``1/sqrt(var+eps)``, ``D x D`` outer products,
    long reductions over masked pixels) overflows fp16 in the early epochs
    and produces ``inf`` gradients.
    """
    decoder    = heads["decoder"]
    mask_token = heads["mask_token"]
    projector  = heads["projector"]

    if fixed_mask is None:
        mask = random_block_mask(view1, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio)
    else:
        mask = fixed_mask

    v1m = apply_mask(view1, mask, mask_token)
    v2m = apply_mask(view2, mask, mask_token)
    z1 = _encode_deepest(encoder, v1m)
    z2 = _encode_deepest(encoder, v2m)
    r1 = decoder(z1)
    r2 = decoder(z2)

    # All loss math in fp32 -- autocast(enabled=False) overrides any outer
    # autocast context so the projector matmul + variance/covariance run
    # in fp32 and stop producing inf grads.
    device_type = z1.device.type
    with torch.amp.autocast(device_type, enabled=False):
        r1f = r1.float()
        r2f = r2.float()
        v1f = view1.float()
        v2f = view2.float()
        mf  = mask.float()
        L_recon = 0.5 * (
            simmim_recon_loss(r1f, v1f, mf, ssl_cfg.loss_kind)
            + simmim_recon_loss(r2f, v2f, mf, ssl_cfg.loss_kind)
        )

        p1 = projector(z1.float().mean(dim=(-2, -1)))
        p2 = projector(z2.float().mean(dim=(-2, -1)))
        L_sim, L_std, L_cov = vicreg_terms(p1, p2)
        L_vicreg = (
            ssl_cfg.lambda_sim * L_sim
            + ssl_cfg.lambda_std * L_std
            + ssl_cfg.lambda_cov * L_cov
        )

        loss = ssl_cfg.w_recon * L_recon + ssl_cfg.w_vicreg * L_vicreg

    metrics = {
        "ssl_loss": float(loss.detach().item()),
        "recon":    float(L_recon.detach().item()),
        "sim":      float(L_sim.detach().item()),
        "std":      float(L_std.detach().item()),
        "cov":      float(L_cov.detach().item()),
        "vicreg":   float(L_vicreg.detach().item()),
    }
    return loss, metrics


def validation_simmim(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    batch: torch.Tensor,
    ssl_cfg: SSLCfg,
) -> dict:
    """Single-view masked-recon validation step + VICReg collapse diagnostics.

    ``batch`` is a single z-scored view. ``L_recon`` (SimMIM) is the primary
    metric; ``L_sim`` is omitted because it requires two views. ``L_std`` and
    ``L_cov`` are batch statistics over projector outputs and *can* be
    computed on a single view -- they're reported here as collapse-detection
    diagnostics on the held-out set, not as a selection criterion.
    """
    decoder    = heads["decoder"]
    mask_token = heads["mask_token"]
    projector  = heads["projector"]

    mask = random_block_mask(batch, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio)
    v_masked = apply_mask(batch, mask, mask_token)
    z = _encode_deepest(encoder, v_masked)
    recon = decoder(z)
    L_recon = simmim_recon_loss(recon, batch, mask, ssl_cfg.loss_kind)

    # Single-view VICReg diagnostics (no L_sim; needs two views).
    p = projector(z.float().mean(dim=(-2, -1))).float()
    B, D = p.shape
    eps = 1e-4
    std = torch.sqrt(p.var(dim=0, unbiased=False) + eps)
    L_std = F.relu(1.0 - std).mean()
    pc = p - p.mean(dim=0, keepdim=True)
    denom = max(B - 1, 1)
    cov = (pc.T @ pc) / denom
    L_cov = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

    return {
        "ssl_loss": float(L_recon.item()),
        "recon":    float(L_recon.item()),
        "std":      float(L_std.item()),
        "cov":      float(L_cov.item()),
    }
