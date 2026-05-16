"""SimMIM + VICReg losses, Fourier auxiliary, and validation metrics.

Includes custom foreground sigmoid reweighting (precedent: VasoMIM, Huang et
al., AAAI 2026) and custom foreground-weighted spatial pooling for the VICReg
branch.
Ref: https://github.com/microsoft/SimMIM
Ref: https://github.com/facebookresearch/mae
Ref: https://github.com/facebookresearch/vicreg
Ref: https://github.com/recursionpharma/maes_microscopy
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SSLCfg
from .masking import apply_mask, random_block_mask


def _encode_at(encoder: nn.Module, x: torch.Tensor, stage_index: int = -1) -> torch.Tensor:
    """Return encoder feature map at ``stage_index``; ``-1`` = deepest."""
    return encoder(x.contiguous())[stage_index]


def _fg_weighted_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    tau: float,
    temp: float,
) -> torch.Tensor:
    """Scale ``mask`` by ``1 + alpha * sigmoid((target - tau) / temp)``.

    Channel-wise ``amax`` over the logit. ``alpha <= 0`` returns ``mask`` unchanged.
    Counters background-focus collapse on sparse foreground.
    Anatomy-weighted recon precedent: VasoMIM (Huang et al., AAAI 2026).
    Ref: https://github.com/microsoft/SimMIM
    """
    if alpha <= 0.0:
        return mask
    fg_logit = ((target - tau) / temp).amax(dim=1, keepdim=True)
    w = 1.0 + alpha * torch.sigmoid(fg_logit)
    return w * mask


def _per_patch_normalize(target: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Normalise each ``patch_size x patch_size`` tile to mean 0 / std 1.

    Breaks channel-mean shortcut. He et al. (CVPR 2022) §4.2.
    Raises ``ValueError`` if spatial dims do not divide by ``patch_size``.
    Ref: https://github.com/facebookresearch/mae
    """
    if patch_size <= 0:
        return target
    B, C, H, W = target.shape
    p = patch_size
    if H % p != 0 or W % p != 0:
        raise ValueError(
            f"per-patch norm: target {H}x{W} not divisible by patch_size={p}"
        )
    t = target.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
    mean = t.mean(dim=(-1, -2), keepdim=True)
    var = t.var(dim=(-1, -2), keepdim=True, unbiased=False)
    t = (t - mean) / (var + 1e-6).sqrt()
    return t.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W).contiguous()


def _fg_recon_metric(
    pred: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
    *,
    per_patch_target_norm: bool = False,
    target_norm_patch_size: int = 8,
) -> torch.Tensor:
    """Masked L1 on foreground pixels only. Eval-time diagnostic; no grad.

    Foreground: ``target_raw.amax(channels) > threshold``. Custom diagnostic;
    exposes background-focus collapse independently of training-time fg weighting.
    """
    fg = (target_raw.amax(dim=1, keepdim=True) > threshold).to(target_raw.dtype)
    if per_patch_target_norm:
        target_for_err = _per_patch_normalize(target_raw, target_norm_patch_size)
    else:
        target_for_err = target_raw
    wm = mask * fg
    err = (pred - target_for_err).abs() * wm
    denom = wm.sum() * pred.shape[1] + 1e-8
    return err.sum() / denom


def _simmim_recon_l1(
    pred, target, mask,
    fg_alpha: float = 0.0, fg_tau: float = 0.0, fg_temp: float = 1.0,
) -> torch.Tensor:
    """Masked L1 (SimMIM Eq. 1). ``fg_alpha > 0`` applies sigmoid fg-reweighting."""
    wm = _fg_weighted_mask(target, mask, fg_alpha, fg_tau, fg_temp)
    err = (pred - target).abs() * wm
    denom = wm.sum() * pred.shape[1] + 1e-8
    return err.sum() / denom

def simmim_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    kind: str = "l1",
    *,
    fg_alpha: float = 0.0,
    fg_tau: float = 0.0,
    fg_temp: float = 1.0,
    per_patch_target_norm: bool = False,
    target_norm_patch_size: int = 8,
) -> torch.Tensor:
    """Masked reconstruction loss. ``kind='l1'`` only.

    ``per_patch_target_norm``: per-tile mean 0 / std 1 (MAE §4.2).
    ``fg_alpha > 0``: sigmoid fg-reweighting; weighted denom preserves scale.
    Ref: https://github.com/microsoft/SimMIM
    """
    if per_patch_target_norm:
        target = _per_patch_normalize(target, target_norm_patch_size)
    if kind == "l1":
        return _simmim_recon_l1(pred, target, mask, fg_alpha, fg_tau, fg_temp)
    raise ValueError(f"Unknown loss_kind={kind!r}; expected 'l1'.")


def fourier_recon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Per-tile FFT L1 on masked blocks. Counters spatial-domain over-smoothing.

    Inspired by CA-MAE (Kraus et al., CVPR 2024). Diverges: CA-MAE compares
    FFT magnitudes only (phase-invariant); this compares full complex spectra
    ``|F_pred - F_target|`` so within-tile shifts are also penalised. Uses
    ``norm='ortho'``.
    Ref: https://github.com/recursionpharma/maes_microscopy/blob/main/loss.py
    """
    B, C, H, W = pred.shape
    p = block_size
    if H % p or W % p:
        raise ValueError(f"fourier loss: block_size={p} must divide {H}x{W}")
    pt = pred.unfold(2, p, p).unfold(3, p, p)   # (B, C, gh, gw, p, p)
    tt = target.unfold(2, p, p).unfold(3, p, p)
    mt = mask.unfold(2, p, p).unfold(3, p, p).amax(dim=(-1, -2))  # (B, 1, gh, gw)
    Fp = torch.fft.fft2(pt, norm="ortho")
    Ft = torch.fft.fft2(tt, norm="ortho")
    diff = (Fp - Ft).abs() * mt.unsqueeze(-1).unsqueeze(-1)
    denom = mt.sum() * C * p * p + 1e-8
    return diff.sum() / denom


def vicreg_terms(z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4):
    """VICReg sim/std/cov terms on ``(B, D)`` projections. Returns ``(L_sim, L_std, L_cov)``.

    Bardes et al. (ICLR 2022). Forced fp32 — avoids fp16 var/cov overflow.
    Ref: https://github.com/facebookresearch/vicreg
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
    """Joint SimMIM + VICReg + Fourier loss on two views. Returns ``(loss, metrics)``.

    ``w_vicreg == 0`` skips the clean-view encoder pass (projector still runs
    under ``no_grad`` for diagnostics). ``fixed_mask=None`` samples a fresh
    block mask. All loss math runs in fp32 under disabled autocast.
    """
    decoder    = heads["decoder"]
    mask_token = heads["mask_token"]
    projector  = heads["projector"]

    if fixed_mask is None:
        mask = random_block_mask(view1, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio)
    else:
        mask = fixed_mask

    # ── Recon branch: encoder sees MASKED views ──────────────────────
    v1m = apply_mask(view1, mask, mask_token)
    v2m = apply_mask(view2, mask, mask_token)
    z1_masked = _encode_at(encoder, v1m, ssl_cfg.head_stage_index)
    z2_masked = _encode_at(encoder, v2m, ssl_cfg.head_stage_index)
    r1 = decoder(z1_masked)
    r2 = decoder(z2_masked)

    # fp32 forced — avoids inf grads from autocast fp16.
    device_type = z1_masked.device.type
    with torch.amp.autocast(device_type, enabled=False):
        r1f = r1.float()
        r2f = r2.float()
        v1f = view1.float()
        v2f = view2.float()
        mf  = mask.float()
        L_recon = 0.5 * (
            simmim_recon_loss(
                r1f, v1f, mf, ssl_cfg.loss_kind,
                fg_alpha=ssl_cfg.fg_weight_alpha,
                fg_tau=ssl_cfg.fg_weight_tau,
                fg_temp=ssl_cfg.fg_weight_temp,
                per_patch_target_norm=ssl_cfg.per_patch_target_norm,
                target_norm_patch_size=ssl_cfg.target_norm_patch_size,
            )
            + simmim_recon_loss(
                r2f, v2f, mf, ssl_cfg.loss_kind,
                fg_alpha=ssl_cfg.fg_weight_alpha,
                fg_tau=ssl_cfg.fg_weight_tau,
                fg_temp=ssl_cfg.fg_weight_temp,
                per_patch_target_norm=ssl_cfg.per_patch_target_norm,
                target_norm_patch_size=ssl_cfg.target_norm_patch_size,
            )
        )
        L_recon_fg = 0.5 * (
            _fg_recon_metric(
                r1f, v1f, mf, ssl_cfg.fg_metric_threshold,
                per_patch_target_norm=ssl_cfg.per_patch_target_norm,
                target_norm_patch_size=ssl_cfg.target_norm_patch_size,
            )
            + _fg_recon_metric(
                r2f, v2f, mf, ssl_cfg.fg_metric_threshold,
                per_patch_target_norm=ssl_cfg.per_patch_target_norm,
                target_norm_patch_size=ssl_cfg.target_norm_patch_size,
            )
        )

        # Fourier auxiliary loss (CA-MAE, Kraus 2024) on masked tiles.
        if ssl_cfg.w_fourier > 0:
            L_fourier = 0.5 * (
                fourier_recon_loss(r1f, v1f, mf, ssl_cfg.mask_block_size)
                + fourier_recon_loss(r2f, v2f, mf, ssl_cfg.mask_block_size)
            )
        else:
            L_fourier = torch.tensor(0.0, device=view1.device)

        # ── VICReg branch: encoder sees CLEAN views ─────────────────
        if ssl_cfg.w_vicreg > 0:
            z1_clean = _encode_at(encoder, view1.contiguous(), ssl_cfg.head_stage_index)
            z2_clean = _encode_at(encoder, view2.contiguous(), ssl_cfg.head_stage_index)
            p1 = projector(z1_clean.float().mean(dim=(-2, -1)))
            p2 = projector(z2_clean.float().mean(dim=(-2, -1)))
            L_sim, L_std, L_cov = vicreg_terms(p1, p2)
            L_vicreg = (
                ssl_cfg.lambda_sim * L_sim
                + ssl_cfg.lambda_std * L_std
                + ssl_cfg.lambda_cov * L_cov
            )
        else:
            # VICReg disabled — run projector on masked features for
            # diagnostic logging only (no grad contribution to loss).
            with torch.no_grad():
                p1 = projector(z1_masked.float().mean(dim=(-2, -1)))
                p2 = projector(z2_masked.float().mean(dim=(-2, -1)))
                L_sim, L_std, L_cov = vicreg_terms(p1, p2)
            L_vicreg = torch.tensor(0.0, device=view1.device)

        loss = (
            ssl_cfg.w_recon * L_recon
            + ssl_cfg.w_vicreg * L_vicreg
            + ssl_cfg.w_fourier * L_fourier
        )

    metrics = {
        "ssl_loss": float(loss.detach().item()),
        "recon":    float(L_recon.detach().item()),
        "recon_fg": float(L_recon_fg.detach().item()),
        "fourier":  float(L_fourier.detach().item()),
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
    """Single-view validation step. Returns a metrics dict.

    Computes ``recon``, ``recon_fg``, ``fourier``, ``std``, ``cov``. ``sim`` is
    omitted (needs two views). ``std`` and ``cov`` serve as collapse diagnostics.
    """
    decoder    = heads["decoder"]
    mask_token = heads["mask_token"]
    projector  = heads["projector"]

    # ── Recon branch: masked view ────────────────────────────────────
    mask = random_block_mask(batch, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio)
    v_masked = apply_mask(batch, mask, mask_token)
    z_masked = _encode_at(encoder, v_masked, ssl_cfg.head_stage_index)
    recon = decoder(z_masked)
    L_recon = simmim_recon_loss(
        recon, batch, mask, ssl_cfg.loss_kind,
        fg_alpha=ssl_cfg.fg_weight_alpha,
        fg_tau=ssl_cfg.fg_weight_tau,
        fg_temp=ssl_cfg.fg_weight_temp,
        per_patch_target_norm=ssl_cfg.per_patch_target_norm,
        target_norm_patch_size=ssl_cfg.target_norm_patch_size,
    )
    L_recon_fg = _fg_recon_metric(
        recon, batch, mask, ssl_cfg.fg_metric_threshold,
        per_patch_target_norm=ssl_cfg.per_patch_target_norm,
        target_norm_patch_size=ssl_cfg.target_norm_patch_size,
    )
    L_fourier = fourier_recon_loss(
        recon.float(), batch.float(), mask.float(), ssl_cfg.mask_block_size,
    )

    # ── VICReg diagnostics: clean view + GeM pooling ──────────────
    z_clean = _encode_at(encoder, batch.contiguous(), ssl_cfg.head_stage_index)
    p = projector(z_clean.float().mean(dim=(-2, -1))).float()
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
        "recon_fg": float(L_recon_fg.item()),
        "fourier":  float(L_fourier.item()),
        "std":      float(L_std.item()),
        "cov":      float(L_cov.item()),
    }
