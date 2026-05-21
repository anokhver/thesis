"""Dice + BCE losses for binary segmentation.

All three callables accept an optional ``loss_mask`` (same shape as
``target``, values in ``{0, 1}``). Pixels where ``loss_mask == 0`` are
ignored: they contribute neither to Dice sums nor to the BCE mean. With
``loss_mask=None`` (default) behaviour is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        if loss_mask is not None:
            pred = pred * loss_mask
            target = target * loss_mask
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Weighted sum of Dice and BCE.

    ``loss = dice_weight * SoftDiceLoss + bce_weight * BCEWithLogitsLoss``.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        # BCE uses 'none' reduction so a loss_mask can be applied per pixel;
        # plain mean is recovered below when loss_mask is None.
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict:
        dice = self.dice_loss(logits, target, loss_mask=loss_mask)
        bce_pixel = self.bce_loss(logits, target)
        if loss_mask is None:
            bce = bce_pixel.mean()
        else:
            B = bce_pixel.size(0)
            bce_pixel = bce_pixel * loss_mask
            denom = loss_mask.view(B, -1).sum(dim=1).clamp_min(1.0)
            bce = (bce_pixel.view(B, -1).sum(dim=1) / denom).mean()
        total = self.dice_weight * dice + self.bce_weight * bce
        return {
            "loss": total,
            "dice_loss": dice,
            "bce_loss": bce,
        }


def compute_dice_metric(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hard Dice coefficient (metric, not loss) for a batch."""
    pred = (torch.sigmoid(logits) > threshold).float()
    if loss_mask is not None:
        pred = pred * loss_mask
        target = target * loss_mask
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return dice.mean()


class JointChannelDiceBCE(nn.Module):
    """Per-channel ``DiceBCELoss`` summed over a 2-channel output.

    Wraps :class:`DiceBCELoss` and applies it independently to
    ``logits[:, 0:1]`` (PRE) and ``logits[:, 1:2]`` (POST). The total
    loss is the *mean* of the two per-channel losses, matching plan
    ``§6`` (option B.1). Per-channel BCE weights can be tuned via
    ``bce_weight_pre`` / ``bce_weight_post`` (option B.2). Dice weight
    is shared.

    Returns a dict with::

        {
            "loss":      mean of pre/post losses        (the backward target)
            "dice_loss": mean of pre/post dice losses
            "bce_loss":  mean of pre/post bce losses
            "pre_loss":  per-channel PRE total loss
            "post_loss": per-channel POST total loss
        }

    ``loss_mask`` may be ``None``, ``(B, 1, H, W)`` (broadcast across
    channels), or ``(B, 2, H, W)`` (channel-specific masking).
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        smooth: float = 1.0,
        *,
        bce_weight_pre: float | None = None,
        bce_weight_post: float | None = None,
    ):
        super().__init__()
        if bce_weight_pre is None:
            bce_weight_pre = bce_weight
        if bce_weight_post is None:
            bce_weight_post = bce_weight
        self.pre_loss = DiceBCELoss(
            dice_weight=dice_weight, bce_weight=bce_weight_pre, smooth=smooth,
        )
        self.post_loss = DiceBCELoss(
            dice_weight=dice_weight, bce_weight=bce_weight_post, smooth=smooth,
        )

    @staticmethod
    def _slice_loss_mask(loss_mask: torch.Tensor | None, ch: int) -> torch.Tensor | None:
        if loss_mask is None:
            return None
        if loss_mask.size(1) == 1:
            return loss_mask
        if loss_mask.size(1) == 2:
            return loss_mask[:, ch : ch + 1]
        raise ValueError(
            f"loss_mask must have 1 or 2 channels; got {loss_mask.size(1)}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict:
        if logits.size(1) != 2 or target.size(1) != 2:
            raise ValueError(
                f"JointChannelDiceBCE expects 2-channel logits/target; "
                f"got logits={tuple(logits.shape)} target={tuple(target.shape)}"
            )
        out_pre = self.pre_loss(
            logits[:, 0:1], target[:, 0:1],
            loss_mask=self._slice_loss_mask(loss_mask, 0),
        )
        out_post = self.post_loss(
            logits[:, 1:2], target[:, 1:2],
            loss_mask=self._slice_loss_mask(loss_mask, 1),
        )
        total = 0.5 * (out_pre["loss"] + out_post["loss"])
        return {
            "loss":      total,
            "dice_loss": 0.5 * (out_pre["dice_loss"] + out_post["dice_loss"]),
            "bce_loss":  0.5 * (out_pre["bce_loss"]  + out_post["bce_loss"]),
            "pre_loss":  out_pre["loss"],
            "post_loss": out_post["loss"],
        }


def compute_dice_metric_per_channel(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-channel hard Dice; returns ``(C,)`` tensor (mean over batch).

    Works for any number of output channels. ``loss_mask`` semantics
    match :class:`JointChannelDiceBCE` (``None``, ``(B,1,H,W)`` or
    ``(B,C,H,W)``).
    """
    pred = (torch.sigmoid(logits) > threshold).float()
    if loss_mask is not None:
        pred = pred * loss_mask
        target = target * loss_mask
    B, C = pred.size(0), pred.size(1)
    pred_flat = pred.view(B, C, -1)
    target_flat = target.view(B, C, -1)
    intersection = (pred_flat * target_flat).sum(dim=2)
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum(dim=2) + target_flat.sum(dim=2) + smooth
    )
    return dice.mean(dim=0)  # (C,)
