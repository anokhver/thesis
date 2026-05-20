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
