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


class TverskyLoss(nn.Module):
    """Differentiable Tversky loss for binary segmentation.

    ``T = TP / (TP + alpha * FP + beta * FN)`` (per-batch-item, then mean).

    - ``alpha = beta = 0.5`` reduces to Dice.
    - ``alpha < beta`` makes false positives cheaper than false negatives,
      i.e. the model is **less** punished for predicting puncta the
      pseudo-labels missed. Use this when you trust the labelled positives
      but suspect the pseudo-labels have missed real ones (typical
      pseudo-label noise regime).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)

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
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
        fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky.mean()


class JointChannelTversky(nn.Module):
    """Per-channel :class:`TverskyLoss` summed over a 2-channel output.

    Mirrors :class:`JointChannelDiceBCE`: applies an independent Tversky
    loss to PRE (channel 0) and POST (channel 1), returns the mean.

    The same ``loss_mask`` conventions apply: ``None``, ``(B, 1, H, W)``
    (broadcast across channels), or ``(B, 2, H, W)`` (channel-specific).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.pre_loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.post_loss = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    @staticmethod
    def _slice_loss_mask(
        loss_mask: torch.Tensor | None, ch: int
    ) -> torch.Tensor | None:
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
                f"JointChannelTversky expects 2-channel logits/target; "
                f"got logits={tuple(logits.shape)} target={tuple(target.shape)}"
            )
        pre_loss = self.pre_loss(
            logits[:, 0:1], target[:, 0:1],
            loss_mask=self._slice_loss_mask(loss_mask, 0),
        )
        post_loss = self.post_loss(
            logits[:, 1:2], target[:, 1:2],
            loss_mask=self._slice_loss_mask(loss_mask, 1),
        )
        total = 0.5 * (pre_loss + post_loss)
        # Provide the same dict shape as JointChannelDiceBCE so the
        # training loop's running-metric dict ("dice_loss", "bce_loss")
        # accumulators don't crash. Tversky has no Dice/BCE breakdown,
        # so we put the total loss in the "dice_loss" slot and zero in
        # "bce_loss" (purely cosmetic; CSV columns stay populated).
        zero = total.detach() * 0.0
        return {
            "loss":      total,
            "dice_loss": total.detach(),
            "bce_loss":  zero,
            "pre_loss":  pre_loss,
            "post_loss": post_loss,
        }


class JointChannelTverskyBCE(nn.Module):
    """Per-channel Tversky + BCE for a 2-channel output.

    Mirrors :class:`JointChannelDiceBCE` but swaps Soft-Dice for
    :class:`TverskyLoss`. The per-channel total is
    ``tversky_weight * Tversky + bce_weight * BCE`` so the BCE term
    provides the per-pixel gradient signal that pure Tversky can lose
    when the prediction saturates to all-zero on sparse targets — the
    failure mode observed in ``training_outputs/joint_2ch/*tversky*``
    runs where ``grad_norm`` collapsed to ~1e-9 after the encoder
    unfreezing step.

    BCE supports an optional ``loss_mask`` with the same conventions as
    :class:`JointChannelDiceBCE` (``None``, ``(B,1,H,W)`` broadcast, or
    ``(B,2,H,W)`` channel-specific). Per-channel BCE weights can be
    tuned via ``bce_weight_pre`` / ``bce_weight_post`` for class
    imbalance.

    Returned dict keys match :class:`JointChannelDiceBCE` so the
    training loop's running-metric accumulators continue to work
    unchanged; ``dice_loss`` holds the Tversky term.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        *,
        tversky_weight: float = 1.0,
        bce_weight: float = 1.0,
        bce_weight_pre: float | None = None,
        bce_weight_post: float | None = None,
    ):
        super().__init__()
        if bce_weight_pre is None:
            bce_weight_pre = bce_weight
        if bce_weight_post is None:
            bce_weight_post = bce_weight
        self.tversky_weight = float(tversky_weight)
        self.bce_weight_pre = float(bce_weight_pre)
        self.bce_weight_post = float(bce_weight_post)
        self.tversky_pre = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        self.tversky_post = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
        # 'none' reduction so loss_mask can be applied per pixel.
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _slice_loss_mask(
        loss_mask: torch.Tensor | None, ch: int
    ) -> torch.Tensor | None:
        if loss_mask is None:
            return None
        if loss_mask.size(1) == 1:
            return loss_mask
        if loss_mask.size(1) == 2:
            return loss_mask[:, ch : ch + 1]
        raise ValueError(
            f"loss_mask must have 1 or 2 channels; got {loss_mask.size(1)}"
        )

    def _bce_term(
        self,
        logits_c: torch.Tensor,
        target_c: torch.Tensor,
        loss_mask_c: torch.Tensor | None,
    ) -> torch.Tensor:
        bce_pixel = self.bce(logits_c, target_c)
        if loss_mask_c is None:
            return bce_pixel.mean()
        B = bce_pixel.size(0)
        bce_pixel = bce_pixel * loss_mask_c
        denom = loss_mask_c.view(B, -1).sum(dim=1).clamp_min(1.0)
        return (bce_pixel.view(B, -1).sum(dim=1) / denom).mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict:
        if logits.size(1) != 2 or target.size(1) != 2:
            raise ValueError(
                f"JointChannelTverskyBCE expects 2-channel logits/target; "
                f"got logits={tuple(logits.shape)} target={tuple(target.shape)}"
            )
        m_pre = self._slice_loss_mask(loss_mask, 0)
        m_post = self._slice_loss_mask(loss_mask, 1)

        tv_pre = self.tversky_pre(logits[:, 0:1], target[:, 0:1], loss_mask=m_pre)
        tv_post = self.tversky_post(logits[:, 1:2], target[:, 1:2], loss_mask=m_post)
        bce_pre = self._bce_term(logits[:, 0:1], target[:, 0:1], m_pre)
        bce_post = self._bce_term(logits[:, 1:2], target[:, 1:2], m_post)

        pre_loss = self.tversky_weight * tv_pre + self.bce_weight_pre * bce_pre
        post_loss = self.tversky_weight * tv_post + self.bce_weight_post * bce_post
        total = 0.5 * (pre_loss + post_loss)
        return {
            "loss":      total,
            "dice_loss": 0.5 * (tv_pre + tv_post),   # Tversky term (kept under
                                                     # the 'dice_loss' key so
                                                     # CSV columns stay aligned)
            "bce_loss":  0.5 * (bce_pre + bce_post),
            "pre_loss":  pre_loss,
            "post_loss": post_loss,
        }
