"""Segmentation visualisation helpers.

Returns Matplotlib figures. Does not call ``plt.show()``.
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


SEG_LOSS_LABELS: dict[str, str] = {
    "train_loss":     r"$\mathcal{L}_\mathrm{total}$ (train)",
    "val_loss":       r"$\mathcal{L}_\mathrm{total}$ (val)",
    "train_dice_loss": r"$\mathcal{L}_\mathrm{Dice}$ (train)",
    "train_bce_loss": r"$\mathcal{L}_\mathrm{BCE}$ (train)",
    "val_dice":       r"Dice (val)",
    "val_metric":     r"Dice (val)",
    "lr_encoder":     r"lr (encoder)",
    "lr_decoder":     r"lr (decoder)",
}


def _to_display(img: np.ndarray | torch.Tensor, ch: int = 0) -> np.ndarray:
    """Reduce ``(C, H, W)`` to ``(H, W)`` float32 single-channel."""
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 3:
        img = img[ch]
    return img.astype(np.float32)


def _normalise_01(img: np.ndarray) -> np.ndarray:
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def plot_seg_overlay(
    image: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    prediction: np.ndarray | torch.Tensor | None = None,
    *,
    channel: int = 0,
    threshold: float = 0.5,
    alpha: float = 0.35,
    title: str = "",
    save_to: str | Path | None = None,
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """Plot image, ground-truth, and optional prediction as overlays."""
    img = _normalise_01(_to_display(image, channel))
    gt = _to_display(mask, 0) if mask is not None else None
    n_cols = 2 if prediction is None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image (ch {})".format(channel))
    axes[0].axis("off")

    if gt is not None:
        axes[1].imshow(img, cmap="gray")
        gt_mask = gt > 0.5
        overlay = np.zeros((*img.shape, 4))
        overlay[gt_mask] = [0, 1, 0, alpha]
        axes[1].imshow(overlay)
        n_px = int(gt_mask.sum())
        axes[1].set_title(f"Pseudo-label ({n_px} px)")
        axes[1].axis("off")

    if prediction is not None:
        pred_arr = _to_display(prediction, 0)
        pred_bin = pred_arr > threshold
        axes[2].imshow(img, cmap="gray")
        overlay_pred = np.zeros((*img.shape, 4))
        overlay_pred[pred_bin] = [1, 0, 0, alpha]
        axes[2].imshow(overlay_pred)
        n_pred = int(pred_bin.sum())
        axes[2].set_title(f"Prediction ({n_pred} px)")
        axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_seg_comparison(
    image: np.ndarray,
    pseudolabel: np.ndarray,
    prediction: np.ndarray,
    *,
    channel_names: Sequence[str] = ("pre", "post", "structural"),
    threshold: float = 0.5,
    alpha: float = 0.4,
    title: str = "",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Plot per-channel image, pseudo-label, prediction, and residual."""
    C = min(image.shape[0], len(channel_names))
    fig, axes = plt.subplots(2, C + 1, figsize=(4 * (C + 1), 8))

    # top row: each channel + pseudo-label overlay on ch0
    for c in range(C):
        ch_img = _normalise_01(image[c])
        axes[0, c].imshow(ch_img, cmap="gray")
        axes[0, c].set_title(channel_names[c])
        axes[0, c].axis("off")

    gt = pseudolabel if pseudolabel.ndim == 2 else pseudolabel[0]
    img0 = _normalise_01(image[0])
    axes[0, C].imshow(img0, cmap="gray")
    overlay = np.zeros((*img0.shape, 4))
    overlay[gt > 0.5] = [0, 1, 0, alpha]
    axes[0, C].imshow(overlay)
    axes[0, C].set_title("Pseudo-label")
    axes[0, C].axis("off")

    # bottom row: prediction probability + binary + overlay + residual
    pred = prediction if prediction.ndim == 2 else prediction[0]
    pred_bin = pred > threshold

    axes[1, 0].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("Pred probability")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_bin, cmap="gray")
    axes[1, 1].set_title(f"Pred binary (τ={threshold})")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img0, cmap="gray")
    overlay_pred = np.zeros((*img0.shape, 4))
    overlay_pred[pred_bin] = [1, 0, 0, alpha]
    axes[1, 2].imshow(overlay_pred)
    axes[1, 2].set_title("Pred overlay")
    axes[1, 2].axis("off")

    # difference
    if C + 1 > 3:
        diff = np.abs(gt.astype(np.float32) - pred_bin.astype(np.float32))
        axes[1, 3].imshow(diff, cmap="RdBu_r", vmin=0, vmax=1)
        axes[1, 3].set_title("| GT − Pred |")
        axes[1, 3].axis("off")
    for ax_idx in range(max(4, C + 1), C + 1):
        axes[1, ax_idx].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_seg_curves(
    csv_path: str | Path,
    *,
    run_label: str = "",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Plot loss, Dice, and LR curves from a metrics CSV."""
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r") as f:
        for row in _csv.DictReader(f):
            rows.append(row)
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    epochs = [int(r["epoch"]) for r in rows]

    def _get(key):
        return [float(r[key]) if r.get(key) else float("nan") for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # loss curves
    train_loss = _get("train_loss")
    val_loss = _get("val_loss") if "val_loss" in rows[0] else None
    axes[0].plot(epochs, train_loss, label="train loss")
    if val_loss:
        axes[0].plot(epochs, val_loss, label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # dice curve
    val_dice = _get("val_dice") if "val_dice" in rows[0] else None
    if val_dice:
        axes[1].plot(epochs, val_dice, color="green", label="val Dice")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("Dice")
        axes[1].set_title("Validation Dice")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # LR schedule
    lr_enc = _get("lr_encoder") if "lr_encoder" in rows[0] else None
    lr_dec = _get("lr_decoder") if "lr_decoder" in rows[0] else None
    if lr_enc:
        axes[2].plot(epochs, lr_enc, label="encoder")
    if lr_dec:
        axes[2].plot(epochs, lr_dec, label="decoder")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate")
    axes[2].set_title("LR schedule")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if run_label:
        fig.suptitle(run_label, fontsize=11, y=1.02)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig


def plot_full_image_result(
    full_image: np.ndarray,
    prob_map: np.ndarray,
    pseudolabel: np.ndarray | None = None,
    *,
    channel: int = 0,
    threshold: float = 0.5,
    alpha: float = 0.3,
    title: str = "",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Plot full-image inference: image | prob map | pred overlay (+ optional pseudo-label)."""
    img = _normalise_01(full_image[channel])
    pred_bin = prob_map > threshold
    n_cols = 4 if pseudolabel is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Image (ch {channel})")
    axes[0].axis("off")

    axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Prediction probability")
    axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    overlay = np.zeros((*img.shape, 4))
    overlay[pred_bin] = [1, 0, 0, alpha]
    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred overlay ({int(pred_bin.sum())} px)")
    axes[2].axis("off")

    if pseudolabel is not None:
        gt = pseudolabel if pseudolabel.ndim == 2 else pseudolabel[0]
        axes[3].imshow(img, cmap="gray")
        gt_overlay = np.zeros((*img.shape, 4))
        gt_overlay[gt > 0.5] = [0, 1, 0, alpha]
        axes[3].imshow(gt_overlay)
        axes[3].set_title(f"Pseudo-label ({int((gt > 0.5).sum())} px)")
        axes[3].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
    return fig
