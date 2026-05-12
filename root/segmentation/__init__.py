"""Shared segmentation utilities for the notebooks under ``root/notebooks/segmentation``.

Provides segmentation-specific config, dataset, losses, model builder,
augmentation, sliding-window inference, and visualisation on top of the
existing ``training`` infrastructure.
"""

from .config import SegTrainCfg
from .dataset import PseudoLabelSegDataset
from .losses import DiceBCELoss, SoftDiceLoss, compute_dice_metric
from .model import build_swinunetr, load_pretrained_encoder_into_swinunetr, count_params
from .augment import SegTrainTransform, SegValTransform
from .inference import sliding_window_predict, predict_full_image
from .viz import (
    plot_seg_overlay,
    plot_seg_comparison,
    plot_seg_curves,
    plot_full_image_result,
    SEG_LOSS_LABELS,
)

__all__ = [
    "SegTrainCfg",
    "PseudoLabelSegDataset",
    "DiceBCELoss", "SoftDiceLoss", "compute_dice_metric",
    "build_swinunetr", "load_pretrained_encoder_into_swinunetr", "count_params",
    "SegTrainTransform", "SegValTransform",
    "sliding_window_predict", "predict_full_image",
    "plot_seg_overlay", "plot_seg_comparison", "plot_seg_curves",
    "plot_full_image_result", "SEG_LOSS_LABELS",
]
