"""Segmentation config, dataset, losses, model, augmentation, inference, viz."""

from .config import SegTrainCfg
from .dataset import PseudoLabelSegDataset, JointChannelSegDataset
from .losses import (
    DiceBCELoss, SoftDiceLoss, compute_dice_metric,
    JointChannelDiceBCE, compute_dice_metric_per_channel,
)
from .model import build_swinunetr, load_pretrained_encoder_into_swinunetr, count_params
from .augment import SegTrainTransform, SegValTransform
from .inference import (
    sliding_window_predict, predict_full_image,
    sliding_window_predict_multichannel, predict_full_image_multichannel,
)
from .viz import (
    plot_seg_overlay,
    plot_seg_comparison,
    plot_seg_curves,
    plot_full_image_result,
    SEG_LOSS_LABELS,
)
from .pseudolabel_io import extract_pseudolabel_archive, discover_full_image_masks
from .colocalisation import SynapseColocCfg, build_synapse_mask

__all__ = [
    "SegTrainCfg",
    "PseudoLabelSegDataset", "JointChannelSegDataset",
    "DiceBCELoss", "SoftDiceLoss", "compute_dice_metric",
    "JointChannelDiceBCE", "compute_dice_metric_per_channel",
    "build_swinunetr", "load_pretrained_encoder_into_swinunetr", "count_params",
    "SegTrainTransform", "SegValTransform",
    "sliding_window_predict", "predict_full_image",
    "sliding_window_predict_multichannel", "predict_full_image_multichannel",
    "plot_seg_overlay", "plot_seg_comparison", "plot_seg_curves",
    "plot_full_image_result", "SEG_LOSS_LABELS",
    "extract_pseudolabel_archive", "discover_full_image_masks",
    "SynapseColocCfg", "build_synapse_mask",
]
