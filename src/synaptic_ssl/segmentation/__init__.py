"""Segmentation config, dataset, losses, model, augmentation, inference, viz."""

from .config import SegTrainCfg, derive_epochs_for_steps, iter1_cfg_from_iter0
from .dataset import PseudoLabelSegDataset
from .dataset_puncta import (
    POST_CHANNEL,
    PRE_CHANNEL,
    PunctaSegDataset,
    positive_patch_fraction,
)
from .losses import DiceBCELoss, SoftDiceLoss, compute_dice_metric
from .model import (
    build_swinunetr,
    count_params,
    load_full_swinunetr_from_ckpt,
    load_pretrained_encoder_into_swinunetr,
)
from .augment import SegTrainTransform, SegValTransform
from .inference import predict_d4_tta, sliding_window_predict, predict_full_image
from .refresh import (
    NEAR_DILATE_PX_DEFAULT,
    POST_AREA_RANGE_DEFAULT,
    PRE_AREA_RANGE_DEFAULT,
    refresh_pseudolabels,
)
from .runner import (
    SegDataState,
    SegModelState,
    SegOptimState,
    build_seg_model,
    run_seg_iter,
    setup_seg_optimizer,
)
from .train_loop import (
    cycle,
    encoder_decoder_lrs,
    freeze_swinvit,
    train_step,
    validate,
)
from .viz import (
    plot_seg_overlay,
    plot_seg_comparison,
    plot_seg_curves,
    plot_full_image_result,
    SEG_LOSS_LABELS,
)

__all__ = [
    "SegTrainCfg", "derive_epochs_for_steps", "iter1_cfg_from_iter0",
    "PseudoLabelSegDataset",
    "PunctaSegDataset", "PRE_CHANNEL", "POST_CHANNEL", "positive_patch_fraction",
    "DiceBCELoss", "SoftDiceLoss", "compute_dice_metric",
    "build_swinunetr",
    "load_pretrained_encoder_into_swinunetr",
    "load_full_swinunetr_from_ckpt",
    "count_params",
    "SegTrainTransform", "SegValTransform",
    "predict_d4_tta", "sliding_window_predict", "predict_full_image",
    "refresh_pseudolabels",
    "PRE_AREA_RANGE_DEFAULT", "POST_AREA_RANGE_DEFAULT", "NEAR_DILATE_PX_DEFAULT",
    "SegDataState", "SegModelState", "SegOptimState",
    "build_seg_model", "setup_seg_optimizer", "run_seg_iter",
    "cycle", "encoder_decoder_lrs", "freeze_swinvit",
    "train_step", "validate",
    "plot_seg_overlay", "plot_seg_comparison", "plot_seg_curves",
    "plot_full_image_result", "SEG_LOSS_LABELS",
]
