"""SwinUNETR fine-tuning configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegTrainCfg:
    """SwinUNETR fine-tuning hyperparameters."""

    epochs: int = 100
    warmup_epochs: int = 5
    base_lr: float = 1e-4
    decoder_lr: float = 1e-3
    weight_decay: float = 0.05
    layer_decay: float = 0.75
    grad_clip_norm: float = 5.0
    freeze_encoder_epochs: int = 3
    save_every_n_epochs: int = 10
    val_metric_key: str = "dice"
    val_metric_direction: str = "max"
    model_save_name: str = "swinunetr_seg_best.pt"

    # loss weights
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    dice_smooth: float = 1.0

    # loss selection (default keeps the original Dice+BCE behaviour)
    # "dice_bce" -> :class:`JointChannelDiceBCE` (uses dice_weight, bce_weight, dice_smooth)
    # "tversky"  -> :class:`JointChannelTversky` (uses tversky_alpha, tversky_beta, dice_smooth)
    loss_type: str = "dice_bce"
    tversky_alpha: float = 0.3   # FP weight (< beta => false positives forgiven)
    tversky_beta: float = 0.7    # FN weight

    # pseudo-label cache
    pseudolabel_cache_dir: str | None = None
