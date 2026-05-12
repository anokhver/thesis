"""Define segmentation training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegTrainCfg:
    """Store SwinUNETR fine-tuning settings."""

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

    # pseudo-label cache
    pseudolabel_cache_dir: str | None = None
