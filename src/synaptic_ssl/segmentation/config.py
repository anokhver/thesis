"""SwinUNETR fine-tuning configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class SegTrainCfg:
    """SwinUNETR fine-tuning hyperparameters for **one** self-training iteration.

    The canonical knobs are **step-based** (``train_steps``,
    ``warmup_steps``, ``freeze_encoder_steps``, ``save_every_n_steps``,
    ``val_every_n_steps``) so the same recipe runs at any batch size
    (laptop BS=16 vs cluster BS=128). Defaults follow ``plan.md`` v2:
    75k steps for iter-0 of the two-stage self-training loop. Iter-1
    uses ``iter1_cfg_from_iter0(cfg)``.

    Output head is **2 channels** (PRE, POST sigmoid) by default. The
    legacy single-channel notebook overrides this to 1.

    Epoch-based fields are kept as a back-compat shim for the existing
    ``train_swinunetr_pseudolabels.ipynb`` only; new code uses
    step-based fields.
    """

    # canonical (step-based)
    train_steps: int = 75_000          # iter-0 budget; iter-1 uses ~45_000
    warmup_steps: int = 4_000          # ~5% of train_steps, linear -> cosine to 0
    freeze_encoder_steps: int = 8_000  # encoder frozen ~10% of iter-0 only
    save_every_n_steps: int = 5_000
    val_every_n_steps: int = 2_500

    # head + identity tag
    out_channels: int = 2              # PRE = ch 0, POST = ch 1
    iter_index: int = 0                # 0 = cold start, 1 = self-train refresh

    # optimiser
    base_lr: float = 1e-4
    decoder_lr: float = 5e-4
    weight_decay: float = 0.05
    layer_decay: float = 0.75
    grad_clip_norm: float = 5.0

    # checkpoint / metric
    val_metric_key: str = "dice"
    val_metric_direction: str = "max"
    model_save_name: str = "swinunetr_seg_best.pt"

    # loss weights
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    dice_smooth: float = 1.0

    # loss selection (default keeps the original Dice+BCE behaviour)
    # "dice_bce"    -> :class:`JointChannelDiceBCE`    (uses dice_weight, bce_weight, dice_smooth)
    # "tversky"     -> :class:`JointChannelTversky`     (uses tversky_alpha, tversky_beta, dice_smooth)
    # "tversky_bce" -> :class:`JointChannelTverskyBCE`  (uses tversky_alpha, tversky_beta,
    #                                                    bce_weight, dice_smooth) -- adds a
    #                  per-pixel BCE term to Tversky to avoid the all-zero saturation
    #                  collapse observed in pure-Tversky runs.
    loss_type: str = "dice_bce"
    tversky_alpha: float = 0.3   # FP weight (< beta => false positives forgiven)
    tversky_beta: float = 0.7    # FN weight

    # pseudo-label cache (legacy single-channel path only)
    pseudolabel_cache_dir: str | None = None

    # back-compat (epoch-based); ``train_swinunetr_pseudolabels.ipynb``
    # still reads these. New notebooks should not.
    epochs: int = 100
    warmup_epochs: int = 5
    freeze_encoder_epochs: int = 3
    save_every_n_epochs: int = 10


def derive_epochs_for_steps(target_steps: int, steps_per_epoch: int) -> int:
    """Smallest epoch count ``E`` such that ``E * steps_per_epoch >= target_steps``."""
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
    return max(1, math.ceil(target_steps / steps_per_epoch))


def iter1_cfg_from_iter0(
    iter0_cfg: SegTrainCfg,
    *,
    train_steps: int = 45_000,
    warmup_steps: int = 2_500,
) -> SegTrainCfg:
    """Derive the iter-1 config from iter-0.

    Per ``plan.md`` v2: iter-1 is warm-started from S0, so the encoder
    does not need a frozen ramp-up (``freeze_encoder_steps = 0``).
    Optimiser + scheduler are rebuilt fresh in the runner; this is
    purely the config payload.
    """
    return replace(
        iter0_cfg,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        freeze_encoder_steps=0,
        iter_index=1,
    )
