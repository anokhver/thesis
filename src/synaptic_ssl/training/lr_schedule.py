"""Layer-wise LR decay param groups and warmup+cosine LR schedule.

Adapted from timm / MAE ``util/lr_decay.py`` for MONAI ``SwinTransformer``
parameter naming (``layers1.0 .. layers4.0``).
Ref: https://github.com/huggingface/pytorch-image-models
"""

from __future__ import annotations

import math

import torch.nn as nn


_NO_DECAY_KEYWORDS = (
    "bias", "norm", "relative_position_bias_table",
    "absolute_pos_embed", "mask_token",
)


def param_groups_layer_decay(
    encoder: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float,
    no_decay_keywords: tuple[str, ...] = _NO_DECAY_KEYWORDS,
) -> list[dict]:
    """Build per-stage AdamW param groups with layer-wise LR decay.

    Stages: ``patch_embed`` -> 0, ``layers{1..4}`` -> 1..4, everything else -> 5.
    LR scale per stage: ``layer_decay ** (5 - stage)``. ``no_decay_keywords``
    names skip weight decay.
    """
    n_stages = 5
    scales = [layer_decay ** (n_stages - d) for d in range(n_stages + 1)]

    def assign_stage(name: str) -> int:
        if name.startswith("patch_embed"):
            return 0
        for i in range(1, 5):
            if name.startswith(f"layers{i}"):
                return i
        return 5

    groups: dict[tuple, dict] = {}
    for name, p in encoder.named_parameters():
        if not p.requires_grad:
            continue
        stage = assign_stage(name)
        no_decay = any(kw in name for kw in no_decay_keywords)
        key = (stage, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * scales[stage],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "stage": stage,
                "no_decay": no_decay,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def make_warmup_cosine(warmup_epochs: int, total_epochs: int):
    """Return an ``LambdaLR`` lambda: linear warmup then cosine decay to 0.

    Stepped once per epoch. For per-step schedules use
    ``make_warmup_cosine_steps``.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def make_warmup_cosine_steps(warmup_steps: int, total_steps: int):
    """Return an ``LambdaLR`` lambda stepped once per optimiser step.

    Linear warmup over ``warmup_steps`` then cosine decay to 0 over the
    remaining ``total_steps - warmup_steps``. BS-invariant: use this when
    the same recipe must run at different batch sizes (e.g. laptop vs
    cluster), where epoch count varies but step budget is fixed.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda
