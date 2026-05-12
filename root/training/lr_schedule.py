"""Build layer-wise LR decay and warmup/cosine schedules.

Adapt timm / MAE ``util/lr_decay.py`` to MONAI ``SwinTransformer``
parameter naming (``layers1.0...layers4.0``).
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
    """Build per-stage AdamW param groups for a MONAI ``SwinTransformer``.

    Stage assignment::

        depth 0 : patch_embed
        depth d : layers{d}            (1..4)
        depth 5 : everything else (final norm, etc.)

    Decay multiplier: ``layer_decay ** (5 - d)``.
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
    """Apply linear warmup for ``warmup_epochs`` then cosine decay to 0."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda
