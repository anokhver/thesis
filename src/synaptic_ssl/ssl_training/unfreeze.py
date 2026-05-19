"""Progressive encoder-unfreeze schedule for SSL pretraining."""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


# Default progressive unfreeze schedule for transfer-learning init sources
# (MoBY, timm ImageNet, local SimMIM checkpoint). Each entry is
# ``[start_epoch, [groups]]`` where ``groups`` lists which encoder parameter
# groups become trainable from that epoch onwards. The schedule is cumulative
# (later entries override earlier ones based on the current epoch).
DEFAULT_UNFREEZE_SCHEDULE: list[list] = [
    [1, ["random", "layers4"]],
    [2, ["random", "layers4", "layers3"]],
    [3, ["random", "layers4", "layers3", "layers2"]],
    [4, ["random", "layers4", "layers3", "layers2", "layers1"]],
    [5, ["random", "layers4", "layers3", "layers2", "layers1", "patch_embed"]],
]

# Default schedule for ``init_source="scratch"``: train every encoder
# parameter from epoch 1. ``"random"`` resolves to all params for scratch
# (see :func:`models.weight_loading.load_pretrained_into_encoder`), so the
# top-down progressive schedule above would only delay learning needlessly.
DEFAULT_SCRATCH_UNFREEZE_SCHEDULE: list[list] = [
    [1, ["random"]],
]

VALID_GROUPS: set[str] = {
    "random", "layers1", "layers2", "layers3", "layers4", "patch_embed",
}


def default_unfreeze_schedule(
    init_source: str,
    freeze_encoder_epochs: int = 0,
) -> list[list]:
    """Pick a sensible default schedule for the given init source.

    For ``"scratch"`` the schedule honours ``freeze_encoder_epochs``: the
    encoder is frozen for the first N epochs and fully unfrozen afterwards.
    For pretrained inits the standard top-down progressive schedule is used.
    """
    if init_source == "scratch":
        if freeze_encoder_epochs > 0:
            return [
                [1, []],
                [freeze_encoder_epochs + 1, ["random"]],
            ]
        return [list(entry) for entry in DEFAULT_SCRATCH_UNFREEZE_SCHEDULE]
    return [list(entry) for entry in DEFAULT_UNFREEZE_SCHEDULE]


def _param_in_group(name: str, group: str, random_init_names: set[str]) -> bool:
    if group == "random":
        return name in random_init_names
    return name.startswith(group)


def apply_unfreeze_schedule(
    encoder: nn.Module,
    epoch: int,
    schedule: Iterable[Iterable],
    random_init_names: set[str],
) -> list[str]:
    """Set ``requires_grad`` on encoder params based on the schedule.

    Returns the list of currently active group names.
    """
    active: list[str] = []
    for start, groups in schedule:
        if epoch >= start:
            active = list(groups)
    active_set = set(active)
    for name, p in encoder.named_parameters():
        p.requires_grad = any(
            _param_in_group(name, g, random_init_names) for g in active_set
        )
    return active


def make_phase_tag(schedule: Iterable[Iterable]) -> dict[int, str]:
    """Generate short phase labels (1-indexed) for each step of the schedule."""
    tags: dict[int, str] = {}
    for i, (_start, groups) in enumerate(schedule, 1):
        groups = list(groups)
        if not groups:
            tags[i] = "frozen"
        elif set(groups) >= VALID_GROUPS or groups == ["random"]:
            tags[i] = "full"
        else:
            tags[i] = "+".join(g.replace("layers", "L") for g in groups)
    return tags

