"""Read-only view of the training-side configuration for the overview page.

The training notebooks own the canonical config files; this module just loads
and groups them for display. Per the repo rules, no value here drives
training — it only mirrors what training already does.
"""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings

PRETRAIN_CONFIG_PATH: Path = (
    Path(settings.BASE_DIR)
    / "notebooks"
    / "training"
    / "checkpoints"
    / "pretrain_ae_config.json"
)

ARCHITECTURE_KEYS: tuple[str, ...] = (
    "in_channels",
    "img_size",
    "feature_size",
    "spatial_dims",
    "patch_size",
    "token_grid",
    "window_size",
    "enc_channels",
    "enc_spatial",
    "n_dec_upsample",
    "dropout_path_rate",
    "use_checkpoint",
)

TRAINING_KEYS: tuple[str, ...] = (
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "bright_weight",
)

REPRODUCIBILITY_KEYS: tuple[str, ...] = (
    "seed",
    "val_split",
)

PREPROCESSING_DEFAULTS: dict[str, str] = {
    "Z-handling": "Maximum Intensity Projection along Z",
    "Per-channel normalization": "percentile [1.0, 99.8], clipped to [0, 1]",
    "Patch size": "128 × 128 px",
    "Patch tiling": "non-overlapping",
    "Empty-patch filter": "mean intensity ≥ 0.01",
    "Input channels": "3 (pre-synaptic, post-synaptic, structural)",
}


def load_pretrain_config() -> dict | None:
    """Return the parsed pretraining config, or None if it is missing/invalid."""
    if not PRETRAIN_CONFIG_PATH.exists():
        return None
    try:
        with PRETRAIN_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def grouped_pretrain_config() -> list[tuple[str, list[tuple[str, object]]]]:
    """Return the pretrain config split into display sections.

    Each entry is `(section_title, [(key, value), ...])`. Sections with no
    matching keys are dropped, and any keys not in a known section are
    collected into a trailing "Other" section so nothing is silently lost.
    """
    config = load_pretrain_config()
    if not config:
        return []

    sections: list[tuple[str, tuple[str, ...]]] = [
        ("Architecture", ARCHITECTURE_KEYS),
        ("Training", TRAINING_KEYS),
        ("Reproducibility", REPRODUCIBILITY_KEYS),
    ]
    seen: set[str] = set()
    out: list[tuple[str, list[tuple[str, object]]]] = []
    for title, keys in sections:
        rows = [(k, config[k]) for k in keys if k in config]
        seen.update(k for k, _ in rows)
        if rows:
            out.append((title, rows))

    leftover = [(k, v) for k, v in config.items() if k not in seen]
    if leftover:
        out.append(("Other", leftover))
    return out
