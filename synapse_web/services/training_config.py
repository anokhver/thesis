"""Read-only view of the training-side configuration for the overview page.

The training notebooks own the canonical config files; this module just
loads and groups them for display. Per the repo rules, no value here drives
training — it only mirrors what training already does.

Robust to two schemas:

* **Flat** (frontend-branch baseline):
  ``{"in_channels": 3, "img_size": 128, "epochs": 200, ...}`` — keys are
  grouped via the hard-coded ``ARCHITECTURE_KEYS``/``TRAINING_KEYS``/
  ``REPRODUCIBILITY_KEYS`` lists.

* **Nested** (integration / master layout):
  ``{"base": {...}, "model": {...}, "train": {...}, "ssl": {...}, ...}``
  — each top-level dict becomes its own section.

Override the config path with the ``PRETRAIN_CONFIG_PATH`` env var; otherwise
the first existing candidate from a short search list wins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from django.conf import settings

_BASE = Path(settings.BASE_DIR)

_CANDIDATE_PATHS: tuple[Path, ...] = (
    _BASE / "notebooks" / "training" / "checkpoints" / "pretrain_ae_config.json",
    _BASE / "configs" / "pretrain_moby" / "default.json",
    _BASE / "configs" / "pretrain_scratch" / "default.json",
)


def _pick_config_path() -> Path:
    override = os.environ.get("PRETRAIN_CONFIG_PATH")
    if override:
        return Path(override)
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return p
    return _CANDIDATE_PATHS[0]


PRETRAIN_CONFIG_PATH: Path = _pick_config_path()


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


def _format_value(value: object) -> object:
    """Make a value template-friendly without losing the underlying number."""
    if isinstance(value, list):
        return ", ".join(str(_format_value(v)) for v in value) or "[]"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return value


def _section_title(key: str) -> str:
    mapping = {
        "base": "Base",
        "data": "Data",
        "model": "Architecture",
        "train": "Training",
        "ssl": "SSL Objective",
        "full_image": "Full-image Inference",
        "unfreeze_schedule": "Unfreeze Schedule",
    }
    return mapping.get(key, key.replace("_", " ").title())


def grouped_pretrain_config() -> list[tuple[str, list[tuple[str, object]]]]:
    """Return the pretrain config split into display sections.

    Schema detection:

    * If any top-level value is a dict, treat each top-level dict as a
      section. Non-dict top-level entries (e.g. flags, lists) are collected
      into an "Overview" section.
    * Otherwise treat the whole config as flat and group via the
      ``ARCHITECTURE_KEYS``/``TRAINING_KEYS``/``REPRODUCIBILITY_KEYS`` lists,
      with anything leftover spilled into "Other".
    """
    config = load_pretrain_config()
    if not config:
        return []

    is_nested = any(isinstance(v, dict) for v in config.values())
    if is_nested:
        return _group_nested(config)
    return _group_flat(config)


def _group_nested(config: dict) -> list[tuple[str, list[tuple[str, object]]]]:
    sections: list[tuple[str, list[tuple[str, object]]]] = []
    overview: list[tuple[str, object]] = []
    for key, value in config.items():
        if isinstance(value, dict):
            rows = [(k, _format_value(v)) for k, v in value.items()]
            if rows:
                sections.append((_section_title(key), rows))
        else:
            overview.append((key, _format_value(value)))
    if overview:
        sections.insert(0, ("Overview", overview))
    return sections


def _group_flat(config: dict) -> list[tuple[str, list[tuple[str, object]]]]:
    groupings: list[tuple[str, tuple[str, ...]]] = [
        ("Architecture", ARCHITECTURE_KEYS),
        ("Training", TRAINING_KEYS),
        ("Reproducibility", REPRODUCIBILITY_KEYS),
    ]
    seen: set[str] = set()
    out: list[tuple[str, list[tuple[str, object]]]] = []
    for title, keys in groupings:
        rows = [(k, _format_value(config[k])) for k in keys if k in config]
        seen.update(k for k, _ in rows)
        if rows:
            out.append((title, rows))

    leftover = [(k, _format_value(v)) for k, v in config.items() if k not in seen]
    if leftover:
        out.append(("Other", leftover))
    return out
