"""JSON config loading for SSL pretraining scripts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from ..training.config import BaseCfg, DataCfg, ModelCfg, TrainCfg, SSLCfg

from .unfreeze import default_unfreeze_schedule


def _unknown_keys(cls, raw: dict | None) -> list[str]:
    if raw is None:
        return []
    fields = {f.name for f in dataclasses.fields(cls)}
    return sorted(set(raw) - fields)


def _build_dataclass(cls, raw: dict | None):
    """Instantiate a dataclass, ignoring unknown keys."""
    if raw is None:
        return cls()
    fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in fields}
    # Convert lists to tuples for tuple-typed fields.
    for f in dataclasses.fields(cls):
        if f.name in filtered and hasattr(f.type, "__origin__") and f.type.__origin__ is tuple:
            filtered[f.name] = tuple(filtered[f.name])
    return cls(**filtered)


def _resolve_path(p: str | None, base: Path) -> str | None:
    """Resolve a relative path against *base*."""
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((base / pp).resolve())


def load_config(config_path: str | Path, root_dir: str | Path) -> dict[str, Any]:
    """Load a JSON config and build all dataclass objects.

    Relative paths in the config are resolved against ``root_dir`` (the
    ``root/`` directory of the repo). Missing top-level keys fall back to
    sensible defaults. The unfreeze schedule defaults depend on
    ``base.init_source``: scratch runs honour ``train.freeze_encoder_epochs``
    and unfreeze the whole encoder afterwards, while pretrained inits use a
    top-down progressive schedule.
    """
    root_dir = Path(root_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    top_allowed = {
        "run_sanity", "run_overfit", "base", "data", "model", "train", "ssl",
        "unfreeze_schedule", "full_image",
    }
    top_unknown = sorted(set(raw) - top_allowed)
    if top_unknown:
        raise ValueError(
            f"Unknown top-level config keys in {config_path}: {top_unknown}."
        )

    section_unknown = {
        "base": _unknown_keys(BaseCfg, raw.get("base")),
        "data": _unknown_keys(DataCfg, raw.get("data")),
        "model": _unknown_keys(ModelCfg, raw.get("model")),
        "train": _unknown_keys(TrainCfg, raw.get("train")),
        "ssl": _unknown_keys(SSLCfg, raw.get("ssl")),
    }
    bad_sections = {k: v for k, v in section_unknown.items() if v}
    if bad_sections:
        details = ", ".join(f"{k}: {v}" for k, v in bad_sections.items())
        raise ValueError(f"Unknown config keys in {config_path}: {details}")

    base_cfg  = _build_dataclass(BaseCfg,  raw.get("base"))
    data_cfg  = _build_dataclass(DataCfg,  raw.get("data"))
    model_cfg = _build_dataclass(ModelCfg, raw.get("model"))
    train_cfg = _build_dataclass(TrainCfg, raw.get("train"))
    ssl_cfg   = _build_dataclass(SSLCfg,   raw.get("ssl"))

    # Resolve paths relative to root/.
    base_cfg.output_root          = _resolve_path(base_cfg.output_root, root_dir)
    base_cfg.pretrained_ckpt_path = _resolve_path(base_cfg.pretrained_ckpt_path, root_dir)
    base_cfg.resume_path          = _resolve_path(base_cfg.resume_path, root_dir)
    data_cfg.data_root            = _resolve_path(data_cfg.data_root, root_dir)
    data_cfg.exclude_patterns_file = _resolve_path(data_cfg.exclude_patterns_file, root_dir)

    # Merge externally defined exclude patterns (file overrides only append).
    if data_cfg.exclude_patterns_file:
        epf = Path(data_cfg.exclude_patterns_file)
        if not epf.is_file():
            raise ValueError(
                f"data.exclude_patterns_file does not exist: {epf}"
            )
        with open(epf, "r", encoding="utf-8") as f:
            ep_raw = json.load(f)
        if isinstance(ep_raw, dict):
            extra = ep_raw.get("exclude_patterns", [])
        elif isinstance(ep_raw, list):
            extra = ep_raw
        else:
            raise ValueError(
                f"{epf} must be a JSON list or an object with key 'exclude_patterns'."
            )
        if not all(isinstance(p, str) for p in extra):
            raise ValueError(
                f"{epf}: all entries in 'exclude_patterns' must be strings."
            )
        # Dedup, preserve order (inlined entries first, file additions after).
        merged: list[str] = []
        seen: set[str] = set()
        for p in list(data_cfg.exclude_patterns) + list(extra):
            if p not in seen:
                seen.add(p)
                merged.append(p)
        data_cfg.exclude_patterns = merged

    # Script-specific settings (optional, with defaults).
    run_sanity  = raw.get("run_sanity", True)
    run_overfit = raw.get("run_overfit", True)
    unfreeze    = raw.get("unfreeze_schedule") or default_unfreeze_schedule(
        base_cfg.init_source, train_cfg.freeze_encoder_epochs,
    )
    full_image  = raw.get("full_image", {"enabled": False})

    # Tuple fields that JSON stores as lists.
    if isinstance(model_cfg.depths, list):
        model_cfg.depths = tuple(model_cfg.depths)
    if isinstance(model_cfg.num_heads, list):
        model_cfg.num_heads = tuple(model_cfg.num_heads)

    if not (0.0 <= data_cfg.val_split < 1.0):
        raise ValueError(f"data.val_split must be in [0, 1), got {data_cfg.val_split}.")
    if data_cfg.batch_size <= 0:
        raise ValueError(f"data.batch_size must be > 0, got {data_cfg.batch_size}.")
    if data_cfg.num_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {data_cfg.num_workers}.")
    if data_cfg.channel_stats_max_samples <= 0:
        raise ValueError(
            f"data.channel_stats_max_samples must be > 0, got {data_cfg.channel_stats_max_samples}."
        )
    if len(data_cfg.channel_names) != model_cfg.in_channels:
        raise ValueError(
            "len(data.channel_names) must equal model.in_channels "
            f"({len(data_cfg.channel_names)} vs {model_cfg.in_channels})."
        )

    if train_cfg.epochs <= 0:
        raise ValueError(f"train.epochs must be > 0, got {train_cfg.epochs}.")
    if train_cfg.warmup_epochs < 0:
        raise ValueError(f"train.warmup_epochs must be >= 0, got {train_cfg.warmup_epochs}.")
    if train_cfg.freeze_encoder_epochs < 0:
        raise ValueError(
            f"train.freeze_encoder_epochs must be >= 0, got {train_cfg.freeze_encoder_epochs}."
        )
    if train_cfg.val_metric_direction not in {"min", "max"}:
        raise ValueError(
            "train.val_metric_direction must be 'min' or 'max', got "
            f"{train_cfg.val_metric_direction!r}."
        )

    valid_val_metrics = {"ssl_loss", "recon", "recon_fg", "fourier", "std", "cov", "vicreg"}
    if train_cfg.val_metric_key not in valid_val_metrics:
        raise ValueError(
            "train.val_metric_key must be one of "
            f"{sorted(valid_val_metrics)}, got {train_cfg.val_metric_key!r}."
        )

    if not (0.0 <= ssl_cfg.mask_ratio < 1.0):
        raise ValueError(f"ssl.mask_ratio must be in [0, 1), got {ssl_cfg.mask_ratio}.")
    if ssl_cfg.mask_block_size <= 0:
        raise ValueError(
            f"ssl.mask_block_size must be > 0, got {ssl_cfg.mask_block_size}."
        )

    n_stages = len(model_cfg.depths) + 1
    head_idx = (
        ssl_cfg.head_stage_index
        if ssl_cfg.head_stage_index >= 0
        else n_stages + ssl_cfg.head_stage_index
    )
    if not (0 <= head_idx < n_stages):
        raise ValueError(
            f"ssl.head_stage_index={ssl_cfg.head_stage_index} is out of range for "
            f"{n_stages} encoder stages (model.depths={model_cfg.depths})."
        )

    return dict(
        base_cfg=base_cfg,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        ssl_cfg=ssl_cfg,
        run_sanity=run_sanity,
        run_overfit=run_overfit,
        unfreeze_schedule=unfreeze,
        full_image=full_image,
    )

