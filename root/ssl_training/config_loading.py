"""JSON config loading for SSL pretraining scripts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from training.config import BaseCfg, DataCfg, ModelCfg, TrainCfg, SSLCfg

from .unfreeze import default_unfreeze_schedule


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

