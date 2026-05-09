"""Configuration dataclasses used by every notebook template.

The notebooks instantiate these once at the top, then pass them around.
Every knob that controls a run lives here; nothing branches on
``isinstance``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BaseCfg:
    seed: int = 42
    output_root: str = "./outputs"
    experiment_name: str = "experiment"
    tag: str = "template"
    method_name: str = "simmim_vicreg"
    init_source: str = "scratch"
    timm_model_name: str = "swin_tiny_patch4_window7_224"
    pretrained_ckpt_path: str | None = None
    resume_path: str | None = None
    dry_run: bool = False


@dataclass
class DataCfg:
    data_root: str = "../../../data/patches_128"
    exclude_patterns: list[str] = field(default_factory=lambda: ["KONTROLA"])
    val_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    channel_names: list[str] = field(
        default_factory=lambda: ["pre_synaptic", "post_synaptic", "structural"]
    )
    channel_stats_max_samples: int = 4096


@dataclass
class ModelCfg:
    in_channels: int = 3
    spatial_dims: int = 2
    img_size: int = 128
    feature_size: int = 96
    patch_size: int = 4
    window_size: int = 7
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout_path_rate: float = 0.1
    use_checkpoint: bool = False


@dataclass
class TrainCfg:
    epochs: int = 200
    warmup_epochs: int = 10
    base_lr: float = 1.5e-4
    head_lr: float = 1.5e-3
    weight_decay: float = 0.05
    layer_decay: float = 0.75
    grad_clip_norm: float = 5.0
    freeze_encoder_epochs: int = 2
    save_every_n_epochs: int = 10
    val_metric_key: str = "ssl_loss"
    val_metric_direction: str = "min"
    encoder_save_name: str = "pretrained_encoder.pt"


@dataclass
class SSLCfg:
    mask_ratio: float = 0.4
    mask_block_size: int = 16
    loss_kind: str = "l1"
    lambda_sim: float = 25.0
    lambda_std: float = 25.0
    lambda_cov: float = 1.0
    w_recon: float = 1.0
    w_vicreg: float = 1.0
    projector_hidden: int = 1024
    projector_dim: int = 1024


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def dump_config(path: str | Path, **cfgs: Any) -> None:
    """Dump a set of named configs to a JSON file next to the run."""
    payload = {name: _to_jsonable(cfg) for name, cfg in cfgs.items()}
    Path(path).write_text(json.dumps(payload, indent=2))
