"""Dataclasses for SSL pretraining configuration."""

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
    pretrained_ckpt_path: str | None = None
    resume_path: str | None = None
    dry_run: bool = False


@dataclass
class DataCfg:
    data_root: str = "../../../data/patches_128"
    exclude_patterns: list[str] = field(default_factory=lambda: ["KONTROLA"])
    val_split: float = 0.1
    batch_size: int = 64
    num_workers: int = 1
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
    # MONAI SwinUNETR enforces ``img_size % patch_size**5 == 0`` (see
    # ``swin_unetr.py:_check_input_size``). At ``img_size=128`` the only
    # value that satisfies that constraint is ``patch_size=2`` (since
    # ``2**5 = 32`` divides 128, but ``4**5 = 1024`` does not). Keep
    # pretraining and fine-tuning on the same value so the patch-embed
    # weights actually transfer between phases.
    patch_size: int = 2
    window_size: int = 7
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    # Low stochastic-depth for sparse-foreground fluorescence microscopy:
    # most pixels are background, so heavy DropPath regularisation removes
    # too much of the limited signal during pretraining.
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = False


@dataclass
class TrainCfg:
    epochs: int = 200
    warmup_epochs: int = 15
    # Base / head LR tuned for MoBY init (Xie et al., 2021): the encoder is
    # already SSL-trained, so a lower base_lr reduces drift away from the
    # contrastive features. From-scratch SimMIM (Xie et al., CVPR 2022) uses
    # ~1.5e-4 base_lr; switch the override in that notebook if needed.
    base_lr: float = 1.0e-4
    head_lr: float = 1.0e-4
    weight_decay: float = 0.05  # AdamW+Swin standard
    # Layer-wise LR decay (LLRD, Clark et al. 2020 / Bao et al. 2022) is an
    # MAE/ViT *fine-tuning* convention; SimMIM-canonical *pretraining* uses
    # uniform LR across layers (layer_decay=1.0). VasoMIM (Huang et al.,
    # AAAI 2026) keeps no LLRD on Swin MIM pretraining.
    layer_decay: float = 1.0
    grad_clip_norm: float = 5.0  # SimMIM (Xie et al., CVPR 2022)
    # Standard transfer-learning warm-up: keep the encoder frozen for the
    # first few epochs so the randomly-initialised SSL heads (decoder +
    # projector) catch up before any encoder gradients flow.
    freeze_encoder_epochs: int = 5
    save_every_n_epochs: int = 25
    val_metric_key: str = "ssl_loss"
    val_metric_direction: str = "min"
    encoder_save_name: str = "pretrained_encoder.pt"


@dataclass
class SSLCfg:
    # SimMIM (Xie et al., CVPR 2022) defaults to 0.6; MAE (He et al., CVPR
    # 2022) to 0.75. 0.50 sits at the low end of the SimMIM-validated 40-70%
    # range -- leaves more unmasked content for the VICReg branch when the
    # joint-embedding loss is on (mask_ratio >= 0.75 hurts std/cov terms
    # because the projector sees too little signal).
    mask_ratio: float = 0.50
    mask_block_size: int = 16
    loss_kind: str = "l1"
    # VICReg internal coefficients (25, 25, 1) are paper-canonical (Bardes
    # et al., ICLR 2022, Table 7) but their *sum* is ~50x the L1 recon term
    # at w_recon=1. No published joint-MIM recipe sums losses at that ratio
    # (iBOT 1:1, SwinUNETR 1:1:1, CAE 1:2). Keep the (25,25,1) ratio inside
    # VICReg -- ratios matter for stability per Bardes Table 7 -- and
    # rescale the whole VICReg branch with ``w_vicreg`` so it matches recon
    # magnitude.
    lambda_sim: float = 25.0
    lambda_std: float = 25.0
    lambda_cov: float = 1.0
    w_recon: float = 1.0
    # VICReg branch OFF by default (pure SimMIM run). Raise to 1.0 to
    # re-enable the joint-embedding loss; see the (25,25,1)-vs-L1 magnitude
    # note above on lambda_*.
    w_vicreg: float = 0.0
    # Fourier-domain auxiliary loss on masked tiles (CA-MAE, Kraus et al.,
    # CVPR 2024, arXiv:2404.10242). Validated on multi-channel fluorescence
    # microscopy to recover high-frequency texture. Note: CA-MAE microscopy
    # ablations used different input normalisation, so transfer to the
    # z-scored sparse inputs here is empirical. Set to 0 to disable.
    w_fourier: float = 0.01
    # Expander (VICReg projector) width / output dimension. Bardes et al.,
    # ICLR 2022, Table 12: top-1 accuracy improves from 55.9% (dim=256) to
    # 68.6% (dim=8192). Width must be >= encoder dim or BN inside the
    # projector trivially satisfies the variance hinge while the encoder
    # collapses. 512 is a compact default; raise above the encoder dim used
    # here (Swin-T stage-4 = feature_size * 8, e.g. 768) when running with
    # ``w_vicreg > 0``.
    projector_hidden: int = 512
    projector_dim: int = 512
    # Foreground-weighted reconstruction. Each masked pixel's error is
    # multiplied by ``1 + alpha * sigmoid((target - tau) / temp)``, with the
    # logit reduced over channels by ``amax`` so a pixel that is bright in
    # any marker is treated as foreground. ``target`` is the per-channel
    # z-scored input, so ``tau=0`` thresholds at the channel mean. The
    # weighted-mean denominator keeps the loss scale roughly invariant to
    # ``alpha``. ``alpha=0`` disables the weighting (recovers plain SimMIM).
    # Motivation: AnatoMask (Li et al., ECCV 2024) and VasoMIM (Huang et
    # al., AAAI 2026) -- on imagery with sparse foreground, plain MSE/L1 is
    # minimised trivially by predicting the background, and pretraining
    # collapses to "all black".
    fg_weight_alpha: float = 0.0
    fg_weight_tau: float = 0.0
    fg_weight_temp: float = 1.0
    # Which encoder feature level feeds the SSL heads (decoder + projector).
    # MONAI's SwinTransformer returns ``len(depths) + 1`` levels; negative
    # indices follow Python semantics (``-1`` = deepest). Keep this explicit
    # in JSON configs to avoid accidental drift between code defaults and
    # experiment settings.
    head_stage_index: int = -1
    # Threshold (in z-scored target space) above which a pixel counts as
    # foreground for the diagnostic recon_fg metric. tau=1 means "one
    # channel std above the channel mean" — a punctum in z-scored inputs.
    fg_metric_threshold: float = 1.0

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
    """Write named configs to ``path`` as JSON."""
    payload = {name: _to_jsonable(cfg) for name, cfg in cfgs.items()}
    Path(path).write_text(json.dumps(payload, indent=2))
