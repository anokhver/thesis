"""Define configuration dataclasses for training runs."""

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
    # Fourier-domain auxiliary loss on masked tiles (CA-MAE, Kraus et al.,
    # CVPR 2024, arXiv:2404.10242). Validated on multi-channel fluorescence
    # microscopy to recover high-frequency texture. ``w_fourier=0`` off.
    w_fourier: float = 0.0
    projector_hidden: int = 1024
    projector_dim: int = 1024
    # Foreground-weighted reconstruction. Each masked pixel's error is
    # multiplied by ``1 + alpha * sigmoid((target - tau) / temp)``, with the
    # logit reduced over channels by ``amax`` so a pixel that is bright in
    # any marker is treated as foreground. ``target`` is the per-channel
    # z-scored input, so ``tau=0`` thresholds at the channel mean. The
    # weighted-mean denominator keeps the loss scale roughly invariant to
    # ``alpha``. ``alpha=0`` disables the weighting (recovers plain SimMIM).
    #
    # Problem motivation: on sparse-foreground medical/microscopy imagery,
    # plain MSE/L1 is trivially minimised by predicting the background,
    # causing pretraining to collapse to "all black". The same failure mode
    # is reported by AnatoMask (Li et al., ECCV 2024) and VasoMIM (Huang
    # et al., AAAI 2026), but those papers address it via *masking-strategy*
    # biasing (AnatoMask: hard-patch ranking; VasoMIM: vessel-aware mask
    # sampling) and/or a *segmentor-consistency* loss (VasoMIM), NOT via
    # per-pixel loss reweighting -- both papers use unweighted MSE for the
    # recon term itself.
    # Mechanism precedent (per-pixel loss reweighting): Focal Loss
    # (Lin et al., ICCV 2017) up-weights hard / minority-class pixels in
    # dense prediction; U-Net (Ronneberger et al., MICCAI 2015) pre-computes
    # a per-pixel weight map ``w(x)`` to emphasise border/sparse pixels in
    # the segmentation loss. The ``1 + alpha * sigmoid`` form here is a
    # custom focal-style reweighting of the SimMIM L1, not directly from
    # AnatoMask/VasoMIM.
    fg_weight_alpha: float = 0.0
    fg_weight_tau: float = 0.0
    fg_weight_temp: float = 1.0
    # Which encoder feature level feeds the SSL heads (decoder + projector).
    # MONAI's SwinTransformer returns ``len(depths) + 1`` levels; -1 is the
    # deepest (and the only one without an ImageNet source when loading
    # timm Swin-T). Use -2 to attach heads at the last fully-pretrained
    # level (768-ch at stride 32 for default Swin-T config).
    head_stage_index: int = -1
    # Per-patch target normalisation (MAE-style; He et al., CVPR 2022 §4.2).
    # When ``per_patch_target_norm=True``, the reconstruction target is
    # split into ``target_norm_patch_size``-sized tiles and each tile is
    # rescaled to mean 0 / std 1 per (sample, channel) before the L1/L2
    # loss is computed. Removes the "predict the channel mean" shortcut on
    # mostly-black inputs (the predicted constant has tile-std 0 while the
    # target has tile-std 1, so the loss never collapses to background).
    per_patch_target_norm: bool = False
    target_norm_patch_size: int = 8
    # Foreground-weighted spatial pooling for the VICReg branch.
    # Before the projector, encoder feature maps are spatially averaged with
    # sigmoid weights ``1 + alpha * sigmoid((input - tau) / temp)`` computed
    # from the *clean* (unmasked) input view at pixel resolution and then
    # downsampled to the feature-map grid. ``alpha=0`` recovers plain
    # ``z.mean(spatial)``. Using the same ``(alpha, tau, temp)`` as
    # ``fg_weight_*`` is a good default; separate knobs are exposed so the
    # recon and VICReg foreground bias can be tuned independently.
    fg_pool_alpha: float = 0.0  #5.0
    fg_pool_tau: float = 0.0    #1.0
    fg_pool_temp: float = 1.0    #0.5
    # Threshold (in z-scored target space) above which a pixel counts as
    # foreground for the diagnostic ``recon_fg`` metric. Independent of the
    # training-time foreground weighting (``fg_weight_*``) so the metric
    # stays comparable across runs that sweep those knobs. ``tau=1`` means
    # "one channel std above the channel mean" -- the rough definition of
    # a punctum in the z-scored microscopy inputs.
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
    """Dump a set of named configs to a JSON file next to the run."""
    payload = {name: _to_jsonable(cfg) for name, cfg in cfgs.items()}
    Path(path).write_text(json.dumps(payload, indent=2))
