"""Build SwinUNETR models and load pretrained encoders.

Wrap MONAI ``SwinUNETR`` and load a SimMIM+VICReg encoder into
``swinViT``. Leave the decoder randomly initialised.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from training.config import ModelCfg


def build_swinunetr(
    model_cfg: ModelCfg,
    out_channels: int = 1,
) -> nn.Module:
    """Instantiate a MONAI SwinUNETR for 2D segmentation."""
    from monai.networks.nets import SwinUNETR

    # Forward every backbone hyper-parameter from ``ModelCfg`` explicitly so
    # the segmentation ``swinViT`` is byte-for-byte identical to the encoder
    # used during SimMIM+VICReg pretraining. Previously ``patch_size``,
    # ``window_size``, ``mlp_ratio`` and ``qkv_bias`` were left at MONAI's
    # defaults, which silently dropped the pretrained ``patch_embed``
    # weights at load time when they didn't match. ``img_size`` was also
    # removed from MONAI ``SwinUNETR`` in recent releases (the divisibility
    # check now runs in ``_check_input_size`` at forward time).
    model = SwinUNETR(
        in_channels=model_cfg.in_channels,
        out_channels=out_channels,
        feature_size=model_cfg.feature_size,
        patch_size=model_cfg.patch_size,
        window_size=model_cfg.window_size,
        depths=list(model_cfg.depths),
        num_heads=list(model_cfg.num_heads),
        mlp_ratio=model_cfg.mlp_ratio,
        qkv_bias=model_cfg.qkv_bias,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=model_cfg.dropout_path_rate,
        use_checkpoint=model_cfg.use_checkpoint,
        spatial_dims=model_cfg.spatial_dims,
    )
    return model


def load_pretrained_encoder_into_swinunetr(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    logger=None,
) -> dict:
    """Load a pretrained encoder checkpoint into ``model.swinViT``.

    Returns a summary dict with load statistics.
    """
    log = logger.info if logger is not None else print

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "encoder_state_dict" in ckpt:
        src_sd = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        # Full SwinUNETR checkpoint — extract swinViT keys
        full_sd = ckpt["model_state_dict"]
        prefix = "swinViT."
        src_sd = {
            k[len(prefix):]: v
            for k, v in full_sd.items()
            if k.startswith(prefix)
        }
        if not src_sd:
            src_sd = full_sd
    else:
        src_sd = ckpt

    target_sd = model.swinViT.state_dict()
    matched = {k: v for k, v in src_sd.items() if k in target_sd and v.shape == target_sd[k].shape}
    missing = sorted(set(target_sd) - set(matched))
    unexpected = sorted(set(src_sd) - set(target_sd))

    result = model.swinViT.load_state_dict(matched, strict=False)

    summary = {
        "n_target_params": len(target_sd),
        "n_loaded": len(matched),
        "n_missing": len(missing),
        "n_unexpected": len(unexpected),
        "channel_mean": ckpt.get("channel_mean"),
        "channel_std": ckpt.get("channel_std"),
        "source_epoch": ckpt.get("epoch"),
        "source_val_metric": ckpt.get("val_metric"),
    }

    log(
        f"[encoder] loaded {summary['n_loaded']}/{summary['n_target_params']} "
        f"encoder params from {Path(ckpt_path).name} | "
        f"missing {summary['n_missing']} | unexpected {summary['n_unexpected']}"
    )

    if summary["n_loaded"] < 0.5 * summary["n_target_params"]:
        warnings.warn(
            f"Only {summary['n_loaded']}/{summary['n_target_params']} encoder "
            f"params loaded. Check checkpoint compatibility."
        )

    return summary


def count_params(model, only_trainable: bool = False) -> int:
    """Count model parameters."""
    if isinstance(model, dict):
        return sum(count_params(v, only_trainable) for v in model.values())
    if isinstance(model, nn.Module):
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    return 0
