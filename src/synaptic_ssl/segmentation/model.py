"""MONAI SwinUNETR builder with pretrained encoder loading.

Encoder weights load into ``swinViT``. Decoder stays random.
Hatamizadeh et al. (2022).

Ref: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
Ref: https://github.com/Project-MONAI/MONAI
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..training.config import ModelCfg


def build_swinunetr(
    model_cfg: ModelCfg,
    out_channels: int = 1,
) -> nn.Module:
    """Build a 2D MONAI ``SwinUNETR``.

    Forwards every backbone hyperparameter from ``ModelCfg`` explicitly so
    ``swinViT`` matches the SimMIM+VICReg pretraining encoder exactly.
    Mismatched ``patch_size`` / ``window_size`` / ``mlp_ratio`` / ``qkv_bias``
    silently drops pretrained ``patch_embed`` at load time.
    """
    from monai.networks.nets import SwinUNETR

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


def load_full_swinunetr_from_ckpt(
    model: nn.Module,
    ckpt_path: str | Path,
    *,
    strict: bool = True,
    logger=None,
) -> dict:
    """Warm-start the **entire** SwinUNETR (encoder + decoder) from a seg ckpt.

    Use this for iter-1 of the self-training loop (Bai 2017 §2.1: warm-start
    S1 from S0 = full model). Distinct from
    :func:`load_pretrained_encoder_into_swinunetr`, which loads only
    ``swinViT`` from an SSL pretraining checkpoint.

    The seg checkpoint produced by
    :func:`synaptic_ssl.training.checkpoints.save_checkpoint`
    stores the full SwinUNETR ``state_dict`` under the (misleadingly
    named) key ``"encoder_state_dict"``.

    Returns a small summary dict including ``channel_mean`` /
    ``channel_std`` so the caller can pin normalisation to the iter-0
    training-time statistics.
    """
    log = logger.info if logger is not None else print

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "encoder_state_dict" not in ckpt:
        raise KeyError(
            f"{ckpt_path}: expected key 'encoder_state_dict' (from save_checkpoint); "
            f"got keys {list(ckpt.keys())}"
        )
    src_sd = ckpt["encoder_state_dict"]
    # Compare head shapes if both are present, so we can emit a useful
    # message instead of a raw state_dict shape mismatch. MONAI's head
    # weight key is ``out.conv.conv.weight`` in current versions; fall
    # back to other plausible spellings.
    tgt_sd = model.state_dict()
    head_keys = ("out.conv.conv.weight", "out.conv.weight", "out.weight")
    for hk in head_keys:
        if hk in src_sd and hk in tgt_sd and src_sd[hk].shape != tgt_sd[hk].shape:
            raise ValueError(
                f"warm-start {ckpt_path}: head shape mismatch at {hk!r} "
                f"(ckpt={tuple(src_sd[hk].shape)}, model={tuple(tgt_sd[hk].shape)}). "
                f"Likely an out_channels mismatch (ckpt={int(src_sd[hk].shape[0])}, "
                f"model={int(tgt_sd[hk].shape[0])}). Rebuild the model with "
                f"SegTrainCfg(out_channels={int(src_sd[hk].shape[0])}) or warm-start "
                f"from a matching iter-0 checkpoint."
            )
    result = model.load_state_dict(src_sd, strict=strict)

    summary = {
        "n_loaded":          len(src_sd),
        "n_missing":         len(getattr(result, "missing_keys", [])),
        "n_unexpected":      len(getattr(result, "unexpected_keys", [])),
        "channel_mean":      ckpt.get("channel_mean"),
        "channel_std":       ckpt.get("channel_std"),
        "source_step":       ckpt.get("global_step", ckpt.get("epoch")),
        "source_val_metric": ckpt.get("val_metric"),
    }
    log(
        f"[warm-start] loaded full SwinUNETR from {Path(ckpt_path).name} | "
        f"params={summary['n_loaded']} missing={summary['n_missing']} "
        f"unexpected={summary['n_unexpected']}"
    )
    if summary["n_missing"] > 0 or summary["n_unexpected"] > 0:
        warnings.warn(
            f"warm-start from {ckpt_path}: missing={result.missing_keys} "
            f"unexpected={result.unexpected_keys}"
        )
    return summary


def count_params(model, only_trainable: bool = False) -> int:
    """Count parameters in a module or dict of modules."""
    if isinstance(model, dict):
        return sum(count_params(v, only_trainable) for v in model.values())
    if isinstance(model, nn.Module):
        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    return 0
