"""Load pretrained Swin weights into MONAI ``SwinTransformer``.

Supports timm ImageNet, MoBY self-supervised, and local SimMIM checkpoints.

Ref: https://github.com/Project-MONAI/MONAI
Ref: https://github.com/microsoft/SimMIM
Ref: https://github.com/SwinTransformer/Transformer-SSL
Ref: https://github.com/huggingface/pytorch-image-models
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from ..training.config import BaseCfg, ModelCfg


def _remap_timm_key_to_monai(k: str) -> str | None:
    """Map a timm Swin key to MONAI ``SwinTransformer`` naming. Return None if no match."""
    if k.startswith(("head.", "norm.", "norm_pre.", "norm_post.", "pre_logits.")):
        return None
    if k.startswith("patch_embed."):
        return k
    if k.startswith("layers."):
        rest = k[len("layers."):]
        idx_str, sep, tail = rest.partition(".")
        if not sep:
            return None
        try:
            i = int(idx_str)
        except ValueError:
            return None
        head, _, _ = tail.partition(".")
        if head == "blocks":
            new_key = f"layers{i + 1}.0.{tail}"
        elif head == "downsample":
            # In timm/Microsoft Swin, ``layers.{i}.downsample`` lives at the
            # END of stage i (dim=embed_dim*2**i -> 2*dim).  In MONAI, the
            # same module lives at ``layers{i+1}.0.downsample`` (also
            # dim=embed_dim*2**i -> 2*dim).  Stage 3 has no downsample.
            new_key = f"layers{i + 1}.0.{tail}"
        else:
            return None
        if ".blocks." in new_key:
            new_key = (
                new_key.replace(".mlp.fc1.", ".mlp.linear1.")
                       .replace(".mlp.fc2.", ".mlp.linear2.")
            )
        return new_key
    return None


def _adapt_first_conv(weight: torch.Tensor, target_in_chans: int) -> torch.Tensor:
    """Average-then-tile ``patch_embed.proj.weight`` to ``target_in_chans``."""
    src_in = weight.shape[1]
    if src_in == target_in_chans:
        return weight
    avg = weight.mean(dim=1, keepdim=True)
    tiled = avg.repeat(1, target_in_chans, 1, 1)
    return tiled * (src_in / target_in_chans)


def convert_timm_to_swinunetr_state_dict(
    timm_sd: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
    target_in_chans: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Remap a timm Swin state-dict to MONAI ``SwinTransformer`` naming.

    Returns ``(remapped_state_dict, summary_dict)``.
    """
    new_sd: dict[str, torch.Tensor] = {}
    skipped, shape_mismatch, copied = [], [], []
    for k, v in timm_sd.items():
        nk = _remap_timm_key_to_monai(k)
        if nk is None:
            skipped.append((k, "no_monai_equivalent"))
            continue
        if nk not in target_sd:
            skipped.append((k, f"missing_in_target:{nk}"))
            continue
        if nk.endswith("patch_embed.proj.weight") and v.dim() == 4:
            v = _adapt_first_conv(v, target_in_chans)
        if v.shape != target_sd[nk].shape:
            shape_mismatch.append((k, nk, tuple(v.shape), tuple(target_sd[nk].shape)))
            continue
        new_sd[nk] = v
        copied.append(nk)
    in_target_not_loaded = sorted(set(target_sd) - set(new_sd))
    summary = {
        "n_target_params":     len(target_sd),
        "n_loaded":            len(copied),
        "n_skipped_in_source": len(skipped),
        "n_shape_mismatch":    len(shape_mismatch),
        "n_random_init":       len(in_target_not_loaded),
        "skipped":             skipped,
        "shape_mismatch":      shape_mismatch,
        "random_init":         in_target_not_loaded,
    }
    return new_sd, summary


def _load_moby_ckpt(
    path: str | Path,
    target_sd: dict[str, torch.Tensor],
    target_in_chans: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a MoBY checkpoint (Xie et al., 2021) and remap to MONAI naming.

    Strips ``module.`` and ``encoder.`` prefixes; drops projector / contrastive
    head keys.
    Ref: https://github.com/SwinTransformer/Transformer-SSL
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "model" in ckpt:
        raw_sd = ckpt["model"]
    else:
        raw_sd = ckpt

    # --- strip ``module.`` DDP prefix, then extract ``encoder.*`` keys ---
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        k_clean = k.replace("module.", "", 1)
        # MoBY stores backbone as encoder.*; strip prefix to get Swin keys
        if k_clean.startswith("encoder."):
            k_clean = k_clean[len("encoder."):]
        else:
            # skip projector, contrastive head, etc.
            continue
        cleaned[k_clean] = v

    return convert_timm_to_swinunetr_state_dict(
        cleaned, target_sd, target_in_chans=target_in_chans,
    )


def _load_local_simmim_ckpt(
    path: str | Path, target_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict]:
    ckpt = torch.load(path, map_location="cpu")
    if "encoder_state_dict" in ckpt:
        src_sd = ckpt["encoder_state_dict"]
    elif "model_state_dict" in ckpt:
        full_sd = ckpt["model_state_dict"]
        prefixed = {
            k[len("swinViT."):]: v
            for k, v in full_sd.items()
            if k.startswith("swinViT.")
        }
        src_sd = prefixed if prefixed else full_sd
    else:
        src_sd = ckpt
    new_sd = {k: v for k, v in src_sd.items() if k in target_sd}
    in_target_not_loaded = sorted(set(target_sd) - set(new_sd))
    summary = {
        "n_target_params":     len(target_sd),
        "n_loaded":            len(new_sd),
        "n_skipped_in_source": len(set(src_sd) - set(target_sd)),
        "n_shape_mismatch":    0,
        "n_random_init":       len(in_target_not_loaded),
        "skipped":             sorted(set(src_sd) - set(target_sd)),
        "shape_mismatch":      [],
        "random_init":         in_target_not_loaded,
    }
    return new_sd, summary


def load_pretrained_into_encoder(
    encoder: nn.Module,
    base_cfg: BaseCfg,
    model_cfg: ModelCfg,
    *,
    logger=None,
) -> dict:
    """Load pretrained weights into ``encoder`` based on ``base_cfg.init_source``.

    Recognised values: ``"scratch"``, ``"timm_imagenet"``, ``"moby"``, ``"local_ckpt"``.
    Returns a summary dict with load statistics.
    """
    log = (logger.info if logger is not None else print)
    if base_cfg.init_source == "scratch":
        target_sd = encoder.state_dict()
        log(f"[init] scratch -- {len(target_sd)} params kept random")
        return {
            "source": "scratch",
            "n_loaded": 0,
            "n_random_init": len(target_sd),
            "n_target_params": len(target_sd),
            "n_shape_mismatch": 0,
            "n_skipped_in_source": 0,
            "random_init": list(target_sd.keys()),
            "shape_mismatch": [],
            "unexpected_keys": [],
            "missing_keys": list(target_sd.keys()),
        }

    target_sd = encoder.state_dict()

    if base_cfg.init_source == "moby":
        if not base_cfg.pretrained_ckpt_path:
            raise ValueError(
                "init_source='moby' but base_cfg.pretrained_ckpt_path is None. "
                "Download the MoBY Swin-T checkpoint from "
                "https://github.com/SwinTransformer/Transformer-SSL and "
                "set pretrained_ckpt_path."
            )
        log(f"[init] loading MoBY checkpoint: {base_cfg.pretrained_ckpt_path}")
        new_sd, summary = _load_moby_ckpt(
            base_cfg.pretrained_ckpt_path, target_sd,
            target_in_chans=model_cfg.in_channels,
        )
    elif base_cfg.init_source == "local_ckpt":
        if not base_cfg.pretrained_ckpt_path:
            raise ValueError(
                "init_source='local_ckpt' but base_cfg.pretrained_ckpt_path is None."
            )
        log(f"[init] loading local checkpoint: {base_cfg.pretrained_ckpt_path}")
        new_sd, summary = _load_local_simmim_ckpt(
            base_cfg.pretrained_ckpt_path, target_sd,
        )
    else:
        raise ValueError(
            f"Unknown init_source={base_cfg.init_source!r}. "
            "Use 'moby' / 'local_ckpt' / 'scratch'."
        )

    incompatible = encoder.load_state_dict(new_sd, strict=False)
    summary["unexpected_keys"] = list(incompatible.unexpected_keys)
    summary["missing_keys"]    = list(incompatible.missing_keys)
    summary["source"]          = base_cfg.init_source

    log(
        f"[init] source={base_cfg.init_source} | "
        f"loaded {summary['n_loaded']}/{summary['n_target_params']} | "
        f"random_init {summary['n_random_init']} | "
        f"shape_mismatch {summary['n_shape_mismatch']} | "
        f"skipped {summary['n_skipped_in_source']}"
    )
    if summary["n_loaded"] < 0.5 * summary["n_target_params"]:
        warnings.warn(
            f"Only {summary['n_loaded']}/{summary['n_target_params']} encoder "
            f"params were loaded from {base_cfg.init_source}. The remap may "
            "have failed; inspect summary['random_init'] / 'shape_mismatch'."
        )
    return summary
