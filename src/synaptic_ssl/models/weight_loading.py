"""Load pretrained Swin weights into MONAI ``SwinTransformer``.

Supports MoBY self-supervised and local SimMIM checkpoints. The internal
timm-to-MONAI key remapping is reused by the MoBY loader because MoBY's
on-disk format is timm-compatible.

Ref: https://github.com/Project-MONAI/MONAI
Ref: https://github.com/microsoft/SimMIM
Ref: https://github.com/SwinTransformer/Transformer-SSL
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

    if not cleaned:
        raise RuntimeError(
            f"MoBY loader found 0 keys with the ``encoder.`` prefix in "
            f"{path}. This usually means the file is the supervised "
            f"ImageNet-22k Swin-T checkpoint (raw timm format, no "
            f"``encoder.`` prefix). Use ``init_source='timm_imagenet'`` "
            "for that file."
        )

    return convert_timm_to_swinunetr_state_dict(
        cleaned, target_sd, target_in_chans=target_in_chans,
    )


def _load_timm_imagenet_ckpt(
    path: str | Path,
    target_sd: dict[str, torch.Tensor],
    target_in_chans: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a raw timm-format Swin checkpoint (e.g. supervised ImageNet-22k).

    Expects keys like ``patch_embed.*`` and ``layers.{i}.blocks.*`` (no
    ``encoder.`` prefix). Strips an optional ``module.`` DDP prefix.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        raw_sd = ckpt

    cleaned = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in raw_sd.items()
    }

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


# Prefixes ``_load_any_ckpt`` tries to strip when probing an unknown
# checkpoint. ``""`` means "leave keys as-is".
_ANY_PREFIXES: tuple[str, ...] = (
    "",
    "module.",
    "encoder.",
    "module.encoder.",
    "backbone.",
    "swinViT.",
    "model.",
)


def _strip_prefix(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(sd)
    return {
        k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
    }


def _direct_match(
    cleaned: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Name-match without remapping. Used by the best-effort loader."""
    new_sd: dict[str, torch.Tensor] = {}
    shape_mismatch: list = []
    skipped: list = []
    for k, v in cleaned.items():
        if k not in target_sd:
            skipped.append((k, "missing_in_target"))
            continue
        if v.shape != target_sd[k].shape:
            shape_mismatch.append((k, k, tuple(v.shape), tuple(target_sd[k].shape)))
            continue
        new_sd[k] = v
    in_target_not_loaded = sorted(set(target_sd) - set(new_sd))
    summary = {
        "n_target_params":     len(target_sd),
        "n_loaded":            len(new_sd),
        "n_skipped_in_source": len(skipped),
        "n_shape_mismatch":    len(shape_mismatch),
        "n_random_init":       len(in_target_not_loaded),
        "skipped":             skipped,
        "shape_mismatch":      shape_mismatch,
        "random_init":         in_target_not_loaded,
    }
    return new_sd, summary


def _load_any_ckpt(
    path: str | Path,
    target_sd: dict[str, torch.Tensor],
    target_in_chans: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Best-effort load of an unknown checkpoint format.

    Tries the usual top-level containers (``model`` / ``state_dict`` /
    ``encoder_state_dict`` / ``model_state_dict``), then several prefix-
    strip variants (``module.``, ``encoder.``, ``swinViT.``, …). For each
    variant it attempts both a direct name-match and a timm-to-MONAI remap
    and keeps the combination with the highest coverage. Never raises on
    low coverage -- the caller opted in to a lenient load.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # ---- unwrap the most common container layouts ----
    raw_sd: dict[str, torch.Tensor]
    if isinstance(ckpt, dict):
        for container in (
            "encoder_state_dict", "model_state_dict", "model", "state_dict",
        ):
            inner = ckpt.get(container)
            if isinstance(inner, dict) and inner:
                raw_sd = inner
                break
        else:
            raw_sd = ckpt
    else:
        raw_sd = ckpt  # tensors at top level

    best_loaded = -1
    best: tuple[dict[str, torch.Tensor], dict] | None = None

    for prefix in _ANY_PREFIXES:
        cleaned = _strip_prefix(raw_sd, prefix)
        if not cleaned:
            continue
        # (a) direct name-match
        d_sd, d_summary = _direct_match(cleaned, target_sd)
        if d_summary["n_loaded"] > best_loaded:
            d_summary["strategy"] = (
                f"direct (strip {prefix!r})" if prefix else "direct (as-is)"
            )
            best = (d_sd, d_summary)
            best_loaded = d_summary["n_loaded"]
        # (b) timm-to-MONAI remap
        try:
            m_sd, m_summary = convert_timm_to_swinunetr_state_dict(
                cleaned, target_sd, target_in_chans=target_in_chans,
            )
        except Exception:
            m_sd, m_summary = None, None
        if m_summary is not None and m_summary["n_loaded"] > best_loaded:
            m_summary["strategy"] = (
                f"timm-remap (strip {prefix!r})" if prefix else "timm-remap (as-is)"
            )
            best = (m_sd, m_summary)
            best_loaded = m_summary["n_loaded"]

    if best is None:
        # No prefix-strip produced any keys at all.
        skipped_list = (
            list(raw_sd.keys()) if isinstance(raw_sd, dict) else []
        )
        empty_summary = {
            "n_target_params":     len(target_sd),
            "n_loaded":            0,
            "n_skipped_in_source": len(skipped_list),
            "n_shape_mismatch":    0,
            "n_random_init":       len(target_sd),
            "skipped":             skipped_list,
            "shape_mismatch":      [],
            "random_init":         sorted(target_sd.keys()),
            "strategy":            "none",
        }
        return {}, empty_summary

    return best


def load_pretrained_into_encoder(
    encoder: nn.Module,
    base_cfg: BaseCfg,
    model_cfg: ModelCfg,
    *,
    logger=None,
) -> dict:
    """Load pretrained weights into ``encoder`` based on ``base_cfg.init_source``.

    Recognised values: ``"scratch"``, ``"moby"``, ``"timm_imagenet"``,
    ``"local_ckpt"``, ``"any"``. ``"any"`` is a best-effort loader that
    tries common container layouts and prefix-strip variants and keeps
    whichever combination produces the highest key coverage; it never
    raises on low coverage. Returns a summary dict with load statistics.
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
    elif base_cfg.init_source == "timm_imagenet":
        if not base_cfg.pretrained_ckpt_path:
            raise ValueError(
                "init_source='timm_imagenet' but base_cfg.pretrained_ckpt_path "
                "is None. Provide the supervised ImageNet (e.g. 22k) Swin-T "
                "checkpoint, such as swin_tiny_patch4_window7_224_22k.pth."
            )
        log(
            f"[init] loading timm/ImageNet checkpoint: "
            f"{base_cfg.pretrained_ckpt_path}"
        )
        new_sd, summary = _load_timm_imagenet_ckpt(
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
    elif base_cfg.init_source == "any":
        if not base_cfg.pretrained_ckpt_path:
            raise ValueError(
                "init_source='any' but base_cfg.pretrained_ckpt_path is None. "
                "Provide a path to the checkpoint you want to best-effort load."
            )
        log(
            f"[init] best-effort loading checkpoint: "
            f"{base_cfg.pretrained_ckpt_path}"
        )
        new_sd, summary = _load_any_ckpt(
            base_cfg.pretrained_ckpt_path, target_sd,
            target_in_chans=model_cfg.in_channels,
        )
        log(f"[init] best-effort strategy: {summary.get('strategy', '?')}")
    else:
        raise ValueError(
            f"Unknown init_source={base_cfg.init_source!r}. "
            "Use 'moby' / 'timm_imagenet' / 'local_ckpt' / 'any' / 'scratch'."
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
        msg = (
            f"Only {summary['n_loaded']}/{summary['n_target_params']} encoder "
            f"params were loaded from {base_cfg.init_source}. The remap may "
            "have failed; inspect summary['random_init'] / 'shape_mismatch'."
        )
        if base_cfg.init_source == "any":
            # User explicitly opted in to a lenient load -- info-level only.
            log(f"[init] {msg}")
        else:
            warnings.warn(msg)
    return summary
