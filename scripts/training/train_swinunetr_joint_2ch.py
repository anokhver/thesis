#!/usr/bin/env python
"""Joint 2-channel (PRE + POST) SwinUNETR fine-tuning script.

Headless equivalent of ``notebooks/segmentation/train_swinunetr_joint_2ch.ipynb``.
Mirrors the structure of ``scripts/training/pretrain_simmim_vicreg.py`` so it can
be submitted as a PBS job (see ``scripts/metacentrum/submit_train_joint_2ch.sh``).

Usage:
    python scripts/training/train_swinunetr_joint_2ch.py \
        --config configs/segmentation/joint_2ch_default.json
    python scripts/training/train_swinunetr_joint_2ch.py \
        --config configs/segmentation/joint_2ch_default.json --dry-run
    python scripts/training/train_swinunetr_joint_2ch.py \
        --config configs/segmentation/joint_2ch_default.json \
        --resume ../data/training_outputs/joint_2ch/<run>/last.pt

Relative paths in the config are resolved against ``root/`` (i.e. ``src/``)
following the same convention as ``pretrain_simmim_vicreg.py``.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless: must precede any pyplot import

import argparse
import csv as _csv
import dataclasses
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
_ROOT = _REPO / "src"
for _p in (_ROOT, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synaptic_ssl.training.config import (
    BaseCfg, DataCfg, ModelCfg, dump_config,
)
from synaptic_ssl.training.seeding import seed_everything
from synaptic_ssl.training.logging import setup_logger, CSVMetricLogger
from synaptic_ssl.training.data import compute_channel_stats
from synaptic_ssl.training.lr_schedule import (
    param_groups_layer_decay, make_warmup_cosine,
)
from synaptic_ssl.training.checkpoints import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
)
from synaptic_ssl.utils_data.patch_dataset import PatchDataset
from synaptic_ssl.segmentation import (
    SegTrainCfg,
    JointChannelSegDataset,
    JointChannelDiceBCE, JointChannelTversky, JointChannelTverskyBCE,
    compute_dice_metric_per_channel,
    build_swinunetr, load_pretrained_encoder_into_swinunetr, count_params,
    SegTrainTransform, SegValTransform,
    sliding_window_predict_multichannel,
    discover_full_image_masks,
    SynapseColocCfg, build_synapse_mask,
)
from synaptic_ssl.utils_data.reassemble import reassemble_image, list_image_indices


# ---------------------------------------------------------------------------
# Lightweight PatchDataset view
# ---------------------------------------------------------------------------
class _IndexedPatchDataset:
    """Wrap a PatchDataset to expose only a subset of records.

    The blueprint notebook uses the same pattern; ``JointChannelSegDataset``
    only needs ``len()``, ``root``, ``channels`` and ``records[idx]``.
    """

    def __init__(self, base_ds: PatchDataset, indices):
        self.root = base_ds.root
        self.channels = base_ds.channels
        self.records = [base_ds.records[i] for i in indices]

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_TOP_KEYS = {
    "run_sanity", "run_overfit", "sessions",
    "base", "data", "model", "seg",
    "pseudolabels", "loss_mask", "colocalisation",
}


def _unknown(cls, raw: dict | None) -> list[str]:
    if raw is None:
        return []
    return sorted(set(raw) - {f.name for f in dataclasses.fields(cls)})


def _build(cls, raw: dict | None):
    if raw is None:
        return cls()
    fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in fields}
    for f in dataclasses.fields(cls):
        if f.name in filtered and hasattr(f.type, "__origin__") and f.type.__origin__ is tuple:
            filtered[f.name] = tuple(filtered[f.name])
    return cls(**filtered)


def _resolve(p: str | None, base: Path) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    return str(pp if pp.is_absolute() else (base / pp).resolve())


def load_config(config_path: str | Path, root_dir: str | Path) -> dict[str, Any]:
    """Load a JSON config and build dataclass objects.

    Path resolution: relative paths are resolved against ``root_dir``
    (the ``src/`` dir) to match ``pretrain_simmim_vicreg.py`` exactly.

    Required top-level keys (all optional except ``pseudolabels``):

    - ``base``           — ``BaseCfg``
    - ``data``           — ``DataCfg``
    - ``model``          — ``ModelCfg``
    - ``seg``            — ``SegTrainCfg``
    - ``pseudolabels``   — ``{archive_path, cache_dir, per_patch_mask_dir}``
    - ``colocalisation`` — ``{prob_thresh, min_distance, match_radius_px, synapse_radius_px}``
    - ``sessions``       — list of session subdir names to use (e.g.
      ``["20251017", "20251030", ...]``)
    - ``run_sanity``, ``run_overfit`` — bool flags
    """
    root_dir = Path(root_dir)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    top_unknown = sorted(set(raw) - _TOP_KEYS)
    if top_unknown:
        raise ValueError(
            f"Unknown top-level config keys in {config_path}: {top_unknown}."
        )
    section_unknown = {
        "base": _unknown(BaseCfg,     raw.get("base")),
        "data": _unknown(DataCfg,     raw.get("data")),
        "model": _unknown(ModelCfg,   raw.get("model")),
        "seg":  _unknown(SegTrainCfg, raw.get("seg")),
    }
    bad = {k: v for k, v in section_unknown.items() if v}
    if bad:
        raise ValueError(
            f"Unknown config keys in {config_path}: "
            + ", ".join(f"{k}: {v}" for k, v in bad.items())
        )

    base_cfg  = _build(BaseCfg,     raw.get("base"))
    data_cfg  = _build(DataCfg,     raw.get("data"))
    model_cfg = _build(ModelCfg,    raw.get("model"))
    seg_cfg   = _build(SegTrainCfg, raw.get("seg"))

    base_cfg.output_root          = _resolve(base_cfg.output_root, root_dir)
    base_cfg.pretrained_ckpt_path = _resolve(base_cfg.pretrained_ckpt_path, root_dir)
    base_cfg.resume_path          = _resolve(base_cfg.resume_path, root_dir)
    data_cfg.data_root            = _resolve(data_cfg.data_root, root_dir)

    pl_raw = raw.get("pseudolabels", {})
    pseudolabels = {
        "mask_root":          _resolve(pl_raw.get("mask_root"), root_dir),
        "per_patch_mask_dir": _resolve(pl_raw.get("per_patch_mask_dir"), root_dir),
    }
    if pseudolabels["mask_root"] is None:
        raise ValueError(
            "pseudolabels.mask_root is required (path to the extracted "
            "`spotiflow/` directory containing one sub-dir per session)."
        )

    coloc_raw = raw.get("colocalisation", {})
    coloc_cfg = SynapseColocCfg(
        prob_thresh       = float(coloc_raw.get("prob_thresh", 0.5)),
        min_distance      = int(coloc_raw.get("min_distance", 2)),
        match_radius_px   = int(coloc_raw.get("match_radius_px", 5)),
        synapse_radius_px = int(coloc_raw.get("synapse_radius_px", 3)),
    )

    # Optional Recipe-A soft loss-masking from per-patch soma+dend tiles.
    # Disabled by default (keeps backward compatibility for configs that
    # don't mention this section).
    lm_raw = raw.get("loss_mask", {}) or {}
    loss_mask_cfg = {
        "enabled":   bool(lm_raw.get("enabled", False)),
        "soma_root": _resolve(lm_raw.get("soma_root"), root_dir),
        "dend_root": _resolve(lm_raw.get("dend_root"), root_dir),
        "blur_sigma": float(lm_raw.get("blur_sigma", 0.0)),
        "floor":      float(lm_raw.get("floor", 0.1)),
    }
    if loss_mask_cfg["enabled"]:
        if not loss_mask_cfg["soma_root"] or not loss_mask_cfg["dend_root"]:
            raise ValueError(
                "loss_mask.enabled is True but soma_root / dend_root not set."
            )

    sessions = raw.get("sessions", None)
    if sessions is not None:
        sessions = list(sessions)

    return {
        "base_cfg":      base_cfg,
        "data_cfg":      data_cfg,
        "model_cfg":     model_cfg,
        "seg_cfg":       seg_cfg,
        "pseudolabels":  pseudolabels,
        "loss_mask_cfg": loss_mask_cfg,
        "coloc_cfg":     coloc_cfg,
        "sessions":      sessions,
        "run_sanity":    bool(raw.get("run_sanity",  True)),
        "run_overfit":   bool(raw.get("run_overfit", True)),
    }


# ---------------------------------------------------------------------------
# Data setup
# ---------------------------------------------------------------------------
def setup_data(
    data_cfg: DataCfg,
    model_cfg: ModelCfg,
    seg_cfg: SegTrainCfg,
    pseudolabels: dict,
    sessions: list[str] | None,
    *,
    seed_generator,
    logger,
    loss_mask_cfg: dict | None = None,
):
    """Load patches + pseudo-labels and return train/val datasets and loaders."""
    # ── Pseudo-label folder (must be pre-extracted) ──
    mask_root = Path(pseudolabels["mask_root"])
    if not mask_root.is_dir():
        raise FileNotFoundError(
            f"pseudolabels.mask_root does not exist or is not a directory: {mask_root}"
        )
    logger.info(f"pseudolabel mask_root = {mask_root}")
    mask_index = discover_full_image_masks(mask_root, sessions=sessions)
    logger.info(f"full-image mask sources indexed = {len(mask_index)}")
    if not mask_index:
        raise RuntimeError(
            f"No mask sources found under {mask_root} for sessions={sessions}."
        )

    # ── Raw patch dataset (nested layout: <session>/<patch>.npy) ──
    raw_dataset = PatchDataset(
        root=data_cfg.data_root,
        exclude_patterns=data_cfg.exclude_patterns,
    )
    if sessions is not None:
        session_set = set(sessions)
        raw_dataset.records = [
            r for r in raw_dataset.records
            if Path(r["filename"]).parts[0] in session_set
        ]
    logger.info(f"raw patches (filtered by sessions) = {len(raw_dataset)}")

    # ── Drop patches whose source MIP has no PRE+POST mask on disk ──
    # ``mask_index`` keys are ``"<session>/<source_stem>"``. The patch dataset
    # may reference source MIPs that the Spotiflow pipeline never produced
    # (e.g. a wall-time crash mid-session). Without this filter such patches
    # crash a DataLoader worker mid-epoch with FileNotFoundError.
    valid_keys = set(mask_index.keys())
    kept, dropped_keys = [], set()
    for r in raw_dataset.records:
        key = f"{Path(r['filename']).parts[0]}/{Path(r['source_npy']).stem}"
        if key in valid_keys:
            kept.append(r)
        else:
            dropped_keys.add(key)
    if dropped_keys:
        logger.warning(
            f"dropped {len(raw_dataset.records) - len(kept)} patches across "
            f"{len(dropped_keys)} source MIPs with no pseudo-labels "
            f"(kept {len(kept)} patches)"
        )
        sample_missing = sorted(dropped_keys)[:5]
        logger.warning(f"  first few missing sources: {sample_missing}")
    raw_dataset.records = kept

    sample = raw_dataset[0]
    logger.info(f"sample shape = {tuple(sample.shape)}  dtype = {sample.dtype}")
    assert sample.ndim == 3 and sample.shape[0] == model_cfg.in_channels
    assert sample.shape[-1] == model_cfg.img_size

    # ── Train/val split + per-channel stats ──
    n_val   = int(len(raw_dataset) * data_cfg.val_split)
    n_train = len(raw_dataset) - n_val
    train_subset, val_subset = random_split(
        raw_dataset, [n_train, n_val], generator=seed_generator,
    )
    logger.info(f"split: train={n_train}  val={n_val}")

    ch_mean, ch_std = compute_channel_stats(
        train_subset, in_channels=model_cfg.in_channels,
        max_samples=data_cfg.channel_stats_max_samples,
    )
    for name, m, s in zip(data_cfg.channel_names, ch_mean.tolist(), ch_std.tolist()):
        logger.info(f"  ch[{name:>14s}] mean={m:.5f}  std={s:.5f}")

    # ── Segmentation datasets ──
    train_patch_ds = _IndexedPatchDataset(raw_dataset, train_subset.indices)
    val_patch_ds   = _IndexedPatchDataset(raw_dataset, val_subset.indices)

    # Loss-mask args (Recipe A). When disabled, JointChannelSegDataset
    # behaves exactly as before (2-tuple).
    lm = loss_mask_cfg or {}
    use_soft_loss_mask = bool(lm.get("enabled", False))
    ds_loss_kwargs = {}
    if use_soft_loss_mask:
        ds_loss_kwargs = dict(
            soma_mask_dir=lm["soma_root"],
            dend_mask_dir=lm["dend_root"],
            loss_mask_blur_sigma=float(lm.get("blur_sigma", 0.0)),
            loss_mask_floor=float(lm.get("floor", 0.1)),
        )
        logger.info(
            f"loss_mask: enabled  soma_root={lm['soma_root']}  "
            f"dend_root={lm['dend_root']}  blur={ds_loss_kwargs['loss_mask_blur_sigma']}  "
            f"floor={ds_loss_kwargs['loss_mask_floor']}"
        )
    else:
        logger.info("loss_mask: disabled (no soma+dend soft weighting)")

    # Soft loss mask requires the train-transform to NOT re-binarise it.
    train_tf = SegTrainTransform(
        ch_mean, ch_std, binarize_loss_mask=not use_soft_loss_mask,
    )
    val_tf = SegValTransform(ch_mean, ch_std)

    per_patch = pseudolabels.get("per_patch_mask_dir")
    train_seg_ds = JointChannelSegDataset(
        train_patch_ds,
        per_patch_mask_dir=per_patch,
        full_image_mask_dir=mask_root,
        transform=train_tf,
        **ds_loss_kwargs,
    )
    val_seg_ds = JointChannelSegDataset(
        val_patch_ds,
        per_patch_mask_dir=per_patch,
        full_image_mask_dir=mask_root,
        transform=val_tf,
        **ds_loss_kwargs,
    )
    logger.info(f"train seg dataset = {len(train_seg_ds)}")
    logger.info(f"val   seg dataset = {len(val_seg_ds)}")

    common = dict(
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.num_workers > 0,
        prefetch_factor=2 if data_cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_seg_ds, batch_size=data_cfg.batch_size,
        shuffle=True, drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_seg_ds, batch_size=data_cfg.batch_size,
        shuffle=False, **common,
    )
    logger.info(
        f"train batches = {len(train_loader)}  val batches = {len(val_loader)}"
    )

    return dict(
        raw_dataset=raw_dataset,
        train_seg_ds=train_seg_ds,
        val_seg_ds=val_seg_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        ch_mean=ch_mean,
        ch_std=ch_std,
        mask_root=mask_root,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(
    base_cfg: BaseCfg,
    model_cfg: ModelCfg,
    device: torch.device,
    *,
    logger,
) -> nn.Module:
    model = build_swinunetr(model_cfg, out_channels=2).to(device)
    logger.info(
        f"SwinUNETR (out_channels=2) total params = "
        f"{count_params(model) / 1e6:.2f} M"
    )

    if base_cfg.pretrained_ckpt_path:
        enc_summary = load_pretrained_encoder_into_swinunetr(
            model, base_cfg.pretrained_ckpt_path, logger=logger,
        )
        n_loaded = enc_summary.get("n_loaded", 0)
        n_target = enc_summary.get("n_target_params", 0) or 1
        frac = n_loaded / n_target
        logger.info(f"encoder load: {n_loaded}/{n_target}  ({frac:.1%})")
        if frac < 0.95:
            logger.warning(
                "encoder load fraction below 95%% -- check ModelCfg matches SSL pretrain"
            )
        if enc_summary.get("channel_mean") is not None:
            logger.info(f"  pretrained channel_mean = {enc_summary['channel_mean']}")
            logger.info(f"  pretrained channel_std  = {enc_summary['channel_std']}")
    else:
        logger.warning("No pretrained_ckpt_path set -- encoder starts from random init")

    return model


# ---------------------------------------------------------------------------
# Optimiser / AMP
# ---------------------------------------------------------------------------
def setup_optimizer(
    model: nn.Module,
    seg_cfg: SegTrainCfg,
    device: torch.device,
    *,
    logger,
):
    encoder_groups = param_groups_layer_decay(
        model.swinViT,
        base_lr=seg_cfg.base_lr,
        weight_decay=seg_cfg.weight_decay,
        layer_decay=seg_cfg.layer_decay,
    )
    encoder_param_ids = {id(p) for p in model.swinViT.parameters()}
    decoder_params = [
        p for p in model.parameters() if id(p) not in encoder_param_ids
    ]
    decoder_group = {
        "params": decoder_params,
        "lr": seg_cfg.decoder_lr,
        "weight_decay": seg_cfg.weight_decay,
    }
    optimizer = torch.optim.AdamW(
        encoder_groups + [decoder_group], betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_warmup_cosine(seg_cfg.warmup_epochs, seg_cfg.epochs),
    )

    # bf16 on Ampere+ (A100/H100), fp16 otherwise.
    use_amp = device.type == "cuda"
    if use_amp:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 8:
            amp_dtype, use_fp16 = torch.bfloat16, False
        else:
            amp_dtype, use_fp16 = torch.float16, True
    else:
        amp_dtype, use_fp16 = torch.float32, False
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)

    logger.info(
        f"AMP: enabled={use_amp}  dtype={amp_dtype}  fp16_scaler={use_fp16}"
    )
    logger.info(f"encoder param groups = {len(encoder_groups)}")
    logger.info(
        f"decoder params       = "
        f"{sum(p.numel() for p in decoder_params) / 1e6:.2f} M"
    )

    return dict(
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        use_amp=use_amp, amp_dtype=amp_dtype, use_fp16=use_fp16,
    )


# ---------------------------------------------------------------------------
# Sanity / overfit
# ---------------------------------------------------------------------------
def _make_loss(seg_cfg: SegTrainCfg) -> nn.Module:
    """Build the per-batch loss according to ``seg_cfg.loss_type``.

    - ``"dice_bce"`` (default): :class:`JointChannelDiceBCE` with
      ``dice_weight`` / ``bce_weight`` / ``dice_smooth``.
    - ``"tversky"``: :class:`JointChannelTversky` with
      ``tversky_alpha`` / ``tversky_beta`` / ``dice_smooth``.
    - ``"tversky_bce"``: :class:`JointChannelTverskyBCE` -- per-channel
      Tversky + BCE, uses ``tversky_alpha`` / ``tversky_beta`` /
      ``bce_weight`` / ``dice_smooth``. Adds the per-pixel BCE term
      that pure Tversky lacks, preventing the all-zero saturation
      collapse seen in early pure-Tversky runs.
    """
    loss_type = (getattr(seg_cfg, "loss_type", "dice_bce") or "dice_bce").lower()
    if loss_type == "tversky":
        return JointChannelTversky(
            alpha=seg_cfg.tversky_alpha,
            beta=seg_cfg.tversky_beta,
            smooth=seg_cfg.dice_smooth,
        )
    if loss_type == "tversky_bce":
        return JointChannelTverskyBCE(
            alpha=seg_cfg.tversky_alpha,
            beta=seg_cfg.tversky_beta,
            smooth=seg_cfg.dice_smooth,
            bce_weight=seg_cfg.bce_weight,
        )
    if loss_type == "dice_bce":
        return JointChannelDiceBCE(
            dice_weight=seg_cfg.dice_weight,
            bce_weight=seg_cfg.bce_weight,
            smooth=seg_cfg.dice_smooth,
        )
    raise ValueError(
        f"Unknown seg.loss_type {loss_type!r}; "
        f"expected 'dice_bce', 'tversky', or 'tversky_bce'."
    )


def run_sanity_checks(
    model: nn.Module,
    data: dict,
    model_cfg: ModelCfg,
    data_cfg: DataCfg,
    seg_cfg: SegTrainCfg,
    device: torch.device,
    save_dir: Path,
    *,
    logger,
):
    train_loader = data["train_loader"]
    batch = next(iter(train_loader))
    if len(batch) == 3:
        batch_img, batch_mask, batch_lm = batch
    else:
        batch_img, batch_mask = batch
        batch_lm = None
    assert batch_img.shape == (
        data_cfg.batch_size, model_cfg.in_channels,
        model_cfg.img_size, model_cfg.img_size,
    ), f"got {tuple(batch_img.shape)}"
    assert batch_mask.shape == (
        data_cfg.batch_size, 2, model_cfg.img_size, model_cfg.img_size,
    ), f"got {tuple(batch_mask.shape)}"
    assert torch.isfinite(batch_img).all()
    logger.info(
        f"[ok] batch shapes: img={tuple(batch_img.shape)} mask={tuple(batch_mask.shape)}"
        + (f" loss_mask={tuple(batch_lm.shape)}" if batch_lm is not None else "")
    )
    logger.info(f"     PRE coverage  = {batch_mask[:, 0].mean().item():.5f}")
    logger.info(f"     POST coverage = {batch_mask[:, 1].mean().item():.5f}")
    if batch_lm is not None:
        logger.info(
            f"     loss-mask mean={float(batch_lm.mean()):.4f}  "
            f"min={float(batch_lm.min()):.4f}  max={float(batch_lm.max()):.4f}"
        )

    # Forward + grad flow
    model.eval()
    with torch.no_grad():
        test_out = model(batch_img[:2].to(device))
    assert test_out.shape == (
        2, 2, model_cfg.img_size, model_cfg.img_size,
    ), f"got {test_out.shape}"
    logger.info(
        f"[ok] forward pass: input {tuple(batch_img[:2].shape)} -> "
        f"output {tuple(test_out.shape)}"
    )
    model.train()

    loss_fn = _make_loss(seg_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    _img = batch_img[:2].to(device)
    _msk = batch_mask[:2].to(device)
    _lm = batch_lm[:2].to(device) if batch_lm is not None else None
    out = model(_img)
    losses = loss_fn(out, _msk, loss_mask=_lm) if _lm is not None else loss_fn(out, _msk)
    losses["loss"].backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_tot = sum(1 for _ in model.parameters())
    opt.step()
    logger.info(f"[ok] grad flow: {n_grad}/{n_tot} params got gradients")

    # Save sanity figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(min(4, data_cfg.batch_size)):
        pre_img = batch_img[i, 0].numpy()
        post_img = batch_img[i, 1].numpy()
        pre_m = batch_mask[i, 0].numpy()
        post_m = batch_mask[i, 1].numpy()
        axes[0, i].imshow(pre_img, cmap="gray")
        axes[0, i].set_title(f"PRE input [{i}]"); axes[0, i].axis("off")
        axes[1, i].imshow(post_img, cmap="gray")
        axes[1, i].set_title(f"POST input [{i}]"); axes[1, i].axis("off")
        axes[2, i].imshow(pre_m, cmap="hot", vmin=0, vmax=1)
        axes[2, i].set_title(f"PRE mask ({int(pre_m.sum())} px)"); axes[2, i].axis("off")
        axes[3, i].imshow(post_m, cmap="hot", vmin=0, vmax=1)
        axes[3, i].set_title(f"POST mask ({int(post_m.sum())} px)"); axes[3, i].axis("off")
    fig.suptitle("Joint 2-channel sanity batch", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "sanity_batch.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"sanity figure saved -> {save_dir / 'sanity_batch.png'}")


def run_overfit_check(
    model_cfg: ModelCfg,
    seg_cfg: SegTrainCfg,
    data: dict,
    device: torch.device,
    save_dir: Path,
    *,
    n_steps: int = 100,
    lr: float = 5e-4,
    logger,
):
    train_loader = data["train_loader"]
    batch = next(iter(train_loader))
    if len(batch) == 3:
        batch_img, batch_mask, batch_lm = batch
    else:
        batch_img, batch_mask = batch
        batch_lm = None
    of_model = build_swinunetr(model_cfg, out_channels=2).to(device)
    of_opt = torch.optim.AdamW(of_model.parameters(), lr=lr)
    of_img = batch_img.to(device); of_msk = batch_mask.to(device)
    of_lm = batch_lm.to(device) if batch_lm is not None else None
    loss_fn = _make_loss(seg_cfg)

    loss_hist, dpre_hist, dpost_hist = [], [], []
    of_model.train()
    for _ in tqdm(range(n_steps), desc="overfit"):
        of_opt.zero_grad(set_to_none=True)
        out = of_model(of_img)
        if of_lm is not None:
            losses = loss_fn(out, of_msk, loss_mask=of_lm)
        else:
            losses = loss_fn(out, of_msk)
        losses["loss"].backward()
        of_opt.step()
        loss_hist.append(losses["loss"].item())
        with torch.no_grad():
            d = compute_dice_metric_per_channel(out, of_msk)
            dpre_hist.append(float(d[0]))
            dpost_hist.append(float(d[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_hist); ax1.set_xlabel("step"); ax1.set_ylabel("loss")
    ax1.set_title("Overfit loss"); ax1.grid(True, alpha=0.3)
    ax2.plot(dpre_hist, label="PRE", color="tab:red")
    ax2.plot(dpost_hist, label="POST", color="tab:blue")
    ax2.set_xlabel("step"); ax2.set_ylabel("Dice"); ax2.set_title("Overfit Dice")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Overfit check: {n_steps} steps on 1 batch")
    fig.tight_layout()
    fig.savefig(save_dir / "overfit_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        f"overfit: loss={loss_hist[-1]:.5f}  dice_pre={dpre_hist[-1]:.4f}  "
        f"dice_post={dpost_hist[-1]:.4f}"
    )
    del of_model, of_opt, of_img, of_msk
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _freeze_encoder(mdl: nn.Module, freeze: bool):
    for p in mdl.swinViT.parameters():
        p.requires_grad = not freeze


def _get_lrs(opt):
    enc_lrs = [g["lr"] for g in opt.param_groups if g.get("stage") is not None]
    dec_lrs = [g["lr"] for g in opt.param_groups if g.get("stage") is None]
    return (
        max(enc_lrs) if enc_lrs else 0.0,
        dec_lrs[0] if dec_lrs else 0.0,
    )


def _trainable_params(mdl):
    return [p for p in mdl.parameters() if p.requires_grad]


def _save_qualitative_grid(
    model, val_seg_ds, qual_idx, *,
    device, use_amp, amp_dtype, save_dir: Path, epoch: int, logger,
):
    """Render a 6-column figure (PRE input, POST input, pred PRE, pred POST,
    pseudo PRE, pseudo POST) for a fixed grid of val patches.
    """
    try:
        model.eval()
        rows = []
        with torch.no_grad():
            for i in qual_idx:
                img, mask = val_seg_ds[i]
                img_t = img.unsqueeze(0).to(device)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = model(img_t)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                rows.append((img.numpy(), mask.numpy(), probs))
        n = len(rows)
        fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n))
        if n == 1:
            axes = axes[None, :]
        col_titles = [
            "PRE input", "POST input",
            "pred PRE", "pred POST",
            "pseudo PRE", "pseudo POST",
        ]
        for r, (img_np, mask_np, prob_np) in enumerate(rows):
            for c, (arr, cmap, title) in enumerate([
                (img_np[0], "gray", col_titles[0]),
                (img_np[1], "gray", col_titles[1]),
                (prob_np[0], "hot", col_titles[2]),
                (prob_np[1], "hot", col_titles[3]),
                (mask_np[0], "hot", col_titles[4]),
                (mask_np[1], "hot", col_titles[5]),
            ]):
                ax = axes[r, c]
                if c < 2:
                    lo, hi = np.percentile(arr, (1, 99))
                    ax.imshow(arr, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-3))
                else:
                    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
                if r == 0:
                    ax.set_title(title)
                ax.axis("off")
        fig.suptitle(f"Qualitative grid @ epoch {epoch}", y=1.001)
        fig.tight_layout()
        out_path = save_dir / "qualitative" / f"epoch_{epoch:03d}.png"
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        model.train()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"qualitative grid failed at epoch {epoch}: {exc}")


def training_loop(
    model: nn.Module,
    data: dict,
    optim_state: dict,
    base_cfg: BaseCfg,
    seg_cfg: SegTrainCfg,
    device: torch.device,
    save_dir: Path,
    *,
    logger,
) -> dict:
    optimizer = optim_state["optimizer"]
    scheduler = optim_state["scheduler"]
    scaler = optim_state["scaler"]
    use_amp = optim_state["use_amp"]
    amp_dtype = optim_state["amp_dtype"]
    use_fp16 = optim_state["use_fp16"]

    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]
    val_seg_ds   = data["val_seg_ds"]
    ch_mean      = data["ch_mean"]
    ch_std       = data["ch_std"]

    loss_fn = _make_loss(seg_cfg)

    # Resume (optional)
    start_epoch = 1
    best_val_metric = float("-inf")
    best_epoch = 0
    if base_cfg.resume_path is not None:
        p = Path(base_cfg.resume_path)
        if p.is_dir():
            p = find_latest_checkpoint(p)
        if p is None or not Path(p).exists():
            logger.warning(
                f"resume path {base_cfg.resume_path!r} not found -- starting fresh"
            )
        else:
            ckpt = load_checkpoint(
                p, encoder=model,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                map_location=device,
            )
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_metric = ckpt.get("val_metric", best_val_metric) or best_val_metric
            best_epoch = int(ckpt.get("epoch", 0))
            logger.info(f"resumed from {p} (epoch {ckpt.get('epoch')})")
    logger.info(f"start_epoch = {start_epoch}  best_val_metric = {best_val_metric}")

    # Qualitative monitoring grid (fixed val indices)
    rng = np.random.default_rng(base_cfg.seed)
    n_qual = min(16, len(val_seg_ds))
    qual_idx = sorted(int(i) for i in rng.choice(len(val_seg_ds), size=n_qual, replace=False))
    logger.info(f"qualitative grid indices = {qual_idx[:8]}... ({n_qual} total)")

    csv_fields = [
        "epoch", "phase",
        "train_loss", "train_dice_loss", "train_bce_loss",
        "train_pre_loss", "train_post_loss",
        "val_loss", "val_dice_pre", "val_dice_post", "val_dice_mean",
        "lr_encoder", "lr_decoder",
        "epoch_time_s", "train_time_s", "val_time_s",
        "best_val_metric", "best_epoch",
        "grad_norm_mean", "grad_norm_max",
    ]
    csv_logger = CSVMetricLogger(save_dir / "metrics.csv", csv_fields)

    total_t0 = time.time()
    better = lambda new, best: new > best

    for epoch in range(start_epoch, seg_cfg.epochs + 1):
        in_warmup = epoch <= seg_cfg.freeze_encoder_epochs
        _freeze_encoder(model, in_warmup)
        phase = "frozen" if in_warmup else "full"

        # ---- TRAIN ----
        model.train()
        running = {
            "loss": 0.0, "dice_loss": 0.0, "bce_loss": 0.0,
            "pre_loss": 0.0, "post_loss": 0.0,
        }
        grads = []
        t_train = time.time()
        pbar = tqdm(
            train_loader,
            desc=f"ep {epoch}/{seg_cfg.epochs} [{phase}]",
            leave=False,
        )
        for batch in pbar:
            if len(batch) == 3:
                img_b, mask_b, lm_b = batch
                lm_b = lm_b.to(device, non_blocking=True)
            else:
                img_b, mask_b = batch
                lm_b = None
            img_b  = img_b.to(device, non_blocking=True)
            mask_b = mask_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                logits = model(img_b)
                losses = (
                    loss_fn(logits, mask_b, loss_mask=lm_b)
                    if lm_b is not None
                    else loss_fn(logits, mask_b)
                )
            if use_fp16:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(
                    _trainable_params(model), max_norm=seg_cfg.grad_clip_norm,
                )
                scaler.step(optimizer); scaler.update()
            else:
                losses["loss"].backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    _trainable_params(model), max_norm=seg_cfg.grad_clip_norm,
                )
                optimizer.step()
            grads.append(float(gn))
            for k in running:
                running[k] += losses[k].item()
            pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")
        n_tb = max(1, len(train_loader))
        train_loss      = running["loss"]      / n_tb
        train_dice_loss = running["dice_loss"] / n_tb
        train_bce_loss  = running["bce_loss"]  / n_tb
        train_pre_loss  = running["pre_loss"]  / n_tb
        train_post_loss = running["post_loss"] / n_tb
        train_time = time.time() - t_train
        scheduler.step()

        # ---- VALIDATE ----
        model.eval()
        val_loss_sum, val_dpre_sum, val_dpost_sum, n_vb = 0.0, 0.0, 0.0, 0
        t_val = time.time()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    img_b, mask_b, lm_b = batch
                    lm_b = lm_b.to(device, non_blocking=True)
                else:
                    img_b, mask_b = batch
                    lm_b = None
                img_b  = img_b.to(device, non_blocking=True)
                mask_b = mask_b.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
                    logits = model(img_b)
                    v_losses = (
                        loss_fn(logits, mask_b, loss_mask=lm_b)
                        if lm_b is not None
                        else loss_fn(logits, mask_b)
                    )
                val_loss_sum += v_losses["loss"].item()
                # Per-channel Dice metric is computed on un-masked predictions
                # so the validation number stays comparable across configs.
                d = compute_dice_metric_per_channel(logits, mask_b)
                val_dpre_sum  += float(d[0])
                val_dpost_sum += float(d[1])
                n_vb += 1
        val_loss      = val_loss_sum / max(1, n_vb)
        val_dice_pre  = val_dpre_sum / max(1, n_vb)
        val_dice_post = val_dpost_sum / max(1, n_vb)
        val_dice_mean = 0.5 * (val_dice_pre + val_dice_post)
        val_time = time.time() - t_val
        val_metric = val_dice_mean

        enc_lr, dec_lr = _get_lrs(optimizer)
        epoch_time = train_time + val_time
        gmean = sum(grads) / max(1, len(grads))
        gmax  = max(grads) if grads else 0.0

        # ---- CHECKPOINTING ----
        improved = ""
        if better(val_metric, best_val_metric):
            best_val_metric = val_metric
            best_epoch = epoch
            improved = " *best*"
            save_checkpoint(
                save_dir / "best_model.pt",
                encoder=model, heads=None,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, val_metric=val_metric, train_loss=train_loss,
                extra={
                    "channel_mean": ch_mean.tolist(),
                    "channel_std":  ch_std.tolist(),
                    "val_dice_pre":  val_dice_pre,
                    "val_dice_post": val_dice_post,
                    "val_dice_mean": val_dice_mean,
                },
            )
        save_checkpoint(
            save_dir / "last.pt",
            encoder=model, heads=None,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, val_metric=val_metric, train_loss=train_loss,
            extra={
                "channel_mean": ch_mean.tolist(),
                "channel_std":  ch_std.tolist(),
            },
        )
        if seg_cfg.save_every_n_epochs and epoch % seg_cfg.save_every_n_epochs == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:04d}.pt",
                encoder=model, heads=None,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                epoch=epoch, val_metric=val_metric, train_loss=train_loss,
            )
            _save_qualitative_grid(
                model, val_seg_ds, qual_idx,
                device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                save_dir=save_dir, epoch=epoch, logger=logger,
            )

        # ---- LOGGING ----
        csv_logger.log(dict(
            epoch=epoch, phase=phase,
            train_loss=train_loss,
            train_dice_loss=train_dice_loss, train_bce_loss=train_bce_loss,
            train_pre_loss=train_pre_loss, train_post_loss=train_post_loss,
            val_loss=val_loss,
            val_dice_pre=val_dice_pre, val_dice_post=val_dice_post,
            val_dice_mean=val_dice_mean,
            lr_encoder=enc_lr, lr_decoder=dec_lr,
            epoch_time_s=epoch_time, train_time_s=train_time, val_time_s=val_time,
            best_val_metric=best_val_metric, best_epoch=best_epoch,
            grad_norm_mean=gmean, grad_norm_max=gmax,
        ))
        logger.info(
            f"ep {epoch:3d}/{seg_cfg.epochs} [{phase}]  "
            f"train={train_loss:.5f} (pre={train_pre_loss:.4f} "
            f"post={train_post_loss:.4f})  val_loss={val_loss:.4f}  "
            f"dice(pre/post/mean)={val_dice_pre:.3f}/{val_dice_post:.3f}/"
            f"{val_dice_mean:.3f}  "
            f"lr(enc/dec)={enc_lr:.2e}/{dec_lr:.2e}  "
            f"t={epoch_time:.1f}s  gn={gmean:.2f}{improved}"
        )
    csv_logger.close()
    logger.info(
        f"total training time = {(time.time() - total_t0) / 60:.1f} min"
    )
    return dict(best_val_metric=best_val_metric, best_epoch=best_epoch)


# ---------------------------------------------------------------------------
# Post-training
# ---------------------------------------------------------------------------
def post_training_flow(
    model: nn.Module,
    data: dict,
    base_cfg: BaseCfg,
    data_cfg: DataCfg,
    model_cfg: ModelCfg,
    coloc_cfg: SynapseColocCfg,
    sessions: list[str] | None,
    device: torch.device,
    save_dir: Path,
    *,
    logger,
):
    """Reload best checkpoint, plot training curves, run one full-MIP
    inference + co-localisation, and save the figures."""
    best_path = save_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        if "encoder_state_dict" in ckpt:
            model.load_state_dict(ckpt["encoder_state_dict"])
        logger.info(
            f"reloaded best model from epoch {ckpt.get('epoch')}  "
            f"dice_mean={ckpt.get('val_dice_mean', ckpt.get('val_metric', float('nan'))):.4f}"
        )
    else:
        logger.warning("no best_model.pt found; using last in-memory weights")

    # ---- Training curves ----
    csv_path = save_dir / "metrics.csv"
    if csv_path.exists():
        rows = list(_csv.DictReader(open(csv_path)))
        if rows:
            ep = [int(r["epoch"]) for r in rows]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
            ax1.plot(ep, [float(r["train_loss"]) for r in rows], label="train total")
            ax1.plot(ep, [float(r["train_pre_loss"])  for r in rows], label="train PRE",  ls="--")
            ax1.plot(ep, [float(r["train_post_loss"]) for r in rows], label="train POST", ls="--")
            ax1.plot(ep, [float(r["val_loss"]) for r in rows], label="val total")
            ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss")
            ax1.legend(); ax1.grid(True, alpha=0.3)
            ax2.plot(ep, [float(r["val_dice_pre"])  for r in rows], label="val Dice PRE")
            ax2.plot(ep, [float(r["val_dice_post"]) for r in rows], label="val Dice POST")
            ax2.plot(ep, [float(r["val_dice_mean"]) for r in rows], label="val Dice mean", lw=2)
            ax2.set_xlabel("epoch"); ax2.set_ylabel("Dice"); ax2.set_title("Dice (vs pseudolabels)")
            ax2.legend(); ax2.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(save_dir / "training_curves.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"training curves -> {save_dir / 'training_curves.png'}")

    # ---- One full-MIP inference + synapse mask ----
    try:
        available = list_image_indices(
            data_cfg.data_root, exclude_patterns=data_cfg.exclude_patterns,
        )
        if not available:
            logger.warning("no full-image indices available; skipping inference demo")
            return
        demo_idx = available[0]
        logger.info(f"demo image index = {demo_idx}")
        full_image, _ = reassemble_image(
            data_cfg.data_root, demo_idx,
            exclude_patterns=data_cfg.exclude_patterns,
        )
        logger.info(f"full image shape = {full_image.shape}")

        ch_mean = data["ch_mean"]; ch_std = data["ch_std"]
        model.eval()
        prob_2ch = sliding_window_predict_multichannel(
            model, full_image,
            patch_size=model_cfg.img_size,
            ch_mean=ch_mean, ch_std=ch_std,
            device=device, overlap=0.5, batch_size=16,
            n_out_channels=2,
        )
        logger.info(
            f"prob_2ch shape = {prob_2ch.shape}  "
            f"PRE [{prob_2ch[0].min():.3f}, {prob_2ch[0].max():.3f}]  "
            f"POST [{prob_2ch[1].min():.3f}, {prob_2ch[1].max():.3f}]"
        )

        # Plot input vs prediction
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for r, name in enumerate(["PRE", "POST"]):
            lo, hi = np.percentile(full_image[r], (1, 99))
            axes[r, 0].imshow(full_image[r], cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-3))
            axes[r, 0].set_title(f"{name} input"); axes[r, 0].axis("off")
            axes[r, 1].imshow(prob_2ch[r], cmap="hot", vmin=0, vmax=1)
            axes[r, 1].set_title(f"{name} prob"); axes[r, 1].axis("off")
        fig.suptitle(f"Full image {demo_idx}: input vs prediction", y=1.001)
        fig.tight_layout()
        fig.savefig(save_dir / "full_image_inference.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        # Synapse mask
        synapse_mask, midpoints, pre_pts, post_pts = build_synapse_mask(
            prob_2ch, coloc_cfg, return_points=True,
        )
        logger.info(
            f"co-localisation: pre={len(pre_pts)}  post={len(post_pts)}  "
            f"matched={len(midpoints)}  mask_px={int(synapse_mask.sum())}"
        )
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, (arr, title) in zip(axes[:2], [
            (full_image[0], "PRE input"),
            (full_image[1], "POST input"),
        ]):
            lo, hi = np.percentile(arr, (1, 99))
            ax.imshow(arr, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-3))
            ax.set_title(title); ax.axis("off")
        lo, hi = np.percentile(full_image[0], (1, 99))
        axes[2].imshow(full_image[0], cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-3))
        overlay = np.zeros((*synapse_mask.shape, 4))
        overlay[synapse_mask > 0] = [1.0, 1.0, 0.0, 0.6]
        axes[2].imshow(overlay)
        axes[2].set_title(
            f"Synapse mask ({int(synapse_mask.sum())} px, {len(midpoints)} sites)"
        )
        axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(save_dir / "synapse_mask.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"post-training figures saved under {save_dir}")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"post-training inference failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint 2-channel (PRE+POST) SwinUNETR fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON config file.")
    p.add_argument("--output-root", type=str, default=None,
                   help="Override base.output_root from config.")
    p.add_argument("--dry-run", action="store_true", default=None,
                   help="Skip the full training loop (still runs setup + sanity).")
    p.add_argument("--resume", type=str, default=None,
                   help="Override base.resume_path from config.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config, root_dir=_ROOT)
    base_cfg     = cfg["base_cfg"]
    data_cfg     = cfg["data_cfg"]
    model_cfg    = cfg["model_cfg"]
    seg_cfg      = cfg["seg_cfg"]
    pseudolabels = cfg["pseudolabels"]
    loss_mask_cfg = cfg["loss_mask_cfg"]
    coloc_cfg    = cfg["coloc_cfg"]
    sessions     = cfg["sessions"]
    run_sanity   = cfg["run_sanity"]
    run_overfit  = cfg["run_overfit"]

    # CLI overrides
    if args.output_root is not None:
        base_cfg.output_root = str(Path(args.output_root).resolve())
    if args.dry_run is not None:
        base_cfg.dry_run = args.dry_run
    if args.resume is not None:
        base_cfg.resume_path = str(Path(args.resume).resolve())

    # Resource cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Output directory and logger
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = (
        Path(base_cfg.output_root)
        / f"{base_cfg.experiment_name}_{base_cfg.tag}_{run_ts}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save_dir = {save_dir}")

    logger = setup_logger("joint_2ch", save_dir / "run.log")
    logger.info(f"experiment = {base_cfg.experiment_name}")
    logger.info(f"tag        = {base_cfg.tag}")
    logger.info(f"method     = {base_cfg.method_name}")
    logger.info(f"init_source= {base_cfg.init_source}")
    logger.info(f"save_dir   = {save_dir}")
    logger.info(f"config     = {args.config}")
    logger.info(f"sessions   = {sessions}")

    dump_config(
        save_dir / "config.json",
        base=base_cfg, data=data_cfg, model=model_cfg, seg=seg_cfg,
        extra={
            "sessions": sessions,
            "pseudolabels": pseudolabels,
            "loss_mask": loss_mask_cfg,
            "colocalisation": dataclasses.asdict(coloc_cfg),
        },
    )
    logger.info("config.json written")

    # Seed and device
    generator = seed_everything(base_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"seed   = {base_cfg.seed}")
    logger.info(f"device = {device}")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"gpu    = {name}  ({mem:.1f} GB)")

    # Data
    data = setup_data(
        data_cfg, model_cfg, seg_cfg, pseudolabels, sessions,
        seed_generator=generator, logger=logger,
        loss_mask_cfg=loss_mask_cfg,
    )
    stats = {
        "channel_names": list(data_cfg.channel_names),
        "mean": data["ch_mean"].tolist(),
        "std":  data["ch_std"].tolist(),
    }
    (save_dir / "channel_stats.json").write_text(json.dumps(stats, indent=2))

    # Sanity / overfit (on a disposable model)
    if run_sanity or run_overfit:
        sanity_model = build_model(base_cfg, model_cfg, device, logger=logger)
        if run_sanity:
            run_sanity_checks(
                sanity_model, data, model_cfg, data_cfg, seg_cfg,
                device, save_dir, logger=logger,
            )
        if run_overfit:
            run_overfit_check(
                model_cfg, seg_cfg, data, device, save_dir, logger=logger,
            )
        del sanity_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.info("sanity/overfit checks skipped")

    # ── Full training ──
    generator = seed_everything(base_cfg.seed)
    model = build_model(base_cfg, model_cfg, device, logger=logger)
    logger.info(f"[reset] model params = {count_params(model) / 1e6:.2f} M")

    optim_state = setup_optimizer(model, seg_cfg, device, logger=logger)

    if base_cfg.dry_run:
        logger.info("dry_run=True -- skipping full training loop")
    else:
        training_loop(
            model, data, optim_state,
            base_cfg, seg_cfg, device, save_dir, logger=logger,
        )

    if not base_cfg.dry_run:
        post_training_flow(
            model, data,
            base_cfg, data_cfg, model_cfg, coloc_cfg, sessions,
            device, save_dir, logger=logger,
        )

    logger.info("done.")


if __name__ == "__main__":
    main()
