"""Step-based training runner for SwinUNETR self-training (iter-0 + iter-1).

Public surface mirrors :mod:`synaptic_ssl.ssl_training.runner` but is
purpose-built for the two-stage puncta self-training recipe (``plan.md``
v2). Each iteration is one call to :func:`run_seg_iter`; iter-1
warm-starts from iter-0 weights with a fresh optimiser and scheduler.

The state containers are kept as plain ``@dataclass`` so callers (CLI
scripts or notebooks) can construct them once and pass them around
without dict-of-dicts spaghetti.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..training.checkpoints import find_latest_checkpoint, load_checkpoint, save_checkpoint
from ..training.config import ModelCfg
from ..training.logging import CSVMetricLogger
from ..training.lr_schedule import make_warmup_cosine_steps, param_groups_layer_decay
from .config import SegTrainCfg
from .losses import DiceBCELoss
from .model import (
    build_swinunetr,
    count_params,
    load_full_swinunetr_from_ckpt,
    load_pretrained_encoder_into_swinunetr,
)
from .train_loop import (
    cycle,
    encoder_decoder_lrs,
    freeze_swinvit,
    train_step,
    validate,
)


CSV_FIELDS = [
    "step", "iter_index", "phase",
    "train_loss", "train_dice_loss", "train_bce_loss", "train_dice",
    "val_loss", "val_dice_loss", "val_bce_loss", "val_dice",
    "lr_encoder", "lr_decoder",
    "grad_norm_mean", "grad_norm_max",
    "best_val_metric", "best_step",
    "step_time_s", "val_time_s",
]


# ---------------------------------------------------------------------------
#  State containers
# ---------------------------------------------------------------------------


@dataclass
class SegDataState:
    """Loaders + per-channel normalisation stats.

    Train/val disjointness is the caller's responsibility. The plan
    holds out a full day-folder (e.g. ``20251220``) to avoid field-level
    leakage from neighbouring patches of the same coverslip; a random
    split would put nearby patches of the same image in both train and
    val and silently inflate ``val_dice``.
    """

    train_loader: DataLoader
    val_loader: DataLoader
    ch_mean: torch.Tensor
    ch_std: torch.Tensor
    train_dataset: object = None
    val_dataset: object = None


@dataclass
class SegModelState:
    model: nn.Module
    load_summary: dict | None = None


@dataclass
class SegOptimState:
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    scaler: torch.amp.GradScaler
    start_step: int = 0
    best_val_metric: float = field(default=float("-inf"))
    best_step: int = 0


# ---------------------------------------------------------------------------
#  Setup
# ---------------------------------------------------------------------------


def build_seg_model(
    model_cfg: ModelCfg,
    seg_cfg: SegTrainCfg,
    device: torch.device,
    *,
    pretrained_ckpt_path: str | Path | None = None,
    warm_start_ckpt_path: str | Path | None = None,
    logger=None,
) -> SegModelState:
    """Build SwinUNETR (``out_channels = seg_cfg.out_channels``) and initialise weights.

    Initialisation precedence (mutually exclusive):

    * ``warm_start_ckpt_path``: load the **full** SwinUNETR (encoder +
      decoder) from a prior segmentation checkpoint. Use this for
      iter-1 of the self-training loop -- Bai 2017 §2.1 warm-starts
      S1 from S0 weights end-to-end.
    * ``pretrained_ckpt_path``: load **only** the encoder (``swinViT``)
      from an SSL pretraining checkpoint; the decoder stays random.
      Use this for iter-0.
    * Neither: both encoder and decoder start from random init (rarely
      what you want for puncta segmentation; warn loudly).
    """
    log = logger.info if logger else (lambda *a, **k: None)
    warn = logger.warning if logger else (lambda *a, **k: None)

    if pretrained_ckpt_path is not None and warm_start_ckpt_path is not None:
        raise ValueError(
            "pretrained_ckpt_path and warm_start_ckpt_path are mutually exclusive; "
            "use the warm-start path for iter-1 and the pretrained path for iter-0"
        )

    model = build_swinunetr(model_cfg, out_channels=seg_cfg.out_channels).to(device)
    log(f"SwinUNETR (out_channels={seg_cfg.out_channels}) "
        f"params = {count_params(model) / 1e6:.2f} M")

    load_summary = None
    if warm_start_ckpt_path is not None:
        load_summary = load_full_swinunetr_from_ckpt(
            model, warm_start_ckpt_path, logger=logger,
        )
    elif pretrained_ckpt_path is not None:
        load_summary = load_pretrained_encoder_into_swinunetr(
            model, pretrained_ckpt_path, logger=logger,
        )
    else:
        warn("[seg] no pretrained / warm-start ckpt -- model starts from scratch")

    return SegModelState(model=model, load_summary=load_summary)


def setup_seg_optimizer(
    model_state: SegModelState,
    seg_cfg: SegTrainCfg,
    device: torch.device,
    *,
    resume_path: str | Path | None = None,
    logger=None,
) -> SegOptimState:
    """Layer-decayed encoder groups + flat decoder group; warmup-cosine over steps.

    Optionally resumes from ``last.pt`` (or any file/dir parseable by
    :func:`find_latest_checkpoint`). The resumed step count is read
    from ``extra["global_step"]`` when present, otherwise from the
    ``epoch`` field of the checkpoint (which the runner writes as the
    global step for clarity).
    """
    log = logger.info if logger else (lambda *a, **k: None)
    warn = logger.warning if logger else (lambda *a, **k: None)
    model = model_state.model

    encoder_groups = param_groups_layer_decay(
        model.swinViT,
        base_lr=seg_cfg.base_lr,
        weight_decay=seg_cfg.weight_decay,
        layer_decay=seg_cfg.layer_decay,
    )
    # param_groups_layer_decay silently drops params with requires_grad=False
    # at construction (lr_schedule.py:46-48). If a caller reuses a model
    # object after a prior run_seg_iter froze swinViT, the new optimiser
    # would never see those tensors. Fail loud rather than silently
    # under-train the encoder.
    if not any(g["params"] for g in encoder_groups):
        raise RuntimeError(
            "setup_seg_optimizer: no encoder params are trainable. "
            "Either model.swinViT was pre-frozen (call freeze_swinvit(model, False) "
            "before setup) or layer-decay groups were built on the wrong module."
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
        optimizer,
        make_warmup_cosine_steps(seg_cfg.warmup_steps, seg_cfg.train_steps),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    start_step = 0
    best_val_metric = (
        float("-inf") if seg_cfg.val_metric_direction == "max" else float("inf")
    )
    best_step = 0

    if resume_path is not None:
        p = Path(resume_path)
        if p.is_dir():
            p = find_latest_checkpoint(p)
        if p is None or not Path(p).exists():
            warn(f"[seg] resume path {resume_path!r} not found -- starting fresh")
        else:
            ckpt = load_checkpoint(
                p, encoder=model,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                map_location=device,
            )
            start_step = int(
                ckpt.get("global_step", ckpt.get("epoch", 0)) or 0
            )
            best_val_metric = ckpt.get("val_metric", best_val_metric) or best_val_metric
            best_step = start_step
            log(f"[seg] resumed from {p} (step {start_step})")

    log(f"encoder param groups = {len(encoder_groups)}  "
        f"decoder params = {sum(p.numel() for p in decoder_params) / 1e6:.2f} M")
    log(f"start_step = {start_step}  best_val_metric = {best_val_metric:.4f}")

    return SegOptimState(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        start_step=start_step,
        best_val_metric=best_val_metric,
        best_step=best_step,
    )


# ---------------------------------------------------------------------------
#  Iteration runner
# ---------------------------------------------------------------------------


def _better(direction: str):
    return (lambda new, best: new > best) if direction == "max" else (lambda new, best: new < best)


def _get_head_out_channels(model) -> int | None:
    """Read the SwinUNETR head out_channels from its state_dict.

    MONAI's SwinUNETR ``out`` is a ``Convolution`` block; the actual
    Conv2d weight lives at ``out.conv.conv.weight`` in the MONAI
    version in use. Fall back to other plausible keys for forward-compat.
    """
    sd = model.state_dict()
    for key in ("out.conv.conv.weight", "out.conv.weight", "out.weight"):
        if key in sd:
            return int(sd[key].shape[0])
    # last-ditch: any 4-D tensor under "out." that looks like a 1x1 conv
    for k, v in sd.items():
        if k.startswith("out.") and k.endswith(".weight") and v.ndim == 4:
            return int(v.shape[0])
    return None


def _save(
    save_path: Path,
    *,
    model: nn.Module,
    optim_state: SegOptimState,
    step: int,
    iter_index: int,
    val_metric: float | None,
    train_loss: float | None,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    extra: dict | None = None,
) -> None:
    out_channels_actual = _get_head_out_channels(model)
    payload_extra = {
        "global_step":  int(step),
        "iter_index":   int(iter_index),
        "out_channels": out_channels_actual,
        "channel_mean": ch_mean.tolist(),
        "channel_std":  ch_std.tolist(),
    }
    if extra:
        payload_extra.update(extra)
    save_checkpoint(
        save_path,
        encoder=model, heads=None,
        optimizer=optim_state.optimizer,
        scheduler=optim_state.scheduler,
        scaler=optim_state.scaler,
        epoch=int(step),               # step stored in the ``epoch`` field for legacy compat
        val_metric=val_metric,
        train_loss=train_loss,
        extra=payload_extra,
    )


def run_seg_iter(
    *,
    seg_cfg: SegTrainCfg,
    data_state: SegDataState,
    model_state: SegModelState,
    optim_state: SegOptimState,
    loss_fn: DiceBCELoss,
    device: torch.device,
    save_dir: str | Path,
    iter_save_name: str,
    logger=None,
    dry_run: bool = False,
) -> dict:
    """Run one self-training iteration in step-mode.

    Trains for ``seg_cfg.train_steps`` optimiser steps (resuming from
    ``optim_state.start_step``). Encoder is frozen while
    ``step < seg_cfg.freeze_encoder_steps`` and unfrozen exactly once
    when that gate fires; this avoids the per-step ``requires_grad``
    toggling that would otherwise litter the inner loop.

    Validates every ``seg_cfg.val_every_n_steps`` and checkpoints to
    ``save_dir/last.pt``. On val-metric improvement also writes
    ``save_dir/<iter_save_name>``. Periodic ``step_<N>.pt`` snapshots
    every ``seg_cfg.save_every_n_steps``.

    Returns the final result dict (best step, best metric, total steps).
    """
    log = logger.info if logger else (lambda *a, **k: None)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        log("[seg] dry_run=True -- skipping training loop")
        return {
            "iter_index": seg_cfg.iter_index,
            "best_step": optim_state.best_step,
            "best_val_metric": optim_state.best_val_metric,
            "final_step": optim_state.start_step,
        }

    model = model_state.model
    better = _better(seg_cfg.val_metric_direction)

    # Encoder freeze gate. We snapshot the desired state and only call
    # freeze_swinvit() when it has to change (saves one `requires_grad`
    # rewrite per parameter per step).
    #
    # DDP caveat: mutating ``requires_grad`` mid-training under
    # DistributedDataParallel rebuilds reducer buckets (or triggers a
    # static-graph error). This runner targets single-process training;
    # if you wrap the model in DDP later, either pre-decide the freeze
    # state before wrapping, or use a DDP-safe progressive unfreeze.
    encoder_currently_frozen = optim_state.start_step < seg_cfg.freeze_encoder_steps
    freeze_swinvit(model, encoder_currently_frozen)
    log(f"[seg iter {seg_cfg.iter_index}] encoder "
        f"{'FROZEN' if encoder_currently_frozen else 'UNFROZEN'} at start "
        f"(step {optim_state.start_step}/{seg_cfg.train_steps})")

    csv_logger = CSVMetricLogger(save_dir / "metrics.csv", CSV_FIELDS)

    data_iter = cycle(data_state.train_loader)
    train_acc_loss = 0.0
    train_acc_dice_l = 0.0
    train_acc_bce_l = 0.0
    train_acc_dice = 0.0
    grad_norms: list[float] = []
    train_step_times: list[float] = []
    n_acc = 0

    val_every = max(1, seg_cfg.val_every_n_steps)
    save_every = max(1, seg_cfg.save_every_n_steps)

    total_t0 = time.time()
    step = optim_state.start_step

    while step < seg_cfg.train_steps:
        # encoder freeze gate (cheap; only mutates when boundary crossed)
        should_freeze = step < seg_cfg.freeze_encoder_steps
        if should_freeze != encoder_currently_frozen:
            freeze_swinvit(model, should_freeze)
            encoder_currently_frozen = should_freeze
            log(f"[seg iter {seg_cfg.iter_index}] step {step}: encoder "
                f"{'FROZEN' if should_freeze else 'UNFROZEN'}")

        t0 = time.time()
        batch = next(data_iter)
        metrics = train_step(
            model, batch,
            optimizer=optim_state.optimizer,
            scheduler=optim_state.scheduler,
            scaler=optim_state.scaler,
            loss_fn=loss_fn,
            grad_clip_norm=seg_cfg.grad_clip_norm,
            device=device,
        )
        train_step_times.append(time.time() - t0)
        train_acc_loss += metrics["loss"]
        train_acc_dice_l += metrics["dice_loss"]
        train_acc_bce_l += metrics["bce_loss"]
        train_acc_dice += metrics["dice"]
        grad_norms.append(metrics["grad_norm"])
        n_acc += 1
        step += 1

        do_validate = (step % val_every == 0) or (step >= seg_cfg.train_steps)
        do_save_periodic = (step % save_every == 0)

        if not (do_validate or do_save_periodic):
            continue

        # ---- summarise the window since the last log ----
        train_loss = train_acc_loss / n_acc
        train_dice_l = train_acc_dice_l / n_acc
        train_bce_l = train_acc_bce_l / n_acc
        train_dice = train_acc_dice / n_acc
        gn_mean = sum(grad_norms) / max(1, len(grad_norms))
        gn_max = max(grad_norms) if grad_norms else 0.0
        step_time_mean = sum(train_step_times) / max(1, len(train_step_times))
        train_acc_loss = train_acc_dice_l = train_acc_bce_l = train_acc_dice = 0.0
        grad_norms.clear()
        train_step_times.clear()
        n_acc = 0

        val_metrics: dict = {}
        val_time = 0.0
        improved = ""
        if do_validate:
            tval = time.time()
            val_metrics = validate(
                model, data_state.val_loader,
                loss_fn=loss_fn, device=device,
            )
            val_time = time.time() - tval

            metric = val_metrics["val_dice"]
            if better(metric, optim_state.best_val_metric):
                optim_state.best_val_metric = metric
                optim_state.best_step = step
                improved = " *best*"
                _save(
                    save_dir / iter_save_name,
                    model=model, optim_state=optim_state,
                    step=step, iter_index=seg_cfg.iter_index,
                    val_metric=metric, train_loss=train_loss,
                    ch_mean=data_state.ch_mean, ch_std=data_state.ch_std,
                )

        # ---- always write last.pt at every log boundary ----
        _save(
            save_dir / "last.pt",
            model=model, optim_state=optim_state,
            step=step, iter_index=seg_cfg.iter_index,
            val_metric=val_metrics.get("val_dice"),
            train_loss=train_loss,
            ch_mean=data_state.ch_mean, ch_std=data_state.ch_std,
        )
        if do_save_periodic:
            _save(
                save_dir / f"step_{step:07d}.pt",
                model=model, optim_state=optim_state,
                step=step, iter_index=seg_cfg.iter_index,
                val_metric=val_metrics.get("val_dice"),
                train_loss=train_loss,
                ch_mean=data_state.ch_mean, ch_std=data_state.ch_std,
            )

        enc_lr, dec_lr = encoder_decoder_lrs(optim_state.optimizer)
        phase = "frozen" if encoder_currently_frozen else "full"

        csv_logger.log(dict(
            step=step, iter_index=seg_cfg.iter_index, phase=phase,
            train_loss=train_loss,
            train_dice_loss=train_dice_l,
            train_bce_loss=train_bce_l,
            train_dice=train_dice,
            val_loss=val_metrics.get("val_loss"),
            val_dice_loss=val_metrics.get("val_dice_loss"),
            val_bce_loss=val_metrics.get("val_bce_loss"),
            val_dice=val_metrics.get("val_dice"),
            lr_encoder=enc_lr, lr_decoder=dec_lr,
            grad_norm_mean=gn_mean, grad_norm_max=gn_max,
            best_val_metric=optim_state.best_val_metric,
            best_step=optim_state.best_step,
            step_time_s=step_time_mean,
            val_time_s=val_time,
        ))

        if do_validate:
            log(
                f"[seg iter {seg_cfg.iter_index}] step {step:7d}/{seg_cfg.train_steps} "
                f"[{phase}]  train={train_loss:.5f}  "
                f"val_loss={val_metrics['val_loss']:.5f}  "
                f"val_dice={val_metrics['val_dice']:.4f}  "
                f"lr(enc/dec)={enc_lr:.2e}/{dec_lr:.2e}  "
                f"gn={gn_mean:.2f}  t/step={step_time_mean*1000:.0f}ms{improved}"
            )
        else:
            log(
                f"[seg iter {seg_cfg.iter_index}] step {step:7d}/{seg_cfg.train_steps} "
                f"[{phase}]  train={train_loss:.5f}  "
                f"lr(enc/dec)={enc_lr:.2e}/{dec_lr:.2e}  "
                f"gn={gn_mean:.2f}  t/step={step_time_mean*1000:.0f}ms (no val)"
            )

    csv_logger.close()
    total_time = time.time() - total_t0
    log(f"[seg iter {seg_cfg.iter_index}] done in {total_time / 60:.1f} min  "
        f"best_step={optim_state.best_step}  "
        f"best_val={optim_state.best_val_metric:.4f}")

    return {
        "iter_index": seg_cfg.iter_index,
        "best_step": optim_state.best_step,
        "best_val_metric": optim_state.best_val_metric,
        "final_step": step,
        "total_seconds": total_time,
    }
