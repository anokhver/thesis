"""Post-training visualisation steps: checkpoint reload, reconstruction, curves."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..training.checkpoints import load_checkpoint
from ..training.masking import apply_mask, random_block_mask
from ..training.sanity_batch import (
    SANITY_TRAIN_INDICES,
    SANITY_VAL_INDICES,
    fixed_single_view_batch,
    fixed_two_view_batch,
)
from ..training.viz import (
    plot_loss_curves,
    plot_recon_panel,
)


# --------------------------------------------------------------------- helpers

def build_run_label(
    base_cfg=None,
    *,
    method_name: str | None = None,
    init_source: str | None = None,
    epoch: int | float | None = None,
    val_metric: float | None = None,
    extra: str | None = None,
) -> str:
    """Build run identifier string for figure labels.

    Accepts ``base_cfg`` or individual fields. Returns ``""`` if nothing supplied.
    """
    if base_cfg is not None:
        method_name = method_name or getattr(base_cfg, "method_name", None)
        init_source = init_source or getattr(base_cfg, "init_source", None)
    parts: list[str] = []
    if method_name:
        parts.append(f"method={method_name}")
    if init_source:
        parts.append(f"init={init_source}")
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    if val_metric is not None:
        parts.append(f"val={val_metric:.4f}")
    if extra:
        parts.append(extra)
    return "  |  ".join(parts)


@torch.no_grad()
def eval_recon_batch(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    view: torch.Tensor,
    mask: torch.Tensor,
    head_stage_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run masked encoder + decoder once. Returns ``(recon, masked_view)``.

    ``head_stage_index`` must match the value used to build
    ``heads['decoder']``; mismatch breaks the decoder input channel count.
    """
    encoder.eval()
    for h in heads.values():
        h.eval()
    v_masked = apply_mask(view, mask, heads["mask_token"])
    z = encoder(v_masked.contiguous())[head_stage_index]
    recon = heads["decoder"](z).float()
    return recon, v_masked.float()


# ---------------------------------------------------------------- step 1: ckpt

def reload_best_checkpoint(
    save_dir: str | Path,
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    device: torch.device,
    *,
    name: str = "best_model.pt",
    logger=None,
) -> dict | None:
    """Reload ``best_model.pt`` if it exists; warn and no-op otherwise.

    Returns the checkpoint dict, or ``None`` when no checkpoint was found.
    """
    save_dir = Path(save_dir)
    best_path = save_dir / name
    log = (logger.info if logger is not None else print)
    warn = (logger.warning if logger is not None else print)
    if best_path.exists():
        ckpt = load_checkpoint(
            best_path, encoder=encoder, heads=heads, map_location=device,
        )
        log(
            f"reloaded {best_path.name}: epoch={ckpt.get('epoch')}  "
            f"val={ckpt.get('val_metric')}"
        )
        return ckpt
    warn("no best_model.pt -- post-training viz uses current weights")
    return None


# ------------------------------------------------------------- step 2/3: recon

def post_training_reconstruction(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    subset,
    transform: Callable,
    ssl_cfg,
    *,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    save_dir: str | Path,
    base_seed: int,
    view: str = "train",
    indices: Sequence[int] | None = None,
    suptitle: str | None = None,
    save_name: str | None = None,
    run_label: str | None = None,
    channel_names: Sequence[str] | None = None,
    device: torch.device | None = None,
    show: bool = True,
):
    """Reconstruct canonical sanity batch and plot.

    ``view='train'``: two-view augmented batch from ``SANITY_TRAIN_INDICES``.
    ``view='val'``: single-view (no aug) from ``SANITY_VAL_INDICES``.
    """
    save_dir = Path(save_dir)
    if device is None:
        device = next(encoder.parameters()).device

    if view == "train":
        idxs = list(indices) if indices is not None else list(SANITY_TRAIN_INDICES)
        x1, _x2, used_idx = fixed_two_view_batch(
            subset, idxs, transform, seed=base_seed + 17,
        )
        x = x1.to(device)
        torch.manual_seed(base_seed + 19)
        title = suptitle or "POST: canonical TRAIN sanity batch"
        out_name = save_name or "post_recon_train.png"
    elif view == "val":
        idxs = list(indices) if indices is not None else list(SANITY_VAL_INDICES)
        x_v, used_idx = fixed_single_view_batch(
            subset, idxs, transform, seed=base_seed + 21,
        )
        x = x_v.to(device)
        torch.manual_seed(base_seed + 22)
        title = suptitle or "POST: canonical VAL patches (no aug)"
        out_name = save_name or "post_recon_val.png"
    else:
        raise ValueError(f"view must be 'train' or 'val', got {view!r}")

    mask = random_block_mask(x, ssl_cfg.mask_block_size, ssl_cfg.mask_ratio).clone()
    recon, v_masked = eval_recon_batch(encoder, heads, x, mask, ssl_cfg.head_stage_index)

    fig = plot_recon_panel(
        x, v_masked, recon, mask, used_idx,
        ch_mean=ch_mean, ch_std=ch_std,
        suptitle=title,
        save_to=save_dir / out_name,
        run_label=run_label,
        channel_names=channel_names,
    )
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------- step 4: curves

def plot_post_training_curves(
    save_dir: str | Path,
    *,
    csv_name: str = "metrics.csv",
    suptitle: str = "Training curves",
    save_name: str = "training_curves.png",
    run_label: str | None = None,
    logger=None,
    show: bool = True,
    **kwargs,
):
    """Plot ``metrics.csv`` training curves if present.

    Extra kwargs forwarded to :func:`training.viz.plot_loss_curves`.
    """
    save_dir = Path(save_dir)
    csv_path = save_dir / csv_name
    if not csv_path.exists():
        warn = (logger.warning if logger is not None else print)
        warn(f"no metrics.csv at {csv_path}")
        return None
    fig = plot_loss_curves(
        csv_path,
        suptitle=suptitle,
        save_to=save_dir / save_name,
        run_label=run_label,
        **kwargs,
    )
    if show:
        plt.show()
    return fig
