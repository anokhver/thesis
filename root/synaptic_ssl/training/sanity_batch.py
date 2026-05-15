"""Fixed sanity-batch indices and overfit-on-batch helper."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .seeding import isolated_rng


# Fixed canonical indices into the *raw* PatchDataset (i.e. before train/val
# split). Picked once and frozen so all sanity figures across templates use
# the same images. If a value is ever out of range for a smaller dataset
# the sanity helpers will gracefully cap to ``len(dataset) - 1``.
SANITY_TRAIN_INDICES: tuple[int, ...] = (3, 4, 50, 200, 400)
SANITY_VAL_INDICES:   tuple[int, ...] = (1, 7, 23)


def _safe_indices(indices: Iterable[int], size: int) -> list[int]:
    return [min(int(i), size - 1) for i in indices if size > 0]


def fixed_two_view_batch(
    subset,
    indices: Iterable[int],
    transform: Callable,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Apply ``transform`` (two-view) at fixed seed. Restore outer RNG on exit."""
    idxs = _safe_indices(indices, len(subset))
    v1, v2 = [], []
    with isolated_rng(seed):
        for i in idxs:
            a, b = transform(subset[i])
            v1.append(a)
            v2.append(b)
    return torch.stack(v1), torch.stack(v2), idxs


def fixed_single_view_batch(
    subset,
    indices: Iterable[int],
    transform: Callable,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """Apply ``transform`` (single-view) at fixed seed. Restore outer RNG on exit."""
    idxs = _safe_indices(indices, len(subset))
    out = []
    with isolated_rng(seed):
        for i in idxs:
            out.append(transform(subset[i]))
    return torch.stack(out), idxs


def overfit_on_batch(
    encoder: nn.Module,
    heads: dict[str, nn.Module],
    view1: torch.Tensor,
    view2: torch.Tensor,
    ssl_cfg,
    *,
    n_steps: int = 500,
    lr: float = 1e-4,
    grad_clip: float = 5.0,
    fixed_mask: torch.Tensor | None = None,
    restore_state: bool = False,
    logger=None,
) -> dict:
    """Run ``n_steps`` of overfit on a fixed two-view batch.

    Mutates encoder + heads in place. Set ``restore_state=True`` to snapshot
    and restore weights + RNG on exit. Returns ``{history, recon_drop, n_steps}``.
    """
    from .losses import compute_simmim_vicreg_loss

    log = (logger.info if logger is not None else print)

    if restore_state:
        from copy import deepcopy
        from .seeding import RNGSnapshot
        rng = RNGSnapshot()
        saved_enc = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        saved_heads = deepcopy({k: v.state_dict() for k, v in heads.items()})

    head_params = [p for h in heads.values() for p in h.parameters()]
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + head_params, lr=lr, weight_decay=0.0,
    )
    encoder.train()
    for h in heads.values():
        h.train()

    history = {"loss": [], "recon": [], "fourier": [], "sim": [], "std": [], "cov": []}
    for step in range(n_steps):
        loss, metrics = compute_simmim_vicreg_loss(
            encoder, heads, view1, view2, ssl_cfg, fixed_mask=fixed_mask,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + head_params, max_norm=grad_clip,
        )
        optimizer.step()
        for k in history:
            history[k].append(metrics[k] if k in metrics else float(loss.item()))

    drop = (
        (history["recon"][0] - history["recon"][-1]) / max(history["recon"][0], 1e-8)
        if history["recon"] else 0.0
    )
    log(
        f"[overfit] {n_steps} steps  "
        f"recon {history['recon'][0]:.4f} -> {history['recon'][-1]:.4f}  "
        f"({drop * 100:.1f}% drop)"
    )

    if restore_state:
        encoder.load_state_dict(saved_enc)
        for k, sd in saved_heads.items():
            heads[k].load_state_dict(sd)
        rng.restore()
        log("[overfit] state restored (encoder + heads + RNG)")

    return {"history": history, "recon_drop": drop, "n_steps": n_steps}
