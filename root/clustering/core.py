"""Embedding extraction and 2-D projection helpers."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


@torch.no_grad()
def extract_pooled_embeddings(
    encoder: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Return globally-pooled deepest-stage features as a CPU tensor."""
    was_training = encoder.training
    encoder.eval()
    feats = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device, non_blocking=True).contiguous()
        z = encoder(batch)[-1]
        z = z.mean(dim=(2, 3))
        feats.append(z.cpu())
    if was_training:
        encoder.train()
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


def reduce_2d(Z: torch.Tensor, *, method: str = "auto", seed: int = 0):
    """Project Z to 2D.

    method: 'auto' | 'umap' | 'pca'. Returns (Z2, used_method).
    Falls back to torch ``pca_lowrank`` if UMAP is unavailable.
    """
    Zn = Z.cpu().numpy()
    if method in ("auto", "umap"):
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(Zn), "umap"
        except Exception:
            if method == "umap":
                raise
    Zc = Z - Z.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(Zc, q=2)
    return (Zc @ V).cpu().numpy(), "pca"
