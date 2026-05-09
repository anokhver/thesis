"""Embedding extraction + collapse / cluster diagnostics.

References
----------
``effective_rank``  -- Roy & Vetterli, "The effective rank: A measure of
    effective dimensionality", EUSIPCO 2007. Defined as
    ``exp(H(p))`` where ``p_i = sigma_i^2 / sum sigma_j^2`` are the
    normalised squared singular values of the centred feature matrix.
``mean_pairwise_cos`` -- Standard collapse diagnostic (e.g. Hua et al.
    "On Feature Decorrelation in Self-Supervised Learning", ICCV 2021).
``reduce_2d``       -- UMAP (McInnes, Healy, Melville, 2018) when the
    package is available, falling back to torch ``pca_lowrank``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def effective_rank(Z: torch.Tensor) -> tuple[float, int, np.ndarray]:
    """Entropy-of-singular-values rank estimate.

    Returns ``(eff_rank, max_possible_rank, singular_values)``.
    """
    if Z.numel() == 0:
        return 0.0, 0, np.zeros(0)
    Zc = (Z - Z.mean(dim=0, keepdim=True)).float()
    S = torch.linalg.svdvals(Zc)
    s2 = (S ** 2) / (S ** 2).sum().clamp_min(1e-12)
    eff = float(torch.exp(-(s2 * torch.log(s2 + 1e-12)).sum()).item())
    r_max = int(min(Z.shape[0] - 1, Z.shape[1]))
    return eff, r_max, S.cpu().numpy()


def mean_pairwise_cos(Z: torch.Tensor, sample: int = 1024) -> float:
    """Mean off-diagonal cosine similarity (close to 1.0 == collapse)."""
    if Z.numel() == 0:
        return float("nan")
    Zn = F.normalize(Z, dim=1)
    n = min(sample, Z.shape[0])
    sub = Zn[torch.randperm(Z.shape[0])[:n]]
    cos = sub @ sub.t()
    mask = ~torch.eye(n, dtype=torch.bool)
    return float(cos[mask].mean().item())


def reduce_2d(Z: torch.Tensor, *, method: str = "auto", seed: int = 0):
    """Project to 2D. ``method``: ``'auto' | 'umap' | 'pca'``.

    Returns ``(Z2, used_method)``.
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
