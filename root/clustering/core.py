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


@torch.no_grad()
def extract_token_embeddings(
    encoder: nn.Module,
    loader: Iterable,
    *,
    device: torch.device,
    max_batches: int | None = None,
    head_stage_index: int = -1,
    tokens_per_image: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Sample per-position patch-token embeddings for the collapse probe.

    Return flattened ``(N_total, C)`` token features so ``effective_rank`` and
    ``mean_pairwise_cos`` measure token diversity instead of pooled averages.
    Avoid background-dominated collapse signals.
    """
    was_training = encoder.training
    encoder.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        batch = batch.to(device, non_blocking=True).contiguous()
        z = encoder(batch)[head_stage_index]
        if z.dim() != 4:
            raise ValueError(
                f"extract_token_embeddings expected (B, C, H, W); got {tuple(z.shape)}"
            )
        B, C, H, W = z.shape
        n_tokens = H * W
        k = min(tokens_per_image, n_tokens)
        idx = torch.stack(
            [torch.randperm(n_tokens, generator=g)[:k] for _ in range(B)],
        )  # (B, k) on CPU
        z_flat = z.reshape(B, C, n_tokens).transpose(1, 2)  # (B, n_tokens, C)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, C).to(z_flat.device)
        z_sub = torch.gather(z_flat, dim=1, index=idx_exp)  # (B, k, C)
        feats.append(z_sub.reshape(B * k, C).cpu().float())
    if was_training:
        encoder.train()
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


def compute_collapse_stats(Z: torch.Tensor) -> dict:
    """Effective rank + mean pairwise cosine on a feature matrix ``Z``.

    Used by the live collapse probe in the SSL training loop. Accepts
    either pooled embeddings (one row per sample) or per-token embeddings
    (multiple rows per sample); the metric is the same.
    """
    if Z.numel() == 0:
        return {
            "N": 0, "D": 0, "effective_rank": float("nan"),
            "r_max": 0, "mean_pairwise_cos": float("nan"),
        }
    eff, r_max, _ = effective_rank(Z)
    mc = mean_pairwise_cos(Z)
    return {
        "N": int(Z.shape[0]),
        "D": int(Z.shape[1]),
        "effective_rank": float(eff),
        "r_max": int(r_max),
        "mean_pairwise_cos": float(mc),
    }


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
