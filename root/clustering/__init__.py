"""Embedding extraction, collapse diagnostics, and patch clustering pipeline."""

from .core import (
    extract_pooled_embeddings, extract_token_embeddings,
    compute_collapse_stats, effective_rank, mean_pairwise_cos, reduce_2d,
)

__all__ = [
    "extract_pooled_embeddings", "extract_token_embeddings",
    "compute_collapse_stats", "effective_rank", "mean_pairwise_cos", "reduce_2d",
]