"""Embedding extraction and patch clustering pipeline."""

from .core import (
    extract_pooled_embeddings, reduce_2d,
)

__all__ = [
    "extract_pooled_embeddings", "reduce_2d",
]