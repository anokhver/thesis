"""Seeding utilities and an RNG snapshot/restore context manager."""

from __future__ import annotations

import random as _py_random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and torch (CPU + CUDA). Returns a torch.Generator."""
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


class RNGSnapshot:
    """Snapshot of all four RNGs (python / numpy / torch CPU / torch CUDA)."""

    def __init__(self) -> None:
        self.py = _py_random.getstate()
        self.np = np.random.get_state()
        self.torch_cpu = torch.get_rng_state()
        self.torch_cuda = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

    def restore(self) -> None:
        _py_random.setstate(self.py)
        np.random.set_state(self.np)
        torch.set_rng_state(self.torch_cpu)
        if self.torch_cuda is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda)


@contextmanager
def isolated_rng(seed: int):
    """Temporarily seed all RNGs and restore the previous state on exit."""
    snap = RNGSnapshot()
    try:
        _py_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        snap.restore()
