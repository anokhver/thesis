"""Checkpoint save / load helpers + latest-checkpoint discovery."""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn


def _heads_state_dict(heads):
    if heads is None:
        return None
    if isinstance(heads, nn.Module):
        return heads.state_dict()
    if isinstance(heads, dict):
        return {k: v.state_dict() for k, v in heads.items()}
    raise TypeError(f"Unsupported heads type: {type(heads)}")


def _load_heads_state_dict(heads, sd):
    if sd is None or heads is None:
        return
    if isinstance(heads, nn.Module):
        heads.load_state_dict(sd)
    else:
        for k, mod in heads.items():
            if k in sd:
                mod.load_state_dict(sd[k])


def save_checkpoint(
    path: str | Path,
    *,
    encoder: nn.Module,
    heads,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    val_metric: float,
    train_loss: float,
    extra: dict | None = None,
) -> None:
    """Persist a full training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":                int(epoch),
        "encoder_state_dict":   encoder.state_dict(),
        "heads_state_dict":     _heads_state_dict(heads),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
        "val_metric":           float(val_metric) if val_metric is not None else None,
        "train_loss":           float(train_loss) if train_loss is not None else None,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    encoder: nn.Module,
    heads=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location=None,
) -> dict:
    """Restore everything that was saved by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    _load_heads_state_dict(heads, ckpt.get("heads_state_dict"))
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


_EPOCH_RE = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)


def find_latest_checkpoint(
    search_root: str | Path,
    *,
    pattern: str = "*.pt",
    prefer_filename: str = "last.pt",
) -> Path | None:
    """Return the most recent checkpoint under ``search_root``.

    Resolution order:
      1. If a file named ``prefer_filename`` exists *anywhere* under
         ``search_root``, the most recently modified one is returned.
      2. Otherwise, files are ranked by the largest ``epoch_<n>`` integer
         in the filename, then by mtime as a tiebreaker.
      3. Otherwise, returns ``None``.
    """
    root = Path(search_root)
    if not root.exists():
        return None
    candidates = list(root.rglob(pattern))
    if not candidates:
        return None
    last = [p for p in candidates if p.name == prefer_filename]
    if last:
        return max(last, key=lambda p: p.stat().st_mtime)

    def key(p: Path) -> tuple:
        m = _EPOCH_RE.search(p.name)
        ep = int(m.group(1)) if m else -1
        return (ep, p.stat().st_mtime)

    return max(candidates, key=key)
