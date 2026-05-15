"""Sliding-window inference for full-image segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]  # root/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils_data.reassemble import reassemble_image  # noqa: E402


def sliding_window_predict(
    model: nn.Module,
    full_image: np.ndarray,
    patch_size: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
) -> np.ndarray:
    """Sliding-window inference on a ``(C, H, W)`` float32 image.

    Returns ``(H, W)`` float32 probability map in ``[0, 1]``.
    """
    C, H, W = full_image.shape
    stride = max(1, int(patch_size * (1 - overlap)))

    ch_mean_np = ch_mean.view(-1, 1, 1).numpy()
    ch_std_np = ch_std.view(-1, 1, 1).numpy()

    # pad image so windows tile evenly
    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else patch_size - H
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else patch_size - W
    padded = np.pad(
        full_image,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    pH, pW = padded.shape[1], padded.shape[2]

    acc = np.zeros((pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)

    # collect all window positions
    positions = []
    for y in range(0, pH - patch_size + 1, stride):
        for x in range(0, pW - patch_size + 1, stride):
            positions.append((y, x))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_pos = positions[i : i + batch_size]
            patches = []
            for y, x in batch_pos:
                p = padded[:, y : y + patch_size, x : x + patch_size].copy()
                p = (p - ch_mean_np) / ch_std_np
                patches.append(p)
            batch_t = torch.from_numpy(np.stack(patches)).float().to(device)
            logits = model(batch_t)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)
            for j, (y, x) in enumerate(batch_pos):
                acc[y : y + patch_size, x : x + patch_size] += probs[j]
                count[y : y + patch_size, x : x + patch_size] += 1.0

    count = np.maximum(count, 1.0)
    prob_map = (acc / count)[:H, :W]
    return prob_map.astype(np.float32)


def predict_full_image(
    model: nn.Module,
    patch_root: str | Path,
    image_index: int,
    patch_size: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
    exclude_patterns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble a full image and run sliding-window prediction.

    Returns ``(full_image, prob_map)`` as numpy arrays.
    """
    full_image, _records = reassemble_image(
        patch_root, image_index, exclude_patterns=exclude_patterns,
    )
    prob_map = sliding_window_predict(
        model, full_image, patch_size, ch_mean, ch_std,
        device, overlap=overlap, batch_size=batch_size,
    )
    return full_image, prob_map
