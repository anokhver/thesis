"""Sliding-window inference for full-image segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..utils_data.reassemble import reassemble_image


# D4 symmetry group: 8 elements built from hflip, vflip, rot90.
# Each entry is (apply_to_input, undo_on_output). Inverses are picked so
# undo(model(apply(x))) is the model output in the original frame.
_D4_TRANSFORMS = (
    (lambda x: x,
     lambda y: y),
    (lambda x: torch.flip(x, dims=(-1,)),
     lambda y: torch.flip(y, dims=(-1,))),
    (lambda x: torch.flip(x, dims=(-2,)),
     lambda y: torch.flip(y, dims=(-2,))),
    (lambda x: torch.flip(x, dims=(-2, -1)),
     lambda y: torch.flip(y, dims=(-2, -1))),
    (lambda x: torch.rot90(x, 1, dims=(-2, -1)),
     lambda y: torch.rot90(y, -1, dims=(-2, -1))),
    (lambda x: torch.rot90(x, 2, dims=(-2, -1)),
     lambda y: torch.rot90(y, -2, dims=(-2, -1))),
    (lambda x: torch.rot90(x, 3, dims=(-2, -1)),
     lambda y: torch.rot90(y, -3, dims=(-2, -1))),
    (lambda x: torch.rot90(torch.flip(x, dims=(-1,)), 1, dims=(-2, -1)),
     lambda y: torch.flip(torch.rot90(y, -1, dims=(-2, -1)), dims=(-1,))),
)


def predict_d4_tta(model: nn.Module, batch_t: torch.Tensor) -> torch.Tensor:
    """Average sigmoid probabilities over the 8 D4 symmetries.

    ``batch_t`` is ``(B, C, H, W)``. Returns ``(B, 1, H, W)`` mean probability.
    Only flips and 90-degree rotations are used; these are the same geometric
    augmentations applied at training time, so they preserve the puncta scale
    and the channel identities (synapsin / PSD / MAP2).
    """
    acc = None
    for apply_fn, undo_fn in _D4_TRANSFORMS:
        xv = apply_fn(batch_t)
        logits = model(xv)
        prob = torch.sigmoid(logits)
        prob = undo_fn(prob)
        acc = prob if acc is None else acc + prob
    return acc / float(len(_D4_TRANSFORMS))


def sliding_window_predict(
    model: nn.Module,
    full_image: np.ndarray,
    patch_size: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
    use_tta: bool = False,
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
            if use_tta:
                probs = predict_d4_tta(model, batch_t).cpu().numpy()[:, 0]
            else:
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
    use_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble a full image and run sliding-window prediction.

    Returns ``(full_image, prob_map)`` as numpy arrays.
    """
    full_image, _records = reassemble_image(
        patch_root, image_index, exclude_patterns=exclude_patterns,
    )
    prob_map = sliding_window_predict(
        model, full_image, patch_size, ch_mean, ch_std,
        device, overlap=overlap, batch_size=batch_size, use_tta=use_tta,
    )
    return full_image, prob_map


def sliding_window_predict_multichannel(
    model: nn.Module,
    full_image: np.ndarray,
    patch_size: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
    n_out_channels: int = 2,
) -> np.ndarray:
    """Multi-channel sliding-window inference on ``(C, H, W)``.

    Identical to :func:`sliding_window_predict` but keeps **all** output
    channels. Returns ``(n_out_channels, H, W)`` float32 in ``[0, 1]``.
    """
    C, H, W = full_image.shape
    stride = max(1, int(patch_size * (1 - overlap)))

    ch_mean_np = ch_mean.view(-1, 1, 1).numpy()
    ch_std_np = ch_std.view(-1, 1, 1).numpy()

    pad_h = (stride - (H - patch_size) % stride) % stride if H > patch_size else patch_size - H
    pad_w = (stride - (W - patch_size) % stride) % stride if W > patch_size else patch_size - W
    padded = np.pad(
        full_image,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )
    pH, pW = padded.shape[1], padded.shape[2]

    acc = np.zeros((n_out_channels, pH, pW), dtype=np.float64)
    count = np.zeros((pH, pW), dtype=np.float64)

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
            if logits.size(1) < n_out_channels:
                raise ValueError(
                    f"model produced {logits.size(1)} channels, requested {n_out_channels}"
                )
            probs = torch.sigmoid(logits[:, :n_out_channels]).cpu().numpy()  # (B, C, H, W)
            for j, (y, x) in enumerate(batch_pos):
                acc[:, y : y + patch_size, x : x + patch_size] += probs[j]
                count[y : y + patch_size, x : x + patch_size] += 1.0

    count = np.maximum(count, 1.0)
    prob_map = (acc / count[None, :, :])[:, :H, :W]
    return prob_map.astype(np.float32)


def predict_full_image_multichannel(
    model: nn.Module,
    patch_root: str | Path,
    image_index: int,
    patch_size: int,
    ch_mean: torch.Tensor,
    ch_std: torch.Tensor,
    device: torch.device,
    overlap: float = 0.5,
    batch_size: int = 16,
    n_out_channels: int = 2,
    exclude_patterns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reassemble a full image and run multi-channel sliding-window prediction.

    Returns ``(full_image (C, H, W), prob_map (n_out_channels, H, W))``.
    """
    full_image, _records = reassemble_image(
        patch_root, image_index, exclude_patterns=exclude_patterns,
    )
    prob_map = sliding_window_predict_multichannel(
        model, full_image, patch_size, ch_mean, ch_std,
        device, overlap=overlap, batch_size=batch_size,
        n_out_channels=n_out_channels,
    )
    return full_image, prob_map
