"""SimMIM block masking + apply-mask-token utility.

Block masking follows Xie et al., "SimMIM: A Simple Framework for Masked
Image Modeling", CVPR 2022 (https://github.com/microsoft/SimMIM): pick
``mask_ratio * (H/block) * (W/block)`` non-overlapping square blocks per
sample uniformly at random. The mask token is replaced in pixel space.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_block_mask(
    img: torch.Tensor, block_size: int, mask_ratio: float,
) -> torch.Tensor:
    """Per-sample grid mask on a ``(B, C, H, W)`` image.

    Returns a ``(B, 1, H, W)`` ``{0, 1}`` mask, broadcast across channels.
    """
    B, _, H, W = img.shape
    if H % block_size or W % block_size:
        raise ValueError(f"block_size={block_size} must divide H={H} and W={W}.")
    gh, gw = H // block_size, W // block_size
    n_blocks = gh * gw
    n_mask = max(1, int(round(n_blocks * mask_ratio)))
    noise = torch.rand(B, n_blocks, device=img.device)
    rank = noise.argsort(dim=1)
    flat = (rank < n_mask).to(img.dtype)
    grid = flat.view(B, 1, gh, gw)
    return F.interpolate(grid, scale_factor=block_size, mode="nearest")


def apply_mask(
    view: torch.Tensor, mask: torch.Tensor, mask_token_module,
) -> torch.Tensor:
    """Replace masked pixels in ``view`` with ``mask_token_module()``."""
    return view * (1.0 - mask) + mask_token_module() * mask
