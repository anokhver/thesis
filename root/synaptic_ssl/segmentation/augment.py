"""Joint image-mask augmentation for segmentation training.

Geometric transforms applied to both image and mask. Intensity transforms
applied to image only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class SegTrainTransform:
    """Joint augmentation for ``(image, mask)`` pairs.

    Image: ``(C, H, W)`` float32 in ``[0, 1]``. Mask: ``(1, H, W)`` float32.
    Returns normalised image and re-binarised mask, same shapes.
    """

    def __init__(
        self,
        ch_mean: torch.Tensor,
        ch_std: torch.Tensor,
        *,
        translate_max: int = 5,
        gauss_noise_sigma: float = 0.005,
        poisson_scale: float = 0.005,
    ):
        self.ch_mean = ch_mean.view(-1, 1, 1)
        self.ch_std = ch_std.view(-1, 1, 1)
        self.translate_max = translate_max
        self.gauss_noise_sigma = gauss_noise_sigma
        self.poisson_scale = poisson_scale

    @staticmethod
    def _flips_rot90(x: torch.Tensor, hflip: bool, vflip: bool, k: int):
        if hflip:
            x = torch.flip(x, dims=(-1,))
        if vflip:
            x = torch.flip(x, dims=(-2,))
        if k:
            x = torch.rot90(x, k=k, dims=(-2, -1))
        return x

    def _translate(self, x: torch.Tensor, tx: int, ty: int):
        if tx == 0 and ty == 0:
            return x
        pad = self.translate_max
        x = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        H, W = x.shape[-2:]
        return x[..., pad - ty : H - pad - ty, pad - tx : W - pad - tx]

    def _noise(self, x: torch.Tensor):
        if self.poisson_scale > 0:
            scale = 1.0 / max(self.poisson_scale, 1e-8)
            x = torch.poisson((x.clamp_min(0) * scale).double()).float() / scale
        if self.gauss_noise_sigma > 0:
            x = x + torch.randn_like(x) * self.gauss_noise_sigma
        return x

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # sample geometric params (shared between image and mask)
        hflip = torch.rand(()) < 0.5
        vflip = torch.rand(()) < 0.5
        k = int(torch.randint(0, 4, ()).item())
        tx = int(torch.randint(-self.translate_max, self.translate_max + 1, ()).item())
        ty = int(torch.randint(-self.translate_max, self.translate_max + 1, ()).item())

        # apply identical geometry to both
        image = self._flips_rot90(image, hflip, vflip, k)
        mask = self._flips_rot90(mask, hflip, vflip, k)
        image = self._translate(image, tx, ty)
        mask = self._translate(mask, tx, ty)

        # image-only augmentation
        image = self._noise(image)

        # normalise image (mask stays binary)
        image = (image - self.ch_mean) / self.ch_std

        # re-binarise mask after interpolation artifacts from translate
        mask = (mask > 0.5).float()

        return image, mask


class SegValTransform:
    """Channel-normalise image. Pass mask through unchanged."""

    def __init__(self, ch_mean: torch.Tensor, ch_std: torch.Tensor):
        self.ch_mean = ch_mean.view(-1, 1, 1)
        self.ch_std = ch_std.view(-1, 1, 1)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image = (image.float() - self.ch_mean) / self.ch_std
        return image, mask
