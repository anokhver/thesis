"""Two-view augmentation and val-only z-score transform for microscopy SSL."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class MicroscopyTwoViewTransform:
    """Produce two augmented views of a ``(C, H, W)`` float tensor in ``[0, 1]``.

    Pipeline: flips/rot90 -> integer translate -> optional Gaussian blur ->
    Poisson + Gaussian noise -> per-channel z-score. Output is two ``(C, H, W)``
    float32 tensors.
    """

    def __init__(
        self,
        ch_mean: torch.Tensor,
        ch_std: torch.Tensor,
        *,
        translate_max: int = 5,
        blur_sigma_max: float = 1.0,
        blur_prob: float = 0.5,
        blur_kernel_size: int = 5,
        gauss_noise_sigma: float = 0.005,
        poisson_scale: float = 0.005,
    ):
        self.ch_mean = ch_mean.view(-1, 1, 1)
        self.ch_std = ch_std.view(-1, 1, 1)
        self.translate_max = translate_max
        self.blur_sigma_max = blur_sigma_max
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        self.gauss_noise_sigma = gauss_noise_sigma
        self.poisson_scale = poisson_scale

    @staticmethod
    def _flips_rot90(x):
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-2,))
        k = int(torch.randint(0, 4, ()).item())
        if k:
            x = torch.rot90(x, k=k, dims=(-2, -1))
        return x

    def _affine_translate(self, x):
        if self.translate_max <= 0:
            return x
        tx = int(torch.randint(-self.translate_max, self.translate_max + 1, ()).item())
        ty = int(torch.randint(-self.translate_max, self.translate_max + 1, ()).item())
        if tx == 0 and ty == 0:
            return x
        pad = self.translate_max
        x = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
        H, W = x.shape[-2:]
        return x[..., pad - ty:H - pad - ty, pad - tx:W - pad - tx]

    def _maybe_blur(self, x):
        if self.blur_prob <= 0 or self.blur_sigma_max <= 0:
            return x
        if torch.rand(()) >= self.blur_prob:
            return x
        sigma = float(torch.empty(()).uniform_(0.1, self.blur_sigma_max).item())
        half = (self.blur_kernel_size - 1) / 2.0
        coords = torch.arange(self.blur_kernel_size, dtype=x.dtype, device=x.device) - half
        kern = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        kern = kern / kern.sum()
        C = x.shape[0]
        kx = kern.view(1, 1, 1, -1).expand(C, 1, 1, -1)
        ky = kern.view(1, 1, -1, 1).expand(C, 1, -1, 1)
        x = x.unsqueeze(0)
        x = F.conv2d(x, kx, padding=(0, self.blur_kernel_size // 2), groups=C)
        x = F.conv2d(x, ky, padding=(self.blur_kernel_size // 2, 0), groups=C)
        return x.squeeze(0)

    def _poisson_gaussian_noise(self, x):
        if self.poisson_scale > 0:
            scale = 1.0 / max(self.poisson_scale, 1e-8)
            x = torch.poisson((x.clamp_min(0) * scale).double()).float() / scale
        if self.gauss_noise_sigma > 0:
            x = x + torch.randn_like(x) * self.gauss_noise_sigma
        return x

    def _one_view(self, x):
        x = self._flips_rot90(x)
        x = self._affine_translate(x)
        x = self._maybe_blur(x)
        x = self._poisson_gaussian_noise(x)
        x = (x - self.ch_mean) / self.ch_std
        return x

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        x = x.float()
        return self._one_view(x), self._one_view(x)


class ValSingleViewTransform:
    """Apply per-channel z-score. No augmentation."""

    def __init__(self, ch_mean: torch.Tensor, ch_std: torch.Tensor):
        self.ch_mean = ch_mean.view(-1, 1, 1)
        self.ch_std = ch_std.view(-1, 1, 1)

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        return (x.float() - self.ch_mean) / self.ch_std
