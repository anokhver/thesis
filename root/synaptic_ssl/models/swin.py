"""Encoder and SSL heads for SimMIM + VICReg pretraining.

Encoder: MONAI ``SwinTransformer`` (Hatamizadeh et al., 2022).
SimMIM decoder: 1x1 Conv + PixelShuffle (Xie et al., CVPR 2022).
VICReg projector: 3-layer BN+ReLU MLP (Bardes et al., ICLR 2022).

Ref: https://github.com/Project-MONAI/MONAI
Ref: https://github.com/microsoft/SimMIM
Ref: https://github.com/facebookresearch/vicreg
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer

from training.config import ModelCfg, SSLCfg


def build_swin_encoder(model_cfg: ModelCfg) -> nn.Module:
    """Build a MONAI ``SwinTransformer`` from ``model_cfg``."""
    patch_size = (model_cfg.patch_size,) * model_cfg.spatial_dims
    window_size = (model_cfg.window_size,) * model_cfg.spatial_dims
    return SwinTransformer(
        in_chans=model_cfg.in_channels,
        embed_dim=model_cfg.feature_size,
        window_size=window_size,
        patch_size=patch_size,
        depths=list(model_cfg.depths),
        num_heads=list(model_cfg.num_heads),
        mlp_ratio=model_cfg.mlp_ratio,
        qkv_bias=model_cfg.qkv_bias,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=model_cfg.dropout_path_rate,
        norm_layer=nn.LayerNorm,
        use_checkpoint=model_cfg.use_checkpoint,
        spatial_dims=model_cfg.spatial_dims,
    )


class SimMIMDecoder(nn.Module):
    """1x1 Conv + PixelShuffle upsampler.

    Xie et al. (CVPR 2022), sec. 3.2.
    Ref: https://github.com/microsoft/SimMIM
    """

    def __init__(self, enc_ch: int, out_channels: int, encoder_stride: int):
        super().__init__()
        self.encoder_stride = encoder_stride
        self.up = nn.Sequential(
            nn.Conv2d(enc_ch, encoder_stride * encoder_stride * out_channels,
                      kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        )

    def forward(self, z):
        return self.up(z)


class MaskToken(nn.Module):
    """Learnable per-channel mask token of shape ``(1, C, 1, 1)``.

    Trunc-normal init, std=0.02.
    Ref: https://github.com/microsoft/SimMIM
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02)

    def forward(self):
        return self.token


class VICRegProjector(nn.Module):
    """3-layer BN+ReLU MLP projector for pooled features.

    Bardes et al. (ICLR 2022), sec. 4.1.
    Ref: https://github.com/facebookresearch/vicreg
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, z_pool):
        return self.net(z_pool)


def _resolve_stage_index(model_cfg: ModelCfg, stage_index: int) -> int:
    """Normalise ``stage_index`` into ``[0, len(depths)]``."""
    n_stages = len(model_cfg.depths) + 1  # patch-embed level + len(depths) downsamples
    idx = stage_index if stage_index >= 0 else n_stages + stage_index
    if not 0 <= idx < n_stages:
        raise ValueError(
            f"head_stage_index={stage_index} out of range for "
            f"{n_stages}-level encoder (depths={model_cfg.depths})."
        )
    return idx


def _encoder_stride_and_channels(
    model_cfg: ModelCfg, stage_index: int = -1,
) -> tuple[int, int, int]:
    """Return ``(channels, spatial, stride)`` at encoder stage ``stage_index``."""
    idx = _resolve_stage_index(model_cfg, stage_index)
    enc_ch = model_cfg.feature_size * (2 ** idx)
    encoder_stride = model_cfg.patch_size * (2 ** idx)
    enc_spatial = model_cfg.img_size // encoder_stride
    if encoder_stride * enc_spatial != model_cfg.img_size:
        raise ValueError(
            f"img_size={model_cfg.img_size} not divisible by "
            f"encoder_stride={encoder_stride} (enc_spatial={enc_spatial})."
        )
    return enc_ch, enc_spatial, encoder_stride


def build_simmim_vicreg_heads(
    model_cfg: ModelCfg, ssl_cfg: SSLCfg,
) -> dict[str, nn.Module]:
    """Build decoder, mask token, and projector. Return dict."""
    enc_ch, _, encoder_stride = _encoder_stride_and_channels(
        model_cfg, ssl_cfg.head_stage_index,
    )
    if model_cfg.img_size % ssl_cfg.mask_block_size != 0:
        raise ValueError(
            f"img_size={model_cfg.img_size} not divisible by "
            f"mask_block_size={ssl_cfg.mask_block_size}."
        )
    return {
        "decoder": SimMIMDecoder(
            enc_ch=enc_ch,
            out_channels=model_cfg.in_channels,
            encoder_stride=encoder_stride,
        ),
        "mask_token": MaskToken(model_cfg.in_channels),
        "projector": VICRegProjector(
            in_dim=enc_ch,
            hidden_dim=ssl_cfg.projector_hidden,
            out_dim=ssl_cfg.projector_dim,
        ),
    }


def count_params(module_or_dict) -> int:
    if isinstance(module_or_dict, nn.Module):
        return sum(p.numel() for p in module_or_dict.parameters())
    return sum(
        p.numel() for m in module_or_dict.values() for p in m.parameters()
    )
