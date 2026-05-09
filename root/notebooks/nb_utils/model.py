"""Encoder + heads (decoder, mask token, projector) for SimMIM + VICReg.

Encoder is MONAI's ``SwinTransformer`` (the encoder backbone of SwinUNETR;
Hatamizadeh et al. 2022). Configured to match timm ``swin_tiny_patch4_window7_224``
so that timm's ImageNet weights load cleanly via the remap in
``weight_loading.py``.

Heads:
- ``SimMIMDecoder``  -- 1x1 Conv + PixelShuffle. The lightweight
  one-layer decoder proposed in Xie et al., "SimMIM: A Simple Framework
  for Masked Image Modeling", CVPR 2022 (sec. 3.2).
- ``MaskToken``      -- standard learnable token, init follows BEiT/SimMIM
  (truncated normal, std=0.02).
- ``VICRegProjector`` -- 3-layer MLP with BN+ReLU between linears, as in
  Bardes et al., "VICReg", ICLR 2022 (sec. 4.1) and the official repo.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer

from .config import ModelCfg, SSLCfg


def build_swin_encoder(model_cfg: ModelCfg) -> nn.Module:
    """Construct the MONAI SwinTransformer encoder from ``model_cfg``."""
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
    """1x1 Conv + PixelShuffle decoder ``(B, C_enc, S, S) -> (B, C_in, H, W)``.

    The minimal one-layer decoder from SimMIM (Xie et al., 2022, sec. 3.2).
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

    Initialisation (truncated normal, std=0.02) follows BEiT / SimMIM.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02)

    def forward(self):
        return self.token


class VICRegProjector(nn.Module):
    """3-layer MLP projector with BN+ReLU between linear layers.

    Matches the projector described in Bardes et al., VICReg, ICLR 2022,
    sec. 4.1, and used in ``main_vicreg.py`` of the official repo.
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


def _encoder_stride_and_channels(model_cfg: ModelCfg) -> tuple[int, int, int]:
    enc_ch = model_cfg.feature_size * (2 ** len(model_cfg.depths))
    encoder_stride = model_cfg.patch_size * (2 ** len(model_cfg.depths))
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
    """Decoder + mask token + projector, returned as a dict of modules."""
    enc_ch, _, encoder_stride = _encoder_stride_and_channels(model_cfg)
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
