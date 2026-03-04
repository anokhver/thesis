"""
WNet3D Architecture — Fully parameterizable version.

Based on CellSeg3D: papers_code/CellSeg3D/.../wnet/model.py

WNet3D cascades two identical U-Net halves:
  - Encoder U-Net (U_enc): input image → K-class segmentation (with softmax)
  - Decoder U-Net (U_dec): segmentation map → reconstructed image (no softmax)

Each U-Net uses a 4-level encoder-decoder with channels [64, 128, 256, 512].
Activation, normalization, dropout, kernel_size, and padding_mode are all
parameterizable via factory functions / constructor args.

ALL BASED ON: https://doi.org/10.7554/eLife.99848.4
"""

import torch
import torch.nn as nn
from typing import List, Optional

# ─── Defaults ────────────────────────────────────────────────────────────────────

DEFAULT_NUM_GROUPS = 4          # GroupNorm groups used throughout the network
DEFAULT_DROPOUT = 0.65          # CellSeg3D default dropout rate
DEFAULT_ACTIVATION = "relu"
DEFAULT_NORM = "group"
DEFAULT_DROPOUT_TYPE = "standard"
DEFAULT_KERNEL_SIZE = 3         # spatial conv kernel size (CellSeg3D uses 3)
DEFAULT_PADDING_MODE = "zeros"  # padding mode for spatial convs
                                # options: 'zeros', 'reflect', 'replicate', 'circular'


# ─── Factory Functions ───────────────────────────────────────────────────────────
# Swap activation / normalization / dropout by changing a string.


def get_activation(activation: str = DEFAULT_ACTIVATION) -> nn.Module:
    """Return an activation layer by name.

    Supported: 'relu', 'leaky_relu', 'elu', 'gelu'.
    """
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=True)
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {activation}")


def get_norm_layer(norm_type: str,
                   num_channels: int,
                   num_groups: int = DEFAULT_NUM_GROUPS) -> nn.Module:
    """Return a normalization layer by name.

    Supported: 'group', 'batch', 'instance'.

    Args:
        norm_type:    which normalization to use.
        num_channels: number of feature channels to normalize.
        num_groups:   groups for GroupNorm (ignored by batch/instance).
    """
    if norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm3d(num_channels)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_channels)
    else:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. Choose 'group', 'batch', or 'instance'."
        )


def get_dropout_layer(p: float = DEFAULT_DROPOUT,
                      dropout_type: str = DEFAULT_DROPOUT_TYPE) -> nn.Module:
    """Return a dropout layer by name.

    Supported:
      'standard' — nn.Dropout3d: drops entire 3D feature channels.
      'spatial'  — nn.Dropout2d: drops entire 2D feature maps (channel-wise).
                   Closest PyTorch equivalent to SpatialDropout3D.

    Args:
        p:            dropout probability.
        dropout_type: 'standard' or 'spatial'.
    """
    if dropout_type == "standard":
        return nn.Dropout3d(p=p)
    elif dropout_type == "spatial":
        return nn.Dropout2d(p=p)
    else:
        raise ValueError(
            f"Unknown dropout_type: {dropout_type}. Choose 'standard' or 'spatial'."
        )


# ─── Building Blocks ────────────────────────────────────────────────────────────


class InBlock(nn.Module):
    """First block at the top of the U-Net encoder.

    Two spatial convolutions, each followed by activation -> dropout -> norm.
    Transforms: in_channels -> out_channels -> out_channels.

    Args:
        kernel_size:  spatial conv kernel size (default 3).
        padding_mode: padding fill strategy ('zeros', 'reflect', 'replicate', 'circular').
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 padding_mode: str = DEFAULT_PADDING_MODE,
                 activation: str = DEFAULT_ACTIVATION,
                 norm_type: str = DEFAULT_NORM,
                 num_groups: int = DEFAULT_NUM_GROUPS,
                 dropout: float = DEFAULT_DROPOUT,
                 dropout_type: str = DEFAULT_DROPOUT_TYPE):
        super().__init__()
        padding = kernel_size // 2  # keeps spatial size unchanged
        self.block = nn.Sequential(
            # Conv 1: expand input channels to out_ch
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, out_ch, num_groups),
            # Conv 2: refine features at out_ch
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, out_ch, num_groups),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Encoder/decoder block used at every level except input and output.

    Two pairs of (spatial conv -> 1x1x1 channel projection),
    each followed by activation -> dropout -> norm.
    Transforms: in_channels -> out_channels.

    Args:
        kernel_size:  spatial conv kernel size (default 3). The 1x1x1 projection
                      always uses kernel_size=1 regardless.
        padding_mode: padding fill strategy for spatial convs.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 padding_mode: str = DEFAULT_PADDING_MODE,
                 activation: str = DEFAULT_ACTIVATION,
                 norm_type: str = DEFAULT_NORM,
                 num_groups: int = DEFAULT_NUM_GROUPS,
                 dropout: float = DEFAULT_DROPOUT,
                 dropout_type: str = DEFAULT_DROPOUT_TYPE):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            # Pair 1: spatial conv + channel projection
            nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),             # 1x1x1 always
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, out_ch, num_groups),
            # Pair 2: spatial conv + channel projection
            nn.Conv3d(out_ch, out_ch, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            nn.Conv3d(out_ch, out_ch, kernel_size=1),            # 1x1x1 always
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, out_ch, num_groups),
        )

    def forward(self, x):
        return self.block(x)


class OutBlock(nn.Module):
    """Final block at the top of the U-Net decoder.

    Two spatial convolutions (intermediate channels configurable, default 64),
    each followed by activation -> dropout -> norm, then a 1x1x1
    projection to the output channels.

    Args:
        kernel_size:  spatial conv kernel size (default 3). The final 1x1x1
                      projection always uses kernel_size=1.
        padding_mode: padding fill strategy for spatial convs.
        mid_channels: intermediate channel count (default 64, matches CellSeg3D).
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 padding_mode: str = DEFAULT_PADDING_MODE,
                 activation: str = DEFAULT_ACTIVATION,
                 norm_type: str = DEFAULT_NORM,
                 num_groups: int = DEFAULT_NUM_GROUPS,
                 dropout: float = DEFAULT_DROPOUT,
                 dropout_type: str = DEFAULT_DROPOUT_TYPE,
                 mid_channels: int = 64):
        super().__init__()
        mid = mid_channels  # intermediate channels before final projection
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, mid, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, mid, num_groups),
            nn.Conv3d(mid, mid, kernel_size=kernel_size,
                      padding=padding, padding_mode=padding_mode),
            get_activation(activation),
            get_dropout_layer(dropout, dropout_type),
            get_norm_layer(norm_type, mid, num_groups),
            nn.Conv3d(mid, out_ch, kernel_size=1),  # 1x1x1 final projection (always 1)
        )

    def forward(self, x):
        return self.block(x)


# ─── U-Net Half ─────────────────────────────────────────────────────────────────


class UNet3DHalf(nn.Module):
    """Single U-Net half of the WNet3D.

    Architecture (depth=4, channels=[64, 128, 256, 512]):
      Encoder:
        InBlock:    in -> 64
        Down 1:     MaxPool3d(2) + DownBlock(64 -> 128)
        Down 2:     MaxPool3d(2) + DownBlock(128 -> 256)
        Bottleneck: MaxPool3d(2) + DownBlock(256 -> 512)
      Decoder:
        Up 1:       ConvTranspose3d(512 -> 256) + cat(skip) + DownBlock(512 -> 256)
        Up 2:       ConvTranspose3d(256 -> 128) + cat(skip) + DownBlock(256 -> 128)
        Up 3:       ConvTranspose3d(128 -> 64)  + cat(skip) + OutBlock(128 -> out)

    Args:
        in_channels:  number of input channels.
        out_channels: number of output channels.
        use_softmax:  apply Softmax(dim=1) on output (True for encoder U-Net).
        kernel_size:  spatial conv kernel size for all blocks (default 3).
        padding_mode: padding fill strategy for all spatial convs.
        activation:   activation function name.
        norm_type:    normalization layer name.
        num_groups:   groups for GroupNorm.
        dropout:      dropout probability.
        dropout_type: dropout variant.
        out_mid_channels: intermediate channels in OutBlock (default 64).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_softmax: bool = False,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 padding_mode: str = DEFAULT_PADDING_MODE,
                 activation: str = DEFAULT_ACTIVATION,
                 norm_type: str = DEFAULT_NORM,
                 num_groups: int = DEFAULT_NUM_GROUPS,
                 dropout: float = DEFAULT_DROPOUT,
                 dropout_type: str = DEFAULT_DROPOUT_TYPE,
                 out_mid_channels: int = 64):
        super().__init__()
        # Channel progression: [64, 128, 256, 512]
        channels = [64, 128, 256, 512]
        self.use_softmax = use_softmax

        # Common kwargs passed to every block
        bkw = dict(kernel_size=kernel_size, padding_mode=padding_mode,
                    activation=activation, norm_type=norm_type,
                    num_groups=num_groups, dropout=dropout,
                    dropout_type=dropout_type)

        # --- Encoder ---
        self.in_block = InBlock(in_channels, channels[0], **bkw)
        self.pool = nn.MaxPool3d(kernel_size=2)

        # 3 encoder stages: 64->128, 128->256, 256->512 (bottleneck)
        self.enc1 = DownBlock(channels[0], channels[1], **bkw)
        self.enc2 = DownBlock(channels[1], channels[2], **bkw)
        self.enc3 = DownBlock(channels[2], channels[3], **bkw)  # bottleneck

        # --- Decoder ---
        # Transposed convolutions for 2x upsampling
        self.up1 = nn.ConvTranspose3d(channels[3], channels[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(channels[2], channels[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(channels[1], channels[0], kernel_size=2, stride=2)

        # Decoder blocks (input = upsampled + skip concatenation -> double channels)
        self.dec1 = DownBlock(channels[3], channels[2], **bkw)      # 512->256
        self.dec2 = DownBlock(channels[2], channels[1], **bkw)      # 256->128
        self.out_block = OutBlock(channels[1], out_channels,
                                  mid_channels=out_mid_channels, **bkw)  # 128->out

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder -- collect skip connections
        s0 = self.in_block(x)                  # [B, 64,  D,   H,   W  ]
        s1 = self.enc1(self.pool(s0))          # [B, 128, D/2, H/2, W/2]
        s2 = self.enc2(self.pool(s1))          # [B, 256, D/4, H/4, W/4]
        bot = self.enc3(self.pool(s2))         # [B, 512, D/8, H/8, W/8]

        # Decoder -- upsample, concat skip, apply block
        x = torch.cat([s2, self.up1(bot)], dim=1)  # [B, 512, ...]
        x = self.dec1(x)                            # [B, 256, ...]

        x = torch.cat([s1, self.up2(x)], dim=1)    # [B, 256, ...]
        x = self.dec2(x)                            # [B, 128, ...]

        x = torch.cat([s0, self.up3(x)], dim=1)    # [B, 128, ...]
        x = self.out_block(x)                       # [B, out, ...]

        if self.use_softmax:
            x = self.softmax(x)

        return x


# ─── WNet3D ─────────────────────────────────────────────────────────────────────


class WNet3D(nn.Module):
    """WNet3D: two cascaded U-Nets for unsupervised 3D segmentation.

    Data flow:
      input image --> encoder U-Net --> K-class probabilities (softmax)
                                              |
                                              v
                            decoder U-Net --> reconstructed image

    All internal layers (activation, norm, dropout, kernel_size, padding_mode)
    are parameterizable.

    Args:
        in_channels:      input image channels (1 for grayscale).
        out_channels:     reconstruction output channels (1 for grayscale).
        num_classes:      segmentation classes K (default 2).
        kernel_size:      spatial conv kernel size for all blocks (default 3).
        padding_mode:     padding fill strategy ('zeros', 'reflect', 'replicate', 'circular').
        activation:       activation function name ('relu', 'leaky_relu', 'elu', 'gelu').
        norm_type:        normalization type ('group', 'batch', 'instance').
        num_groups:       groups for GroupNorm (default 4).
        dropout:          dropout probability (default 0.65).
        dropout_type:     dropout variant ('standard', 'spatial').
        out_mid_channels: intermediate channels in the OutBlock (default 64).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 num_classes: int = 2,
                 kernel_size: int = DEFAULT_KERNEL_SIZE,
                 padding_mode: str = DEFAULT_PADDING_MODE,
                 activation: str = DEFAULT_ACTIVATION,
                 norm_type: str = DEFAULT_NORM,
                 num_groups: int = DEFAULT_NUM_GROUPS,
                 dropout: float = DEFAULT_DROPOUT,
                 dropout_type: str = DEFAULT_DROPOUT_TYPE,
                 out_mid_channels: int = 64):
        super().__init__()

        common = dict(kernel_size=kernel_size, padding_mode=padding_mode,
                      activation=activation, norm_type=norm_type,
                      num_groups=num_groups, dropout=dropout,
                      dropout_type=dropout_type,
                      out_mid_channels=out_mid_channels)

        # Encoder U-Net: image -> K-class soft segmentation
        self.encoder = UNet3DHalf(
            in_channels=in_channels,
            out_channels=num_classes,
            use_softmax=True,
            **common,
        )

        # Decoder U-Net: K-class segmentation -> reconstructed image
        self.decoder = UNet3DHalf(
            in_channels=num_classes,
            out_channels=out_channels,
            use_softmax=False,
            **common,
        )

    def forward(self, x):
        """Returns (segmentation, reconstruction)."""
        seg = self.encoder(x)
        rec = self.decoder(seg)
        return seg, rec

    def forward_encoder(self, x):
        """Encoder only -- returns K-class probability map."""
        return self.encoder(x)

    def forward_decoder(self, seg):
        """Decoder only -- reconstructs image from segmentation."""
        return self.decoder(seg)
