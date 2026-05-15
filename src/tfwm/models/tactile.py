"""Tactile encoder and decoder using PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from tfwm.types import Tactile, TactileLatent


class TactileEncoder(nn.Module):
    """CNN encoder for tactile sensor readings."""

    def __init__(
        self,
        features: Sequence[int] = (32, 64, 128),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        d_latent: int = 128,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent

        conv_layers = []
        in_channels = 1
        for out_channels, kernel_size in zip(features, kernel_sizes, strict=False):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.encoder = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(in_features=features[-1] * 2 * 2, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=d_latent),
        )

    def forward(self, tactile: Tactile) -> TactileLatent:
        if tactile.dim() == 2:
            x = tactile.unsqueeze(0).unsqueeze(1)
        elif tactile.dim() == 3:
            x = tactile.unsqueeze(1)
        else:
            x = tactile

        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


class TactileDecoder(nn.Module):
    """CNN decoder for tactile sensor readings."""

    def __init__(
        self,
        features: Sequence[int] = (128, 64, 32),
        kernel_sizes: Sequence[int] = (3, 3, 3),
        output_shape: tuple[int, int] = (16, 16),
        d_latent: int = 64,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.fc = nn.Sequential(
            nn.Linear(in_features=d_latent, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=256, out_features=features[0] * 2 * 2),
            nn.ReLU(),
        )

        in_channels = features[0]
        convt_layers = []
        for out_channels, kernel_size in zip(features[1:], kernel_sizes[1:], strict=False):
            convt_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1,
                )
            )
            convt_layers.append(nn.ReLU())
            in_channels = out_channels

        self.decoders = nn.Sequential(*convt_layers)
        self.output_conv = nn.Conv2d(in_channels=features[-1], out_channels=1, kernel_size=1)

    def forward(self, latent: TactileLatent) -> Tactile:
        if latent.dim() == 1:
            x = latent.unsqueeze(0)
            squeeze_batch = True
        else:
            x = latent
            squeeze_batch = False

        x = self.fc(x)
        h, w = self.output_shape
        x = x.view(x.shape[0], -1, h // 8, w // 8)
        x = self.decoders(x)
        x = self.output_conv(x)
        if x.shape[-2:] != self.output_shape:
            x = F.interpolate(x, size=self.output_shape, mode="bilinear", align_corners=False)
        x = torch.sigmoid(x)

        if squeeze_batch:
            return x[0, 0]
        return x[:, 0]
