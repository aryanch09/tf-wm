"""Vision models using PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tfwm.types import Vision


class VisionEncoder(nn.Module):
    """CNN encoder for vision inputs."""

    def __init__(
        self,
        features: Sequence[int] = (32, 64, 128, 256),
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        d_latent: int | None = None,
        input_channels: int = 3,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_latent is None:
            d_latent = 256 if latent_dim is None else latent_dim
        elif latent_dim is not None and latent_dim != d_latent:
            raise ValueError("d_latent and latent_dim must match when both are provided")
        self.d_latent = d_latent
        layers = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(features, kernel_sizes, strict=False):
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[-1] * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, d_latent),
        )

    def forward(self, vision: Vision) -> Tensor:
        if vision.dim() == 3:
            x = vision.unsqueeze(0)
        else:
            x = vision
        x = self.encoder(x)
        return self.fc(x)


class VisionDecoder(nn.Module):
    """CNN decoder for vision reconstruction."""

    def __init__(
        self,
        features: Sequence[int] = (256, 128, 64, 32),
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        output_shape: tuple[int, int, int] = (3, 64, 64),
        d_latent: int | None = None,
        latent_dim: int | None = None,
        output_channels: int | None = None,
    ) -> None:
        super().__init__()
        if output_channels is not None:
            output_shape = (output_channels, output_shape[1], output_shape[2])
        if d_latent is None:
            d_latent = 256 if latent_dim is None else latent_dim
        elif latent_dim is not None and latent_dim != d_latent:
            raise ValueError("d_latent and latent_dim must match when both are provided")
        self.output_shape = output_shape
        c, h, w = output_shape
        self.fc = nn.Sequential(
            nn.Linear(d_latent, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, features[0] * 4 * 4),
            nn.ReLU(),
        )
        convt_layers = []
        in_channels = features[0]
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
        self.decoder = nn.Sequential(*convt_layers)
        self.output_conv = nn.Conv2d(in_channels=features[-1], out_channels=c, kernel_size=1)

    def forward(self, latent: Tensor) -> Vision:
        if latent.dim() == 1:
            x = latent.unsqueeze(0)
            squeeze_batch = True
        else:
            x = latent
            squeeze_batch = False

        x = self.fc(x)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.decoder(x)
        if x.shape[-2:] != (self.output_shape[1], self.output_shape[2]):
            x = F.interpolate(
                x,
                size=(self.output_shape[1], self.output_shape[2]),
                mode="bilinear",
                align_corners=False,
            )
        x = torch.sigmoid(self.output_conv(x))

        if squeeze_batch:
            return x[0]
        return x
