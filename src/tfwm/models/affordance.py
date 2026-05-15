"""Affordance prediction models using PyTorch."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tfwm.types import Action, AffordanceLogits, GateScore, Proprio, TactileLatent


class AffordancePredictor(nn.Module):
    """Predicts affordances from tactile latents."""

    def __init__(
        self,
        n_affordances: int = 10,
        d_latent: int = 128,
        hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        layers = []
        input_dim = d_latent
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_affordances))
        self.network = nn.Sequential(*layers)

    def forward(self, tactile_latent: TactileLatent) -> AffordanceLogits:
        if tactile_latent.dim() == 1:
            tactile_latent = tactile_latent.unsqueeze(0)
        return self.network(tactile_latent)


class TactileGatingNetwork(nn.Module):
    """Gating network for multi-sensor fusion."""

    def __init__(
        self,
        n_sensors: int = 2,
        d_latent: int = 128,
        d_proprio: int = 12,
        hidden_dims: Sequence[int] = (64, 32),
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        input_dim = n_sensors * d_latent + d_proprio
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_sensors))
        self.network = nn.Sequential(*layers)

    def forward(self, tactile_latents: Tensor, proprio: Proprio) -> GateScore:
        if tactile_latents.dim() == 1:
            tactile_latents = tactile_latents.unsqueeze(0)

        if tactile_latents.dim() == 2 and tactile_latents.shape[0] == self.n_sensors:
            # Single-sample multi-sensor input: shape (n_sensors, d_latent)
            tactile_latents = tactile_latents.unsqueeze(0)

        if tactile_latents.dim() == 3:
            x = tactile_latents.view(tactile_latents.shape[0], -1)
        else:
            x = tactile_latents

        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        x = torch.cat([x, proprio], dim=-1)
        return self.network(x)


class ActionPredictor(nn.Module):
    """Predicts actions from current state."""

    def __init__(
        self,
        d_latent: int = 128,
        d_proprio: int = 12,
        d_action: int = 6,
        hidden_dims: Sequence[int] = (128, 64),
    ) -> None:
        super().__init__()
        input_dim = d_latent + d_proprio
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, d_action))
        self.network = nn.Sequential(*layers)

    def forward(self, tactile_latent: TactileLatent, proprio: Proprio) -> Action:
        if tactile_latent.dim() == 1:
            tactile_latent = tactile_latent.unsqueeze(0)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        x = torch.cat([tactile_latent, proprio], dim=-1)
        return torch.tanh(self.network(x))
