"""Top-level world-model wiring."""

from __future__ import annotations

import torch
from torch import nn

from tfwm.types import Action, LatentDist, Tactile, WorldModelOutput


class TactileWorldModel(nn.Module):
    """Composable TF-WM wrapper around encoder, dynamics, heads, and gate."""

    def __init__(
        self,
        tactile_encoder: nn.Module,
        dynamics: nn.Module,
        gate: nn.Module,
        tactile_decoder: nn.Module | None = None,
        contact_head: nn.Module | None = None,
        affordance_head: nn.Module | None = None,
        object_state_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.tactile_encoder = tactile_encoder
        self.dynamics = dynamics
        self.gate = gate
        self.tactile_decoder = tactile_decoder
        self.contact_head = contact_head
        self.affordance_head = affordance_head
        self.object_state_head = object_state_head

    def _as_latent_dist(self, encoded: torch.Tensor | LatentDist) -> LatentDist:
        if isinstance(encoded, dict):
            return encoded
        return {"mu": encoded, "logvar": torch.zeros_like(encoded)}

    def forward(
        self,
        tactile: Tactile,
        actions: Action,
        context: torch.Tensor | None = None,
    ) -> WorldModelOutput:
        """Encode tactile history, roll out dynamics, and run prediction heads."""

        latent = self._as_latent_dist(self.tactile_encoder(tactile))
        rollout = self.dynamics.rollout(latent["mu"], actions, context=context)
        z_pred, aux = rollout
        pred_tactile = self.tactile_decoder(z_pred) if self.tactile_decoder is not None else None
        pred_contact = (
            self.contact_head(z_pred).argmax(dim=-1) if self.contact_head is not None else None
        )
        pred_affordance = self.affordance_head(z_pred) if self.affordance_head is not None else None
        pred_object_state = (
            self.object_state_head(z_pred) if self.object_state_head is not None else None
        )
        gate_score = self.gate(z_pred, aux)
        return WorldModelOutput(
            latent=latent,
            pred_tactile=pred_tactile,
            pred_contact=pred_contact,
            pred_affordance=pred_affordance,
            pred_object_state=pred_object_state,
            gate_score=gate_score,
        )
