"""Safety filter for action sequences.

Provides a Control Barrier Function (CBF)-inspired safety projection that
clips actions likely to produce excessive contact forces or joint-limit
violations, as predicted by the latent dynamics model.
"""

from __future__ import annotations

import torch
from torch import nn

from tfwm.errors import SafetyError


class SafetyFilter(nn.Module):
    """Project actions to the safety set predicted by the dynamics model.

    A proposed action is first evaluated by rolling the dynamics forward one
    step.  If the predicted contact force (approximated as the contact-event
    logit magnitude) exceeds ``force_threshold``, the action magnitude is
    scaled down to satisfy the constraint.

    Args:
        dynamics: Module with a compatible ``rollout`` method.
        force_threshold: Contact-logit magnitude threshold.
        max_scale_down: Minimum fraction by which an action can be scaled.
        hard_stop_threshold: If predicted force exceeds this, raises
            :class:`~tfwm.errors.SafetyError` (e-stop).

    Shape:
        - ``action``: ``(B, d_a)``
        - ``z_t``:    ``(B, 1, d_z)`` or ``(B, d_z)``
        - Returns:    ``(B, d_a)`` safe action.
    """

    def __init__(
        self,
        dynamics: nn.Module,
        force_threshold: float = 5.0,
        max_scale_down: float = 0.1,
        hard_stop_threshold: float = 20.0,
    ) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.force_threshold = force_threshold
        self.max_scale_down = max_scale_down
        self.hard_stop_threshold = hard_stop_threshold

    @torch.no_grad()
    def forward(self, action: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """Filter ``action`` for safety given current latent state ``z_t``.

        Args:
            action: Proposed action ``(B, d_a)``.
            z_t: Current latent ``(B, d_z)`` or ``(B, 1, d_z)``.

        Returns:
            Safety-filtered action ``(B, d_a)``.

        Raises:
            SafetyError: If predicted force exceeds ``hard_stop_threshold``.
        """
        if z_t.dim() == 2:
            z_t = z_t.unsqueeze(1)
        a = action.unsqueeze(1)  # (B, 1, d_a)
        _, aux = self.dynamics.rollout(z_t, a)
        pred_force = aux["contact_logits"].abs().max(dim=-1)[0].squeeze(1)  # (B,)

        if (pred_force > self.hard_stop_threshold).any():
            raise SafetyError(
                f"Predicted contact force {pred_force.max().item():.2f} exceeds "
                f"hard-stop threshold {self.hard_stop_threshold}."
            )

        over_threshold = pred_force > self.force_threshold  # (B,)
        if over_threshold.any():
            scale = (self.force_threshold / pred_force.clamp_min(1e-6)).clamp(
                self.max_scale_down, 1.0
            )
            action = action * scale.unsqueeze(-1)

        return action
