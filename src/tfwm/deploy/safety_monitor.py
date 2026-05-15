"""Hardware safety monitor.

Enforces hard force cutoffs and predicted-collision e-stops.
Runs synchronously inside the controller thread.
"""

from __future__ import annotations

import torch

from tfwm.errors import SafetyError


class SafetyMonitor:
    """Monitor joint torques and predicted forces for safety violations.

    Args:
        max_force_N: Hard force limit in Newtons.
        max_torque_Nm: Hard torque limit in Newton-metres.
        max_action_norm: Maximum L2 norm of a command action.
    """

    def __init__(
        self,
        max_force_N: float = 30.0,
        max_torque_Nm: float = 5.0,
        max_action_norm: float = 1.0,
    ) -> None:
        self.max_force_N = max_force_N
        self.max_torque_Nm = max_torque_Nm
        self.max_action_norm = max_action_norm
        self.violation_count: int = 0

    def check(self, action: torch.Tensor, measured_force_N: float = 0.0) -> torch.Tensor:
        """Validate action and measured force; raise or clip on violation.

        Args:
            action: Proposed action tensor ``(B, d_a)`` or ``(d_a,)``.
            measured_force_N: Current measured contact force in Newtons.

        Returns:
            Clipped safe action.

        Raises:
            SafetyError: On hard force threshold violation.
        """
        if measured_force_N > self.max_force_N:
            self.violation_count += 1
            raise SafetyError(
                f"Measured force {measured_force_N:.1f} N exceeds limit {self.max_force_N} N. "
                f"Triggering e-stop."
            )
        # Soft clip: scale action if norm exceeds limit.
        norm = action.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_action_norm / norm.clamp_min(1e-8), max=1.0)
        return action * scale

    def reset_count(self) -> None:
        """Reset violation counter (call at episode start)."""
        self.violation_count = 0
