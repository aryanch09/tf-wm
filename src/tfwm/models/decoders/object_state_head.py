"""Object state regression head: latent → (position, orientation, velocity)."""

from __future__ import annotations

import torch
from torch import nn


class ObjectStateHead(nn.Module):
    """Regress continuous object state from latent.

    Args:
        d_latent: Input latent dimension.
        d_state: Output state dimension (e.g. 13 for pos+quat+vel).
        hidden_dim: Hidden layer width.
        normalise_quat: If ``True``, L2-normalises the quaternion slice
            ``out[..., 3:7]`` at inference time (no-op during training).

    Shape:
        - Input  ``z``: ``(B, T, d_z)``
        - Output state: ``(B, T, d_state)``
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_state: int = 13,
        hidden_dim: int = 128,
        normalise_quat: bool = True,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.normalise_quat = normalise_quat
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_state),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return predicted object state.

        Args:
            z: Latent tensor ``(B, T, d_z)``.

        Returns:
            State tensor ``(B, T, d_state)``.
        """
        out = self.net(z)
        if self.normalise_quat and not self.training and self.d_state >= 7:
            quat = out[..., 3:7]
            out = torch.cat([out[..., :3], nn.functional.normalize(quat, dim=-1), out[..., 7:]], dim=-1)
        return out
