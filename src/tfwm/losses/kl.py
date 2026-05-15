"""KL-divergence losses for variational latent dynamics."""

from __future__ import annotations

import torch
from torch import nn


class KLDivergenceLoss(nn.Module):
    """KL divergence between posterior and prior Gaussian distributions.

    Supports both standard (posterior ‖ N(0,1)) and RSSM-style
    (posterior ‖ prior) modes with linear KL annealing.

    Args:
        free_nats: KL terms below this value are zeroed (free information).
        reduction: ``"mean"`` or ``"sum"``.

    Shape:
        - ``posterior_mu``:    ``(B, T, d_z)``
        - ``posterior_logvar``: ``(B, T, d_z)``
        - ``prior_mu``:        ``(B, T, d_z)`` or ``None`` → N(0,1) prior
        - ``prior_logvar``:    ``(B, T, d_z)`` or ``None``
        - Returns: scalar.
    """

    def __init__(self, free_nats: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.free_nats = free_nats
        self.reduction = reduction

    def forward(
        self,
        posterior_mu: torch.Tensor,
        posterior_logvar: torch.Tensor,
        prior_mu: torch.Tensor | None = None,
        prior_logvar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """KL( posterior ‖ prior ).

        Args:
            posterior_mu: Posterior mean ``(B, T, d_z)``.
            posterior_logvar: Posterior log-variance ``(B, T, d_z)``.
            prior_mu: Prior mean; defaults to zeros (standard Normal).
            prior_logvar: Prior log-variance; defaults to zeros (unit var).

        Returns:
            Scalar KL loss after free-nats clipping.
        """
        if prior_mu is None:
            prior_mu = torch.zeros_like(posterior_mu)
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(posterior_logvar)

        # KL( N(mu1, sigma1^2) ‖ N(mu0, sigma0^2) ) per-dimension.
        kl = 0.5 * (
            prior_logvar - posterior_logvar
            + (posterior_logvar.exp() + (posterior_mu - prior_mu).pow(2)) / prior_logvar.exp()
            - 1.0
        )  # (B, T, d_z)

        # Free-nats: treat KL < free_nats as zero to avoid over-regularisation
        # during early training (Kingma et al. 2016).
        kl = kl.clamp(min=self.free_nats / kl.shape[-1])

        if self.reduction == "mean":
            return kl.mean()
        return kl.sum()


def kl_divergence_standard(
    mu: torch.Tensor, logvar: torch.Tensor, free_nats: float = 0.0
) -> torch.Tensor:
    """KL( N(mu, exp(logvar)) ‖ N(0,1) ) — functional convenience."""
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_nats > 0:
        kl = kl.clamp(min=free_nats / mu.shape[-1])
    return kl.mean()
