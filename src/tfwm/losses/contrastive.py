"""InfoNCE / CPC contrastive loss for tactile representation learning.

Following the Contrastive Predictive Coding (CPC) setup: anchors are latents
at time ``t``, positives are future latents at time ``t+k``, and negatives are
drawn from within the same batch (cross-episode) and from time-shifted
positions within the same episode.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """InfoNCE (NT-Xent) loss for latent tactile sequences.

    Negatives per anchor consist of:
    - ``n_time_negatives``: other timesteps in the same episode.
    - ``n_cross_negatives``: latents from different episodes in the batch.
    Total negatives = ``n_time_negatives + n_cross_negatives``.

    Args:
        temperature: Softmax temperature ``τ``.
        n_time_negatives: Negatives drawn from time-shifted positions.
        n_cross_negatives: Negatives drawn from other episodes in the batch.
        k_steps: Prediction horizon (anchor predicts ``t + k_steps``).

    Shape:
        - ``anchors``: ``(B, T, d_z)``
        - ``positives``: ``(B, T, d_z)`` (shifted by ``k_steps``)
        - Returns: scalar loss.

    References:
        van den Oord et al. (2018) "Representation Learning with Contrastive
        Predictive Coding." arXiv:1807.03748.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_time_negatives: int = 32,
        n_cross_negatives: int = 32,
        k_steps: int = 1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.n_time_negatives = n_time_negatives
        self.n_cross_negatives = n_cross_negatives
        self.k_steps = k_steps

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            anchors: Latent states ``(B, T, d_z)``.
            positives: Target latents ``(B, T, d_z)``.

        Returns:
            Scalar InfoNCE loss averaged over valid anchor positions.
        """
        B, T, d_z = anchors.shape
        if B * T < 2:
            return anchors.new_zeros(())

        # L2-normalise for cosine similarity.
        anc = F.normalize(anchors, dim=-1)  # (B, T, d_z)
        pos = F.normalize(positives, dim=-1)

        losses: list[torch.Tensor] = []
        for b in range(B):
            for t in range(T - self.k_steps):
                query = anc[b, t]              # (d_z,)
                key   = pos[b, t + self.k_steps]  # (d_z,) — positive

                negs: list[torch.Tensor] = []

                # Time negatives (within episode, excluding the positive window).
                valid_times = [i for i in range(T) if abs(i - (t + self.k_steps)) > 1]
                if valid_times:
                    idx = torch.randint(len(valid_times), (self.n_time_negatives,))
                    chosen = [valid_times[i] for i in idx.tolist()]
                    negs.append(pos[b][chosen])

                # Cross-episode negatives.
                other_eps = [i for i in range(B) if i != b]
                if other_eps:
                    ep_idx = torch.randint(len(other_eps), (self.n_cross_negatives,))
                    t_idx  = torch.randint(T, (self.n_cross_negatives,))
                    cross  = torch.stack([pos[other_eps[ep_idx[i].item()], t_idx[i]] for i in range(self.n_cross_negatives)])
                    negs.append(cross)

                if not negs:
                    continue

                all_keys = torch.cat([key.unsqueeze(0), torch.cat(negs, dim=0)], dim=0)  # (1+N, d_z)
                logits   = (query.unsqueeze(0) @ all_keys.T) / self.temperature  # (1, 1+N)
                label    = torch.zeros(1, dtype=torch.long, device=logits.device)
                losses.append(F.cross_entropy(logits, label))

        if not losses:
            return anchors.new_zeros(())
        return torch.stack(losses).mean()
