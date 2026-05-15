"""Dynamics models for TF-WM.

Implements a Recurrent State Space Model (RSSM) with a GRU backbone and a
variational latent transition. The ``rollout`` method satisfies the
``Dynamics`` Protocol from :mod:`tfwm.types`, making this compatible with
:class:`tfwm.planning.cem_mpc.CEMMPCPlanner` and
:class:`tfwm.models.world_model.TactileWorldModel`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from tfwm.errors import ShapeError
from tfwm.types import LatentDist

# Use plain torch.Tensor in signatures; the project type aliases (_Tensor,
# Action) are jaxtyping annotations that Pyright cannot validate statically.
_Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Helper: stochastic projection head
# ---------------------------------------------------------------------------

class _GaussianHead(nn.Module):
    """Project a feature vector to (mu, logvar)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.logvar = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> LatentDist:
        return {"mu": self.mu(x), "logvar": self.logvar(x).clamp(-10.0, 2.0)}


# ---------------------------------------------------------------------------
# Deterministic dynamics (MLP baseline)
# ---------------------------------------------------------------------------

class DeterministicDynamics(nn.Module):
    """Single-step MLP transition model (no recurrence).

    Args:
        d_latent: Latent dimension ``d_z``.
        d_action: _Tensor dimension ``d_a``.
        hidden_dims: Hidden layer widths.

    Shape:
        - ``z``: ``(B, T, d_z)``
        - ``actions``: ``(B, H, d_a)``
        - Returns: ``(z_pred [B, H, d_z], aux)``
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_action: int = 6,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_action = d_action
        layers: list[nn.Module] = []
        in_dim = d_latent + d_action
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = _GaussianHead(in_dim, d_latent)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> LatentDist:
        x = torch.cat([z, action], dim=-1)
        feat = self.trunk(x)
        return self.head(feat)

    def rollout(
        self,
        z0: _Tensor,
        actions: _Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[_Tensor, dict[str, torch.Tensor]]:
        """Roll out H steps from z0.

        Args:
            z0: Shape ``(B, 1, d_z)`` or ``(B, d_z)``.
            actions: Shape ``(B, H, d_a)``.
            context: Ignored (deterministic model has no context path).

        Returns:
            ``(z_pred [B, H, d_z], aux)`` where ``aux`` contains
            ``mu``, ``logvar``, ``contact_logits``, ``affordance_logits``.
        """
        del context
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)
        B = z0.shape[0]
        H = actions.shape[1]

        zs, mus, logvars = [], [], []
        z = z0[:, -1]
        for h in range(H):
            dist = self.forward(z, actions[:, h])
            z = dist["mu"]
            zs.append(z)
            mus.append(dist["mu"])
            logvars.append(dist["logvar"])

        z_pred = torch.stack(zs, dim=1)
        aux: dict[str, torch.Tensor] = {
            "mu": torch.stack(mus, dim=1),
            "logvar": torch.stack(logvars, dim=1),
            "contact_logits": torch.zeros(B, H, 5, device=z_pred.device, dtype=z_pred.dtype),
            "affordance_logits": torch.zeros(B, H, 8, device=z_pred.device, dtype=z_pred.dtype),
        }
        return z_pred, aux


# ---------------------------------------------------------------------------
# RSSM — the primary dynamics model
# ---------------------------------------------------------------------------

class RSSMDynamics(nn.Module):
    """Recurrent State Space Model (RSSM) for latent tactile dynamics.

    Architecture mirrors Dreamer-v3: a GRU computes a deterministic recurrent
    state ``h_t``; a stochastic head projects ``(h_t, action_t)`` to
    ``(mu_t, logvar_t)`` for the next latent ``z_{t+1}``.

    Args:
        d_latent: Stochastic latent size ``d_z``.
        d_action: _Tensor dimensionality ``d_a``.
        d_recurrent: GRU hidden size (deterministic state ``h``).
        d_hidden: MLP hidden size for the stochastic head.
        n_contact_classes: Number of contact-event categories.
        n_affordance_classes: Number of affordance categories.

    Shape:
        - ``z0``: ``(B, 1, d_z)`` or ``(B, d_z)``
        - ``actions``: ``(B, H, d_a)``
        - Returns ``(z_pred [B, H, d_z], aux)``
    """

    def __init__(
        self,
        d_latent: int = 128,
        d_action: int = 6,
        d_recurrent: int = 256,
        d_hidden: int = 256,
        n_contact_classes: int = 5,
        n_affordance_classes: int = 8,
    ) -> None:
        super().__init__()
        self.d_latent = d_latent
        self.d_recurrent = d_recurrent
        self.n_contact_classes = n_contact_classes
        self.n_affordance_classes = n_affordance_classes

        # GRU input: concat(z_t, a_t)
        self.gru = nn.GRUCell(input_size=d_latent + d_action, hidden_size=d_recurrent)

        # Stochastic prior: h_t → (mu, logvar)
        self.prior_head = nn.Sequential(
            nn.Linear(d_recurrent, d_hidden), nn.SiLU(),
        )
        self.prior_gaussian = _GaussianHead(d_hidden, d_latent)

        # Posterior: (h_t, x_t) → (mu, logvar), used during training with observation
        self.posterior_head = nn.Sequential(
            nn.Linear(d_recurrent + d_latent, d_hidden), nn.SiLU(),
        )
        self.posterior_gaussian = _GaussianHead(d_hidden, d_latent)

        # Auxiliary prediction heads
        self.contact_head = nn.Linear(d_recurrent, n_contact_classes)
        self.affordance_head = nn.Linear(d_recurrent, n_affordance_classes)

    # ------------------------------------------------------------------
    # Single-step prior transition
    # ------------------------------------------------------------------

    def prior_step(
        self, z: torch.Tensor, action: torch.Tensor, h: torch.Tensor
    ) -> tuple[LatentDist, torch.Tensor]:
        """Advance one step under the prior (no observation).

        Args:
            z: Stochastic state ``(B, d_z)``.
            action: _Tensor ``(B, d_a)``.
            h: Deterministic state ``(B, d_rec)``.

        Returns:
            ``(prior_dist, h_next)``
        """
        inp = torch.cat([z, action], dim=-1)
        h_next = self.gru(inp, h)
        feat = self.prior_head(h_next)
        return self.prior_gaussian(feat), h_next

    def posterior_step(
        self, obs_z: torch.Tensor, h: torch.Tensor
    ) -> LatentDist:
        """Compute posterior given an observation latent ``obs_z``.

        Args:
            obs_z: Encoder output latent ``(B, d_z)``.
            h: Deterministic state ``(B, d_rec)``.

        Returns:
            Posterior ``LatentDist``.
        """
        feat = self.posterior_head(torch.cat([h, obs_z], dim=-1))
        return self.posterior_gaussian(feat)

    @staticmethod
    def _rsample(dist: LatentDist) -> torch.Tensor:
        """Reparameterised sample from a Gaussian ``LatentDist``."""
        std = (0.5 * dist["logvar"]).exp()
        return dist["mu"] + std * torch.randn_like(std)

    # ------------------------------------------------------------------
    # Teacher-forced forward (training)
    # ------------------------------------------------------------------

    def forward_teacher(
        self,
        obs_latents: torch.Tensor,
        actions: _Tensor,
        h0: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced rollout: posterior at every step.

        Args:
            obs_latents: Encoded observations ``(B, T, d_z)``.
            actions: _Tensors ``(B, T, d_a)``.
            h0: Initial recurrent state; zeros if ``None``.

        Returns:
            Dict with keys ``prior_mu``, ``prior_logvar``, ``posterior_mu``,
            ``posterior_logvar``, ``z_posterior [B, T, d_z]``,
            ``contact_logits [B, T, K]``, ``affordance_logits [B, T, K]``.
        """
        B, T, _ = obs_latents.shape
        device, dtype = obs_latents.device, obs_latents.dtype
        h = h0 if h0 is not None else torch.zeros(B, self.d_recurrent, device=device, dtype=dtype)
        z = torch.zeros(B, self.d_latent, device=device, dtype=dtype)

        prior_mus, prior_lvs, post_mus, post_lvs, z_posts = [], [], [], [], []
        contact_logits_list, affordance_logits_list = [], []

        for t in range(T):
            prior_dist, h = self.prior_step(z, actions[:, t], h)
            post_dist = self.posterior_step(obs_latents[:, t], h)
            z = self._rsample(post_dist)

            prior_mus.append(prior_dist["mu"])
            prior_lvs.append(prior_dist["logvar"])
            post_mus.append(post_dist["mu"])
            post_lvs.append(post_dist["logvar"])
            z_posts.append(z)
            contact_logits_list.append(self.contact_head(h))
            affordance_logits_list.append(self.affordance_head(h))

        return {
            "prior_mu": torch.stack(prior_mus, dim=1),
            "prior_logvar": torch.stack(prior_lvs, dim=1),
            "posterior_mu": torch.stack(post_mus, dim=1),
            "posterior_logvar": torch.stack(post_lvs, dim=1),
            "z_posterior": torch.stack(z_posts, dim=1),
            "contact_logits": torch.stack(contact_logits_list, dim=1),
            "affordance_logits": torch.stack(affordance_logits_list, dim=1),
        }

    # ------------------------------------------------------------------
    # Free-running rollout (planning / eval)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(
        self,
        z0: _Tensor,
        actions: _Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[_Tensor, dict[str, torch.Tensor]]:
        """Open-loop rollout from ``z0`` for H steps under the prior.

        Args:
            z0: Shape ``(B, 1, d_z)`` or ``(B, d_z)``.
            actions: Shape ``(B, H, d_a)``.
            context: Optional context (used to initialise ``h``).

        Returns:
            ``(z_pred [B, H, d_z], aux)`` where ``aux`` holds
            ``mu``, ``logvar``, ``contact_logits``, ``affordance_logits``.

        Raises:
            ShapeError: If ``z0`` or ``actions`` have wrong rank.
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)
        if z0.dim() != 3:
            raise ShapeError(f"Expected z0 shape (B, T, d_z), got {tuple(z0.shape)}")
        if actions.dim() != 3:
            raise ShapeError(f"Expected actions shape (B, H, d_a), got {tuple(actions.shape)}")

        B, _, _ = z0.shape
        H = actions.shape[1]
        device, dtype = z0.device, z0.dtype

        # Initialise recurrent state from context if provided (B, d_rec)
        if context is not None and context.shape[-1] == self.d_recurrent:
            h = context.reshape(B, self.d_recurrent)
        else:
            h = torch.zeros(B, self.d_recurrent, device=device, dtype=dtype)

        z = z0[:, -1]  # (B, d_z)
        zs, mus, logvars, contacts, affordances = [], [], [], [], []

        for step in range(H):
            prior_dist, h = self.prior_step(z, actions[:, step], h)
            z = self._rsample(prior_dist)
            zs.append(z)
            mus.append(prior_dist["mu"])
            logvars.append(prior_dist["logvar"])
            contacts.append(self.contact_head(h))
            affordances.append(self.affordance_head(h))

        z_pred = torch.stack(zs, dim=1)  # (B, H, d_z)
        aux: dict[str, torch.Tensor] = {
            "mu": torch.stack(mus, dim=1),
            "logvar": torch.stack(logvars, dim=1),
            "contact_logits": torch.stack(contacts, dim=1),
            "affordance_logits": torch.stack(affordances, dim=1),
        }
        return z_pred, aux


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class EnsembleDynamics(nn.Module):
    """Ensemble of ``RSSMDynamics`` models for epistemic uncertainty.

    Args:
        n_members: Number of ensemble members.
        **kwargs: Forwarded to :class:`RSSMDynamics`.

    Shape:
        Same as :class:`RSSMDynamics`.  The returned ``logvar`` also encodes
        inter-member disagreement via the law of total variance.
    """

    def __init__(self, n_members: int = 5, **kwargs: Any) -> None:
        super().__init__()
        self.members = nn.ModuleList([RSSMDynamics(**kwargs) for _ in range(n_members)])

    def rollout(
        self,
        z0: _Tensor,
        actions: _Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[_Tensor, dict[str, torch.Tensor]]:
        """Ensemble rollout; returns mean prediction + total uncertainty.

        Returns:
            ``(z_pred [B, H, d_z], aux)`` where ``aux["logvar"]`` reflects
            both aleatoric and epistemic uncertainty.
        """
        all_z, all_mu, all_lv, all_contact, all_aff = [], [], [], [], []
        for member in self.members:
            z_pred, aux = member.rollout(z0, actions, context)
            all_z.append(z_pred)
            all_mu.append(aux["mu"])
            all_lv.append(aux["logvar"])
            all_contact.append(aux["contact_logits"])
            all_aff.append(aux["affordance_logits"])

        stacked_mu = torch.stack(all_mu, dim=0)   # (M, B, H, d_z)
        stacked_lv = torch.stack(all_lv, dim=0)
        # Law of total variance: E[Var] + Var[E]
        mean_mu = stacked_mu.mean(0)
        aleatoric = stacked_lv.exp().mean(0)
        epistemic = stacked_mu.var(0).clamp_min(1e-8)
        total_var = aleatoric + epistemic
        total_logvar = total_var.log()

        return mean_mu, {
            "mu": mean_mu,
            "logvar": total_logvar,
            "contact_logits": torch.stack(all_contact, dim=0).mean(0),
            "affordance_logits": torch.stack(all_aff, dim=0).mean(0),
        }


# ---------------------------------------------------------------------------
# Legacy alias kept for backward compat with existing tests
# ---------------------------------------------------------------------------

class DynamicsModel(nn.Module):
    """Thin wrapper kept for backward compatibility.

    Prefer :class:`RSSMDynamics` for new code.
    """

    def __init__(self, d_latent: int = 128, d_action: int = 6) -> None:
        super().__init__()
        self.d_latent = d_latent
        self._rssm = RSSMDynamics(d_latent=d_latent, d_action=d_action)

    def forward(self, tactile_latent: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor) -> LatentDist:
        del proprio  # legacy signature; ignored
        if tactile_latent.dim() == 2:
            tactile_latent = tactile_latent.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)
        z_pred, aux = self._rssm.rollout(tactile_latent, action)
        return {"mu": z_pred[:, 0], "logvar": aux["logvar"][:, 0]}

    def rollout(
        self,
        z0: _Tensor,
        actions: _Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[_Tensor, dict[str, torch.Tensor]]:
        return self._rssm.rollout(z0, actions, context)


class PriorDynamicsModel(nn.Module):
    """Unconditional Gaussian prior for KL regularisation."""

    def __init__(self, d_latent: int = 128, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_latent
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.head = _GaussianHead(in_dim, d_latent)

    def forward(self, prev_latent: torch.Tensor) -> LatentDist:
        if prev_latent.dim() == 2:
            prev_latent = prev_latent.unsqueeze(0)
        return self.head(self.trunk(prev_latent))
