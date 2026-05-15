"""Multi-horizon loss aggregator.

Implements the weighted geometric discount from Section 7 of COPILOT.md:

    L_total = Σ_{h=1..H} γ^(h-1) * L_h

where ``L_h`` is the loss evaluated at prediction horizon ``h`` and ``γ`` is
the discount factor (default 0.85).

The aggregator also supports per-loss-term weights (``alpha_*``) and linear
KL annealing / scheduled sampling state injection so the Phase 1 trainer can
delegate all loss bookkeeping to a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from tfwm.losses.affordance import AffordanceLoss, ObjectStateLoss
from tfwm.losses.contact_bce import ContactBCELoss
from tfwm.losses.contrastive import InfoNCELoss
from tfwm.losses.kl import KLDivergenceLoss
from tfwm.losses.tactile_recon import TactileReconLoss


@dataclass
class LossWeights:
    """Scalar multipliers for each loss term (all default to 1.0 or off)."""

    alpha_recon: float = 1.0
    alpha_cpc:   float = 1.0
    alpha_event: float = 0.1
    alpha_kl:    float = 0.1
    alpha_aff:   float = 0.0
    alpha_obj:   float = 0.0


class MultiHorizonLoss(nn.Module):
    """Geometrically discounted multi-horizon loss aggregator.

    Args:
        weights: :class:`LossWeights` instance.
        gamma: Horizon discount factor (0 < γ ≤ 1).
        kl_loss: Optional :class:`~tfwm.losses.kl.KLDivergenceLoss`.
        recon_loss: Optional :class:`~tfwm.losses.tactile_recon.TactileReconLoss`.
        cpc_loss: Optional :class:`~tfwm.losses.contrastive.InfoNCELoss`.
        contact_loss: Optional :class:`~tfwm.losses.contact_bce.ContactBCELoss`.
        affordance_loss: Optional :class:`~tfwm.losses.affordance.AffordanceLoss`.
        object_state_loss: Optional :class:`~tfwm.losses.affordance.ObjectStateLoss`.
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        gamma: float = 0.85,
        kl_loss: KLDivergenceLoss | None = None,
        recon_loss: TactileReconLoss | None = None,
        cpc_loss: InfoNCELoss | None = None,
        contact_loss: ContactBCELoss | None = None,
        affordance_loss: AffordanceLoss | None = None,
        object_state_loss: ObjectStateLoss | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.gamma = gamma
        self.kl_loss         = kl_loss         or KLDivergenceLoss()
        self.recon_loss      = recon_loss      or TactileReconLoss()
        self.cpc_loss        = cpc_loss        or InfoNCELoss()
        self.contact_loss    = contact_loss    or ContactBCELoss()
        self.affordance_loss = affordance_loss or AffordanceLoss()
        self.obj_state_loss  = object_state_loss or ObjectStateLoss()
        # KL annealing state (updated externally by trainer).
        self.kl_weight: float = 0.0

    def _discount_weights(self, H: int, device: torch.device) -> torch.Tensor:
        """Return γ^{h-1} for h = 1..H, normalised to sum to 1."""
        w = torch.tensor(
            [self.gamma ** h for h in range(H)],
            device=device, dtype=torch.float32,
        )
        return w / w.sum()

    def forward(
        self,
        *,
        # Reconstruction
        pred_tactile: torch.Tensor | None = None,
        gt_tactile: torch.Tensor | None = None,
        # Contrastive
        anchors: torch.Tensor | None = None,
        positives: torch.Tensor | None = None,
        # Contact events
        contact_logits: torch.Tensor | None = None,
        contact_targets: torch.Tensor | None = None,
        # KL (RSSM)
        posterior_mu: torch.Tensor | None = None,
        posterior_logvar: torch.Tensor | None = None,
        prior_mu: torch.Tensor | None = None,
        prior_logvar: torch.Tensor | None = None,
        # Affordances
        affordance_logits: torch.Tensor | None = None,
        affordance_targets: torch.Tensor | None = None,
        # Object state
        pred_object_state: torch.Tensor | None = None,
        gt_object_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute and aggregate all active loss terms.

        All horizon tensors are expected to have shape ``(B, H, ...)``.
        The method applies the geometric discount along the ``H`` dimension.

        Returns:
            ``(total_loss, metrics_dict)`` where ``metrics_dict`` maps each
            active term name to its un-weighted scalar value.
        """
        device = (
            pred_tactile.device if pred_tactile is not None
            else (anchors.device if anchors is not None else torch.device("cpu"))
        )
        total: torch.Tensor = torch.zeros((), device=device)
        metrics: dict[str, float] = {}

        w = self.weights

        # ── Reconstruction ────────────────────────────────────────────
        if pred_tactile is not None and gt_tactile is not None and w.alpha_recon > 0:
            H = pred_tactile.shape[1]
            disc = self._discount_weights(H, device)
            loss = sum(
                disc[h] * self.recon_loss(pred_tactile[:, h], gt_tactile[:, h])
                for h in range(H)
            )
            metrics["loss_recon"] = float(loss)
            total = total + w.alpha_recon * loss

        # ── InfoNCE / CPC ─────────────────────────────────────────────
        if anchors is not None and positives is not None and w.alpha_cpc > 0:
            loss = self.cpc_loss(anchors, positives)
            metrics["loss_cpc"] = float(loss)
            total = total + w.alpha_cpc * loss

        # ── Contact event BCE ─────────────────────────────────────────
        if contact_logits is not None and contact_targets is not None and w.alpha_event > 0:
            loss = self.contact_loss(contact_logits, contact_targets)
            metrics["loss_contact"] = float(loss)
            total = total + w.alpha_event * loss

        # ── KL divergence ─────────────────────────────────────────────
        if posterior_mu is not None and w.alpha_kl > 0 and self.kl_weight > 0:
            loss = self.kl_loss(posterior_mu, posterior_logvar or torch.zeros_like(posterior_mu),
                                prior_mu, prior_logvar)
            metrics["loss_kl"] = float(loss)
            total = total + w.alpha_kl * self.kl_weight * loss

        # ── Affordances ───────────────────────────────────────────────
        if affordance_logits is not None and affordance_targets is not None and w.alpha_aff > 0:
            loss = self.affordance_loss(affordance_logits, affordance_targets)
            metrics["loss_affordance"] = float(loss)
            total = total + w.alpha_aff * loss

        # ── Object state ──────────────────────────────────────────────
        if pred_object_state is not None and gt_object_state is not None and w.alpha_obj > 0:
            loss = self.obj_state_loss(pred_object_state, gt_object_state)
            metrics["loss_object_state"] = float(loss)
            total = total + w.alpha_obj * loss

        metrics["loss_total"] = float(total)
        return total, metrics
