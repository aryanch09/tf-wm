"""Loss functions for TF-WM."""

from __future__ import annotations

from tfwm.losses.affordance import AffordanceLoss
from tfwm.losses.contact_bce import ContactBCELoss
from tfwm.losses.contrastive import InfoNCELoss
from tfwm.losses.kl import KLDivergenceLoss
from tfwm.losses.multi_horizon import MultiHorizonLoss
from tfwm.losses.tactile_recon import TactileReconLoss

# Keep legacy functional API available.
from tfwm.losses.functional import kl_divergence, reconstruction_loss

__all__ = [
    "TactileReconLoss",
    "InfoNCELoss",
    "ContactBCELoss",
    "KLDivergenceLoss",
    "AffordanceLoss",
    "MultiHorizonLoss",
    "kl_divergence",
    "reconstruction_loss",
]
