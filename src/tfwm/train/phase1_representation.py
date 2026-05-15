"""Phase 1: Tactile representation learning.

Trains the tactile encoder and RSSM dynamics jointly with:
  - Tactile reconstruction (MSE)
  - InfoNCE / CPC contrastive
  - Contact event BCE
  - KL divergence with linear warmup (free-nats)
  - Optional affordance / object-state losses (disabled by default)

Run via CLI::

    tfwm train phase=phase1 task=inhand_reorient train.batch_size=256

Or directly::

    python -m tfwm.cli.train phase=phase1
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader, random_split

from tfwm.data.dataset import EpisodeDataset, SequenceDataset
from tfwm.losses.multi_horizon import LossWeights, MultiHorizonLoss
from tfwm.models.decoders.contact_event_head import ContactEventHead
from tfwm.models.decoders.tactile_decoder import TactileDecoder
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.train.trainer_base import TrainerBase


class Phase1Trainer(TrainerBase):
    """Phase 1 representation learning trainer.

    Args:
        encoder: Tactile encoder (default: :class:`~tfwm.models.tactile_encoder.hybrid.HybridTactileEncoder`).
        dynamics: RSSM dynamics model.
        tactile_decoder: Reconstruction head.
        contact_head: Contact event classifier.
        loss_fn: :class:`~tfwm.losses.multi_horizon.MultiHorizonLoss` instance.
        kl_anneal_steps: Steps to ramp KL weight from 0 → 1.
        scheduled_sampling_start: Initial teacher-forcing probability (1.0 = full TF).
        scheduled_sampling_end: Final teacher-forcing probability.
        scheduled_sampling_steps: Steps over which to decay ε.
        **kwargs: Forwarded to :class:`~tfwm.train.trainer_base.TrainerBase`.
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        dynamics: RSSMDynamics,
        tactile_decoder: TactileDecoder,
        contact_head: ContactEventHead,
        loss_fn: MultiHorizonLoss,
        kl_anneal_steps: int = 20_000,
        scheduled_sampling_start: float = 1.0,
        scheduled_sampling_end: float = 0.1,
        scheduled_sampling_steps: int = 100_000,
        **kwargs: Any,
    ) -> None:
        modules = {
            "encoder": encoder,
            "dynamics": dynamics,
            "tactile_decoder": tactile_decoder,
            "contact_head": contact_head,
        }
        super().__init__(model=modules, **kwargs)  # type: ignore[arg-type]
        self.encoder = encoder
        self.dynamics = dynamics
        self.tactile_decoder = tactile_decoder
        self.contact_head = contact_head
        self.loss_fn = loss_fn
        self.kl_anneal_steps = max(kl_anneal_steps, 1)
        self.ss_start = scheduled_sampling_start
        self.ss_end = scheduled_sampling_end
        self.ss_steps = max(scheduled_sampling_steps, 1)

    # ------------------------------------------------------------------
    # Scheduled-sampling epsilon
    # ------------------------------------------------------------------

    @property
    def _epsilon(self) -> float:
        """Teacher-forcing probability at current step."""
        frac = min(self.global_step / self.ss_steps, 1.0)
        return self.ss_start - frac * (self.ss_start - self.ss_end)

    @property
    def _kl_weight(self) -> float:
        """Linear KL annealing weight."""
        return min(self.global_step / self.kl_anneal_steps, 1.0)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """One gradient step for Phase 1.

        Args:
            batch: Dict with keys ``tactile (B,T,N,C)``, ``actions (B,T,d_a)``,
                and optionally ``contact_events (B,T)``.
            step: Current global step.

        Returns:
            Metrics dict with scalar loss terms.
        """
        tactile = batch["tactile"]   # (B, T, N, C)
        actions = batch["actions"]   # (B, T, d_a)

        B, T, N, C = tactile.shape

        self.optimizer.zero_grad(set_to_none=True)

        # ── Encode observations ────────────────────────────────────────
        enc_out = self.encoder(tactile)  # LatentDist; mu/logvar (B, T, d_z)
        obs_latents = enc_out["mu"]

        # ── Teacher-forced RSSM rollout ────────────────────────────────
        rssm_out = self.dynamics.forward_teacher(obs_latents, actions)
        z_post = rssm_out["z_posterior"]  # (B, T, d_z)

        # ── Reconstruct tactile from posterior ─────────────────────────
        pred_tactile = self.tactile_decoder(z_post)  # (B, T, N, C)

        # ── Contact events ────────────────────────────────────────────
        contact_logits = self.contact_head(z_post)  # (B, T, K)
        contact_targets = batch.get("contact_events")

        # ── Update KL annealing weight ─────────────────────────────────
        self.loss_fn.kl_weight = self._kl_weight

        # ── Aggregate losses ──────────────────────────────────────────
        # Expand pred/gt tactile to (B, T, 1, N, C) with H=1 for MultiHorizonLoss.
        total_loss, metrics = self.loss_fn(
            pred_tactile=z_post.unsqueeze(2),          # proxy: predict latent
            gt_tactile=obs_latents.unsqueeze(2),
            anchors=obs_latents[:, :-1],
            positives=obs_latents[:, 1:],
            contact_logits=contact_logits if contact_targets is not None else None,
            contact_targets=contact_targets,
            posterior_mu=rssm_out["posterior_mu"],
            posterior_logvar=rssm_out["posterior_logvar"],
            prior_mu=rssm_out["prior_mu"],
            prior_logvar=rssm_out["prior_logvar"],
        )

        # Add explicit reconstruction in tactile space.
        recon = torch.nn.functional.mse_loss(pred_tactile, tactile)
        total_loss = total_loss + self.loss_fn.weights.alpha_recon * recon
        metrics["loss_tactile_recon"] = float(recon)
        metrics["epsilon"] = self._epsilon
        metrics["kl_weight"] = self._kl_weight

        total_loss.backward()
        self._clip_and_step()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """Validation step: compute reconstruction MSE.

        Args:
            batch: Same structure as :meth:`training_step`.
            step: Current global step.

        Returns:
            ``{"val_tactile_mse": float}``.
        """
        tactile = batch["tactile"]
        actions = batch["actions"]
        enc_out = self.encoder(tactile)
        rssm_out = self.dynamics.forward_teacher(enc_out["mu"], actions)
        pred = self.tactile_decoder(rssm_out["z_posterior"])
        return {"val_tactile_mse": float(torch.nn.functional.mse_loss(pred, tactile).item())}


# ---------------------------------------------------------------------------
# Factory — build from Hydra config
# ---------------------------------------------------------------------------

def build_phase1_trainer(cfg: DictConfig) -> Phase1Trainer:
    """Construct a :class:`Phase1Trainer` from a Hydra config.

    Args:
        cfg: Resolved top-level Hydra config.

    Returns:
        Configured :class:`Phase1Trainer`.
    """
    enc_cfg = cfg.model.tactile_encoder
    dyn_cfg = cfg.model.dynamics
    train_cfg = cfg.train

    device = torch.device(cfg.get("device", "cpu"))

    encoder = HybridTactileEncoder(
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_latent=enc_cfg.get("d_latent", 128),
    )
    dynamics = RSSMDynamics(
        d_latent=dyn_cfg.get("d_latent", 128),
        d_action=dyn_cfg.get("d_action", 6),
    )
    tactile_decoder = TactileDecoder(
        d_latent=enc_cfg.get("d_latent", 128),
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
    )
    contact_head = ContactEventHead(d_latent=enc_cfg.get("d_latent", 128))

    loss_weights = LossWeights(**{k: v for k, v in train_cfg.loss.items()})
    loss_fn = MultiHorizonLoss(weights=loss_weights, gamma=0.85)

    all_params = (
        list(encoder.parameters())
        + list(dynamics.parameters())
        + list(tactile_decoder.parameters())
        + list(contact_head.parameters())
    )
    optimizer = optim.AdamW(all_params, lr=train_cfg.get("learning_rate", 1e-3), weight_decay=1e-4)

    # Placeholder dataset — real usage passes data via CLI / Hydra.
    from tfwm.data.synth import SyntheticTactileDataset
    dataset = SyntheticTactileDataset(
        n_episodes=100,
        episode_length=50,
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_action=dyn_cfg.get("d_action", 6),
    )
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False, num_workers=0)

    return Phase1Trainer(
        encoder=encoder,
        dynamics=dynamics,
        tactile_decoder=tactile_decoder,
        contact_head=contact_head,
        loss_fn=loss_fn,
        kl_anneal_steps=train_cfg.kl.get("anneal_steps", 20_000),
        scheduled_sampling_start=train_cfg.scheduled_sampling.get("start_epsilon", 1.0),
        scheduled_sampling_end=train_cfg.scheduled_sampling.get("end_epsilon", 0.1),
        scheduled_sampling_steps=train_cfg.scheduled_sampling.get("decay_steps", 100_000),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(cfg.get("output_dir", "outputs/phase1")),
        device=device,
        max_steps=train_cfg.get("max_steps", 100_000),
        log_every=train_cfg.get("log_every", 100),
        eval_every=train_cfg.get("eval_every", 5_000),
        save_every=train_cfg.get("save_every", 10_000),
        run_name=cfg.get("run_name", "phase1"),
        cfg=dict(cfg),
    )
