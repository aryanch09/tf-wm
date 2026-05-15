"""Phase 2: VOI gating training.

Trains the gating head to decide *when* to query vision, with the frozen
(or slow-EMA) tactile encoder and RSSM dynamics from Phase 1.

The loss has two components:
  - Gate BCE: binary label = "did querying vision reduce prediction error?"
  - Budget regulariser: lambda_budget * mean(gate_scores) penalises over-querying

Hypothesis H-B: VOI gating should reduce camera queries by ≥40 % while
preserving <5 % success-rate drop relative to always-on vision.

Run via::

    tfwm train phase=phase2 task=inhand_reorient
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from tfwm.data.synth import SyntheticTactileDataset
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.gating.voi_gate import VOIGate
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.train.trainer_base import TrainerBase


class Phase2GatingTrainer(TrainerBase):
    """Phase 2: learn when to query the vision stream.

    Args:
        encoder: Frozen tactile encoder from Phase 1.
        dynamics: Frozen RSSM dynamics from Phase 1.
        gate: VOI or Hybrid gate to train.
        lambda_budget: Weight on the camera-query budget penalty.
        gate_target_rate: Desired average gate-activation rate (used for
            adaptive lambda scheduling).
        freeze_encoder: If ``True``, no gradients flow into ``encoder``.
        freeze_dynamics: If ``True``, no gradients flow into ``dynamics``.
        **kwargs: Forwarded to :class:`~tfwm.train.trainer_base.TrainerBase`.
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        dynamics: RSSMDynamics,
        gate: nn.Module,
        lambda_budget: float = 0.1,
        gate_target_rate: float = 0.5,
        freeze_encoder: bool = True,
        freeze_dynamics: bool = True,
        **kwargs: Any,
    ) -> None:
        modules: dict[str, nn.Module] = {"gate": gate}
        if not freeze_encoder:
            modules["encoder"] = encoder
        if not freeze_dynamics:
            modules["dynamics"] = dynamics

        super().__init__(model=modules, **kwargs)  # type: ignore[arg-type]
        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.gate = gate

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        if freeze_dynamics:
            for p in self.dynamics.parameters():
                p.requires_grad_(False)
            self.dynamics.eval()

        self.lambda_budget = lambda_budget
        self.gate_target_rate = gate_target_rate

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """Compute gate BCE + budget penalty and step the gate.

        Args:
            batch: Dict with ``tactile (B,T,N,C)`` and ``actions (B,T,d_a)``.
            step: Current global step.

        Returns:
            Metrics dict.
        """
        tactile = batch["tactile"]   # (B, T, N, C)
        actions = batch["actions"]   # (B, T, d_a)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            enc_out = self.encoder(tactile)
            obs_latents = enc_out["mu"]   # (B, T, d_z)
            rssm_out = self.dynamics.forward_teacher(obs_latents, actions)

        z_post = rssm_out["z_posterior"]   # (B, T, d_z)
        z_prior = rssm_out.get("z_prior", z_post)

        # Prediction error per timestep: higher = more need for vision.
        pred_err = F.mse_loss(z_prior, z_post, reduction="none").mean(-1)  # (B, T)

        # Binary gate target: 1 where prediction error exceeds median.
        with torch.no_grad():
            threshold = pred_err.median()
            gate_target = (pred_err > threshold).float()   # (B, T)

        # Gate forward.
        dyn_aux = {k: rssm_out[k] for k in ("posterior_logvar", "prior_logvar") if k in rssm_out}
        aux = {"logvar": dyn_aux.get("posterior_logvar", torch.zeros_like(z_post))}
        gate_scores = self.gate(z_post, aux)   # (B, T)

        loss_bce = F.binary_cross_entropy(gate_scores, gate_target)
        loss_budget = self.lambda_budget * gate_scores.mean()
        total_loss = loss_bce + loss_budget

        total_loss.backward()
        self._clip_and_step()

        avg_rate = float(gate_scores.detach().mean())
        return {
            "loss_gate_bce": float(loss_bce),
            "loss_budget": float(loss_budget),
            "loss_total": float(total_loss),
            "gate_rate": avg_rate,
        }

    def validation_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        tactile = batch["tactile"]
        actions = batch["actions"]
        with torch.no_grad():
            enc_out = self.encoder(tactile)
            obs_latents = enc_out["mu"]
            rssm_out = self.dynamics.forward_teacher(obs_latents, actions)
            z_post = rssm_out["z_posterior"]
            aux = {"logvar": rssm_out.get("posterior_logvar", torch.zeros_like(z_post))}
            gate_scores = self.gate(z_post, aux)
        return {"val_gate_rate": float(gate_scores.mean())}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_phase2_trainer(cfg: DictConfig, phase1_ckpt: Path | None = None) -> Phase2GatingTrainer:
    """Construct a :class:`Phase2GatingTrainer` from a Hydra config.

    Args:
        cfg: Resolved top-level Hydra config.
        phase1_ckpt: Optional path to a Phase 1 checkpoint to warm-start encoder/dynamics.

    Returns:
        Configured :class:`Phase2GatingTrainer`.
    """
    enc_cfg = cfg.model.tactile_encoder
    dyn_cfg = cfg.model.dynamics
    gate_cfg = cfg.model.get("gating", {})
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

    if phase1_ckpt is not None and phase1_ckpt.exists():
        state = torch.load(phase1_ckpt, map_location=device)
        mods = state.get("modules", {})
        if "encoder" in mods:
            encoder.load_state_dict(mods["encoder"])
        if "dynamics" in mods:
            dynamics.load_state_dict(mods["dynamics"])

    gate: nn.Module = VOIGate(
        d_latent=enc_cfg.get("d_latent", 128),
        threshold=gate_cfg.get("threshold", 0.5),
    )

    optimizer = optim.AdamW(gate.parameters(), lr=train_cfg.get("learning_rate", 3e-4))

    dataset = SyntheticTactileDataset(
        n_episodes=100,
        episode_length=50,
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_action=dyn_cfg.get("d_action", 6),
    )
    n_val = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False, num_workers=0)

    return Phase2GatingTrainer(
        encoder=encoder,
        dynamics=dynamics,
        gate=gate,
        lambda_budget=train_cfg.get("lambda_budget", 0.1),
        gate_target_rate=train_cfg.get("gate_target_rate", 0.5),
        freeze_encoder=train_cfg.get("freeze_encoder", True),
        freeze_dynamics=train_cfg.get("freeze_dynamics", True),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(cfg.get("output_dir", "outputs/phase2")),
        device=device,
        max_steps=train_cfg.get("max_steps", 50_000),
        log_every=train_cfg.get("log_every", 100),
        eval_every=train_cfg.get("eval_every", 2_000),
        save_every=train_cfg.get("save_every", 5_000),
        run_name=cfg.get("run_name", "phase2_gating"),
        cfg=dict(cfg),
    )
