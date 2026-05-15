"""Phase 5: Sim-to-real fine-tuning with limited real-robot data.

Target: H-C — ≤500 real episodes reduces the sim-to-real gap to < 10 %.

Strategy:
  - Freeze all layers except a small adapter MLP on top of the tactile encoder
    and the last RSSM GRU step.
  - Use a low learning rate with cosine schedule.
  - Optional: domain-adversarial alignment loss (gradient reversal between
    sim and real latents).

Run via::

    tfwm train phase=phase5 data.real_episodes=data/real task=peg_in_hole
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from tfwm.data.synth import SyntheticTactileDataset
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.train.trainer_base import TrainerBase


class _GradReverse(torch.autograd.Function):
    """Gradient reversal layer for domain alignment."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lam * grad, None


class _DomainDiscriminator(nn.Module):
    """Binary MLP: sim (0) vs. real (1)."""

    def __init__(self, d_latent: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        z_rev = _GradReverse.apply(z, lam)
        return self.net(z_rev)


class Phase5FineTuneTrainer(TrainerBase):
    """Phase 5 sim-to-real fine-tuning trainer.

    Args:
        encoder: Tactile encoder (partially unfrozen).
        dynamics: RSSM (partially unfrozen).
        adapter: Small MLP adapter appended after the frozen encoder.
        n_freeze_encoder_layers: Number of TCN blocks to keep frozen
            (0 = fine-tune all, -1 = freeze all).
        use_domain_align: If ``True``, train a domain discriminator with
            gradient reversal.
        lam_domain: Weight of the domain-alignment adversarial loss.
        real_loader: DataLoader over real-robot episodes.
        **kwargs: Forwarded to :class:`~tfwm.train.trainer_base.TrainerBase`.
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        dynamics: RSSMDynamics,
        adapter: nn.Module,
        n_freeze_encoder_layers: int = -1,
        use_domain_align: bool = False,
        lam_domain: float = 0.1,
        real_loader: DataLoader | None = None,  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        modules: dict[str, nn.Module] = {"adapter": adapter}
        if n_freeze_encoder_layers != -1:
            modules["encoder"] = encoder
        if use_domain_align:
            disc = _DomainDiscriminator(d_latent=adapter.out_features if hasattr(adapter, "out_features") else 128)
            modules["discriminator"] = disc
            self._disc: _DomainDiscriminator | None = disc
        else:
            self._disc = None

        super().__init__(model=modules, **kwargs)  # type: ignore[arg-type]

        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.adapter = adapter
        self.lam_domain = lam_domain
        self.real_iter: Any = None
        if real_loader is not None:
            self.real_iter = iter(real_loader)
        self._real_loader = real_loader

        # Freeze encoder layers.
        if n_freeze_encoder_layers == -1:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        for p in self.dynamics.parameters():
            p.requires_grad_(False)

        # Scheduler: cosine annealing over full budget.
        max_steps = kwargs.get("max_steps", 10_000)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=1e-6)

    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """Fine-tune step using mixed sim + real batches.

        Args:
            batch: Sim batch with ``tactile (B,T,N,C)`` and ``actions (B,T,d_a)``.
            step: Current global step.

        Returns:
            Metrics dict.
        """
        sim_tactile = batch["tactile"]
        sim_actions = batch["actions"]

        self.optimizer.zero_grad(set_to_none=True)

        # ── Sim encode + reconstruct ───────────────────────────────────
        enc_sim = self.encoder(sim_tactile)
        z_sim = enc_sim["mu"]
        z_sim_adapted = self.adapter(z_sim)

        with torch.no_grad():
            rssm_out = self.dynamics.forward_teacher(z_sim, sim_actions)
        pred_z = rssm_out["z_posterior"]
        loss_recon = F.mse_loss(z_sim_adapted, pred_z)

        total_loss = loss_recon
        metrics: dict[str, float] = {"loss_sim_recon": float(loss_recon)}

        # ── Real data (if available) ───────────────────────────────────
        if self._real_loader is not None and self._disc is not None:
            real_batch = self._next_real()
            real_tactile = real_batch["tactile"].to(self.device)
            enc_real = self.encoder(real_tactile)
            z_real = enc_real["mu"]
            z_real_adapted = self.adapter(z_real)

            # Domain alignment: make sim and real latents indistinguishable.
            # GRL causes encoder to produce domain-invariant features.
            B_s = z_sim_adapted.shape[0]
            B_r = z_real_adapted.shape[0]
            T = min(z_sim_adapted.shape[1], z_real_adapted.shape[1])
            sim_flat = z_sim_adapted[:, :T].reshape(B_s * T, -1)
            real_flat = z_real_adapted[:, :T].reshape(B_r * T, -1)

            sim_labels = torch.zeros(B_s * T, 1, device=self.device)
            real_labels = torch.ones(B_r * T, 1, device=self.device)

            all_z = torch.cat([sim_flat, real_flat], dim=0)
            all_labels = torch.cat([sim_labels, real_labels], dim=0)
            disc_preds = self._disc(all_z)
            loss_domain = F.binary_cross_entropy_with_logits(disc_preds, all_labels)
            total_loss = total_loss + self.lam_domain * loss_domain
            metrics["loss_domain"] = float(loss_domain)

        total_loss.backward()
        self._clip_and_step()
        self.scheduler.step()
        metrics["loss_total"] = float(total_loss)
        metrics["lr"] = self.scheduler.get_last_lr()[0]
        return metrics

    def _next_real(self) -> dict[str, torch.Tensor]:
        try:
            return next(self.real_iter)
        except StopIteration:
            self.real_iter = iter(self._real_loader)
            return next(self.real_iter)

    def validation_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        tactile = batch["tactile"]
        actions = batch["actions"]
        with torch.no_grad():
            enc_out = self.encoder(tactile)
            z = enc_out["mu"]
            z_adapted = self.adapter(z)
            rssm_out = self.dynamics.forward_teacher(z, actions)
        loss = F.mse_loss(z_adapted, rssm_out["z_posterior"])
        return {"val_finetune_mse": float(loss)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_phase5_trainer(
    cfg: DictConfig,
    phase1_ckpt: Path | None = None,
) -> Phase5FineTuneTrainer:
    """Construct a :class:`Phase5FineTuneTrainer` from a Hydra config.

    Args:
        cfg: Resolved top-level Hydra config.
        phase1_ckpt: Optional path to a Phase 1 (or Phase 4) checkpoint.

    Returns:
        Configured :class:`Phase5FineTuneTrainer`.
    """
    enc_cfg = cfg.model.tactile_encoder
    dyn_cfg = cfg.model.dynamics
    train_cfg = cfg.train

    device = torch.device(cfg.get("device", "cpu"))
    d_latent = enc_cfg.get("d_latent", 128)

    encoder = HybridTactileEncoder(
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_latent=d_latent,
    )
    dynamics = RSSMDynamics(
        d_latent=dyn_cfg.get("d_latent", d_latent),
        d_action=dyn_cfg.get("d_action", 6),
    )

    if phase1_ckpt is not None and phase1_ckpt.exists():
        state = torch.load(phase1_ckpt, map_location=device)
        mods = state.get("modules", {})
        if "encoder" in mods:
            encoder.load_state_dict(mods["encoder"])
        if "dynamics" in mods:
            dynamics.load_state_dict(mods["dynamics"])

    # Lightweight adapter: two-layer residual MLP.
    adapter = nn.Sequential(
        nn.Linear(d_latent, d_latent),
        nn.SiLU(),
        nn.Linear(d_latent, d_latent),
    )

    optimizer = optim.AdamW(adapter.parameters(), lr=train_cfg.get("learning_rate", 1e-4))

    dataset = SyntheticTactileDataset(
        n_episodes=100,
        episode_length=50,
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_action=dyn_cfg.get("d_action", 6),
    )
    n_val = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 8), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 8), shuffle=False, num_workers=0)

    return Phase5FineTuneTrainer(
        encoder=encoder,
        dynamics=dynamics,
        adapter=adapter,
        n_freeze_encoder_layers=train_cfg.get("n_freeze_encoder_layers", -1),
        use_domain_align=train_cfg.get("use_domain_align", False),
        lam_domain=train_cfg.get("lam_domain", 0.1),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(cfg.get("output_dir", "outputs/phase5")),
        device=device,
        max_steps=train_cfg.get("max_steps", 10_000),
        log_every=train_cfg.get("log_every", 50),
        eval_every=train_cfg.get("eval_every", 1_000),
        save_every=train_cfg.get("save_every", 2_000),
        run_name=cfg.get("run_name", "phase5_finetune"),
        cfg=dict(cfg),
    )
