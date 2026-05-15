"""Phase 4: SAC policy learning in latent space.

Trains the :class:`~tfwm.policies.sac.SACPolicy` and
:class:`~tfwm.policies.sac.TwinCritic` with a frozen world model as the
environment model.  All experience is collected in the latent world-model
rollout (Dyna-style), so no real environment is required for this phase.

Loss terms follow standard SAC:
  - Critic loss:  MSE to Bellman target (soft, clipped double-Q)
  - Actor loss:   min(Q) − α * log_prob
  - Temperature:  auto-tuned via dual gradient descent on target entropy

Run via::

    tfwm train phase=phase4 task=peg_in_hole
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from tfwm.data.buffer import ReplayBuffer
from tfwm.data.synth import SyntheticTactileDataset
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.policies.sac import SACPolicy, TwinCritic
from tfwm.train.trainer_base import TrainerBase


class Phase4PolicyTrainer(TrainerBase):
    """Phase 4: off-policy SAC in the world-model latent space.

    Args:
        encoder: Frozen tactile encoder.
        dynamics: Frozen RSSM for imagination rollouts.
        actor: SAC actor.
        critic: Twin Q-critic.
        target_critic: Lagging copy of ``critic``.
        gamma: Discount factor.
        tau: Polyak averaging rate for target critic.
        init_alpha: Initial SAC temperature.
        target_entropy: Entropy target (defaults to ``-d_action``).
        n_imagination_steps: Latent rollout horizon per real transition.
        replay_buffer: Experience replay.
        **kwargs: Forwarded to :class:`~tfwm.train.trainer_base.TrainerBase`.
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        dynamics: RSSMDynamics,
        actor: SACPolicy,
        critic: TwinCritic,
        target_critic: TwinCritic,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        target_entropy: float | None = None,
        n_imagination_steps: int = 5,
        replay_buffer: ReplayBuffer | None = None,
        **kwargs: Any,
    ) -> None:
        modules: dict[str, nn.Module] = {
            "actor": actor,
            "critic": critic,
        }
        super().__init__(model=modules, **kwargs)  # type: ignore[arg-type]

        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic.to(self.device)

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        for p in self.dynamics.parameters():
            p.requires_grad_(False)
        self.dynamics.eval()
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.gamma = gamma
        self.tau = tau
        self.n_imagination_steps = n_imagination_steps
        self.replay_buffer = replay_buffer

        # SAC temperature (log-parameterised for stability).
        self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)).log())
        d_action = actor.mu_head.out_features
        self.target_entropy = target_entropy if target_entropy is not None else -float(d_action)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Separate optimisers for actor and critic.
        self.critic_optimizer = optim.AdamW(critic.parameters(), lr=3e-4)
        self.actor_optimizer = optim.AdamW(actor.parameters(), lr=3e-4)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # ------------------------------------------------------------------
    # Target critic soft update
    # ------------------------------------------------------------------

    def _soft_update(self) -> None:
        for p, p_tgt in zip(self.critic.parameters(), self.target_critic.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """One SAC update step (imagination rollout as environment).

        Args:
            batch: Dict with ``tactile (B,T,N,C)`` and ``actions (B,T,d_a)``.
            step: Current global step.

        Returns:
            Metrics dict.
        """
        tactile = batch["tactile"]   # (B, T, N, C)
        actions = batch["actions"]   # (B, T, d_a)

        with torch.no_grad():
            enc_out = self.encoder(tactile)
            obs_latents = enc_out["mu"]    # (B, T, d_z)

        # Use the last encoded latent as the initial state.
        z0 = obs_latents[:, -1]   # (B, d_z)

        # ── Imagination rollout ────────────────────────────────────────
        imagined_z, imagined_a, imagined_r = self._imagine(z0)

        # Flatten imagination (B*H, d_z), (B*H, d_a).
        B, H = imagined_z.shape[:2]
        z_flat = imagined_z[:, :-1].reshape(B * (H - 1), -1)
        z_next_flat = imagined_z[:, 1:].reshape(B * (H - 1), -1)
        a_flat = imagined_a[:, :-1].reshape(B * (H - 1), -1)
        r_flat = imagined_r[:, :-1].reshape(B * (H - 1))

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            a_next, lp_next = self.actor.log_prob(z_next_flat)
            q_next = self.target_critic.min_q(z_next_flat, a_next)
            target_q = r_flat.unsqueeze(1) + self.gamma * (q_next - self.alpha * lp_next.unsqueeze(1))

        q1, q2 = self.critic(z_flat, a_flat)
        loss_critic = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ── Actor update ───────────────────────────────────────────────
        a_new, lp_new = self.actor.log_prob(z_flat.detach())
        q_val = self.critic.min_q(z_flat.detach(), a_new)
        loss_actor = (self.alpha * lp_new - q_val.squeeze(-1)).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ── Temperature update ─────────────────────────────────────────
        loss_alpha = -(self.log_alpha * (lp_new.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.alpha_optimizer.step()

        # ── Target critic soft update ──────────────────────────────────
        self._soft_update()

        return {
            "loss_critic": float(loss_critic),
            "loss_actor": float(loss_actor),
            "loss_alpha": float(loss_alpha),
            "alpha": float(self.alpha),
            "mean_reward": float(r_flat.mean()),
        }

    @torch.no_grad()
    def _imagine(
        self, z0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out the world model using the current actor.

        Args:
            z0: Initial latent state ``(B, d_z)``.

        Returns:
            ``(imagined_z, imagined_a, imagined_r)`` each of shape ``(B, H, ...)``.
        """
        B, d_z = z0.shape
        H = self.n_imagination_steps
        zs = torch.zeros(B, H + 1, d_z, device=self.device)
        actions = torch.zeros(B, H, self.actor.mu_head.out_features, device=self.device)
        rewards = torch.zeros(B, H, device=self.device)

        zs[:, 0] = z0
        for t in range(H):
            a = self.actor.act({"latent": zs[:, t]})
            actions[:, t] = a
            a_expanded = a.unsqueeze(1)   # (B, 1, d_a)
            z_expanded = zs[:, t].unsqueeze(1)   # (B, 1, d_z)
            z_next, _ = self.dynamics.rollout(z_expanded, a_expanded)
            zs[:, t + 1] = z_next[:, 0]
            # Simple reward: negative norm of latent change (placeholder).
            rewards[:, t] = -torch.norm(zs[:, t + 1] - zs[:, t], dim=-1)

        return zs, actions, rewards

    def validation_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        tactile = batch["tactile"]
        with torch.no_grad():
            enc_out = self.encoder(tactile)
            z0 = enc_out["mu"][:, -1]
            _, _, rewards = self._imagine(z0)
        return {"val_mean_reward": float(rewards.mean())}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_phase4_trainer(
    cfg: DictConfig,
    phase1_ckpt: Path | None = None,
) -> Phase4PolicyTrainer:
    """Construct a :class:`Phase4PolicyTrainer` from a Hydra config.

    Args:
        cfg: Resolved top-level Hydra config.
        phase1_ckpt: Optional path to a Phase 1 checkpoint.

    Returns:
        Configured :class:`Phase4PolicyTrainer`.
    """
    enc_cfg = cfg.model.tactile_encoder
    dyn_cfg = cfg.model.dynamics
    pol_cfg = cfg.model.get("policy", {})
    train_cfg = cfg.train

    device = torch.device(cfg.get("device", "cpu"))
    d_latent = enc_cfg.get("d_latent", 128)
    d_action = dyn_cfg.get("d_action", 6)

    encoder = HybridTactileEncoder(
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_latent=d_latent,
    )
    dynamics = RSSMDynamics(
        d_latent=dyn_cfg.get("d_latent", d_latent),
        d_action=d_action,
    )

    if phase1_ckpt is not None and phase1_ckpt.exists():
        state = torch.load(phase1_ckpt, map_location=device)
        mods = state.get("modules", {})
        if "encoder" in mods:
            encoder.load_state_dict(mods["encoder"])
        if "dynamics" in mods:
            dynamics.load_state_dict(mods["dynamics"])

    actor = SACPolicy(
        d_latent=d_latent,
        d_action=d_action,
        hidden_dim=pol_cfg.get("hidden_dim", 256),
        n_layers=pol_cfg.get("n_layers", 3),
    )
    critic = TwinCritic(d_latent=d_latent, d_action=d_action)
    import copy
    target_critic = copy.deepcopy(critic)

    # Dummy optimizer — Phase4PolicyTrainer manages its own per-component optimisers.
    dummy_opt = optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)

    dataset = SyntheticTactileDataset(
        n_episodes=100,
        episode_length=50,
        n_taxels=enc_cfg.get("n_taxels", 16),
        n_channels=enc_cfg.get("n_channels", 1),
        d_action=d_action,
    )
    n_val = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False, num_workers=0)

    return Phase4PolicyTrainer(
        encoder=encoder,
        dynamics=dynamics,
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        gamma=train_cfg.get("gamma", 0.99),
        tau=train_cfg.get("tau", 0.005),
        init_alpha=train_cfg.get("init_alpha", 0.2),
        target_entropy=train_cfg.get("target_entropy", None),
        n_imagination_steps=train_cfg.get("n_imagination_steps", 5),
        optimizer=dummy_opt,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(cfg.get("output_dir", "outputs/phase4")),
        device=device,
        max_steps=train_cfg.get("max_steps", 200_000),
        log_every=train_cfg.get("log_every", 100),
        eval_every=train_cfg.get("eval_every", 5_000),
        save_every=train_cfg.get("save_every", 10_000),
        run_name=cfg.get("run_name", "phase4_policy"),
        cfg=dict(cfg),
    )
