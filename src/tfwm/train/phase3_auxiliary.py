"""Phase 3: Auxiliary decoder supervision.

Fine-tunes the latent representation with additional self-supervised heads:
  - AffordanceHead (discrete affordance classification)
  - ObjectStateHead (continuous object state regression)

The encoder and RSSM are optionally frozen; only the decoder heads (and
optionally the top encoder layers) receive gradients.

Run via::

    tfwm train phase=phase3 task=peg_in_hole
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
from tfwm.losses.affordance import AffordanceLoss
from tfwm.models.decoders.affordance_head import AffordanceHead
from tfwm.models.decoders.object_state_head import ObjectStateHead
from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.train.trainer_base import TrainerBase


class Phase3AuxiliaryTrainer(TrainerBase):
    """Phase 3: train auxiliary prediction heads on top of a frozen world model.

    Args:
        encoder: Tactile encoder (frozen or fine-tuned).
        dynamics: RSSM dynamics (frozen or fine-tuned).
        affordance_head: Affordance classifier.
        object_state_head: Continuous object-state regressor.
        affordance_loss: :class:`~tfwm.losses.affordance.AffordanceLoss`.
        alpha_affordance: Weight for affordance loss.
        alpha_obj_state: Weight for object-state MSE.
        freeze_encoder: If ``True``, encoder weights are frozen.
        freeze_dynamics: If ``True``, dynamics weights are frozen.
        **kwargs: Forwarded to :class:`~tfwm.train.trainer_base.TrainerBase`.
    """

    def __init__(
        self,
        encoder: HybridTactileEncoder,
        dynamics: RSSMDynamics,
        affordance_head: AffordanceHead,
        object_state_head: ObjectStateHead,
        affordance_loss: AffordanceLoss,
        alpha_affordance: float = 1.0,
        alpha_obj_state: float = 1.0,
        freeze_encoder: bool = True,
        freeze_dynamics: bool = True,
        **kwargs: Any,
    ) -> None:
        modules: dict[str, nn.Module] = {
            "affordance_head": affordance_head,
            "object_state_head": object_state_head,
        }
        if not freeze_encoder:
            modules["encoder"] = encoder
        if not freeze_dynamics:
            modules["dynamics"] = dynamics

        super().__init__(model=modules, **kwargs)  # type: ignore[arg-type]

        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.affordance_head = affordance_head
        self.object_state_head = object_state_head
        self.affordance_loss = affordance_loss
        self.alpha_affordance = alpha_affordance
        self.alpha_obj_state = alpha_obj_state

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()
        if freeze_dynamics:
            for p in self.dynamics.parameters():
                p.requires_grad_(False)
            self.dynamics.eval()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        """Compute auxiliary losses and update heads.

        Args:
            batch: Dict with ``tactile (B,T,N,C)``, ``actions (B,T,d_a)``,
                and optionally ``affordance_labels (B,T)`` and
                ``object_states (B,T,d_state)``.
            step: Current global step.

        Returns:
            Metrics dict.
        """
        tactile = batch["tactile"]
        actions = batch["actions"]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            enc_out = self.encoder(tactile)
            obs_latents = enc_out["mu"]
            rssm_out = self.dynamics.forward_teacher(obs_latents, actions)

        z_post = rssm_out["z_posterior"]   # (B, T, d_z)

        total_loss = torch.zeros(1, device=self.device)
        metrics: dict[str, float] = {}

        # ── Affordance loss ────────────────────────────────────────────
        aff_logits = self.affordance_head(z_post)   # (B, T, K)
        aff_labels = batch.get("affordance_labels")
        if aff_labels is not None:
            loss_aff = self.affordance_loss(aff_logits, aff_labels)
            total_loss = total_loss + self.alpha_affordance * loss_aff
            metrics["loss_affordance"] = float(loss_aff)
        else:
            # Self-supervised: predict cluster assignments via pseudo-labels.
            B, T, K = aff_logits.shape
            pseudo = aff_logits.detach().argmax(-1)   # (B, T)
            loss_aff = F.cross_entropy(aff_logits.view(B * T, K), pseudo.view(B * T))
            total_loss = total_loss + self.alpha_affordance * loss_aff
            metrics["loss_affordance"] = float(loss_aff)

        # ── Object-state MSE ──────────────────────────────────────────
        obj_pred = self.object_state_head(z_post)   # (B, T, d_state)
        obj_gt = batch.get("object_states")
        if obj_gt is not None:
            loss_obj = F.mse_loss(obj_pred, obj_gt)
            total_loss = total_loss + self.alpha_obj_state * loss_obj
            metrics["loss_obj_state"] = float(loss_obj)

        metrics["loss_total"] = float(total_loss)
        total_loss.backward()
        self._clip_and_step()
        return metrics

    def validation_step(
        self, batch: dict[str, torch.Tensor], step: int
    ) -> dict[str, float]:
        tactile = batch["tactile"]
        actions = batch["actions"]
        with torch.no_grad():
            enc_out = self.encoder(tactile)
            rssm_out = self.dynamics.forward_teacher(enc_out["mu"], actions)
            z_post = rssm_out["z_posterior"]
            obj_pred = self.object_state_head(z_post)
        obj_gt = batch.get("object_states")
        if obj_gt is not None:
            return {"val_obj_mse": float(F.mse_loss(obj_pred, obj_gt))}
        return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_phase3_trainer(cfg: DictConfig, phase1_ckpt: Path | None = None) -> Phase3AuxiliaryTrainer:
    """Construct a :class:`Phase3AuxiliaryTrainer` from a Hydra config.

    Args:
        cfg: Resolved top-level Hydra config.
        phase1_ckpt: Optional path to a Phase 1 checkpoint.

    Returns:
        Configured :class:`Phase3AuxiliaryTrainer`.
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

    n_affordances = train_cfg.get("n_affordances", 8)
    affordance_head = AffordanceHead(d_latent=d_latent, n_affordances=n_affordances)
    object_state_head = ObjectStateHead(d_latent=d_latent, d_state=train_cfg.get("d_state", 7))
    affordance_loss = AffordanceLoss(n_affordances=n_affordances)

    trainable_params = (
        list(affordance_head.parameters())
        + list(object_state_head.parameters())
    )
    optimizer = optim.AdamW(trainable_params, lr=train_cfg.get("learning_rate", 1e-3))

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

    return Phase3AuxiliaryTrainer(
        encoder=encoder,
        dynamics=dynamics,
        affordance_head=affordance_head,
        object_state_head=object_state_head,
        affordance_loss=affordance_loss,
        alpha_affordance=train_cfg.get("alpha_affordance", 1.0),
        alpha_obj_state=train_cfg.get("alpha_obj_state", 1.0),
        freeze_encoder=train_cfg.get("freeze_encoder", True),
        freeze_dynamics=train_cfg.get("freeze_dynamics", True),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(cfg.get("output_dir", "outputs/phase3")),
        device=device,
        max_steps=train_cfg.get("max_steps", 30_000),
        log_every=train_cfg.get("log_every", 100),
        eval_every=train_cfg.get("eval_every", 2_000),
        save_every=train_cfg.get("save_every", 5_000),
        run_name=cfg.get("run_name", "phase3_auxiliary"),
        cfg=dict(cfg),
    )
