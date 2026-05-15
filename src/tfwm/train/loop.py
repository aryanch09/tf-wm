"""Training loop for TF-WM."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tfwm.data.dataset import EpisodeDataset
from tfwm.models.affordance import AffordancePredictor
from tfwm.models.dynamics import DynamicsModel
from tfwm.models.tactile import TactileEncoder
from tfwm.utils.logging import log_metrics, setup_logging
from tfwm.utils.reproducibility import save_manifest


class TrainingLoop:
    """Training loop for TF-WM models."""

    def __init__(
        self,
        encoder: TactileEncoder,
        dynamics: DynamicsModel,
        affordance: AffordancePredictor,
        dataset: EpisodeDataset,
        output_dir: Path,
        device: torch.device | str = "cpu",
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.affordance = affordance.to(self.device)
        self.dataset = dataset
        self.output_dir = output_dir
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.affordance.parameters()),
            lr=lr,
        )
        self.criterion = nn.MSELoss()
        self.logger = setup_logging()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.encoder.train()
        self.dynamics.train()
        self.affordance.train()

        metrics = {"loss": 0.0, "steps": 0}
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            tactile = batch["tactile"].to(self.device)
            proprio = batch["proprio"].to(self.device)
            actions = batch["actions"].to(self.device)

            latent = self.encoder(tactile)
            dist = self.dynamics(latent, proprio, actions)
            pred_latent = dist["mu"]
            loss = self.criterion(pred_latent, latent.detach())
            loss.backward()
            self.optimizer.step()

            metrics["loss"] += loss.item()
            metrics["steps"] += 1

        metrics["loss"] /= max(metrics["steps"], 1)
        log_metrics(metrics, step=epoch)
        return metrics

    def evaluate(self, dataset: EpisodeDataset) -> dict[str, float]:
        self.encoder.eval()
        self.dynamics.eval()
        self.affordance.eval()

        metrics = {"eval_loss": 0.0, "steps": 0}
        loader = DataLoader(dataset, batch_size=self.dataloader.batch_size)
        with torch.no_grad():
            for batch in loader:
                tactile = batch["tactile"].to(self.device)
                proprio = batch["proprio"].to(self.device)
                actions = batch["actions"].to(self.device)
                latent = self.encoder(tactile)
                dist = self.dynamics(latent, proprio, actions)
                pred_latent = dist["mu"]
                loss = self.criterion(pred_latent, latent)
                metrics["eval_loss"] += loss.item()
                metrics["steps"] += 1

        metrics["eval_loss"] /= max(metrics["steps"], 1)
        return metrics

    def save_checkpoint(self, epoch: int) -> None:
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "affordance": self.affordance.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.output_dir / f"checkpoint_{epoch}.pt")

    def save_manifest(self, manifest: dict[str, object]) -> None:
        save_manifest(manifest, self.output_dir)
