"""Base trainer class for all TF-WM training phases.

Provides checkpoint save/load, gradient clipping, wandb / rich logging, and
the reproducibility manifest.  Each phase sub-classes this and overrides
:meth:`training_step` and :meth:`validation_step`.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

from tfwm.utils.logging import setup_logging
from tfwm.utils.reproducibility import (
    generate_manifest,
    save_manifest,
    update_manifest_end_time,
)


class TrainerBase:
    """Abstract base for TF-WM training phases.

    Sub-classes must implement :meth:`training_step` and optionally
    :meth:`validation_step`.  Call :meth:`fit` to run the full loop.

    Args:
        model: The ``nn.Module`` (or module collection) being trained.
        optimizer: Pre-constructed optimiser.
        train_loader: Training data loader.
        val_loader: Optional validation data loader.
        output_dir: Directory for checkpoints and manifests.
        device: Compute device.
        max_steps: Maximum training step budget.
        log_every: Log to wandb / console every N steps.
        eval_every: Run validation every N steps.
        save_every: Save checkpoint every N steps.
        grad_clip: Max gradient norm; ``0.0`` disables clipping.
        precision: ``"fp32"``, ``"bf16"``, or ``"fp16"``.
        run_name: W&B run name.
        cfg: Resolved Hydra config (logged to W&B and manifest).
    """

    def __init__(
        self,
        model: nn.Module | dict[str, nn.Module],
        optimizer: optim.Optimizer,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        output_dir: Path = Path("outputs/run"),
        device: torch.device | str = "cpu",
        max_steps: int = 100_000,
        log_every: int = 100,
        eval_every: int = 5_000,
        save_every: int = 10_000,
        grad_clip: float = 10.0,
        precision: str = "fp32",
        run_name: str = "tfwm_run",
        cfg: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.grad_clip = grad_clip
        self.run_name = run_name
        self.cfg = cfg or {}
        self.global_step = 0

        # Model setup — accept either a single module or a dict of modules.
        if isinstance(model, dict):
            self.modules: dict[str, nn.Module] = {k: v.to(self.device) for k, v in model.items()}
            self.model: nn.Module | None = None
        else:
            self.model = model.to(self.device)
            self.modules = {}

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = setup_logging()

        # AMP scalar.
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.amp_dtype = dtype_map.get(precision, torch.float32)
        self.scaler: torch.cuda.amp.GradScaler | None = (
            torch.cuda.amp.GradScaler() if precision == "fp16" else None
        )

        self._manifest = generate_manifest(run_name, self.cfg, self.output_dir, seed=0)
        save_manifest(self._manifest, self.output_dir)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def training_step(self, batch: dict[str, torch.Tensor], step: int) -> dict[str, float]:
        """Compute loss, backprop, step optimiser. Return metrics dict."""

    def validation_step(self, batch: dict[str, torch.Tensor], step: int) -> dict[str, float]:
        """Optional validation logic. Override to activate."""
        return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _set_train(self) -> None:
        if self.model is not None:
            self.model.train()
        for m in self.modules.values():
            m.train()

    def _set_eval(self) -> None:
        if self.model is not None:
            self.model.eval()
        for m in self.modules.values():
            m.eval()

    def _clip_and_step(self) -> None:
        params = (
            list(self.model.parameters()) if self.model is not None
            else [p for m in self.modules.values() for p in m.parameters()]
        )
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def _log(self, metrics: dict[str, float], prefix: str = "train") -> None:
        prefixed = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if _WANDB and wandb.run is not None:
            wandb.log(prefixed, step=self.global_step)

    def save_checkpoint(self, tag: str = "") -> None:
        """Serialise model weights and optimiser state."""
        suffix = f"_{tag}" if tag else f"_step{self.global_step}"
        path = self.output_dir / f"checkpoint{suffix}.pt"
        state: dict[str, Any] = {"global_step": self.global_step, "optimizer": self.optimizer.state_dict()}
        if self.model is not None:
            state["model"] = self.model.state_dict()
        else:
            state["modules"] = {k: v.state_dict() for k, v in self.modules.items()}
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        """Restore weights and optimiser from checkpoint."""
        state = torch.load(path, map_location=self.device)
        self.global_step = state.get("global_step", 0)
        self.optimizer.load_state_dict(state["optimizer"])
        if "model" in state and self.model is not None:
            self.model.load_state_dict(state["model"])
        elif "modules" in state:
            for k, sd in state["modules"].items():
                if k in self.modules:
                    self.modules[k].load_state_dict(sd)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, float]:
        """Run the training loop until ``max_steps`` is reached.

        Returns:
            Final validation metrics (empty dict if no val loader).
        """
        t0 = time.perf_counter()
        train_iter = iter(self.train_loader)
        final_val: dict[str, float] = {}

        while self.global_step < self.max_steps:
            # ── fetch batch (reshuffle on exhaustion) ──────────────────
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            batch = self._to_device(batch)
            self._set_train()

            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                metrics = self.training_step(batch, self.global_step)

            self.global_step += 1

            # ── logging ────────────────────────────────────────────────
            if self.global_step % self.log_every == 0:
                elapsed = time.perf_counter() - t0
                metrics["steps_per_sec"] = self.log_every / max(elapsed, 1e-6)
                self._log(metrics, prefix="train")
                t0 = time.perf_counter()

            # ── validation ─────────────────────────────────────────────
            if self.val_loader is not None and self.global_step % self.eval_every == 0:
                self._set_eval()
                val_metrics: dict[str, float] = {}
                with torch.no_grad():
                    for val_batch in self.val_loader:
                        val_batch = self._to_device(val_batch)
                        step_metrics = self.validation_step(val_batch, self.global_step)
                        for k, v in step_metrics.items():
                            val_metrics[k] = val_metrics.get(k, 0.0) + v
                n = len(self.val_loader)
                val_metrics = {k: v / max(n, 1) for k, v in val_metrics.items()}
                self._log(val_metrics, prefix="val")
                final_val = val_metrics

            # ── checkpoint ─────────────────────────────────────────────
            if self.global_step % self.save_every == 0:
                self.save_checkpoint()

        self.save_checkpoint(tag="final")
        update_manifest_end_time(self.output_dir)
        return final_val
