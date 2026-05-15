"""Exponential Moving Average (EMA) of model weights."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMACallback:
    """Maintain an EMA shadow copy of model parameters.

    The shadow weights are not trained directly; they track a running average
    of the online model and are used at evaluation time for stable predictions.

    Args:
        model: The online model to track.
        decay: EMA decay rate (higher = slower tracking, e.g. 0.9999).

    Usage::

        ema = EMACallback(model, decay=0.9999)
        for batch in loader:
            loss = model(batch)
            loss.backward()
            opt.step()
            ema.update()

        with ema.average_parameters():
            val_loss = model(val_batch)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        """Update shadow weights after each optimiser step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self) -> None:
        """Copy shadow weights into the model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore the online weights after evaluation."""
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

    def average_parameters(self):  # type: ignore[return]
        """Context manager: temporarily apply EMA weights."""
        class _Ctx:
            def __enter__(inner_self) -> None:
                self.apply_shadow()
            def __exit__(inner_self, *_: object) -> None:
                self.restore()
        return _Ctx()
