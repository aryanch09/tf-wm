"""Tactile sensor implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from mujoco import MjData, MjModel

from tfwm.types import Tactile


@dataclass
class TactileSensorConfig:
    """Configuration for tactile sensor."""

    site_name: str
    n_taxels: tuple[int, int]
    taxel_spacing: float
    contact_threshold: float = 1e-3


class TactileSensor(ABC):
    """Abstract base class for tactile sensors."""

    def __init__(self, config: TactileSensorConfig):
        self.config = config
        self.reading = np.zeros(config.n_taxels, dtype=np.float32)

    @abstractmethod
    def update(self, model: MjModel, data: MjData) -> None:
        pass

    @abstractmethod
    def get_reading(self) -> Tactile:
        pass

    def reset(self) -> None:
        self.reading = np.zeros(self.config.n_taxels, dtype=np.float32)


class GelSightSensor(TactileSensor):
    """GelSight-style tactile sensor simulation."""

    def __init__(self, config: TactileSensorConfig):
        super().__init__(config)
        self.contact_forces = np.zeros(config.n_taxels, dtype=np.float32)

    def update(self, model: MjModel, data: MjData) -> None:
        contact_force = np.zeros(3, dtype=np.float32)
        total_force = np.linalg.norm(contact_force)
        if total_force > self.config.contact_threshold:
            self.contact_forces = np.full(
                self.config.n_taxels,
                total_force / float(np.prod(np.array(self.config.n_taxels, dtype=np.float32))),
                dtype=np.float32,
            )
        else:
            self.contact_forces = np.zeros(self.config.n_taxels, dtype=np.float32)

    def get_reading(self) -> Tactile:
        max_force = 10.0
        reading = np.clip(self.contact_forces / max_force, 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(reading)


class BioTacSensor(TactileSensor):
    """BioTac-style tactile sensor simulation."""

    def __init__(self, config: TactileSensorConfig):
        super().__init__(config)
        self.impedance = np.ones(config.n_taxels, dtype=np.float32) * 0.5

    def update(self, model: MjModel, data: MjData) -> None:
        has_contact = True
        if has_contact:
            self.impedance = np.ones(self.config.n_taxels, dtype=np.float32) * 0.3
        else:
            self.impedance = np.ones(self.config.n_taxels, dtype=np.float32) * 0.5

    def get_reading(self) -> Tactile:
        return torch.from_numpy(self.impedance)
