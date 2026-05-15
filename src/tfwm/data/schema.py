"""Data schema definitions using Pydantic."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SensorData(BaseModel):
    """Sensor data at a single timestep."""

    tactile: list[float] | None = None
    vision: list[float] | None = None
    proprio: list[float] | None = None
    contact_events: list[int] | None = None
    timestamp: float


class ActionData(BaseModel):
    """Action data at a single timestep."""

    action: list[float] = Field(..., description="Action vector")
    timestamp: float


class EpisodeMetadata(BaseModel):
    """Metadata for an episode."""

    task_name: str
    sim_name: str
    seed: int
    randomization_seed: int
    success: bool
    steps: int
    duration_s: float
    camera_queries: int
    slip_events: int
    force_violations: int


class Episode(BaseModel):
    """Complete episode data."""

    metadata: EpisodeMetadata
    observations: list[SensorData]
    actions: list[ActionData]
    rewards: list[float]
    dones: list[bool]

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def n_steps(self) -> int:
        return len(self)
