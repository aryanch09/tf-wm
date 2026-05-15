"""MuJoCo simulation environment with tactile sensors."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from mujoco import MjData, MjModel

from tfwm.sim.sensors import BioTacSensor, GelSightSensor, TactileSensorConfig
from tfwm.types import Action, Proprio, Tactile, Vision


class TFWMSimEnv:
    """MuJoCo simulation environment for tactile-first world models."""

    def __init__(
        self,
        model_path: str,
        task_config: dict,
        sensor_configs: list[dict],
    ):
        """Initialize simulation environment."""
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.task_config = task_config
        self.dt = self.model.opt.timestep
        self.tactile_sensors = [self._build_sensor(cfg) for cfg in sensor_configs]

    def _build_sensor(self, config: dict) -> object:
        sensor_type = config.get("type", "gelsight")
        sensor_cfg = TactileSensorConfig(
            site_name=config["site_name"],
            n_taxels=tuple(config.get("n_taxels", (8, 8))),
            taxel_spacing=float(config.get("taxel_spacing", 0.001)),
            contact_threshold=float(config.get("contact_threshold", 1e-3)),
        )

        if sensor_type == "biotac":
            return BioTacSensor(sensor_cfg)
        return GelSightSensor(sensor_cfg)

    def reset(self) -> tuple[Vision, Proprio, Tactile]:
        """Reset environment to initial state."""
        self.data = MjData(self.model)
        for sensor in self.tactile_sensors:
            sensor.reset()
        return self._get_obs()

    def step(self, action: Action) -> tuple[Vision, Proprio, Tactile, float, bool]:
        """Step environment forward."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.data.ctrl[:] = np.asarray(action, dtype=np.float64).reshape(self.data.ctrl.shape)
        mujoco.mj_step(self.model, self.data)

        for sensor in self.tactile_sensors:
            sensor.update(self.model, self.data)

        vision, proprio, tactile = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()
        return vision, proprio, tactile, reward, done

    def _get_obs(self) -> tuple[Vision, Proprio, Tactile]:
        vision = torch.zeros((3, 64, 64), dtype=torch.float32)
        proprio = torch.from_numpy(
            np.concatenate([self.data.qpos, self.data.qvel], axis=0).astype(np.float32)
        )
        tactile = torch.stack([sensor.get_reading() for sensor in self.tactile_sensors])
        return vision, proprio, tactile

    def _compute_reward(self) -> float:
        return 0.0

    def _is_done(self) -> bool:
        return False
