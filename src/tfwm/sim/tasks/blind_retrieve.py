"""Blind retrieval from a fabric pouch (no vision)."""

from __future__ import annotations

from typing import Any

import numpy as np

from tfwm.sim.base_env import BaseEnv


class BlindRetrieveEnv(BaseEnv):
    """Blind object retrieval from a closed pouch using tactile exploration.

    The robot must locate a target object (by shape/texture) inside a pouch
    without any vision.  Success requires grasping the correct object class.

    Args:
        n_objects: Number of objects inside the pouch.
        target_object_id: Index of the target object class.
        episode_length: Maximum steps.
        tactile_noise_std: Sensor noise standard deviation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 6,
        n_objects: int = 5,
        target_object_id: int = 0,
        episode_length: int = 300,
        tactile_noise_std: float = 0.02,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_taxels = n_taxels
        self.n_channels = n_channels
        self.d_action = d_action
        self.d_proprio = d_proprio
        self.n_objects = n_objects
        self.target_id = target_object_id
        self.episode_length = episode_length
        self.noise_std = tactile_noise_std
        self._rng = np.random.default_rng(seed)
        # Each object has a unique tactile signature (fingerprint).
        self._signatures = self._rng.random((n_objects, n_taxels * n_channels))
        self._current_contact = -1
        self._step = 0

        from gymnasium.spaces import Box, Dict
        self.observation_space = Dict({
            "tactile": Box(0.0, 1.0, shape=(n_taxels, n_channels), dtype=np.float32),
            "proprio": Box(-np.inf, np.inf, shape=(d_proprio,), dtype=np.float32),
        })
        self.action_space = Box(-1.0, 1.0, shape=(d_action,), dtype=np.float32)

    def requires_vision(self, t: int) -> bool:
        return False  # Vision always occluded in this task.

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_contact = -1
        self._step = 0
        self._signatures = self._rng.random((self.n_objects, self.n_taxels * self.n_channels))
        return self._get_obs(), {"sim_time": 0.0}

    def step(self, action: np.ndarray):
        self._step += 1
        # Action determines which object the finger contacts.
        explore = action[:2]
        obj_idx = int(np.argmin([np.linalg.norm(explore - self._rng.random(2)) for _ in range(self.n_objects)]))
        self._current_contact = obj_idx
        grasp = bool(action[2] > 0.5)  # Grasp command.
        success = grasp and (obj_idx == self.target_id)
        reward = 1.0 if success else (-0.1 if grasp else 0.0)
        terminated = grasp
        truncated = self._step >= self.episode_length
        obs = self._get_obs()
        info = {"contact_events": int(self._current_contact >= 0), "success": bool(success),
                "camera_queried": False, "sim_time": self._step * 0.01}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        if self._current_contact < 0:
            raw = np.zeros(self.n_taxels * self.n_channels)
        else:
            raw = self._signatures[self._current_contact]
        noisy = np.clip(raw + self._rng.normal(0, self.noise_std, raw.shape), 0, 1)
        tactile = noisy.reshape(self.n_taxels, self.n_channels).astype(np.float32)
        proprio = np.zeros(self.d_proprio, dtype=np.float32)
        proprio[0] = float(self._current_contact)
        return {"tactile": tactile, "proprio": proprio}
