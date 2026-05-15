"""Abstract MuJoCo-backed gymnasium environment for TF-WM tasks.

Provides a thin, reusable base that wires MuJoCo 3.x physics to the
Gymnasium API used by the rest of the codebase.

Key features:
  - Touch sensor readout via ``<sensor type="touch">`` elements in the XML
    (spatially-resolved via per-site cutoff radius)
  - Mocap body control for Cartesian 6-DoF tool manipulation
  - Joint actuator control for finger-based dexterous manipulation
  - Deterministic seeding and optional vision occlusion
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import mujoco
import numpy as np

from tfwm.sim.base_env import BaseEnv


# ---------------------------------------------------------------------------
# Quaternion helpers — MuJoCo convention [w, x, y, z]
# ---------------------------------------------------------------------------

def _random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion [w, x, y, z] via Shoemake sampling."""
    u1, u2, u3 = rng.random(3)
    x = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    y = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion product in [w, x, y, z] convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _quat_err_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle in degrees between two unit quaternions [w,x,y,z]."""
    dot = float(np.abs(np.dot(q1, q2)))
    return math.degrees(2 * math.acos(min(dot, 1.0)))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MuJoCoBaseEnv(BaseEnv):
    """Abstract MuJoCo-backed task environment with touch-sensor tactile.

    Subclasses provide the MuJoCo XML string and implement :meth:`reset` and
    :meth:`step`.  Tactile readings come from ``<sensor type="touch">``
    elements in the XML (one per taxel), spatially resolved via a per-site
    cutoff radius.

    Args:
        xml: Full MuJoCo XML model string.
        touch_sensor_names: Ordered list of touch sensor names in the XML —
            must have exactly ``n_taxels`` entries.
        mocap_body_name: Name of the mocap body (for Cartesian control tasks).
            ``None`` if not needed.
        n_taxels: Number of tactile taxels.
        n_channels: Channels per taxel (1 = scalar normal force).
        d_proprio: Proprioception vector dimension.
        d_action: Action dimension.
        episode_length: Max steps before truncation.
        occlusion_prob: Probability that vision is occluded at each episode.
        tactile_noise_std: Additive Gaussian noise on normalised taxel readings.
        max_force: Newtons corresponding to a saturated (1.0) taxel reading.
        n_substeps: Physics steps per control step.
        seed: RNG seed.
    """

    def __init__(
        self,
        xml: str,
        touch_sensor_names: list[str],
        mocap_body_name: str | None = None,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 6,
        episode_length: int = 200,
        occlusion_prob: float = 0.0,
        tactile_noise_std: float = 0.01,
        max_force: float = 5.0,
        n_substeps: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__()

        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)

        self.n_taxels = n_taxels
        self.n_channels = n_channels
        self.d_proprio = d_proprio
        self.d_action = d_action
        self.episode_length = episode_length
        self.occlusion_prob = occlusion_prob
        self.noise_std = tactile_noise_std
        self._max_force = max_force
        self._n_substeps = n_substeps
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._vision_occluded = False

        if len(touch_sensor_names) != n_taxels:
            raise ValueError(
                f"touch_sensor_names has {len(touch_sensor_names)} entries but n_taxels={n_taxels}"
            )
        self._touch_adrs: list[int] = []
        for name in touch_sensor_names:
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"Touch sensor '{name}' not found in XML")
            self._touch_adrs.append(int(self._model.sensor_adr[sid]))

        self._mocap_idx: int = -1
        if mocap_body_name is not None:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
            if bid < 0:
                raise ValueError(f"Body '{mocap_body_name}' not found in XML")
            self._mocap_idx = int(self._model.body_mocapid[bid])

        from gymnasium.spaces import Box, Dict
        self.observation_space = Dict({
            "tactile": Box(0.0, 1.0, shape=(n_taxels, n_channels), dtype=np.float32),
            "proprio": Box(-np.inf, np.inf, shape=(d_proprio,), dtype=np.float32),
        })
        self.action_space = Box(-1.0, 1.0, shape=(d_action,), dtype=np.float32)

    # ------------------------------------------------------------------
    # BaseEnv protocol
    # ------------------------------------------------------------------

    def requires_vision(self, t: int) -> bool:
        return not self._vision_occluded

    @abstractmethod
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        ...

    @abstractmethod
    def step(self, action: np.ndarray):
        ...

    # ------------------------------------------------------------------
    # Tactile readout
    # ------------------------------------------------------------------

    def _read_taxels(self) -> np.ndarray:
        """Read touch sensor sensordata and return normalised (B,C) array."""
        raw = np.array(
            [self._data.sensordata[a] for a in self._touch_adrs],
            dtype=np.float32,
        )
        out = np.clip(raw / self._max_force, 0.0, 1.0)
        if self.noise_std > 0:
            out = np.clip(
                out + self._rng.normal(0.0, self.noise_std, out.shape).astype(np.float32),
                0.0, 1.0,
            )
        return out.reshape(self.n_taxels, self.n_channels)

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action_mocap(
        self,
        action: np.ndarray,
        pos_scale: float = 0.01,
        rot_scale: float = 0.05,
        workspace: float = 0.35,
    ) -> None:
        """Integrate Cartesian velocity action into the mocap target pose.

        Args:
            action: Shape ``(d_action,)`` in ``[-1, 1]``.  First 3 dims are
                translational velocity, dims 3-5 are angular velocity.
            pos_scale: Metres per unit action per control step.
            rot_scale: Radians per unit action per control step.
            workspace: Half-side of the cubic workspace in metres.
        """
        pos = self._data.mocap_pos[self._mocap_idx].copy()
        pos += np.clip(action[:3], -1.0, 1.0).astype(np.float64) * pos_scale
        self._data.mocap_pos[self._mocap_idx] = np.clip(pos, -workspace, workspace)

        if len(action) >= 6:
            ang = np.clip(action[3:6], -1.0, 1.0).astype(np.float64) * rot_scale
            angle = float(np.linalg.norm(ang))
            if angle > 1e-8:
                axis = ang / angle
                ha = angle / 2.0
                dq = np.array([np.cos(ha), *(np.sin(ha) * axis)])
                q = self._data.mocap_quat[self._mocap_idx]
                new_q = _quat_mul(q, dq)
                self._data.mocap_quat[self._mocap_idx] = new_q / np.linalg.norm(new_q)

    def _apply_action_joints(self, action: np.ndarray) -> None:
        """Map action in ``[-1, 1]`` to actuator control range.

        Excess action dimensions (beyond model.nu) are silently ignored.
        """
        n_act = int(self._model.nu)
        for i in range(min(len(action), n_act)):
            lo = float(self._model.actuator_ctrlrange[i, 0])
            hi = float(self._model.actuator_ctrlrange[i, 1])
            self._data.ctrl[i] = 0.5 * (lo + hi) + 0.5 * float(action[i]) * (hi - lo)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    def _step_physics(self) -> None:
        """Advance the physics simulation by ``n_substeps`` steps."""
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

    def _joint_qpos_adr(self, joint_name: str) -> int:
        """Return the qpos starting index for a named joint."""
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint '{joint_name}' not found in XML")
        return int(self._model.jnt_qposadr[jid])

    def _body_id(self, body_name: str) -> int:
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found in XML")
        return bid
