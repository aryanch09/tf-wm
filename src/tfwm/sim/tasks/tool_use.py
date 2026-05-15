"""Tool-use contact task — MuJoCo physics backend.

A rigid probe (mocap body, kinematically controlled via 6-DoF Cartesian
velocity) must make sustained contact with a fixed target object.  The target
sphere is fixed to the world (no joint), but its position is re-randomised at
every episode reset by writing to ``model.body_pos``.

16 touch sensors on the probe shaft give spatially-resolved force feedback:
4 rows × 4 circumferential sites.  The characteristic contact pattern
(which row and sector light up) encodes the approach direction.

Task success: ``max(taxel_readings) > contact_threshold`` for
``success_hold_steps`` consecutive steps.

Hypothesis H-A: TF-WM achieves ≥15 % absolute success improvement over
vision-first baselines under 50 % occlusion.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from tfwm.sim.mujoco_env import MuJoCoBaseEnv

# ---------------------------------------------------------------------------
# MuJoCo XML — probe tool (mocap) + fixed target sphere + 16 touch sensors
# ---------------------------------------------------------------------------
#
# Probe: cylinder, radius 8 mm, half-length 40 mm, tip points in -z.
# Target: sphere, radius 20 mm, fixed to world (no joint).
#   Starting position randomised in reset() via model.body_pos.
#
# Touch sensor layout — 4 rows × 4 sites:
#   Row 0: z = -0.035  (nearest to tip)
#   Row 1: z = -0.025
#   Row 2: z = -0.015
#   Row 3: z = -0.005  (furthest from tip)
#   Circumferential: 0°, 90°, 180°, 270° at r = 8 mm
# ---------------------------------------------------------------------------

_XML = """
<mujoco model="tool_use">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="1 1 0.1" rgba="0.75 0.75 0.75 1" condim="1"/>
    <!-- Target object: sphere fixed to world (no joint) -->
    <body name="target" pos="0.07 0 0.02">
      <geom type="sphere" size="0.020" rgba="0.9 0.3 0.2 1" condim="1"/>
    </body>
    <!-- Probe tool (mocap — Cartesian velocity control) -->
    <body name="probe" pos="0 0 0.15" mocap="true">
      <geom type="cylinder" size="0.008 0.040" rgba="0.3 0.7 0.35 1"
            condim="4" friction="0.5 0.005 0.0001"/>
      <!-- Row 0: z = -0.035 (near tip) -->
      <site name="t0"  pos=" 0.008  0      -0.035" size="0.003"/>
      <site name="t1"  pos=" 0      0.008  -0.035" size="0.003"/>
      <site name="t2"  pos="-0.008  0      -0.035" size="0.003"/>
      <site name="t3"  pos=" 0     -0.008  -0.035" size="0.003"/>
      <!-- Row 1: z = -0.025 -->
      <site name="t4"  pos=" 0.008  0      -0.025" size="0.003"/>
      <site name="t5"  pos=" 0      0.008  -0.025" size="0.003"/>
      <site name="t6"  pos="-0.008  0      -0.025" size="0.003"/>
      <site name="t7"  pos=" 0     -0.008  -0.025" size="0.003"/>
      <!-- Row 2: z = -0.015 -->
      <site name="t8"  pos=" 0.008  0      -0.015" size="0.003"/>
      <site name="t9"  pos=" 0      0.008  -0.015" size="0.003"/>
      <site name="t10" pos="-0.008  0      -0.015" size="0.003"/>
      <site name="t11" pos=" 0     -0.008  -0.015" size="0.003"/>
      <!-- Row 3: z = -0.005 -->
      <site name="t12" pos=" 0.008  0      -0.005" size="0.003"/>
      <site name="t13" pos=" 0      0.008  -0.005" size="0.003"/>
      <site name="t14" pos="-0.008  0      -0.005" size="0.003"/>
      <site name="t15" pos=" 0     -0.008  -0.005" size="0.003"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="fs_t0"  site="t0"  cutoff="0.015"/>
    <touch name="fs_t1"  site="t1"  cutoff="0.015"/>
    <touch name="fs_t2"  site="t2"  cutoff="0.015"/>
    <touch name="fs_t3"  site="t3"  cutoff="0.015"/>
    <touch name="fs_t4"  site="t4"  cutoff="0.015"/>
    <touch name="fs_t5"  site="t5"  cutoff="0.015"/>
    <touch name="fs_t6"  site="t6"  cutoff="0.015"/>
    <touch name="fs_t7"  site="t7"  cutoff="0.015"/>
    <touch name="fs_t8"  site="t8"  cutoff="0.015"/>
    <touch name="fs_t9"  site="t9"  cutoff="0.015"/>
    <touch name="fs_t10" site="t10" cutoff="0.015"/>
    <touch name="fs_t11" site="t11" cutoff="0.015"/>
    <touch name="fs_t12" site="t12" cutoff="0.015"/>
    <touch name="fs_t13" site="t13" cutoff="0.015"/>
    <touch name="fs_t14" site="t14" cutoff="0.015"/>
    <touch name="fs_t15" site="t15" cutoff="0.015"/>
  </sensor>
</mujoco>
"""

_TOUCH_NAMES = [f"fs_t{i}" for i in range(16)]

_PROBE_HALF_LENGTH = 0.040  # metres from body centre to tip
_TARGET_RADIUS     = 0.020  # metres
_WORKSPACE         = 0.20   # half-side of cubic workspace


class ToolUseEnv(MuJoCoBaseEnv):
    """MuJoCo tool-use: probe a fixed target sphere with the tool tip.

    Args:
        success_dist_m: Not used — success is determined by touch sensors
            exceeding ``contact_threshold`` (physically grounded).
        success_hold_steps: Steps above contact threshold for success.
        contact_threshold: Normalised taxel reading (0-1) for contact.
        occlusion_prob: Vision occlusion probability per episode.
        tactile_noise_std: Additive noise on normalised readings.
        workspace_radius: Radius of the valid target placement region.
        seed: RNG seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 6,
        episode_length: int = 300,
        success_dist_m: float = 0.01,   # kept for API compat
        success_hold_steps: int = 5,
        contact_threshold: float = 0.05,
        occlusion_prob: float = 0.5,
        tactile_noise_std: float = 0.015,
        workspace_radius: float = 0.12,
        seed: int = 0,
    ) -> None:
        if n_taxels != 16:
            raise ValueError("ToolUseEnv requires n_taxels=16")
        super().__init__(
            xml=_XML,
            touch_sensor_names=_TOUCH_NAMES,
            mocap_body_name="probe",
            n_taxels=16,
            n_channels=n_channels,
            d_proprio=d_proprio,
            d_action=d_action,
            episode_length=episode_length,
            occlusion_prob=occlusion_prob,
            tactile_noise_std=tactile_noise_std,
            max_force=3.0,
            seed=seed,
        )
        self.success_hold_steps = success_hold_steps
        self.contact_threshold = contact_threshold
        self._workspace = workspace_radius
        self._hold_count = 0
        self._target_pos = np.array([0.07, 0.0, 0.02])

        # Cache target body id for position re-randomisation
        self._target_bid = self._body_id("target")

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        # Randomise target position in a disk in the z=target_z plane
        for _ in range(30):
            angle = self._rng.uniform(0, 2 * np.pi)
            radius = self._rng.uniform(0.04, self._workspace)
            cand = np.array([radius * np.cos(angle), radius * np.sin(angle), _TARGET_RADIUS])
            if np.linalg.norm(cand[:2]) > 0.03:
                self._target_pos = cand
                break
        self._model.body_pos[self._target_bid] = self._target_pos

        # Probe starts centred above the target with a small random offset
        offset = self._rng.uniform(-0.03, 0.03, 2)
        probe_start = np.array([
            self._target_pos[0] + offset[0],
            self._target_pos[1] + offset[1],
            0.15,
        ])
        self._data.mocap_pos[self._mocap_idx] = probe_start
        self._data.mocap_quat[self._mocap_idx] = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self._model, self._data)

        self._hold_count = 0
        self._step_count = 0
        self._vision_occluded = self._rng.random() < self.occlusion_prob

        dist = float(np.linalg.norm(probe_start[:2] - self._target_pos[:2]))
        return self._get_obs(), {
            "target_pos": self._target_pos.tolist(),
            "dist_to_target": dist,
        }

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._apply_action_mocap(action, pos_scale=0.005, rot_scale=0.02, workspace=0.25)
        self._step_physics()

        obs = self._get_obs()

        # Success determined by physics: contact force from touch sensors
        in_contact = bool(np.max(obs["tactile"]) > self.contact_threshold)
        self._hold_count = self._hold_count + 1 if in_contact else 0
        episode_success = self._hold_count >= self.success_hold_steps

        # Reward: approach reward + contact bonus + success bonus
        probe_pos = self._data.mocap_pos[self._mocap_idx].copy()
        tip_pos = probe_pos.copy()
        tip_pos[2] -= _PROBE_HALF_LENGTH
        dist = float(np.linalg.norm(tip_pos - self._target_pos))

        reward = -dist + 0.5 * float(in_contact) + 10.0 * float(episode_success) - 0.005

        dt = self._n_substeps * float(self._model.opt.timestep)
        info = {
            "dist_to_target": dist,
            "in_contact": in_contact,
            "contact_force": float(np.max(obs["tactile"]) * self._max_force),
            "success": bool(episode_success),
            "camera_queried": self.requires_vision(self._step_count),
            "slip_flag": False,
            "sim_time": self._step_count * dt,
        }
        return obs, float(reward), episode_success, self._step_count >= self.episode_length, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        probe_pos = self._data.mocap_pos[self._mocap_idx].astype(np.float32)
        tip_pos = probe_pos.copy()
        tip_pos[2] -= _PROBE_HALF_LENGTH
        proprio = np.zeros(self.d_proprio, dtype=np.float32)
        proprio[:3] = probe_pos
        proprio[3:6] = self._target_pos.astype(np.float32)
        proprio[6:9] = (tip_pos - self._target_pos.astype(np.float32))
        return {"tactile": self._read_taxels(), "proprio": proprio}
