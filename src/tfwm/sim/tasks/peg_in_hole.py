"""Peg-in-hole insertion task — MuJoCo physics backend.

The peg is a kinematic (mocap) body controlled via 6-DoF Cartesian velocity.
The hole fixture is a static structure with four box walls forming a square
aperture (±10 mm in both lateral axes).  Real contact forces propagate from
the peg geom into the 16 touch sensors arranged in two circumferential rings.

Spatially-resolved contact feedback distinguishes the direction of lateral
misalignment (which sensor ring lights up), making this a demanding testbed
for the tactile-first world model.

Hypothesis H-A: TF-WM achieves ≥15 % absolute success gain over vision-first
baselines under 80 % occlusion.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from tfwm.sim.mujoco_env import MuJoCoBaseEnv, _random_quat

# ---------------------------------------------------------------------------
# MuJoCo XML — cylindrical peg (mocap) + square-hole fixture + 16 sensors
# ---------------------------------------------------------------------------
#
# Coordinate convention:
#   +z = up, gravity = -z
#   Hole top surface at z = 0; hole depth = 60 mm → base plate at z = -0.060
#   Peg body centre starts at z = 0.10; tip at z = 0.06 (6 cm above hole)
#
# Touch sensor layout (2 rings of 8):
#   Ring A: z = -0.033 relative to peg centre (near tip)
#   Ring B: z = -0.013 relative to peg centre (upper shaft)
#   8 sites per ring at angles 0, 45, 90, 135, 180, 225, 270, 315 degrees
# ---------------------------------------------------------------------------

_r = 0.009       # peg radius in metres
_COS45 = 0.7071  # cos(45°) = sin(45°)

_XML = f"""
<mujoco model="peg_in_hole">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <!-- Hole fixture (fixed to world — no joint) -->
    <body name="fixture" pos="0 0 0">
      <!-- Base plate below hole -->
      <geom type="box" size="0.045 0.045 0.005" pos="0 0 -0.065"
            rgba="0.5 0.5 0.5 1" condim="1"/>
      <!-- 4 walls forming the square aperture (10 mm inner half-side) -->
      <geom type="box" size="0.005 0.025 0.030" pos=" 0.015 0 -0.030"
            rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.005 0.025 0.030" pos="-0.015 0 -0.030"
            rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.010 0.005 0.030" pos="0  0.015 -0.030"
            rgba="0.5 0.5 0.5 1"/>
      <geom type="box" size="0.010 0.005 0.030" pos="0 -0.015 -0.030"
            rgba="0.5 0.5 0.5 1"/>
    </body>
    <!-- Peg (mocap-controlled, tip points in -z direction) -->
    <body name="peg" pos="0 0 0.10" mocap="true">
      <geom type="cylinder" size="{_r} 0.04" rgba="0.2 0.55 0.9 1"
            condim="4" friction="0.5 0.005 0.0001"/>
      <!-- Ring A — near tip (z = -0.033 from peg centre) -->
      <site name="t0"  pos=" {_r}    0        -0.033" size="0.003"/>
      <site name="t1"  pos=" {_r*_COS45:.4f}  {_r*_COS45:.4f}  -0.033" size="0.003"/>
      <site name="t2"  pos=" 0        {_r}    -0.033" size="0.003"/>
      <site name="t3"  pos="-{_r*_COS45:.4f}  {_r*_COS45:.4f}  -0.033" size="0.003"/>
      <site name="t4"  pos="-{_r}    0        -0.033" size="0.003"/>
      <site name="t5"  pos="-{_r*_COS45:.4f} -{_r*_COS45:.4f}  -0.033" size="0.003"/>
      <site name="t6"  pos=" 0       -{_r}    -0.033" size="0.003"/>
      <site name="t7"  pos=" {_r*_COS45:.4f} -{_r*_COS45:.4f}  -0.033" size="0.003"/>
      <!-- Ring B — upper shaft (z = -0.013 from peg centre) -->
      <site name="t8"  pos=" {_r}    0        -0.013" size="0.003"/>
      <site name="t9"  pos=" {_r*_COS45:.4f}  {_r*_COS45:.4f}  -0.013" size="0.003"/>
      <site name="t10" pos=" 0        {_r}    -0.013" size="0.003"/>
      <site name="t11" pos="-{_r*_COS45:.4f}  {_r*_COS45:.4f}  -0.013" size="0.003"/>
      <site name="t12" pos="-{_r}    0        -0.013" size="0.003"/>
      <site name="t13" pos="-{_r*_COS45:.4f} -{_r*_COS45:.4f}  -0.013" size="0.003"/>
      <site name="t14" pos=" 0       -{_r}    -0.013" size="0.003"/>
      <site name="t15" pos=" {_r*_COS45:.4f} -{_r*_COS45:.4f}  -0.013" size="0.003"/>
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

_PEG_HALF_LENGTH = 0.04   # metres
_MAX_INSERTION    = 0.05   # task done when peg tip ≤ -max_insertion below z=0


class PegInHoleEnv(MuJoCoBaseEnv):
    """MuJoCo peg-in-hole insertion with spatially-resolved tactile feedback.

    The peg is controlled via 6-DoF Cartesian velocity (mocap body).  Contact
    forces with the hole walls are read by 16 touch sensors arranged in two
    circumferential rings near the peg tip.

    Args:
        hole_radius: Unused (kept for API compatibility; hole geometry uses
            a fixed 10 mm square aperture matching the peg 9 mm radius).
        peg_radius: Unused (geometry is fixed in the XML).
        max_insertion_depth: Insertion depth in metres for task success.
        episode_length: Max steps per episode.
        occlusion_prob: Vision occlusion probability per episode.
        tactile_noise_std: Additive noise on normalised readings.
        seed: RNG seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 6,
        hole_radius: float = 0.01,
        peg_radius: float = 0.009,
        max_insertion_depth: float = _MAX_INSERTION,
        episode_length: int = 200,
        occlusion_prob: float = 0.8,
        tactile_noise_std: float = 0.01,
        seed: int = 0,
    ) -> None:
        if n_taxels != 16:
            raise ValueError("PegInHoleEnv requires n_taxels=16")
        super().__init__(
            xml=_XML,
            touch_sensor_names=_TOUCH_NAMES,
            mocap_body_name="peg",
            n_taxels=16,
            n_channels=n_channels,
            d_proprio=d_proprio,
            d_action=d_action,
            episode_length=episode_length,
            occlusion_prob=occlusion_prob,
            tactile_noise_std=tactile_noise_std,
            max_force=5.0,
            seed=seed,
        )
        self.max_depth = max_insertion_depth
        self._peg_bid = self._body_id("peg")

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        # Random lateral start offset: up to 3 × clearance (1 mm) from hole centre
        max_off = 3 * (0.010 - _r)   # 3 mm
        lateral = self._rng.uniform(-max_off, max_off, 2)
        self._data.mocap_pos[self._mocap_idx] = np.array([lateral[0], lateral[1], 0.10])
        self._data.mocap_quat[self._mocap_idx] = np.array([1.0, 0.0, 0.0, 0.0])

        mujoco.mj_forward(self._model, self._data)

        self._step_count = 0
        self._vision_occluded = self._rng.random() < self.occlusion_prob

        return self._get_obs(), {"sim_time": 0.0, "lateral_offset_mm": float(np.linalg.norm(lateral) * 1000)}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._apply_action_mocap(action, pos_scale=0.002, rot_scale=0.02, workspace=0.15)
        self._step_physics()

        peg_z = float(self._data.mocap_pos[self._mocap_idx][2])
        tip_z = peg_z - _PEG_HALF_LENGTH
        peg_xy = self._data.mocap_pos[self._mocap_idx][:2].copy()
        lateral = float(np.linalg.norm(peg_xy))
        insertion = -tip_z   # positive when tip is below z=0

        success = insertion >= self.max_depth
        reward = float(insertion / self.max_depth) - 0.5 * lateral / 0.01
        obs = self._get_obs()

        dt = self._n_substeps * float(self._model.opt.timestep)
        info = {
            "insertion_depth": float(max(insertion, 0.0)),
            "lateral_error": lateral,
            "contact_events": int(np.any(obs["tactile"] > 0.05)),
            "slip_flag": bool(np.any(obs["tactile"] > 0.7)),
            "camera_queried": self.requires_vision(self._step_count),
            "sim_time": self._step_count * dt,
            "success": bool(success),
        }
        return obs, float(reward), success, self._step_count >= self.episode_length, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        pos = self._data.mocap_pos[self._mocap_idx].astype(np.float32)
        quat = self._data.mocap_quat[self._mocap_idx].astype(np.float32)
        proprio = np.zeros(self.d_proprio, dtype=np.float32)
        proprio[:3] = pos
        proprio[3:7] = quat
        proprio[7] = float(max(-pos[2] - _PEG_HALF_LENGTH, 0.0))   # insertion depth
        return {"tactile": self._read_taxels(), "proprio": proprio}
