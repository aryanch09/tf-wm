"""In-hand reorientation task — MuJoCo physics backend.

4-finger pinch grasp of a sphere.  Each finger has two slide joints:
  - *grip*: radial squeeze toward the sphere centre
  - *z-slide*: axial translation for differential rolling

The 8 actuators (4 grip + 4 z-slide) are mapped from the first 8 dims of
the action vector.  Any remaining action dimensions are silently ignored.

Tactile feedback: 16 touch sensors (4 per finger, 2×2 grid) give spatially-
resolved contact forces via MuJoCo's ``<touch>`` sensor with a 15 mm cutoff.

Hypothesis H-A: TF-WM achieves ≥15 % success improvement over vision-first
baselines under heavy occlusion.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from tfwm.sim.mujoco_env import MuJoCoBaseEnv, _quat_err_deg, _random_quat

# ---------------------------------------------------------------------------
# MuJoCo XML — 4-finger spherical grasp, 16 touch sensors
# ---------------------------------------------------------------------------

_XML = """
<mujoco model="inhand_reorient">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <!-- Object: sphere with free joint -->
    <body name="object" pos="0 0 0">
      <joint name="obj_free" type="free"/>
      <geom type="sphere" size="0.03" mass="0.08" condim="4"
            friction="1.5 0.005 0.0001" rgba="0.85 0.45 0.1 1"/>
    </body>
    <!-- Finger 1: grip along -x, rest at x=+0.06 -->
    <body name="f1" pos="0.06 0 0">
      <joint name="f1_grip" type="slide" axis="-1 0 0" range="0 0.05"/>
      <joint name="f1_z"    type="slide" axis="0 0 1"  range="-0.02 0.02"/>
      <geom type="capsule" size="0.012" fromto="0 0 -0.025 0 0 0.025"
            mass="0.04" condim="4" friction="1.5 0.005 0.0001" rgba="0.4 0.6 0.8 1"/>
      <site name="t0"  pos="-0.012  0.006 -0.012" size="0.003"/>
      <site name="t1"  pos="-0.012 -0.006 -0.012" size="0.003"/>
      <site name="t2"  pos="-0.012  0.006  0.012" size="0.003"/>
      <site name="t3"  pos="-0.012 -0.006  0.012" size="0.003"/>
    </body>
    <!-- Finger 2: grip along +x, rest at x=-0.06 -->
    <body name="f2" pos="-0.06 0 0">
      <joint name="f2_grip" type="slide" axis="1 0 0" range="0 0.05"/>
      <joint name="f2_z"    type="slide" axis="0 0 1" range="-0.02 0.02"/>
      <geom type="capsule" size="0.012" fromto="0 0 -0.025 0 0 0.025"
            mass="0.04" condim="4" friction="1.5 0.005 0.0001" rgba="0.4 0.6 0.8 1"/>
      <site name="t4"  pos=" 0.012  0.006 -0.012" size="0.003"/>
      <site name="t5"  pos=" 0.012 -0.006 -0.012" size="0.003"/>
      <site name="t6"  pos=" 0.012  0.006  0.012" size="0.003"/>
      <site name="t7"  pos=" 0.012 -0.006  0.012" size="0.003"/>
    </body>
    <!-- Finger 3: grip along -y, rest at y=+0.06 -->
    <body name="f3" pos="0 0.06 0">
      <joint name="f3_grip" type="slide" axis="0 -1 0" range="0 0.05"/>
      <joint name="f3_z"    type="slide" axis="0 0 1"  range="-0.02 0.02"/>
      <geom type="capsule" size="0.012" fromto="0 0 -0.025 0 0 0.025"
            mass="0.04" condim="4" friction="1.5 0.005 0.0001" rgba="0.4 0.6 0.8 1"/>
      <site name="t8"  pos=" 0.006 -0.012 -0.012" size="0.003"/>
      <site name="t9"  pos="-0.006 -0.012 -0.012" size="0.003"/>
      <site name="t10" pos=" 0.006 -0.012  0.012" size="0.003"/>
      <site name="t11" pos="-0.006 -0.012  0.012" size="0.003"/>
    </body>
    <!-- Finger 4: grip along +y, rest at y=-0.06 -->
    <body name="f4" pos="0 -0.06 0">
      <joint name="f4_grip" type="slide" axis="0 1 0" range="0 0.05"/>
      <joint name="f4_z"    type="slide" axis="0 0 1" range="-0.02 0.02"/>
      <geom type="capsule" size="0.012" fromto="0 0 -0.025 0 0 0.025"
            mass="0.04" condim="4" friction="1.5 0.005 0.0001" rgba="0.4 0.6 0.8 1"/>
      <site name="t12" pos=" 0.006  0.012 -0.012" size="0.003"/>
      <site name="t13" pos="-0.006  0.012 -0.012" size="0.003"/>
      <site name="t14" pos=" 0.006  0.012  0.012" size="0.003"/>
      <site name="t15" pos="-0.006  0.012  0.012" size="0.003"/>
    </body>
  </worldbody>
  <actuator>
    <position name="f1_grip_act" joint="f1_grip" kp="300" ctrlrange="0 0.05"/>
    <position name="f1_z_act"    joint="f1_z"    kp="100" ctrlrange="-0.02 0.02"/>
    <position name="f2_grip_act" joint="f2_grip" kp="300" ctrlrange="0 0.05"/>
    <position name="f2_z_act"    joint="f2_z"    kp="100" ctrlrange="-0.02 0.02"/>
    <position name="f3_grip_act" joint="f3_grip" kp="300" ctrlrange="0 0.05"/>
    <position name="f3_z_act"    joint="f3_z"    kp="100" ctrlrange="-0.02 0.02"/>
    <position name="f4_grip_act" joint="f4_grip" kp="300" ctrlrange="0 0.05"/>
    <position name="f4_z_act"    joint="f4_z"    kp="100" ctrlrange="-0.02 0.02"/>
  </actuator>
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

# Joint initial positions: fingers lightly contact the sphere
_GRIP_INIT = 0.018   # metres — inner capsule face at sphere surface
_GRIP_JOINTS = ["f1_grip", "f2_grip", "f3_grip", "f4_grip"]
_Z_JOINTS = ["f1_z", "f2_z", "f3_z", "f4_z"]


class InHandReorientEnv(MuJoCoBaseEnv):
    """MuJoCo in-hand reorientation: rotate grasped sphere to goal quaternion.

    Args:
        n_taxels: Must be 16 (4 fingers × 4 sites).
        success_thresh_deg: Orientation error threshold for success.
        success_hold_steps: Consecutive steps below threshold for success.
        occlusion_prob: Probability that vision is occluded at each episode.
        tactile_noise_std: Additive noise on normalised taxel readings.
        seed: RNG seed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 16,
        episode_length: int = 200,
        success_thresh_deg: float = 15.0,
        success_hold_steps: int = 10,
        occlusion_prob: float = 0.0,
        tactile_noise_std: float = 0.01,
        seed: int = 0,
    ) -> None:
        if n_taxels != 16:
            raise ValueError("InHandReorientEnv requires n_taxels=16")
        super().__init__(
            xml=_XML,
            touch_sensor_names=_TOUCH_NAMES,
            mocap_body_name=None,
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
        self.success_thresh_deg = success_thresh_deg
        self.success_hold_steps = success_hold_steps
        self._hold_count = 0
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Cache joint qpos addresses
        self._obj_qadr = self._joint_qpos_adr("obj_free")
        self._grip_adrs = [self._joint_qpos_adr(j) for j in _GRIP_JOINTS]
        self._z_adrs = [self._joint_qpos_adr(j) for j in _Z_JOINTS]

    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self._model, self._data)

        # Random initial object orientation
        init_quat = _random_quat(self._rng)
        self._data.qpos[self._obj_qadr:self._obj_qadr + 3] = 0.0   # centred
        self._data.qpos[self._obj_qadr + 3:self._obj_qadr + 7] = init_quat

        # Fingers start in light contact
        for adr in self._grip_adrs:
            self._data.qpos[adr] = _GRIP_INIT
        for adr in self._z_adrs:
            self._data.qpos[adr] = 0.0

        # Match ctrl to qpos
        self._data.ctrl[0] = _GRIP_INIT   # f1_grip
        self._data.ctrl[2] = _GRIP_INIT   # f2_grip
        self._data.ctrl[4] = _GRIP_INIT   # f3_grip
        self._data.ctrl[6] = _GRIP_INIT   # f4_grip

        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        self._goal_quat = _random_quat(self._rng)
        self._hold_count = 0
        self._step_count = 0
        self._vision_occluded = self._rng.random() < self.occlusion_prob

        return self._get_obs(), {"goal_quat": self._goal_quat.tolist()}

    def step(self, action: np.ndarray):
        self._step_count += 1
        self._apply_action_joints(action)
        self._step_physics()

        obj_quat = self._data.qpos[self._obj_qadr + 3:self._obj_qadr + 7].copy()
        err_deg = _quat_err_deg(obj_quat, self._goal_quat)

        success = err_deg < self.success_thresh_deg
        self._hold_count = self._hold_count + 1 if success else 0
        episode_success = self._hold_count >= self.success_hold_steps

        reward = -err_deg / 180.0 + float(episode_success) * 10.0
        obs = self._get_obs()

        dt = self._n_substeps * float(self._model.opt.timestep)
        info = {
            "orientation_error_deg": float(err_deg),
            "success": bool(episode_success),
            "contact_events": int(np.any(obs["tactile"] > 0.05)),
            "slip_flag": bool(np.any(obs["tactile"] > 0.6)),
            "camera_queried": self.requires_vision(self._step_count),
            "sim_time": self._step_count * dt,
        }
        return obs, float(reward), episode_success, self._step_count >= self.episode_length, info

    def _get_obs(self) -> dict[str, np.ndarray]:
        obj_pos = self._data.qpos[self._obj_qadr:self._obj_qadr + 3].astype(np.float32)
        obj_quat = self._data.qpos[self._obj_qadr + 3:self._obj_qadr + 7].astype(np.float32)
        proprio = np.zeros(self.d_proprio, dtype=np.float32)
        proprio[:3] = obj_pos
        proprio[3:7] = obj_quat
        end = min(11, self.d_proprio)
        proprio[7:end] = self._goal_quat[:end - 7].astype(np.float32)
        return {"tactile": self._read_taxels(), "proprio": proprio}
