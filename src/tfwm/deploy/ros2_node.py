"""ROS 2 deployment node for TF-WM.

Subscribes to sensor topics, runs the world-model inference pipeline, and
publishes actions to the robot controller.  Designed for a three-topic
architecture matching the real-time loop in
:mod:`tfwm.deploy.realtime_loop`:

  Subscriptions:
    - ``/tactile`` (sensor_msgs/Float32MultiArray): taxel readings
    - ``/proprio``  (sensor_msgs/Float32MultiArray): joint state
    - ``/camera``   (sensor_msgs/Image):            RGB frame (optional)

  Publications:
    - ``/action``  (sensor_msgs/Float32MultiArray): commanded action
    - ``/gate``    (std_msgs/Float32):              current gate score

Usage::

    ros2 run tfwm tfwm_node --ros-args \\
        -p checkpoint:=/path/to/checkpoint_final.pt \\
        -p n_taxels:=16 \\
        -p d_action:=6

.. note::

    This file is intentionally written to be importable without a ROS 2
    installation; missing rclpy is caught at import time and the module
    degrades to a stub that raises ``ImportError`` only when
    :class:`TFWMNode` is instantiated.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Float32MultiArray, Image
    from std_msgs.msg import Float32

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]

from tfwm.models.dynamics import RSSMDynamics
from tfwm.models.gating.voi_gate import VOIGate
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.policies.mpc_policy import MPCPolicy


class TFWMNode(Node):  # type: ignore[misc]
    """ROS 2 node that runs TF-WM inference and publishes actions.

    Args:
        checkpoint: Path to a TF-WM checkpoint (modules dict format).
        n_taxels: Number of tactile taxels.
        n_channels: Tactile channels per taxel.
        d_proprio: Proprioception dimension.
        d_action: Action dimension.
        d_latent: Latent state dimension.
        control_hz: Target controller frequency (Hz).
    """

    def __init__(
        self,
        checkpoint: Path | str = "checkpoint_final.pt",
        n_taxels: int = 16,
        n_channels: int = 1,
        d_proprio: int = 12,
        d_action: int = 6,
        d_latent: int = 128,
        control_hz: float = 100.0,
    ) -> None:
        if not _ROS2_AVAILABLE:
            raise ImportError(
                "rclpy is not installed. Install ROS 2 and source its setup.bash."
            )
        super().__init__("tfwm_node")

        self._n_taxels = n_taxels
        self._n_channels = n_channels
        self._d_proprio = d_proprio
        self._d_action = d_action
        self._d_latent = d_latent
        self._control_hz = control_hz
        self._lock = threading.Lock()

        # ── Model ──────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = HybridTactileEncoder(
            n_taxels=n_taxels, n_channels=n_channels, d_latent=d_latent
        ).to(self.device).eval()
        self._dynamics = RSSMDynamics(d_latent=d_latent, d_action=d_action).to(self.device).eval()
        self._gate = VOIGate(d_latent=d_latent).to(self.device).eval()
        self._policy = MPCPolicy(d_latent=d_latent, d_action=d_action).to(self.device).eval()

        self._load_checkpoint(Path(checkpoint))

        # ── Latent state ───────────────────────────────────────────────
        self._h = torch.zeros(1, self._dynamics.d_recurrent, device=self.device)
        self._z = torch.zeros(1, d_latent, device=self.device)

        # ── Latest sensor readings ─────────────────────────────────────
        self._latest_tactile: np.ndarray = np.zeros((n_taxels, n_channels), dtype=np.float32)
        self._latest_proprio: np.ndarray = np.zeros(d_proprio, dtype=np.float32)
        self._latest_vision: np.ndarray | None = None

        # ── ROS subscriptions ──────────────────────────────────────────
        self.create_subscription(Float32MultiArray, "/tactile", self._tactile_cb, 10)
        self.create_subscription(Float32MultiArray, "/proprio", self._proprio_cb, 10)
        self.create_subscription(Image, "/camera", self._camera_cb, 10)

        # ── ROS publications ───────────────────────────────────────────
        self._action_pub = self.create_publisher(Float32MultiArray, "/action", 10)
        self._gate_pub = self.create_publisher(Float32, "/gate", 10)

        # ── Control timer ──────────────────────────────────────────────
        period = 1.0 / control_hz
        self.create_timer(period, self._control_cb)
        self.get_logger().info(
            f"TFWMNode ready — device={self.device}, control={control_hz:.0f} Hz"
        )

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _tactile_cb(self, msg: Any) -> None:
        data = np.array(msg.data, dtype=np.float32)
        expected = self._n_taxels * self._n_channels
        if data.size != expected:
            self.get_logger().warn(f"tactile size mismatch: {data.size} != {expected}")
            return
        with self._lock:
            self._latest_tactile = data.reshape(self._n_taxels, self._n_channels)

    def _proprio_cb(self, msg: Any) -> None:
        data = np.array(msg.data, dtype=np.float32)
        if data.size < self._d_proprio:
            data = np.pad(data, (0, self._d_proprio - data.size))
        with self._lock:
            self._latest_proprio = data[:self._d_proprio]

    def _camera_cb(self, msg: Any) -> None:
        with self._lock:
            self._latest_vision = np.frombuffer(msg.data, dtype=np.uint8).copy()

    # ------------------------------------------------------------------
    # Control callback — runs at control_hz
    # ------------------------------------------------------------------

    def _control_cb(self) -> None:
        with self._lock:
            tactile = torch.from_numpy(self._latest_tactile).to(self.device)
            proprio = torch.from_numpy(self._latest_proprio).to(self.device)

        tactile_t = tactile.unsqueeze(0).unsqueeze(0)   # (1, 1, N, C)

        with torch.no_grad():
            enc_out = self._encoder(tactile_t)
            z_obs = enc_out["mu"][:, 0]   # (1, d_z)

            # Compute gate score.
            z_seq = z_obs.unsqueeze(1)   # (1, 1, d_z)
            gate_aux = {"logvar": enc_out.get("logvar", torch.zeros_like(z_seq))}
            gate_score = self._gate(z_seq, gate_aux)[0, 0].item()

            # Simple prior update (one-step GRU in RSSM).
            action_prev = torch.zeros(1, self._d_action, device=self.device)
            prior, h_new = self._dynamics.prior_step(z_obs, action_prev, self._h)
            self._h = h_new
            self._z = prior["mu"]

            # Plan action.
            action = self._policy.act({"latent": self._z})

        # Publish action.
        action_msg = Float32MultiArray()
        action_msg.data = action.cpu().numpy().flatten().tolist()
        self._action_pub.publish(action_msg)

        # Publish gate score.
        gate_msg = Float32()
        gate_msg.data = float(gate_score)
        self._gate_pub.publish(gate_msg)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            self.get_logger().warn(f"checkpoint not found: {ckpt_path}, using random weights")
            return
        state = torch.load(ckpt_path, map_location=self.device)
        mods = state.get("modules", {})
        if "encoder" in mods:
            self._encoder.load_state_dict(mods["encoder"])
        if "dynamics" in mods:
            self._dynamics.load_state_dict(mods["dynamics"])
        if "gate" in mods:
            self._gate.load_state_dict(mods["gate"])
        self.get_logger().info(f"loaded checkpoint from {ckpt_path}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main(args: list[str] | None = None) -> None:
    if not _ROS2_AVAILABLE:
        raise ImportError(
            "rclpy is not installed. Source your ROS 2 workspace before running this node."
        )
    rclpy.init(args=args)
    node = TFWMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
