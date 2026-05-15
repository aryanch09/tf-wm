"""Real-time control loop for hardware deployment.

Architecture (three threads):
  - **Tactile thread** @ sensor rate (up to 5 kHz): reads sensor, fills ring buffer.
  - **Controller thread** @ 100–200 Hz: encodes latest tactile window, runs
    the gate, conditionally waits for vision, then calls the planner.
  - **Vision thread**: dormant until woken by gate decision; returns within
    one control step or the planner replans without vision.

Safety: force cutoff and predicted-collision e-stop via
:class:`~tfwm.deploy.safety_monitor.SafetyMonitor`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from tfwm.errors import SafetyError
from tfwm.utils.timing import Timer, get_timer


@dataclass
class LoopConfig:
    """Real-time loop configuration."""

    tactile_rate_hz: int = 1000
    controller_rate_hz: int = 200
    tactile_window_size: int = 50     # steps of tactile history fed to encoder
    vision_timeout_s: float = 0.005  # max wait for vision (= 1 control step @ 200 Hz)
    device: str = "cpu"


class RingBuffer:
    """Thread-safe ring buffer for streaming sensor data."""

    def __init__(self, capacity: int, shape: tuple[int, ...]) -> None:
        self._buf = np.zeros((capacity, *shape), dtype=np.float32)
        self._ptr = 0
        self._lock = threading.Lock()
        self._capacity = capacity

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf[self._ptr % self._capacity] = frame
            self._ptr += 1

    def latest(self, n: int) -> np.ndarray:
        with self._lock:
            end = self._ptr
            start = max(end - n, 0)
            indices = [i % self._capacity for i in range(start, end)]
            return self._buf[indices].copy()


class RealtimeLoop:
    """Three-thread real-time control loop.

    Args:
        encoder: Tactile encoder module (eval mode).
        planner: Planner with ``plan(z_t, goal)`` API.
        gate: Gate module.
        safety_monitor: Safety monitor (optional).
        sensor: Object with ``.read() -> np.ndarray`` method.
        goal: Task goal latent ``(d_z,)``.
        cfg: :class:`LoopConfig`.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        planner: Any,
        gate: torch.nn.Module,
        safety_monitor: Any | None,
        sensor: Any,
        goal: torch.Tensor,
        cfg: LoopConfig | None = None,
    ) -> None:
        self.encoder = encoder.eval()
        self.planner = planner
        self.gate = gate.eval()
        self.safety_monitor = safety_monitor
        self.sensor = sensor
        self.goal = goal
        self.cfg = cfg or LoopConfig()
        self.device = torch.device(self.cfg.device)

        # Infer taxel shape from first read.
        sample = sensor.read()
        if isinstance(sample, torch.Tensor):
            shape = tuple(sample.shape[-2:])
        else:
            shape = (16, 1)
        self._tactile_buf = RingBuffer(
            capacity=self.cfg.tactile_window_size * 10, shape=shape
        )
        self._vision_frame: torch.Tensor | None = None
        self._vision_event = threading.Event()
        self._stop_event = threading.Event()
        self._latest_action: np.ndarray = np.zeros(6, dtype=np.float32)

    # ------------------------------------------------------------------
    # Thread targets
    # ------------------------------------------------------------------

    def _tactile_thread(self) -> None:
        dt = 1.0 / self.cfg.tactile_rate_hz
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            frame = self.sensor.read()
            if isinstance(frame, torch.Tensor):
                frame = frame.squeeze().numpy()
            self._tactile_buf.push(frame)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _vision_thread(self) -> None:
        while not self._stop_event.is_set():
            self._vision_event.wait(timeout=1.0)
            if self._stop_event.is_set():
                break
            # Simulate vision capture (replace with real camera call).
            self._vision_frame = torch.zeros(1, 3, 224, 224)
            self._vision_event.clear()

    def _controller_thread(self) -> None:
        dt = 1.0 / self.cfg.controller_rate_hz
        ctl_timer = get_timer("controller_step")
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            with Timer("controller_step", stats=ctl_timer):
                self._control_step()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _control_step(self) -> None:
        window = self._tactile_buf.latest(self.cfg.tactile_window_size)
        if len(window) == 0:
            return
        t = torch.from_numpy(window).unsqueeze(0).to(self.device)  # (1, T, N, C)
        if t.dim() == 3:
            t = t.unsqueeze(-1)
        with torch.no_grad():
            enc = self.encoder(t)
            z_t = enc["mu"]  # (1, T, d_z)
            dummy_aux = {"logvar": torch.zeros_like(z_t)}
            gate_score = self.gate(z_t, dummy_aux)

        if gate_score[:, -1].item() > self.gate.threshold:
            self._vision_event.set()
            self._vision_event.wait(timeout=self.cfg.vision_timeout_s)

        with torch.no_grad():
            action = self.planner.plan(z_t, self.goal.to(self.device).unsqueeze(0))

        if self.safety_monitor is not None:
            try:
                action = self.safety_monitor.check(action)
            except SafetyError:
                action = torch.zeros_like(action)

        self._latest_action = action.squeeze().cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all threads."""
        self._threads = [
            threading.Thread(target=self._tactile_thread, daemon=True, name="tactile"),
            threading.Thread(target=self._vision_thread, daemon=True, name="vision"),
            threading.Thread(target=self._controller_thread, daemon=True, name="controller"),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        """Signal all threads to stop and join."""
        self._stop_event.set()
        self._vision_event.set()
        for t in self._threads:
            t.join(timeout=2.0)

    @property
    def action(self) -> np.ndarray:
        """Latest computed action ``(d_a,)``."""
        return self._latest_action.copy()
