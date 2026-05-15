"""RH20T dataset adapter — robot episodes with force-torque and proprioception.

RH20T (2023) contains 110 k robot manipulation episodes across 140+ tasks,
with multi-modal sensory data: RGB-D, 6-DoF force-torque, proprioception,
and audio.  Used here for Phase 5 sim-to-real fine-tuning.

Download: https://rh20t.github.io/  (requires registration, ~5 TB RGB)
Expected layout after extraction::

    <root>/
      episode_0001/
        rgb_static.mp4   (or frame_*.jpg)
        robot_obs.npy    (T, 20) — EE pose + joint states
        actions.npy      (T, 7)  — delta EE + gripper
        ft_sensor.npy    (T, 6)  — [Fx,Fy,Fz,Tx,Ty,Tz] in N / N·m
        metadata.json
      episode_0002/
        ...

Force-torque → 16 virtual taxels:
  The 6-D F/T vector is projected onto 16 pre-defined basis vectors that
  represent the 4×4 spatial layout of the TF-WM probe sensor.  Each basis
  vector encodes the contact normal expected at that taxel site.  Magnitude
  is normalised by ``max_force_n``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

# 16 unit normals for the 4×4 taxel grid (probe sensor, in end-effector frame)
# Row 0–3: z positions [-0.035, -0.025, -0.015, -0.005]
# Cols 0–3: azimuths [0°, 90°, 180°, 270°]
_AZIMUTHS = np.array([0.0, 90.0, 180.0, 270.0]) * (np.pi / 180.0)
_TAXEL_NORMALS = np.zeros((16, 3), dtype=np.float32)
for _row in range(4):
    for _col, _az in enumerate(_AZIMUTHS):
        _TAXEL_NORMALS[_row * 4 + _col] = [np.cos(_az), np.sin(_az), 0.0]


def _ft_to_taxels(ft: np.ndarray, max_force_n: float = 20.0) -> np.ndarray:
    """Project 6-D F/T reading onto 16 virtual taxels.

    Args:
        ft: Shape ``(6,)`` — [Fx, Fy, Fz, Tx, Ty, Tz].
        max_force_n: Saturation force in Newtons.

    Returns:
        Shape ``(16,)`` normalised to [0, 1].
    """
    force = ft[:3].astype(np.float32)
    projections = _TAXEL_NORMALS @ force          # (16,)
    return np.clip(projections / max_force_n, 0.0, 1.0)


class RH20TDataset(Dataset):
    """PyTorch Dataset for RH20T episodes.

    Args:
        root: Path to extracted RH20T data directory.
        split_json: Optional JSON file with ``{"train": [...], "val": [...], ...}``
                    episode folder names.  If ``None``, all episodes are used.
        split: Which split key to load from ``split_json``.
        seq_length: Steps per sample window (``None`` = full episode).
        stride: Sliding window stride.
        max_force_n: Saturation force for F/T normalisation (Newtons).
        d_proprio: Proprioception vector length (padded/truncated).
    """

    def __init__(
        self,
        root: str | Path,
        split_json: str | Path | None = None,
        split: str = "train",
        seq_length: int | None = 50,
        stride: int = 10,
        max_force_n: float = 20.0,
        d_proprio: int = 12,
    ) -> None:
        self.root = Path(root)
        self.seq_length = seq_length
        self.stride = stride
        self.max_force_n = max_force_n
        self.d_proprio = d_proprio

        if not self.root.exists():
            raise FileNotFoundError(
                f"RH20T data not found at {self.root}. "
                "Download from https://rh20t.github.io/ and extract here."
            )

        if split_json is not None:
            with open(split_json) as f:
                keys = json.load(f)[split]
            episode_dirs = [self.root / k for k in keys if (self.root / k).exists()]
        else:
            episode_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())

        self._windows: list[tuple[Path, int, int]] = []
        for ep_dir in episode_dirs:
            actions_path = ep_dir / "actions.npy"
            if not actions_path.exists():
                continue
            T = np.load(actions_path, mmap_mode="r").shape[0]
            if seq_length is None:
                self._windows.append((ep_dir, 0, T))
            else:
                for start in range(0, T - seq_length + 1, stride):
                    self._windows.append((ep_dir, start, start + seq_length))

        if not self._windows:
            raise FileNotFoundError(f"No valid episodes found under {self.root}")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_dir, start, end = self._windows[idx]

        ft_path = ep_dir / "ft_sensor.npy"
        if ft_path.exists():
            ft = np.load(ft_path, mmap_mode="r")[start:end]   # (T, 6)
            tactile = np.stack([_ft_to_taxels(ft[t], self.max_force_n) for t in range(len(ft))])
        else:
            T = end - start
            tactile = np.zeros((T, 16), dtype=np.float32)

        proprio_path = ep_dir / "robot_obs.npy"
        if proprio_path.exists():
            raw = np.load(proprio_path, mmap_mode="r")[start:end].astype(np.float32)
            T, D = raw.shape
            proprio = np.zeros((T, self.d_proprio), dtype=np.float32)
            cols = min(D, self.d_proprio)
            proprio[:, :cols] = raw[:, :cols]
        else:
            T = end - start
            proprio = np.zeros((T, self.d_proprio), dtype=np.float32)

        actions_raw = np.load(ep_dir / "actions.npy", mmap_mode="r")[start:end].astype(np.float32)

        return {
            "tactile": torch.from_numpy(tactile).unsqueeze(-1),    # (T, 16, 1)
            "proprio": torch.from_numpy(proprio),                   # (T, d_proprio)
            "actions": torch.from_numpy(actions_raw),               # (T, d_action)
            "rewards": torch.zeros(len(tactile)),
            "dones": torch.zeros(len(tactile), dtype=torch.bool),
        }

    # ------------------------------------------------------------------

    def iter_episodes(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield full episodes as numpy dicts (no windowing)."""
        seen: set[Path] = set()
        for ep_dir, _, _ in self._windows:
            if ep_dir in seen:
                continue
            seen.add(ep_dir)
            ft = np.load(ep_dir / "ft_sensor.npy") if (ep_dir / "ft_sensor.npy").exists() else None
            tactile = (
                np.stack([_ft_to_taxels(ft[t], self.max_force_n) for t in range(len(ft))])
                if ft is not None
                else np.zeros((0, 16), dtype=np.float32)
            )
            yield {
                "tactile": tactile,
                "actions": np.load(ep_dir / "actions.npy") if (ep_dir / "actions.npy").exists() else None,
            }
