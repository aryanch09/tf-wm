"""REASSEMBLE dataset adapter — contact-rich assembly manipulation.

REASSEMBLE (arXiv:2502.05086, 2025) provides 4 551 demonstrations of NIST
assembly board tasks (peg insertions, gear meshing, connector assembly) with:
  - Multi-view RGB cameras
  - 6-DoF force/torque sensor
  - Microphone (audio)
  - Event camera
  - Joint proprioception

This adapter uses force/torque + proprioception only, converting the 6-D F/T
to 16 virtual taxels (same spatial basis as RH20TDataset).  Used for Phase 1
pretraining and Phase 5 sim-to-real on contact-rich insertion tasks.

Expected layout::

    <root>/
      episode_0001/
        ft.npy          (T, 6)  force/torque [Fx,Fy,Fz,Tx,Ty,Tz]
        proprio.npy     (T, d_p) joint angles / EE pose
        actions.npy     (T, d_a)
        success.txt     "1" or "0"
        task.txt        task identifier string
      ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Shared F/T → taxel projection (same as rh20t.py)
_AZIMUTHS = np.array([0.0, 90.0, 180.0, 270.0]) * (np.pi / 180.0)
_TAXEL_NORMALS = np.zeros((16, 3), dtype=np.float32)
for _row in range(4):
    for _col, _az in enumerate(_AZIMUTHS):
        _TAXEL_NORMALS[_row * 4 + _col] = [np.cos(_az), np.sin(_az), 0.0]


def _ft_to_taxels(ft: np.ndarray, max_force_n: float = 30.0) -> np.ndarray:
    """Project (T, 6) F/T array → (T, 16) normalised virtual taxels."""
    force = ft[:, :3].astype(np.float32)             # (T, 3)
    proj = force @ _TAXEL_NORMALS.T                  # (T, 16)
    return np.clip(proj / max_force_n, 0.0, 1.0)


class REASSEMBLEDataset(Dataset):
    """PyTorch Dataset for the REASSEMBLE contact-rich assembly corpus.

    Args:
        root: Path to extracted REASSEMBLE data.
        split_json: Optional ``{"train": [...], "val": [...]}`` file.
        split: Which split to load.
        tasks: Task name filters (``None`` = all).
        success_only: Exclude failed demonstrations.
        seq_length: Sliding window length (``None`` = full episode).
        stride: Sliding window stride.
        max_force_n: F/T saturation for normalisation.
        d_proprio: Proprioception output dimension.
    """

    def __init__(
        self,
        root: str | Path,
        split_json: str | Path | None = None,
        split: str = "train",
        tasks: list[str] | None = None,
        success_only: bool = True,
        seq_length: int | None = 50,
        stride: int = 10,
        max_force_n: float = 30.0,
        d_proprio: int = 12,
    ) -> None:
        self.root = Path(root)
        self.max_force_n = max_force_n
        self.d_proprio = d_proprio
        self.seq_length = seq_length
        self.stride = stride

        if not self.root.exists():
            raise FileNotFoundError(
                f"REASSEMBLE data not found at {self.root}. "
                "See https://arxiv.org/abs/2502.05086 for download instructions."
            )

        if split_json is not None:
            with open(split_json) as f:
                allowed_names = set(json.load(f).get(split, []))
        else:
            allowed_names = None

        episode_dirs: list[Path] = []
        for ep_dir in sorted(self.root.iterdir()):
            if not ep_dir.is_dir():
                continue
            if allowed_names is not None and ep_dir.name not in allowed_names:
                continue

            if tasks:
                task_file = ep_dir / "task.txt"
                task_name = task_file.read_text().strip() if task_file.exists() else ""
                if not any(t in task_name for t in tasks):
                    continue

            if success_only:
                suc_file = ep_dir / "success.txt"
                if suc_file.exists() and suc_file.read_text().strip() == "0":
                    continue

            if (ep_dir / "ft.npy").exists():
                episode_dirs.append(ep_dir)

        if not episode_dirs:
            raise FileNotFoundError(
                f"No valid episodes found under {self.root} "
                f"(split={split}, success_only={success_only})"
            )

        self._windows: list[tuple[Path, int, int]] = []
        for ep_dir in episode_dirs:
            T = np.load(ep_dir / "ft.npy", mmap_mode="r").shape[0]
            if seq_length is None:
                self._windows.append((ep_dir, 0, T))
            else:
                for s in range(0, T - seq_length + 1, stride):
                    self._windows.append((ep_dir, s, s + seq_length))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_dir, start, end = self._windows[idx]
        T = end - start

        ft_raw = np.load(ep_dir / "ft.npy", mmap_mode="r")[start:end]   # (T, 6)
        tactile = _ft_to_taxels(ft_raw, self.max_force_n)                # (T, 16)

        proprio_path = ep_dir / "proprio.npy"
        if proprio_path.exists():
            raw = np.load(proprio_path, mmap_mode="r")[start:end].astype(np.float32)
            proprio = np.zeros((T, self.d_proprio), dtype=np.float32)
            cols = min(raw.shape[1], self.d_proprio)
            proprio[:, :cols] = raw[:, :cols]
        else:
            proprio = np.zeros((T, self.d_proprio), dtype=np.float32)

        actions_path = ep_dir / "actions.npy"
        if actions_path.exists():
            actions = np.load(actions_path, mmap_mode="r")[start:end].astype(np.float32)
        else:
            actions = np.zeros((T, 6), dtype=np.float32)

        suc_file = ep_dir / "success.txt"
        success = float(suc_file.read_text().strip()) if suc_file.exists() else 0.0
        dones = np.zeros(T, dtype=bool)
        if T > 0:
            dones[-1] = True

        return {
            "tactile": torch.from_numpy(tactile).unsqueeze(-1),  # (T, 16, 1)
            "proprio": torch.from_numpy(proprio),                 # (T, d_proprio)
            "actions": torch.from_numpy(actions),                 # (T, d_action)
            "rewards": torch.tensor([0.0] * (T - 1) + [success]),
            "dones": torch.from_numpy(dones),
        }
