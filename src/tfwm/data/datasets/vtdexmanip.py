"""VTDexManip dataset adapter — vision-tactile dexterous manipulation.

VTDexManip (OpenReview 2024, GitHub: LQTS/VTDexManip) provides 10 daily
manipulation tasks across 182 objects with paired RGB and fingertip tactile
sensor readings from a 5-finger hand.

Download: https://github.com/LQTS/VTDexManip
Expected layout::

    <root>/
      task_<name>/
        episode_<id>/
          tactile.npy    (T, n_sensors) — raw tactile readings
          vision.npy     (T, H, W, C)  — RGB frames
          proprio.npy    (T, d_p)      — joint positions + velocities
          actions.npy    (T, d_a)
          labels.json    — {success, task_name, object_id}

Tactile normalisation: raw readings divided by ``max_taxel`` (dataset max),
then clipped to [0, 1] and resampled to 16 taxels via linear interpolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_MAX_TAXEL = 4096.0   # typical raw max for resistive pressure sensors


def _resample_taxels(raw: np.ndarray, n_out: int = 16) -> np.ndarray:
    """Resample ``raw`` (T, n_in) → (T, n_out) via linear interpolation."""
    T, n_in = raw.shape
    if n_in == n_out:
        return raw.astype(np.float32)
    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    out = np.zeros((T, n_out), dtype=np.float32)
    for t in range(T):
        out[t] = np.interp(x_out, x_in, raw[t].astype(np.float32))
    return out


class VTDexManipDataset(Dataset):
    """PyTorch Dataset for VTDexManip episodes.

    Args:
        root: Path to extracted VTDexManip data directory.
        tasks: Task names to include (``None`` = all found).
        split: ``"train"`` | ``"val"`` | ``"test"`` — loaded from
               ``<root>/splits.json`` if present, otherwise random 80/10/10.
        seq_length: Window length (``None`` = full episode).
        stride: Sliding window stride.
        n_taxels: Output taxel count (default 16 to match TF-WM schema).
        d_proprio: Proprioception dimension (padded/truncated).
        success_only: If ``True``, exclude failed episodes.
    """

    def __init__(
        self,
        root: str | Path,
        tasks: list[str] | None = None,
        split: str = "train",
        seq_length: int | None = 50,
        stride: int = 10,
        n_taxels: int = 16,
        d_proprio: int = 12,
        success_only: bool = False,
    ) -> None:
        self.root = Path(root)
        self.n_taxels = n_taxels
        self.d_proprio = d_proprio
        self.seq_length = seq_length
        self.stride = stride

        if not self.root.exists():
            raise FileNotFoundError(
                f"VTDexManip data not found at {self.root}. "
                "Clone https://github.com/LQTS/VTDexManip and set root accordingly."
            )

        # Discover task directories
        all_task_dirs = sorted(p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("task_"))
        if tasks:
            all_task_dirs = [p for p in all_task_dirs if any(t in p.name for t in tasks)]

        # Load or build split index
        splits_path = self.root / "splits.json"
        if splits_path.exists():
            with open(splits_path) as f:
                split_index: dict[str, list[str]] = json.load(f)
            allowed = set(split_index.get(split, []))
        else:
            allowed = None   # use all

        episode_dirs: list[Path] = []
        for task_dir in all_task_dirs:
            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                if allowed is not None and ep_dir.name not in allowed:
                    continue
                if success_only:
                    lbl_path = ep_dir / "labels.json"
                    if lbl_path.exists():
                        with open(lbl_path) as f:
                            if not json.load(f).get("success", True):
                                continue
                if (ep_dir / "tactile.npy").exists():
                    episode_dirs.append(ep_dir)

        if not episode_dirs:
            raise FileNotFoundError(f"No episodes found under {self.root} for split='{split}'")

        self._windows: list[tuple[Path, int, int]] = []
        for ep_dir in episode_dirs:
            T = np.load(ep_dir / "tactile.npy", mmap_mode="r").shape[0]
            if seq_length is None:
                self._windows.append((ep_dir, 0, T))
            else:
                for s in range(0, T - seq_length + 1, stride):
                    self._windows.append((ep_dir, s, s + seq_length))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_dir, start, end = self._windows[idx]

        raw_tactile = np.load(ep_dir / "tactile.npy", mmap_mode="r")[start:end]
        normed = np.clip(raw_tactile.astype(np.float32) / _MAX_TAXEL, 0.0, 1.0)
        tactile = _resample_taxels(normed, self.n_taxels)   # (T, 16)

        raw_proprio = np.load(ep_dir / "proprio.npy", mmap_mode="r")[start:end].astype(np.float32)
        T, D = raw_proprio.shape
        proprio = np.zeros((T, self.d_proprio), dtype=np.float32)
        proprio[:, :min(D, self.d_proprio)] = raw_proprio[:, :min(D, self.d_proprio)]

        actions = np.load(ep_dir / "actions.npy", mmap_mode="r")[start:end].astype(np.float32)

        return {
            "tactile": torch.from_numpy(tactile).unsqueeze(-1),  # (T, 16, 1)
            "proprio": torch.from_numpy(proprio),                 # (T, d_proprio)
            "actions": torch.from_numpy(actions),                 # (T, d_action)
            "rewards": torch.zeros(T),
            "dones": torch.zeros(T, dtype=torch.bool),
        }
