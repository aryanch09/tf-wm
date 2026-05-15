"""Sparsh tactile pretraining dataset adapter.

Sparsh (Meta FAIR, 2024) is a self-supervised tactile representation dataset
with 460 k tactile images from DIGIT, GelSight 2017, and GelSight Mini sensors.

HuggingFace repo: facebook/sparsh  (or local mirror at data/external/sparsh/)

Usage in Phase 1 (tactile encoder pretraining)::

    ds = SparshDataset(root="data/external/sparsh", split="train", n_taxels=16)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    # yields {"tactile": Tensor[B, 16, 1], "sensor_type": List[str]}

Tactile images are mean-pooled into a 4×4 grid (16 scalar values) to match
the TF-WM taxel schema.  Normalisation: raw pixel mean / 255, clipped [0,1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


_GRID = 4   # 4×4 = 16 taxels


def _image_to_taxels(img: np.ndarray, n: int = _GRID) -> np.ndarray:
    """Mean-pool a (H, W[, C]) image into (n*n,) normalised taxel readings."""
    if img.ndim == 3:
        img = img.mean(axis=2)   # (H, W)
    img = img.astype(np.float32) / 255.0
    H, W = img.shape
    th, tw = H // n, W // n
    out = np.zeros(n * n, dtype=np.float32)
    for r in range(n):
        for c in range(n):
            patch = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            out[r * n + c] = float(patch.mean())
    return np.clip(out, 0.0, 1.0)


class SparshDataset(Dataset):
    """PyTorch Dataset wrapping the Sparsh tactile pretraining corpus.

    Args:
        root: Directory containing the downloaded Sparsh data.
              Expected layout: ``<root>/<sensor_type>/<split>/*.png``
              where sensor_type ∈ {digit, gelsight_2017, gelsight_mini}.
        split: ``"train"`` | ``"val"`` | ``"test"``.
        n_taxels: Number of taxels after pooling (must be a perfect square).
        sensor_types: Which sensor sub-sets to include (``None`` = all).
        max_samples: Cap total samples (useful for quick debugging).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        n_taxels: int = 16,
        sensor_types: list[str] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.n_taxels = n_taxels
        self.grid = int(n_taxels ** 0.5)
        assert self.grid * self.grid == n_taxels, "n_taxels must be a perfect square"

        _default_types = ["digit", "gelsight_2017", "gelsight_mini"]
        self.sensor_types = sensor_types or _default_types

        self._entries: list[tuple[Path, str]] = []   # (image_path, sensor_type)
        for stype in self.sensor_types:
            folder = self.root / stype / split
            if not folder.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for p in sorted(folder.glob(ext)):
                    self._entries.append((p, stype))

        if not self._entries:
            raise FileNotFoundError(
                f"No Sparsh images found under {self.root}. "
                "Run: python scripts/download_datasets.py --datasets sparsh"
            )

        if max_samples is not None:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(self._entries), min(max_samples, len(self._entries)), replace=False)
            self._entries = [self._entries[i] for i in sorted(idx)]

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        path, stype = self._entries[idx]
        try:
            from PIL import Image
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((64, 64, 3), dtype=np.uint8)

        taxels = _image_to_taxels(img, self.grid)   # (16,)
        return {
            "tactile": torch.from_numpy(taxels).unsqueeze(-1),  # (16, 1)
            "sensor_type": stype,
        }

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def iter_taxel_arrays(self) -> Iterator[np.ndarray]:
        """Yield normalised taxel arrays without loading into memory."""
        for path, _ in self._entries:
            try:
                from PIL import Image
                img = np.array(Image.open(path).convert("RGB"))
            except Exception:
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            yield _image_to_taxels(img, self.grid)

    @classmethod
    def from_huggingface(
        cls,
        root: str | Path,
        split: str = "train",
        n_taxels: int = 16,
        repo_id: str = "facebook/sparsh",
        **kwargs,
    ) -> "SparshDataset":
        """Download from HuggingFace Hub if ``root`` does not exist, then load.

        Requires: ``pip install huggingface_hub``
        """
        root = Path(root)
        if not root.exists():
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=str(root), repo_type="dataset")
            except ImportError as e:
                raise ImportError("pip install huggingface_hub") from e
        return cls(root=root, split=split, n_taxels=n_taxels, **kwargs)
