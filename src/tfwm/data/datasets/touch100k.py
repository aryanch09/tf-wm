"""Touch100k dataset adapter — tactile + vision + language pairs.

Touch100k (CoCa-Lab, 2024 — arXiv:2406.03813) contains 100 k GelSight tactile
images paired with RGB vision frames and natural-language descriptions at
sentence and phrase granularity.  Used for Phase 3 auxiliary head training
(affordance prediction, object state estimation from tactile+language).

HuggingFace: cocacola-lab/Touch100k  (or local mirror)
Homepage: https://cocacola-lab.github.io/Touch100k/

Expected HuggingFace dataset fields (streaming mode)::

    {
      "tactile_image": PIL.Image  (H×W grayscale GelSight)
      "rgb_image":     PIL.Image  (H×W×3 scene RGB)
      "sentence":      str        (full description)
      "phrases":       list[str]  (sub-phrase annotations)
      "object_class":  str
      "contact_type":  str        (e.g. "pinch", "press", "slide")
    }

Tactile image → 16 taxels via 4×4 mean-pool (same as SparshDataset).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


_GRID = 4
_CONTACT_TYPES = ["pinch", "press", "slide", "roll", "twist", "hold", "other"]
_CONTACT2IDX: dict[str, int] = {ct: i for i, ct in enumerate(_CONTACT_TYPES)}


def _image_to_taxels(img: np.ndarray, n: int = _GRID) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32) / 255.0
    H, W = img.shape
    th, tw = H // n, W // n
    out = np.zeros(n * n, dtype=np.float32)
    for r in range(n):
        for c in range(n):
            out[r * n + c] = float(img[r * th:(r + 1) * th, c * tw:(c + 1) * tw].mean())
    return np.clip(out, 0.0, 1.0)


class Touch100kDataset(IterableDataset):
    """Streaming PyTorch Dataset for Touch100k (HuggingFace backend).

    Each sample is a single tactile frame (not an episode), suitable for
    Phase 3 auxiliary head training.

    Args:
        split: ``"train"`` | ``"validation"`` | ``"test"``.
        repo_id: HuggingFace dataset repository ID.
        local_dir: If provided, load from local mirror instead of hub.
        n_taxels: Taxel count after pooling (must be a perfect square).
        max_samples: Cap total samples.
        streaming: Use HuggingFace streaming mode (avoids full download).
    """

    def __init__(
        self,
        split: str = "train",
        repo_id: str = "cocacola-lab/Touch100k",
        local_dir: str | Path | None = None,
        n_taxels: int = 16,
        max_samples: int | None = None,
        streaming: bool = True,
    ) -> None:
        self.split = split
        self.repo_id = repo_id
        self.local_dir = Path(local_dir) if local_dir else None
        self.n_taxels = n_taxels
        self.grid = int(n_taxels ** 0.5)
        self.max_samples = max_samples
        self.streaming = streaming

    def _load_hf_dataset(self):
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e

        if self.local_dir and self.local_dir.exists():
            return load_dataset(
                str(self.local_dir),
                split=self.split,
                streaming=self.streaming,
            )
        return load_dataset(
            self.repo_id,
            split=self.split,
            streaming=self.streaming,
        )

    def __iter__(self) -> Iterator[dict]:
        ds = self._load_hf_dataset()
        count = 0
        for sample in ds:
            if self.max_samples is not None and count >= self.max_samples:
                break
            yield self._process(sample)
            count += 1

    def _process(self, sample: dict) -> dict[str, torch.Tensor | int | str]:
        # Tactile image → taxel array
        tac_img = sample.get("tactile_image")
        if tac_img is not None:
            if hasattr(tac_img, "convert"):
                tac_arr = np.array(tac_img.convert("L"))
            else:
                tac_arr = np.asarray(tac_img)
            taxels = _image_to_taxels(tac_arr, self.grid)
        else:
            taxels = np.zeros(self.n_taxels, dtype=np.float32)

        # Contact type → class index
        contact_str = sample.get("contact_type", "other")
        contact_idx = _CONTACT2IDX.get(contact_str, _CONTACT2IDX["other"])

        return {
            "tactile": torch.from_numpy(taxels).unsqueeze(-1),  # (16, 1)
            "sentence": sample.get("sentence", ""),
            "contact_type": contact_idx,
            "object_class": sample.get("object_class", ""),
        }

    @classmethod
    def from_local(cls, root: str | Path, **kwargs) -> "Touch100kDataset":
        """Load from a locally downloaded mirror."""
        return cls(local_dir=root, streaming=False, **kwargs)


class Touch100kEpisodicDataset(Dataset):
    """Map-style wrapper for pre-downloaded Touch100k (non-streaming).

    Use this when the full dataset is locally available on disk as
    ``<root>/train/*.json`` or similar flat structure.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        n_taxels: int = 16,
        max_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.grid = int(n_taxels ** 0.5)
        self.n_taxels = n_taxels

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Touch100k split directory not found: {split_dir}. "
                "Run: python scripts/download_datasets.py --datasets touch100k"
            )

        import glob
        self._files = sorted(glob.glob(str(split_dir / "*.png")))
        if max_samples:
            self._files = self._files[:max_samples]

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from PIL import Image
        img = np.array(Image.open(self._files[idx]).convert("L"))
        taxels = _image_to_taxels(img, self.grid)
        return {"tactile": torch.from_numpy(taxels).unsqueeze(-1)}
