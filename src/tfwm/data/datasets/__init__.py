"""External dataset adapters for TF-WM pretraining and fine-tuning.

Each adapter normalises its native format into the TF-WM Episode schema
(tactile: [0,1] float32, proprio, actions, rewards, dones) and provides a
PyTorch Dataset for direct use in the training pipeline.

Recommended usage per training phase:

  Phase 1 (representation)  : SparshDataset, Touch100kDataset
  Phase 3 (auxiliary heads) : Touch100kDataset (language labels)
  Phase 5 (sim-to-real)     : RH20TDataset, VTDexManipDataset, REASSEMBLEDataset

Download all datasets with::

    python scripts/download_datasets.py --datasets all --root data/external
"""

from .sparsh import SparshDataset
from .rh20t import RH20TDataset
from .vtdexmanip import VTDexManipDataset
from .touch100k import Touch100kDataset
from .reassemble import REASSEMBLEDataset

__all__ = [
    "SparshDataset",
    "RH20TDataset",
    "VTDexManipDataset",
    "Touch100kDataset",
    "REASSEMBLEDataset",
]
