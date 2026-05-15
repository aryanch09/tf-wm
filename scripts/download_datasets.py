#!/usr/bin/env python3
"""Download external datasets for TF-WM training.

Usage::

    # Download everything (into data/external/ by default)
    python scripts/download_datasets.py --datasets all

    # Download specific datasets
    python scripts/download_datasets.py --datasets sparsh touch100k

    # Custom root directory
    python scripts/download_datasets.py --datasets sparsh --root /mnt/data/tfwm

Available datasets:
  sparsh       460 k tactile images (DIGIT/GelSight) — Phase 1 pretraining
  touch100k    100 k tactile+vision+language pairs    — Phase 3 auxiliary heads
  rh20t        110 k robot episodes (F/T, RGB, prop.) — Phase 5 sim-to-real
  vtdexmanip   Vision-tactile dexterous manipulation  — Phase 5 sim-to-real
  reassemble   4.5 k contact-rich assembly demos      — Phase 1 + Phase 5

Notes:
  - Sparsh and Touch100k are downloaded from HuggingFace Hub.
  - RH20T requires manual registration at https://rh20t.github.io/
    Download the HDF5 archive there and extract to --root/rh20t/.
  - VTDexManip is cloned from GitHub (data bundled in repo).
  - REASSEMBLE: check arXiv:2502.05086 for access links.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-dataset download helpers
# ---------------------------------------------------------------------------

def _pip_check(package: str, import_name: str | None = None) -> None:
    import importlib
    try:
        importlib.import_module(import_name or package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def download_sparsh(root: Path) -> None:
    dest = root / "sparsh"
    if dest.exists():
        print(f"[sparsh] Already exists at {dest} — skipping.")
        return
    _pip_check("huggingface_hub")
    from huggingface_hub import snapshot_download
    print("[sparsh] Downloading from HuggingFace Hub (facebook/sparsh) …")
    snapshot_download(
        repo_id="facebook/sparsh",
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"[sparsh] Saved to {dest}")


def download_touch100k(root: Path) -> None:
    dest = root / "touch100k"
    if dest.exists():
        print(f"[touch100k] Already exists at {dest} — skipping.")
        return
    _pip_check("huggingface_hub")
    from huggingface_hub import snapshot_download
    print("[touch100k] Downloading from HuggingFace Hub (cocacola-lab/Touch100k) …")
    snapshot_download(
        repo_id="cocacola-lab/Touch100k",
        repo_type="dataset",
        local_dir=str(dest),
    )
    print(f"[touch100k] Saved to {dest}")


def download_rh20t(root: Path) -> None:
    dest = root / "rh20t"
    print(
        "[rh20t] RH20T requires manual registration and download.\n"
        "  1. Visit https://rh20t.github.io/ and register for access.\n"
        "  2. Download the HDF5 archives (~5 TB total, or a subset task).\n"
        f"  3. Extract to: {dest}/\n"
        "  The directory should contain episode_XXXX/ subdirectories.\n"
        "  Skipping automatic download."
    )


def download_vtdexmanip(root: Path) -> None:
    dest = root / "vtdexmanip"
    if dest.exists():
        print(f"[vtdexmanip] Already exists at {dest} — skipping.")
        return
    print("[vtdexmanip] Cloning VTDexManip from GitHub …")
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/LQTS/VTDexManip.git",
        str(dest),
    ])
    print(f"[vtdexmanip] Cloned to {dest}")


def download_reassemble(root: Path) -> None:
    dest = root / "reassemble"
    print(
        "[reassemble] REASSEMBLE (arXiv:2502.05086) does not yet have a\n"
        "  public download link in the repository.\n"
        "  1. Check https://arxiv.org/abs/2502.05086 for the latest access instructions.\n"
        f"  2. Extract data to: {dest}/\n"
        "  Expected structure: episode_XXXX/ with ft.npy, proprio.npy, actions.npy.\n"
        "  Skipping automatic download."
    )


# ---------------------------------------------------------------------------
# Split-file generation helpers
# ---------------------------------------------------------------------------

def _make_splits(root: Path, dataset: str, train: float = 0.8, val: float = 0.1) -> None:
    """Create a train/val/test splits.json for datasets that don't provide one."""
    import json, random
    ds_dir = root / dataset
    episodes = sorted(p.name for p in ds_dir.iterdir() if p.is_dir())
    if not episodes:
        return
    random.seed(42)
    random.shuffle(episodes)
    n = len(episodes)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": episodes[:n_train],
        "val": episodes[n_train:n_train + n_val],
        "test": episodes[n_train + n_val:],
    }
    out_path = ds_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[{dataset}] Generated splits → {out_path} "
          f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_DATASETS = ["sparsh", "touch100k", "rh20t", "vtdexmanip", "reassemble"]

_DOWNLOADERS = {
    "sparsh": download_sparsh,
    "touch100k": download_touch100k,
    "rh20t": download_rh20t,
    "vtdexmanip": download_vtdexmanip,
    "reassemble": download_reassemble,
}

_NEEDS_SPLITS = {"rh20t", "reassemble"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        metavar="DS",
        help="Dataset names or 'all'",
    )
    parser.add_argument(
        "--root",
        default="data/external",
        help="Root directory for downloads (default: data/external)",
    )
    parser.add_argument(
        "--make-splits",
        action="store_true",
        help="Generate train/val/test splits.json after download",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    targets: list[str] = _ALL_DATASETS if "all" in args.datasets else args.datasets
    unknown = set(targets) - set(_ALL_DATASETS)
    if unknown:
        parser.error(f"Unknown datasets: {unknown}. Choose from: {_ALL_DATASETS}")

    for ds in targets:
        print(f"\n{'='*60}")
        _DOWNLOADERS[ds](root)
        if args.make_splits and ds in _NEEDS_SPLITS:
            _make_splits(root, ds)

    print("\nDone. Configure data paths in configs/data/*.yaml")


if __name__ == "__main__":
    main()
