"""Reproducibility manifest generation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import wandb
except ImportError:
    wandb = None


def get_git_info() -> dict[str, str]:
    """Get git repository information.

    Returns:
        Dict with git_sha, git_dirty, etc.
    """

    def _git(args: list[str]) -> str:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()

    try:
        sha = _git(["rev-parse", "HEAD"])
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
        dirty = bool(_git(["status", "--porcelain"]))
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
        branch = "unknown"
        dirty = False

    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "git_branch": branch,
    }


def get_package_versions() -> dict[str, str]:
    """Get versions of key packages.

    Returns:
        Dict of package names to versions.
    """
    import importlib.metadata

    packages = [
        "torch",
        "hydra-core",
        "omegaconf",
        "wandb",
        "einops",
        "rich",
        "gymnasium",
        "pydantic",
        "numpy",
        "dvc",
    ]

    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    return versions


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information.

    Returns:
        Dict with GPU model, etc.
    """
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
    }

    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_model"] = torch.cuda.get_device_name(0)
    else:
        info["cuda"] = None
        info["gpu_count"] = 0
        info["gpu_model"] = None

    return info


def generate_manifest(
    run_name: str,
    config: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """Generate reproducibility manifest.

    Args:
        run_name: Name of the run.
        config: Resolved config dict.
        output_dir: Output directory.
        seed: Random seed.

    Returns:
        Manifest dict.
    """
    import time

    # Config hash
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()

    manifest = {
        "run_name": run_name,
        **get_git_info(),
        "config_hash": config_hash,
        **get_hardware_info(),
        "seed": seed,
        "package_versions": get_package_versions(),
        "hostname": platform.node(),
        "start_time": time.time(),  # Will be updated
        "end_time": None,
        "artifacts": {},
    }

    return manifest


def save_manifest(manifest: dict[str, Any], output_dir: Path) -> None:
    """Save manifest to run.yaml.

    Args:
        manifest: Manifest dict.
        output_dir: Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)


def update_manifest_end_time(output_dir: Path) -> None:
    """Update manifest with end time.

    Args:
        output_dir: Output directory.
    """
    import time

    manifest_path = output_dir / "run.yaml"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        manifest["end_time"] = time.time()
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)


def log_manifest_to_wandb(manifest: dict[str, Any]) -> None:
    """Log manifest to wandb.

    Args:
        manifest: Manifest dict.
    """
    if wandb:
        # Flatten nested dicts for wandb
        flat_manifest = {}
        for key, value in manifest.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_manifest[f"{key}.{subkey}"] = subvalue
            else:
                flat_manifest[key] = value
        wandb.log(flat_manifest)
