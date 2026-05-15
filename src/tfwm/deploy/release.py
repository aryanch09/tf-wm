"""Deployment packaging utilities."""

from __future__ import annotations

from pathlib import Path


def create_deployment_package(source_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "tfwm_package.zip"
    return package_path
