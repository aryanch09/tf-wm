"""Tests for reproducibility utilities."""

import tempfile
from pathlib import Path

from tfwm.utils.reproducibility import generate_manifest, save_manifest, update_manifest_end_time


def test_generate_manifest():
    """Test manifest generation."""
    config = {"seed": 42, "model": {"d_z": 128}}
    output_dir = Path("/tmp/test")

    manifest = generate_manifest(
        run_name="test_run",
        config=config,
        output_dir=output_dir,
        seed=42,
    )

    assert manifest["run_name"] == "test_run"
    assert manifest["seed"] == 42
    assert "config_hash" in manifest
    assert "package_versions" in manifest


def test_save_and_update_manifest():
    """Test saving and updating manifest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        manifest = {"run_name": "test", "start_time": 1000.0, "end_time": None}

        save_manifest(manifest, output_dir)
        manifest_path = output_dir / "run.yaml"
        assert manifest_path.exists()

        update_manifest_end_time(output_dir)

        # Check end_time was updated
        import yaml

        with open(manifest_path) as f:
            updated = yaml.safe_load(f)
        assert updated["end_time"] is not None
        assert updated["end_time"] > 1000.0
