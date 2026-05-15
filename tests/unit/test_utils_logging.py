"""Tests for logging utilities."""

import tempfile
from pathlib import Path

from tfwm.utils.logging import setup_logging


def test_setup_logging_basic():
    """Test basic logging setup."""
    logger = setup_logging(level="INFO")
    assert logger.level == 20  # INFO level


def test_setup_logging_with_file():
    """Test logging with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logging(level="DEBUG", log_file=log_file)

        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content


def test_setup_logging_wandb_disabled():
    """Test logging setup when wandb not available."""
    # Mock wandb not available
    import tfwm.utils.logging as log_mod

    original_wandb = log_mod.wandb
    log_mod.wandb = None

    try:
        logger = setup_logging(wandb_project="test")
        # Should not raise
        assert logger is not None
    finally:
        log_mod.wandb = original_wandb
