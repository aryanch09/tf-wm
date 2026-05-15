"""Logging utilities with rich and wandb integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

try:
    import wandb
except ImportError:
    wandb = None


console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> logging.Logger:
    """Set up logging with rich console and optional file/wandb.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file to log to.
        wandb_project: Optional wandb project name.
        wandb_entity: Optional wandb entity.
        wandb_mode: Wandb mode (online, offline, disabled).

    Returns:
        Configured logger.
    """
    logger = logging.getLogger("tfwm")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Rich console handler
    rich_handler = RichHandler(console=console, show_time=True, show_level=True)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    # Wandb setup
    if wandb_project and wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            settings=wandb.Settings(start_method="fork"),
        )

    return logger


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to console and wandb.

    Args:
        metrics: Dict of metric names to values.
        step: Optional step number for wandb.
    """
    # Console logging
    metric_str = " | ".join(
        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()
    )
    console.print(f"[bold blue]Metrics:[/bold blue] {metric_str}")

    # Wandb logging
    if wandb:
        wandb.log(metrics, step=step)


def log_artifact(name: str, path: Path | str, type: str = "file") -> None:
    """Log artifact to wandb.

    Args:
        name: Artifact name.
        path: Path to artifact.
        type: Artifact type.
    """
    if wandb:
        artifact = wandb.Artifact(name, type=type)
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
