"""Deployment CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tfwm.utils.logging import setup_logging

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run deployment."""

    del cfg
    logger = setup_logging()
    logger.info("Deployment entrypoint initialized; provide a release bundle for hardware rollout.")


if __name__ == "__main__":
    main()
