"""Data collection CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tfwm.utils.logging import setup_logging
from tfwm.utils.seeding import set_seed

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run data collection."""

    set_seed(int(cfg.seed))
    logger = setup_logging()
    logger.info(
        "Collection entrypoint initialized; configure a simulator backend to record rollouts."
    )


if __name__ == "__main__":
    main()
