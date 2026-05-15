"""Training CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils.logging import setup_logging
from ..utils.reproducibility import generate_manifest, save_manifest
from ..utils.seeding import set_seed

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra config.
    """
    # Set up
    set_seed(cfg.seed)
    logger = setup_logging()

    # Generate manifest
    manifest = generate_manifest(
        run_name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        output_dir=Path(str(cfg.output_dir)),
        seed=cfg.seed,
    )
    save_manifest(manifest, Path(str(cfg.output_dir)))

    logger.info(f"Starting training with config: {cfg}")

    logger.info("Training manifest written. Use the training loop APIs for supervised smoke runs.")


if __name__ == "__main__":
    main()
