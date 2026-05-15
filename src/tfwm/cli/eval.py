"""Evaluation CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from tfwm.eval.benchmark import (
    benchmark_report,
    evaluate_claims,
    load_json_episodes,
    write_report_artifacts,
)
from tfwm.utils.logging import setup_logging
from tfwm.utils.seeding import set_seed

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run evaluation."""

    set_seed(int(cfg.seed))
    logger = setup_logging()
    eval_cfg = cfg.eval
    data_dir = Path(hydra.utils.to_absolute_path(str(eval_cfg.data_dir)))
    output_dir = Path(hydra.utils.to_absolute_path(str(eval_cfg.output_dir)))
    episodes = load_json_episodes(data_dir, pattern=str(eval_cfg.pattern))
    claims = OmegaConf.to_container(eval_cfg.get("claims", []), resolve=True)
    report = benchmark_report(
        episodes,
        metric_names=list(eval_cfg.metrics),
        bootstrap_samples=int(eval_cfg.bootstrap_samples),
        confidence=float(eval_cfg.confidence),
        seed=int(cfg.seed),
    )
    claim_checks = evaluate_claims(report["metrics"], claims) if claims else []
    json_path, markdown_path = write_report_artifacts(
        report,
        output_dir,
        title=str(eval_cfg.title),
        claim_checks=claim_checks,
    )
    logger.info("Benchmark complete: %s", json_path)
    logger.info("Markdown report: %s", markdown_path)


if __name__ == "__main__":
    main()
