"""Top-level TF-WM command dispatcher."""

from __future__ import annotations

import sys
from collections.abc import Callable


def _run_hydra_entrypoint(entrypoint: Callable[[], None], argv: list[str]) -> None:
    original = sys.argv[:]
    try:
        sys.argv = [original[0], *argv]
        entrypoint()
    finally:
        sys.argv = original


def main() -> None:
    """Dispatch ``tfwm <command>`` to the matching CLI module."""

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = "train, collect, eval, deploy, viz"
        raise SystemExit(f"Usage: tfwm <command> [overrides]\nAvailable commands: {commands}")

    command = sys.argv[1]
    args = sys.argv[2:]
    if command == "train":
        from tfwm.cli.train import main as train_main

        _run_hydra_entrypoint(train_main, args)
        return
    if command == "collect":
        from tfwm.cli.collect import main as collect_main

        _run_hydra_entrypoint(collect_main, args)
        return
    if command == "eval":
        from tfwm.cli.eval import main as eval_main

        _run_hydra_entrypoint(eval_main, args)
        return
    if command == "deploy":
        from tfwm.cli.deploy import main as deploy_main

        _run_hydra_entrypoint(deploy_main, args)
        return
    if command == "viz":
        from tfwm.cli.viz import main as viz_main

        _run_hydra_entrypoint(viz_main, args)
        return
    raise SystemExit(f"Unknown tfwm command: {command}")


if __name__ == "__main__":
    main()
