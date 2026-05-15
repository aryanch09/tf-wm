#!/usr/bin/env python3
"""Run a compact local health check for TF-WM development."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--pytest-args", nargs="*", default=["tests/unit"])
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    _run([sys.executable, "-m", "compileall", "-q", "src", "scripts"])
    if not args.skip_tests:
        _run([sys.executable, "-m", "pytest", *args.pytest_args])


if __name__ == "__main__":
    main()
