#!/usr/bin/env python3
"""Create deterministic train/val/test split files from a dataset manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/synthetic/manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(row["path"] for row in rows) + "\n")


def main() -> None:
    args = parse_args()
    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1:
        raise SystemExit("--train and --val must leave a non-empty test split")
    if not args.manifest.exists():
        raise SystemExit(f"Manifest does not exist: {args.manifest}")

    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    train_end = int(len(rows) * args.train)
    val_end = train_end + int(len(rows) * args.val)
    splits = {
        "train.txt": rows[:train_end],
        "val.txt": rows[train_end:val_end],
        "test.txt": rows[val_end:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, split_rows in splits.items():
        _write_split(args.output_dir / filename, split_rows)
        print(f"{filename}: {len(split_rows)} episodes")


if __name__ == "__main__":
    main()
