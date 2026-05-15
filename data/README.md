# TF-WM Data Directory

This directory is the local workspace for datasets used by TF-WM experiments.
Generated or downloaded data should stay out of source control unless it is a
tiny fixture that is required for tests.

## Layout

- `raw/`: source episodes from simulators, robots, or synthetic generators.
- `processed/`: normalized tensors, windowed sequences, cached features, and
  model-ready shards.
- `splits/`: deterministic train/validation/test split files.

## Quick Synthetic Dataset

```bash
python scripts/generate_synthetic_dataset.py --episodes 32 --steps 64
python scripts/inspect_dataset.py data/raw/synthetic
python scripts/make_splits.py --manifest data/raw/synthetic/manifest.csv
```

The synthetic generator writes JSON episodes compatible with
`tfwm.data.episode.Episode.from_json`.
