# Data Pipeline

The data pipeline is designed to support simulator rollouts, robot logs, and
small synthetic datasets used for local development.

## Episode Schema

Each episode contains:

- `metadata`: task name, simulator or hardware source, seeds, success flag, and
  diagnostic counts.
- `observations`: tactile values, optional vision features, proprioception,
  contact events, and timestamps.
- `actions`: timestamped control vectors.
- `rewards`: scalar reward per step.
- `dones`: terminal flags.

The canonical schema lives in `tfwm.data.schema`.

## Local Development Dataset

Generate a smoke-test dataset:

```bash
python scripts/generate_synthetic_dataset.py --episodes 32 --steps 64
python scripts/inspect_dataset.py data/raw/synthetic
python scripts/make_splits.py --manifest data/raw/synthetic/manifest.csv
```

This creates JSON episodes under `data/raw/synthetic`, a manifest, and split
files under `data/splits`.

## Expected Production Flow

1. Collect raw rollouts into `data/raw/<source>/<run-id>`.
2. Validate every episode against `tfwm.data.schema.Episode`.
3. Produce deterministic split files.
4. Normalize and window sequences into `data/processed/<dataset-id>`.
5. Record preprocessing settings in experiment configs.

Generated datasets should not be committed unless they are tiny fixtures.

## Benchmark Inputs

Benchmark episodes should be immutable once reported. Store the source episodes,
manifest, split files, and generated report together so a table in a paper can
be traced back to exact rollout IDs.
