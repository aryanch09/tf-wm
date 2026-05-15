#!/usr/bin/env bash
# Launch a Hydra multi-run sweep over dynamics and gating variants.
# Requires submitit (SLURM) or runs locally with --multirun.
set -euo pipefail

uv run python -m tfwm.cli.train \
  --multirun \
  model/dynamics=deterministic,vae,ensemble \
  model/gating=variance,voi,hybrid \
  task=inhand_reorient \
  seed=0,1,2 \
  "hydra/launcher=basic"
