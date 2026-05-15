# TF-WM: Tactile-First World Models for In-Hand and Occluded Manipulation

[![CI](https://github.com/aryanch09/tf-wm/actions/workflows/ci.yml/badge.svg)](https://github.com/aryanch09/tf-wm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/tf-wm/tfwm/branch/main/graph/badge.svg)](https://codecov.io/gh/tf-wm/tfwm)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://aryanch09.github.io/tf-wm/)

A publication-grade robotics research codebase implementing tactile-first world models for manipulation tasks under occlusion and limited vision.

## Key Features

- **Tactile-First Dynamics**: Learn world models primarily from high-frequency tactile streams, using vision only when gated by uncertainty/value-of-information.
- **Latent MPC Planning**: Cross-entropy method (CEM) and model-predictive control in tactile latent space.
- **Sim-to-Real Transfer**: Domain randomization, noise modeling, and adversarial alignment for real-world deployment.
- **Modular Architecture**: Extensible sensor abstractions, pluggable encoders/dynamics/policies.
- **Reproducible Research**: Hydra configs, seeded RNGs, automatic reproducibility manifests.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/aryanch09/tf-wm.git
cd tf-wm
uv sync
pre-commit install
```

## Quick Start

Train a tactile encoder and dynamics model on simulated in-hand reorientation:

```bash
# Generate a local synthetic dataset and benchmark report
python scripts/generate_synthetic_dataset.py --episodes 12 --steps 48
python scripts/run_benchmark.py

# Phase 1: Representation learning
tfwm train phase=phase1 task=inhand_reorient

# Phase 2: Gating
tfwm train phase=phase2 task=inhand_reorient

# Evaluate through the Hydra CLI
tfwm eval eval.data_dir=data/raw/synthetic
```

## Project Structure

```
tf-wm/
├── src/tfwm/                 # Core package
│   ├── sensors/              # Tactile/vision/proprio sensor abstractions
│   ├── data/                 # Episode buffers, datasets, collation
│   ├── sim/                  # Simulation environments
│   ├── models/               # Encoders, dynamics, decoders, gating
│   ├── planning/             # MPC, hierarchical planners
│   ├── policies/             # BC, SAC, PPO policies
│   ├── train/                # Training phases and callbacks
│   ├── eval/                 # Benchmarks and metrics
│   ├── deploy/               # Real-time deployment
│   └── utils/                # Seeding, logging, reproducibility
├── configs/                  # Hydra configurations
├── benchmarks/               # Benchmark cards and claim targets
├── data/                     # Local raw/processed/split workspace
├── notebooks/                # Analysis notebooks
├── tests/                    # Unit, integration, property tests
├── docs/                     # MkDocs documentation
└── scripts/                  # Utility scripts
```

## Research Claims

We demonstrate that tactile-first world models achieve:

- **H-A**: ≥15% absolute success-rate improvement over vision-first baselines on occluded in-hand reorientation.
- **H-B**: VOI gating reduces camera queries by ≥40% with <5% success-rate drop.
- **H-C**: Sim-to-real gap ≤10% with ≤500 real fine-tuning episodes.

Each claim must be backed by the benchmark report artifacts described in
`docs/benchmarks.md` and `benchmarks/inhand_reorient_occluded.yaml`.

## Citation

```bibtex
@misc{tfwm2026,
  title={Tactile-First World Models for In-Hand and Occluded Manipulation},
  author={TF-WM Contributors},
  year={2026},
  url={https://github.com/aryanch09/tf-wm}
}
```

## License

Apache 2.0
