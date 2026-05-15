# TF-WM Build Specification

This repository implements **Tactile-First World Models for In-Hand and Occluded
Manipulation (TF-WM)**.

The binding operating contract for future agents is the master prompt supplied by the
project owner. In short:

- Treat tactile contact streams as the primary state variable.
- Keep the canonical `src/tfwm` API contracts stable.
- Use Python 3.11, PyTorch, Hydra/OmegaConf, W&B-compatible manifests, DVC-compatible
  data layout, pytest, ruff, black, mypy, and MkDocs.
- Do not commit secrets, weights, or datasets.
- Every public module should have mirrored tests and reproducibility-aware behavior.
- Prefer conservative, typed, documented implementations over speculative dependencies.

## Binding Core Contracts

- `tfwm.types.Tactile`: `(batch, time, taxel, channel)`.
- `tfwm.types.TactileLatent`: `(batch, time, d_z)`.
- `TactileSensor.read() -> Tactile`.
- `Dynamics.rollout(z0, actions, context=None) -> (z_pred, aux)`.
- Gates return query scores in `[0, 1]` and expose thresholded query decisions.
- Planners return the first-step action with shape `(B, d_a)`.

## Implementation Order

1. Foundations: packaging, configs, types, utilities, data modules.
2. Sim and sensors: tactile sensor abstractions, base envs, noise models, synthetic data.
3. Models: tactile encoders, dynamics, decoders, gates, world-model wiring.
4. Training: phase-one representation learning and CLI manifests.
5. Gating and auxiliary heads.
6. Planning and policies.
7. Evaluation suite and reporting.
8. Real robot deployment loop.

When adding dependencies, update `pyproject.toml` intentionally and document why.
