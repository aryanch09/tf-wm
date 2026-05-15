# Training Recipes

TF-WM uses phased training so tactile representations, latent dynamics, and
gating policies can be debugged independently.

## Phase 1: Representation Learning

Goal: learn tactile encoders and dynamics on dense tactile streams.

```bash
tfwm train phase=phase1 task=inhand_reorient
```

Recommended checks:

- tactile reconstruction loss decreases on train and validation splits;
- latent rollout error is stable across sequence lengths;
- contact and slip event probes correlate with generated labels.

## Phase 2: Vision Gating

Goal: query vision only when it reduces uncertainty or improves task value.

```bash
tfwm train phase=phase2 task=inhand_reorient
```

Recommended checks:

- camera-query rate drops against a vision-always baseline;
- success does not collapse under occlusion;
- gating decisions are calibrated against uncertainty spikes and slip events.

## Phase 3: Planning

Goal: use the learned world model in latent MPC.

```bash
tfwm eval benchmark=inhand_reorient_occluded
```

Recommended checks:

- CEM action candidates stay within hardware limits;
- planning horizon is long enough to improve reward but short enough for
  real-time control;
- failures are logged with tactile traces and query decisions.
