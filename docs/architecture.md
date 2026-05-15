# Architecture

TF-WM is organized around tactile-first world modeling for manipulation under
occlusion. The code is split into narrow modules so research ideas can move
without rewriting the entire stack.

## Package Boundaries

- `tfwm.sensors`: typed sensor abstractions, including taxel arrays.
- `tfwm.data`: schemas, episodes, collation, datasets, loaders, augmentation,
  and domain randomization.
- `tfwm.sim`: simulator-facing environments, tactile sensor models, and plugin
  adapters.
- `tfwm.models`: tactile and vision encoders, latent dynamics, affordance
  models, and gating networks.
- `tfwm.planning`: model-predictive control and CEM planners.
- `tfwm.train`: training loops and phase orchestration.
- `tfwm.eval`: metrics, benchmark runners, and result aggregation.
- `tfwm.deploy`: release packaging and deployment checks.

## Data Flow

1. Rollouts produce timestamped observations, actions, rewards, and dones.
2. Episodes are validated through the Pydantic schema in `tfwm.data.schema`.
3. Dataset wrappers convert episodes into tensors for sequence training.
4. Encoders map tactile, vision, and proprioceptive observations into latent
   states.
5. Dynamics predict future latent states under candidate action sequences.
6. Gating modules decide when vision queries are worth their cost.
7. MPC chooses actions that maximize task reward while respecting uncertainty.

## Research Surface

The intended extension points are tactile representation learning, uncertainty
calibration, vision-query value estimation, and sim-to-real transfer. Each of
those can be changed independently if the schema and tensor contracts stay
stable.
