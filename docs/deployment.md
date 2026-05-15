# Deployment

Deployment is the path from research checkpoints to a real-time control loop.

## Release Checklist

- model checkpoint exported with matching config;
- normalization statistics stored with the release;
- sensor calibration version recorded;
- action limits checked against the target hand;
- planning latency benchmarked on target hardware;
- fallback policy defined for sensor dropouts and force violations.

## Runtime Loop

1. Read tactile, proprioceptive, and optional camera observations.
2. Normalize observations with release statistics.
3. Update latent state through the tactile encoder and dynamics model.
4. Estimate value of information for vision.
5. Query vision only when the gating policy requests it.
6. Plan actions with MPC and enforce safety constraints.
7. Log observations, actions, rewards, and diagnostics.

The deployment module should stay conservative: predictable failure behavior is
more important than squeezing out one more benchmark point.
