#!/usr/bin/env bash
# Benchmark CEM-MPC latency (H=20, N=256) on available hardware.
# Expected: <30 ms on A100; logs warning if exceeded.
set -euo pipefail

echo "==> Benchmarking CEM-MPC (H=20, N=256)..."
uv run python -c "
import time, torch, statistics
from tfwm.models.dynamics import RSSMDynamics
from tfwm.planning.cem_mpc import CEMMPCPlanner, CEMMPCConfig
from tfwm.utils.timing import cuda_sync_timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

dyn = RSSMDynamics(d_latent=128, d_action=6).to(device)
cfg = CEMMPCConfig(horizon=20, num_candidates=256, action_dim=6)
planner = CEMMPCPlanner(dyn, cfg)
z = torch.randn(1, 1, 128, device=device)
goal = torch.randn(1, 128, device=device)

# Warm-up.
for _ in range(5):
    planner.plan(z, goal)

times = []
for _ in range(20):
    with cuda_sync_timer('cem') as t:
        planner.plan(z, goal)
    times.append(t.elapsed_ms)

mean_ms = statistics.mean(times)
p95_ms  = sorted(times)[int(0.95 * len(times))]
print(f'CEM mean: {mean_ms:.2f} ms  p95: {p95_ms:.2f} ms')
if mean_ms > 30.0 and device == 'cuda':
    print('WARNING: CEM mean exceeds 30 ms SLA on GPU!')
else:
    print('Latency SLA: PASS')
"
