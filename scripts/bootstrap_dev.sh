#!/usr/bin/env bash
# Bootstrap development environment.
# Usage: bash scripts/bootstrap_dev.sh
set -euo pipefail

echo "==> Installing uv..."
pip install --quiet uv

echo "==> Syncing project dependencies..."
uv sync

echo "==> Installing pre-commit hooks..."
uv run pre-commit install

echo "==> Running smoke test..."
uv run python -c "
import torch
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
enc = HybridTactileEncoder(n_taxels=8, n_channels=1, d_latent=32, tcn_hidden=32, d_model=32, n_heads=4)
x = torch.randn(2, 5, 8, 1)
out = enc(x)
print('Encoder smoke test: mu shape =', out['mu'].shape)
print('Bootstrap COMPLETE')
"
