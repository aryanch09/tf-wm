#!/usr/bin/env bash
# Export encoder + dynamics to ONNX for embedded deployment.
set -euo pipefail

CHECKPOINT=${1:-outputs/phase4/checkpoint_final.pt}
OUT_DIR=${2:-outputs/onnx}

echo "==> Exporting to ONNX from checkpoint: ${CHECKPOINT}"
mkdir -p "${OUT_DIR}"

uv run python -c "
import torch
from pathlib import Path
from tfwm.models.tactile_encoder.hybrid import HybridTactileEncoder
from tfwm.deploy.quantize import export_onnx

enc = HybridTactileEncoder(n_taxels=16, n_channels=1, d_latent=128, tcn_hidden=128, d_model=128, n_heads=4)
enc.eval()

dummy = torch.randn(1, 10, 16, 1)

class EncoderMuOnly(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def forward(self, x): return self.enc(x)['mu']

export_onnx(
    EncoderMuOnly(enc),
    (dummy,),
    path=Path('${OUT_DIR}/encoder.onnx'),
    input_names=['tactile'],
    output_names=['latent_mu'],
    dynamic_axes={'tactile': {0: 'batch', 1: 'time'}, 'latent_mu': {0: 'batch', 1: 'time'}},
)
print('ONNX export complete: ${OUT_DIR}/encoder.onnx')
"
