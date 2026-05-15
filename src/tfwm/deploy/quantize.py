"""Model quantisation for edge deployment.

Converts trained TF-WM encoder + dynamics to INT8 or BF16 for deployment
on embedded hardware (NVIDIA Jetson, Orin, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def quantize_dynamic(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Apply dynamic quantisation to linear layers.

    Args:
        model: The module to quantise.
        dtype: Quantisation dtype (``torch.qint8`` or ``torch.float16``).

    Returns:
        Quantised module.
    """
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


def export_torchscript(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    path: Path,
) -> None:
    """Trace and export a module to TorchScript.

    Args:
        model: Module to export (must be in eval mode).
        example_inputs: Tuple of example tensors for tracing.
        path: Output ``.pt`` file path.
    """
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)
    torch.jit.save(traced, str(path))


def export_onnx(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    path: Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, Any] | None = None,
    opset_version: int = 17,
) -> None:
    """Export a module to ONNX format.

    Args:
        model: Module in eval mode.
        example_inputs: Tuple of example input tensors.
        path: Output ``.onnx`` file path.
        input_names: ONNX input node names.
        output_names: ONNX output node names.
        dynamic_axes: Dynamic axis specification.
        opset_version: ONNX opset version.
    """
    model.eval()
    torch.onnx.export(
        model,
        example_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
