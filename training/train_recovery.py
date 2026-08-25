#!/usr/bin/env python3
"""Layerwise channel-scale recovery for a frozen Q2/Q4 matrix.

Inputs are safetensors with `weight`, optional `bias`, and calibration tensors
`input`/`teacher_output`. This bounded stage is used after sensitivity-driven
quantization and before end-to-end logit distillation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from recovery_modules import ChannelScaleRecovery, reconstruction_loss
from run_ledger import GpuRun, require_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized-layer", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hadamard", action="store_true")
    args = parser.parse_args()
    require_budget(args.ledger, args.reserved_gpu_hours)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise SystemExit("install training/requirements.in before recovery") from error

    device = torch.device("cuda")
    layer = load_file(args.quantized_layer)
    calibration = load_file(args.calibration)
    module = ChannelScaleRecovery(
        layer["weight"].to(device),
        layer.get("bias", None).to(device) if "bias" in layer else None,
        hadamard=args.hadamard,
    ).to(device)
    inputs = calibration["input"].to(device)
    teacher = calibration["teacher_output"].to(device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=0.0)

    with GpuRun(args.ledger, "layer-recovery", args.gpus, sys.argv):
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            student = module(inputs)
            loss = reconstruction_loss(student, teacher)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"step={step} loss={loss.item():.8f}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            module.correction_tensors(),
            args.output,
            metadata={
                "format": "ctox.recovery.channel-scales.v1",
                "hadamard": str(args.hadamard).lower(),
            },
        )


if __name__ == "__main__":
    main()
