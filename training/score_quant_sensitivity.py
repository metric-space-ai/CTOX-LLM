#!/usr/bin/env python3
"""Score activation-weighted Q2 and Q4 errors for every planned matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from quantization import quantize_dequantize
from run_ledger import GpuRun, require_budget


def packed_bytes(dtype: str, elements: int) -> int:
    block_bytes = 18 if dtype == "q2_b64" else 34
    return math.ceil(elements / 64) * block_bytes


def fixed_q4(name: str) -> bool:
    return (
        name == "lm_head.weight"
        or (name.startswith("mtp.") and ".self_attn." in name)
        or name.endswith(".self_attn.k_proj.weight")
        or name.endswith(".self_attn.v_proj.weight")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_tensor(torch: Any, source: Any, name: str, input_mean_sq: Any, output_mean_sq: Any, device: str, rows_per_chunk: int) -> dict[str, float]:
    tensor_slice = source.get_slice(name)
    rows, columns = tensor_slice.get_shape()
    input_energy = input_mean_sq.to(device=device, dtype=torch.float32)
    signal_energy = float(output_mean_sq.double().sum().item())
    errors = {"q2_b64": 0.0, "q4_b64": 0.0}
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        weight = tensor_slice[row_start:row_end].to(device=device, dtype=torch.float32)
        for dtype in errors:
            reconstructed = quantize_dequantize(torch, weight, dtype)
            difference = reconstructed - weight
            errors[dtype] += float((difference.square() * input_energy).double().sum().item())
            del reconstructed, difference
        del weight
    denominator = max(signal_energy, 1e-30)
    return {
        "q2_expected_error": errors["q2_b64"],
        "q4_expected_error": errors["q4_b64"],
        "q2_relative_error": errors["q2_b64"] / denominator,
        "q4_relative_error": errors["q4_b64"] / denominator,
        "quality_gain": max(errors["q2_b64"] - errors["q4_b64"], 0.0),
        "signal_energy": signal_energy,
    }


def run(args: argparse.Namespace, torch: Any, safe_open: Any) -> None:
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    quantized = [
        entry
        for entry in plan["tensors"]
        if entry["source_shard"] is not None and entry["dtype"] in {"q2_b64", "q4_b64"}
    ]
    shards = sorted({entry["source_shard"] for entry in quantized})
    candidates = []
    with ExitStack() as stack:
        stats = stack.enter_context(safe_open(args.stats, framework="pt", device="cpu"))
        sources = {
            shard: stack.enter_context(
                safe_open(args.checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in shards
        }
        stats_keys = set(stats.keys())
        for index, entry in enumerate(quantized, 1):
            name = entry["name"]
            elements = math.prod(entry["shape"])
            candidate: dict[str, Any] = {
                "name": name,
                "q2_bytes": packed_bytes("q2_b64", elements),
                "q4_bytes": packed_bytes("q4_b64", elements),
                "fixed_q4": fixed_q4(name),
                "planned_dtype": entry["dtype"],
            }
            input_key = f"{name}.input_mean_sq"
            output_key = f"{name}.output_mean_sq"
            if input_key not in stats_keys or output_key not in stats_keys:
                candidate.update({"observed": False, "quality_gain": 0.0})
            else:
                candidate.update(
                    score_tensor(
                        torch,
                        sources[entry["source_shard"]],
                        name,
                        stats.get_tensor(input_key),
                        stats.get_tensor(output_key),
                        args.device,
                        args.rows_per_chunk,
                    )
                )
                candidate["observed"] = True
            candidates.append(candidate)
            print(f"[{index}/{len(quantized)}] {name}", flush=True)
    document = {
        "format": "ctox.q2q4.sensitivity.v1",
        "model": plan["model"],
        "revision": plan["revision"],
        "activation_stats_sha256": file_sha256(args.stats),
        "estimator": "diagonal-input-covariance",
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    args = parser.parse_args()
    if args.rows_per_chunk <= 0:
        raise SystemExit("--rows-per-chunk must be positive")
    require_budget(args.ledger, args.reserved_gpu_hours)
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before scoring sensitivity") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")
    with GpuRun(args.ledger, "quant-sensitivity", 1, sys.argv):
        run(args, torch, safe_open)


if __name__ == "__main__":
    main()
