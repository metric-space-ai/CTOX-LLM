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


def row_group_document(
    group_index: int,
    row_start: int,
    row_end: int,
    columns: int,
    q2_error: float,
    q4_error: float,
    signal_energy: float,
) -> dict[str, Any]:
    elements = (row_end - row_start) * columns
    denominator = max(signal_energy, 1e-30)
    return {
        "group_index": group_index,
        "row_start": row_start,
        "row_end": row_end,
        "q2_bytes": packed_bytes("q2_b64", elements),
        "q4_bytes": packed_bytes("q4_b64", elements),
        "q2_expected_error": q2_error,
        "q4_expected_error": q4_error,
        "q2_relative_error": q2_error / denominator,
        "q4_relative_error": q4_error / denominator,
        "quality_gain": max(q2_error - q4_error, 0.0),
        "signal_energy": signal_energy,
    }


def score_tensor(
    torch: Any,
    source: Any,
    name: str,
    input_mean_sq: Any | None,
    output_mean_sq: Any | None,
    row_count: Any | None,
    device: str,
    rows_per_chunk: int,
    row_group_size: int | None = None,
) -> dict[str, Any]:
    tensor_slice = source.get_slice(name)
    rows, columns = tensor_slice.get_shape()
    if (input_mean_sq is None) == (row_count is None):
        raise ValueError(f"{name} requires exactly one activation weighting mode")
    input_energy = (
        input_mean_sq.to(device=device, dtype=torch.float32)
        if input_mean_sq is not None
        else None
    )
    row_frequency = (
        row_count.to(device=device, dtype=torch.float64) if row_count is not None else None
    )
    signal_energy = (
        float(output_mean_sq.double().sum().item()) if output_mean_sq is not None else 0.0
    )
    analytic_signal_energy = 0.0
    errors = {"q2_b64": 0.0, "q4_b64": 0.0}
    group_accumulators: dict[int, dict[str, float | int]] = {}
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        weight = tensor_slice[row_start:row_end].to(device=device, dtype=torch.float32)
        if input_energy is not None:
            row_signal = (weight.square() * input_energy).double().sum(dim=1)
        else:
            frequencies = row_frequency[row_start:row_end]
            row_signal = weight.double().square().sum(dim=1) * frequencies
        analytic_signal_energy += float(row_signal.sum().item())
        for dtype in errors:
            reconstructed = quantize_dequantize(torch, weight, dtype)
            difference = reconstructed - weight
            if input_energy is not None:
                row_error = (difference.square() * input_energy).double().sum(dim=1)
            else:
                row_error = difference.double().square().sum(dim=1) * frequencies
            errors[dtype] += float(row_error.sum().item())
            if row_group_size is not None:
                first_group = row_start // row_group_size
                last_group = (row_end - 1) // row_group_size
                for group_index in range(first_group, last_group + 1):
                    state = group_accumulators.setdefault(
                        group_index,
                        {
                            "row_start": group_index * row_group_size,
                            "row_end": min((group_index + 1) * row_group_size, rows),
                            "q2_b64": 0.0,
                            "q4_b64": 0.0,
                            "signal": 0.0,
                        },
                    )
                    local_start = max(group_index * row_group_size, row_start) - row_start
                    local_end = min((group_index + 1) * row_group_size, row_end) - row_start
                    state[dtype] = float(state[dtype]) + float(
                        row_error[local_start:local_end].sum().item()
                    )
            del reconstructed, difference
        if row_group_size is not None:
            first_group = row_start // row_group_size
            last_group = (row_end - 1) // row_group_size
            for group_index in range(first_group, last_group + 1):
                state = group_accumulators[group_index]
                local_start = max(group_index * row_group_size, row_start) - row_start
                local_end = min((group_index + 1) * row_group_size, row_end) - row_start
                state["signal"] = float(state["signal"]) + float(
                    row_signal[local_start:local_end].sum().item()
                )
        del weight
    if output_mean_sq is None:
        signal_energy = analytic_signal_energy
    denominator = max(signal_energy, 1e-30)
    result: dict[str, Any] = {
        "q2_expected_error": errors["q2_b64"],
        "q4_expected_error": errors["q4_b64"],
        "q2_relative_error": errors["q2_b64"] / denominator,
        "q4_relative_error": errors["q4_b64"] / denominator,
        "quality_gain": max(errors["q2_b64"] - errors["q4_b64"], 0.0),
        "signal_energy": signal_energy,
    }
    if row_group_size is not None:
        result["row_group_size"] = row_group_size
        result["row_groups"] = [
            row_group_document(
                group_index,
                int(state["row_start"]),
                int(state["row_end"]),
                columns,
                float(state["q2_b64"]),
                float(state["q4_b64"]),
                float(state["signal"]),
            )
            for group_index, state in sorted(group_accumulators.items())
        ]
    return result


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
            row_count_key = f"{name}.row_count"
            has_input = input_key in stats_keys
            has_rows = row_count_key in stats_keys
            if not has_input and not has_rows:
                candidate.update({"observed": False, "quality_gain": 0.0})
            else:
                group_size = (
                    args.row_group_size
                    if name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}
                    else None
                )
                candidate.update(
                    score_tensor(
                        torch,
                        sources[entry["source_shard"]],
                        name,
                        stats.get_tensor(input_key) if has_input else None,
                        stats.get_tensor(output_key) if output_key in stats_keys else None,
                        stats.get_tensor(row_count_key) if has_rows else None,
                        args.device,
                        args.rows_per_chunk,
                        group_size,
                    )
                )
                candidate["observed"] = True
                candidate["activation_weighting"] = (
                    "token-row-frequency" if has_rows else "diagonal-input-covariance"
                )
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
    parser.add_argument("--row-group-size", type=int, default=256)
    args = parser.parse_args()
    if args.rows_per_chunk <= 0:
        raise SystemExit("--rows-per-chunk must be positive")
    if args.row_group_size <= 0:
        raise SystemExit("--row-group-size must be positive")
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
