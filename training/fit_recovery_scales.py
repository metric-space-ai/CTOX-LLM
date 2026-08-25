#!/usr/bin/env python3
"""Fit complete activation-weighted s_in/s_out recovery initializers.

The logical Q2/Q4 codes and their FP16 block scales are regenerated from the
frozen BF16 source on every bounded row chunk. They are never optimized. Only
positive input/output channel corrections are fitted by deterministic
alternating least squares and written using the exact recovery tensor names in
the quantization plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from quantization import quantize_dequantize
from run_ledger import GpuRun, require_budget


QUANTIZED_DTYPES = frozenset({"q2_b64", "q4_b64", "mixed_q2_q4_b64"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantized_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        entry
        for entry in plan["tensors"]
        if entry["source_shard"] is not None and entry["dtype"] in QUANTIZED_DTYPES
    ]


def quant_dtype_ranges(
    entry: dict[str, Any],
    row_start: int,
    row_end: int,
) -> list[tuple[int, int, str]]:
    if not 0 <= row_start < row_end <= entry["shape"][0]:
        raise ValueError(f"invalid row range [{row_start}, {row_end}) for {entry['name']}")
    if entry["dtype"] != "mixed_q2_q4_b64":
        return [(row_start, row_end, entry["dtype"])]
    ranges = []
    covered = row_start
    for segment in entry.get("segments", []):
        start = max(row_start, int(segment["row_start"]))
        stop = min(row_end, int(segment["row_end"]))
        if start >= stop:
            continue
        if start != covered:
            raise ValueError(f"mixed row segments leave a gap in {entry['name']}")
        ranges.append((start, stop, str(segment["dtype"])))
        covered = stop
    if covered != row_end:
        raise ValueError(f"mixed row segments do not cover {entry['name']}")
    return ranges


def dequantized_rows(
    torch: Any,
    source: Any,
    entry: dict[str, Any],
    row_start: int,
    row_end: int,
    device: str,
) -> tuple[Any, Any]:
    weight = source.get_slice(entry["name"])[row_start:row_end].to(
        device=device,
        dtype=torch.float32,
    )
    reconstructed = torch.empty_like(weight)
    for start, stop, dtype in quant_dtype_ranges(entry, row_start, row_end):
        local_start = start - row_start
        local_stop = stop - row_start
        reconstructed[local_start:local_stop] = quantize_dequantize(
            torch,
            weight[local_start:local_stop],
            dtype,
        )
    return weight, reconstructed


def weighted_error_metrics(
    torch: Any,
    source: Any,
    entry: dict[str, Any],
    s_in: Any,
    s_out: Any,
    column_weight: Any | None,
    row_weight: Any | None,
    rows_per_chunk: int,
    device: str,
) -> dict[str, float]:
    error = 0.0
    signal = 0.0
    rows = int(entry["shape"][0])
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(rows, row_start + rows_per_chunk)
        weight, reconstructed = dequantized_rows(
            torch, source, entry, row_start, row_end, device
        )
        student = (
            reconstructed
            * s_in.unsqueeze(0)
            * s_out[row_start:row_end].unsqueeze(1)
        )
        difference_sq = (student - weight).double().square()
        signal_sq = weight.double().square()
        if column_weight is not None:
            weighting = column_weight.double().unsqueeze(0)
        else:
            weighting = row_weight[row_start:row_end].double().unsqueeze(1)
        error += float((difference_sq * weighting).sum().item())
        signal += float((signal_sq * weighting).sum().item())
        del weight, reconstructed, student, difference_sq, signal_sq
    return {
        "weighted_squared_error": error,
        "signal_energy": signal,
        "relative_error": error / max(signal, 1e-30),
    }


def fit_matrix(
    torch: Any,
    source: Any,
    stats: Any,
    stats_keys: set[str],
    entry: dict[str, Any],
    iterations: int,
    rows_per_chunk: int,
    scale_min: float,
    scale_max: float,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    name = entry["name"]
    rows, columns = map(int, entry["shape"])
    input_key = f"{name}.input_mean_sq"
    row_key = f"{name}.row_count"
    has_columns = input_key in stats_keys
    has_rows = row_key in stats_keys
    if has_columns == has_rows:
        raise ValueError(f"{name} requires exactly one activation weighting mode")
    column_weight = (
        stats.get_tensor(input_key).to(device=device, dtype=torch.float32)
        if has_columns
        else None
    )
    row_weight = (
        stats.get_tensor(row_key).to(device=device, dtype=torch.float32)
        if has_rows
        else None
    )
    if column_weight is not None and tuple(column_weight.shape) != (columns,):
        raise ValueError(f"{input_key} shape does not match {name}")
    if row_weight is not None and tuple(row_weight.shape) != (rows,):
        raise ValueError(f"{row_key} shape does not match {name}")

    s_in = torch.ones(columns, device=device, dtype=torch.float32)
    s_out = torch.ones(rows, device=device, dtype=torch.float32)
    baseline = weighted_error_metrics(
        torch,
        source,
        entry,
        s_in,
        s_out,
        column_weight,
        row_weight,
        rows_per_chunk,
        device,
    )
    iteration_evidence = []
    epsilon = torch.finfo(torch.float32).tiny
    for iteration in range(iterations):
        next_out = torch.empty_like(s_out)
        input_numerator = torch.zeros_like(s_in)
        input_denominator = torch.zeros_like(s_in)
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            weight, reconstructed = dequantized_rows(
                torch, source, entry, row_start, row_end, device
            )
            scaled_input = reconstructed * s_in.unsqueeze(0)
            if column_weight is None:
                output_numerator = (weight * scaled_input).sum(dim=1)
                output_denominator = scaled_input.square().sum(dim=1)
            else:
                output_numerator = (
                    weight * scaled_input * column_weight.unsqueeze(0)
                ).sum(dim=1)
                output_denominator = (
                    scaled_input.square() * column_weight.unsqueeze(0)
                ).sum(dim=1)
            output_scale = torch.where(
                output_denominator > epsilon,
                output_numerator / output_denominator,
                torch.ones_like(output_denominator),
            ).clamp(scale_min, scale_max)
            if row_weight is not None:
                output_scale = torch.where(
                    row_weight[row_start:row_end] > 0,
                    output_scale,
                    torch.ones_like(output_scale),
                )
            next_out[row_start:row_end] = output_scale
            scaled_output = reconstructed * output_scale.unsqueeze(1)
            if row_weight is None:
                input_numerator += (scaled_output * weight).sum(dim=0)
                input_denominator += scaled_output.square().sum(dim=0)
            else:
                weights = row_weight[row_start:row_end].unsqueeze(1)
                input_numerator += (scaled_output * weight * weights).sum(dim=0)
                input_denominator += (scaled_output.square() * weights).sum(dim=0)
            del weight, reconstructed, scaled_input, scaled_output
        next_in = torch.where(
            input_denominator > epsilon,
            input_numerator / input_denominator,
            torch.ones_like(input_denominator),
        ).clamp(scale_min, scale_max)
        active_inputs = (
            column_weight > 0
            if column_weight is not None
            else torch.ones_like(next_in, dtype=torch.bool)
        )
        if not bool(active_inputs.any()):
            raise ValueError(f"{name} has no observed input channels")
        next_in = torch.where(active_inputs, next_in, torch.ones_like(next_in))
        normalization = next_in[active_inputs].log().mean().exp()
        next_in = (next_in / normalization).clamp(scale_min, scale_max)
        next_out = (next_out * normalization).clamp(scale_min, scale_max)
        next_in = torch.where(active_inputs, next_in, torch.ones_like(next_in))
        if row_weight is not None:
            next_out = torch.where(row_weight > 0, next_out, torch.ones_like(next_out))
        maximum_log_change = max(
            float((next_in / s_in).log().abs().max().item()),
            float((next_out / s_out).log().abs().max().item()),
        )
        s_in, s_out = next_in, next_out
        iteration_evidence.append(
            {
                "iteration": iteration + 1,
                "maximum_log_scale_change": maximum_log_change,
            }
        )

    # Persisted recovery scales are FP16. Evaluate that exact representation,
    # not the transient FP32 fit.
    stored_in = s_in.to(torch.float16).float()
    stored_out = s_out.to(torch.float16).float()
    recovered = weighted_error_metrics(
        torch,
        source,
        entry,
        stored_in,
        stored_out,
        column_weight,
        row_weight,
        rows_per_chunk,
        device,
    )
    report = {
        "name": name,
        "dtype": entry["dtype"],
        "shape": entry["shape"],
        "activation_weighting": (
            "diagonal_input_covariance" if has_columns else "token_row_frequency"
        ),
        "baseline": baseline,
        "recovered": recovered,
        "error_ratio": recovered["weighted_squared_error"]
        / max(baseline["weighted_squared_error"], 1e-30),
        "iterations": iteration_evidence,
        "s_in_min": float(stored_in.min().item()),
        "s_in_max": float(stored_in.max().item()),
        "s_out_min": float(stored_out.min().item()),
        "s_out_max": float(stored_out.max().item()),
    }
    return stored_in.to(torch.float16).cpu(), stored_out.to(torch.float16).cpu(), report


def run(args: argparse.Namespace, torch: Any, safe_open: Any, save_file: Any) -> None:
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite recovery output or report")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    entries = quantized_entries(plan)
    stop = len(entries) if args.max_tensors is None else args.start_index + args.max_tensors
    selected = entries[args.start_index:stop]
    if not selected:
        raise SystemExit("recovery tensor selection is empty")
    shards = sorted({entry["source_shard"] for entry in selected})
    corrections = {}
    reports = []
    with ExitStack() as stack:
        stats = stack.enter_context(safe_open(args.stats, framework="pt", device="cpu"))
        sources = {
            shard: stack.enter_context(
                safe_open(args.checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in shards
        }
        stats_keys = set(stats.keys())
        for index, entry in enumerate(selected, args.start_index + 1):
            s_in, s_out, evidence = fit_matrix(
                torch,
                sources[entry["source_shard"]],
                stats,
                stats_keys,
                entry,
                args.iterations,
                args.rows_per_chunk,
                args.scale_min,
                args.scale_max,
                args.device,
            )
            corrections[f"{entry['name']}.s_in"] = s_in
            corrections[f"{entry['name']}.s_out"] = s_out
            reports.append(evidence)
            print(
                f"[{index}/{len(entries)}] {entry['name']} error_ratio={evidence['error_ratio']:.8f}",
                flush=True,
            )

    complete = args.start_index == 0 and len(selected) == len(entries)
    expected = {
        f"{entry['name']}.{suffix}"
        for entry in entries
        for suffix in ("s_in", "s_out")
    }
    if complete and set(corrections) != expected:
        raise RuntimeError("complete recovery output does not match planned correction tensors")
    report_document = {
        "format": "ctox.recovery.scale-fit-report.v1",
        "status": "complete" if complete else "partial_smoke",
        "model": plan["model"],
        "revision": plan["revision"],
        "plan_sha256": sha256(args.plan),
        "activation_stats_sha256": sha256(args.stats),
        "algorithm": "activation-weighted-positive-alternating-least-squares",
        "iterations": args.iterations,
        "scale_min": args.scale_min,
        "scale_max": args.scale_max,
        "rows_per_chunk": args.rows_per_chunk,
        "start_index": args.start_index,
        "selected_tensors": len(selected),
        "planned_tensors": len(entries),
        "matrices": reports,
    }
    report_bytes = (json.dumps(report_document, indent=2, sort_keys=True) + "\n").encode()
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_report = args.report.with_name(f".{args.report.name}.tmp")
    save_file(
        corrections,
        temporary_output,
        metadata={
            "format": "ctox.recovery.channel-scales.v2",
            "status": "complete" if complete else "partial_smoke",
            "model": plan["model"],
            "revision": plan["revision"],
            "plan_sha256": report_document["plan_sha256"],
            "activation_stats_sha256": report_document["activation_stats_sha256"],
            "report_sha256": report_sha256,
            "fixed_logical_qcodes": "true",
        },
    )
    temporary_report.write_bytes(report_bytes)
    temporary_output.replace(args.output)
    temporary_report.replace(args.report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tensors", type=int)
    args = parser.parse_args()
    if args.iterations <= 0 or args.rows_per_chunk <= 0:
        raise SystemExit("--iterations and --rows-per-chunk must be positive")
    if not 0 < args.scale_min < 1 < args.scale_max:
        raise SystemExit("recovery scale range must contain 1")
    if args.start_index < 0 or (args.max_tensors is not None and args.max_tensors <= 0):
        raise SystemExit("invalid tensor selection")
    require_budget(args.ledger, args.reserved_gpu_hours)
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise SystemExit("install training/requirements.in before fitting recovery scales") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")
    with GpuRun(args.ledger, "recovery-scale-fit", args.gpus, sys.argv):
        run(args, torch, safe_open, save_file)


if __name__ == "__main__":
    main()
