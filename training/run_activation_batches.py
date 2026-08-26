#!/usr/bin/env python3
"""Collect and verify immutable activation-statistics batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from plan_activation_batches import activation_batches
from run_teacher_batches import cache_environment, gpu_weight_memory_for_batch
from select_activation_calibration import load_jsonl, load_token_counts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_batch_matches(
    artifact: Path,
    verification: dict[str, Any],
    batch: dict[str, Any],
    batch_plan_sha256: str,
    quant_plan_sha256: str,
    model: str,
    revision: str,
    provenance_sha256: str,
) -> bool:
    return (
        verification.get("status") == "passed"
        and verification.get("artifact_sha256") == sha256(artifact)
        and verification.get("batch_plan_sha256") == batch_plan_sha256
        and verification.get("quant_plan_sha256") == quant_plan_sha256
        and verification.get("model") == model
        and verification.get("revision") == revision
        and verification.get("local_model_provenance_sha256") == provenance_sha256
        and int(verification.get("batch_index", -1)) == int(batch["batch_index"])
        and int(verification.get("samples", -1)) == int(batch["samples"])
        and int(verification.get("sequence_tokens", -1))
        == int(batch["sequence_tokens"])
    )


def validate_batch_plan(
    batch_plan: dict[str, Any],
    input_path: Path,
) -> list[dict[str, Any]]:
    """Rebuild the immutable batch schedule from its still-hashed token plans."""

    if batch_plan.get("format") != "ctox.activation-batch-plan.v1":
        raise ValueError("unsupported activation batch-plan format")
    resolved_input = input_path.expanduser().resolve()
    if Path(str(batch_plan.get("input", ""))).expanduser().resolve() != resolved_input:
        raise ValueError("activation input path differs from the batch plan")
    input_bytes = resolved_input.read_bytes()
    if hashlib.sha256(input_bytes).hexdigest() != batch_plan.get("input_sha256"):
        raise ValueError("activation input does not match the batch plan")
    records = load_jsonl(resolved_input)
    record_ids = {str(record["id"]) for record in records}
    if len(record_ids) != len(records):
        raise ValueError("activation input contains duplicate sample ids")

    cache_plan_records = batch_plan.get("cache_plans")
    if not isinstance(cache_plan_records, list) or not cache_plan_records:
        raise ValueError("activation batch plan has no token-count sources")
    cache_plan_paths = []
    for record in cache_plan_records:
        if not isinstance(record, dict):
            raise ValueError("activation token-count source is not an object")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        encoded = path.read_bytes()
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        if actual_sha256 != record.get("sha256"):
            raise ValueError(
                f"activation token-count plan changed: {path} is {actual_sha256}, "
                f"expected {record.get('sha256')}"
            )
        cache_plan_paths.append(path)
    token_counts = load_token_counts(cache_plan_paths, record_ids)

    limits = batch_plan.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("activation batch plan has no limits")
    rebuilt = activation_batches(
        records,
        token_counts,
        int(limits["max_samples"]),
        int(limits["max_batch_tokens"]),
        int(limits["max_sequence_tokens"]),
    )
    if rebuilt != batch_plan.get("batches"):
        raise ValueError("activation batches differ from their token-count sources")
    summary = {
        "batches": len(rebuilt),
        "samples": sum(int(batch["samples"]) for batch in rebuilt),
        "sequence_tokens": sum(int(batch["sequence_tokens"]) for batch in rebuilt),
    }
    if summary != batch_plan.get("summary"):
        raise ValueError("activation batch summary differs from the rebuilt schedule")
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--reserved-gpu-hours", type=float, default=4.0)
    parser.add_argument("--gpu-weight-memory-gib", type=int, default=16)
    parser.add_argument(
        "--long-context-gpu-weight-memory-gib",
        type=int,
        help="deterministic lower weight placement for batches at or above the token threshold",
    )
    parser.add_argument(
        "--long-context-threshold-tokens",
        type=int,
        default=65_536,
    )
    parser.add_argument("--cpu-offload-memory-gib", type=int, default=96)
    parser.add_argument("--mtp-device", default="cuda:1")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    args = parser.parse_args()

    batch_plan_bytes = args.batch_plan.read_bytes()
    batch_plan = json.loads(batch_plan_bytes)
    try:
        batches = validate_batch_plan(batch_plan, args.input)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    quant_plan_sha256 = sha256(args.plan)
    provenance_sha256 = sha256(args.local_model_provenance)
    batch_plan_sha256 = hashlib.sha256(batch_plan_bytes).hexdigest()
    end_batch = len(batches) if args.end_batch is None else args.end_batch
    if not 0 <= args.start_batch < end_batch <= len(batches):
        raise SystemExit("invalid activation batch range")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = cache_environment(os.environ, args.hf_home)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for batch in batches[args.start_batch:end_batch]:
        index = int(batch["batch_index"])
        gpu_weight_memory_gib = gpu_weight_memory_for_batch(
            args.gpu_weight_memory_gib,
            args.long_context_gpu_weight_memory_gib,
            args.long_context_threshold_tokens,
            int(batch["maximum_sample_tokens"]),
        )
        artifact = args.output_root / f"{args.output_prefix}-batch-{index:03d}-v1.safetensors"
        verification_path = (
            args.output_root
            / f"{args.output_prefix}-batch-{index:03d}-v1-verification-v1.json"
        )
        temporary = artifact.with_name(f".{artifact.name}.tmp")
        if artifact.exists() and verification_path.exists():
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            if completed_batch_matches(
                artifact,
                verification,
                batch,
                batch_plan_sha256,
                quant_plan_sha256,
                args.model,
                args.revision,
                provenance_sha256,
            ):
                print(f"batch={index} status=verified-skip", flush=True)
                continue
            raise SystemExit(f"activation batch {index} has non-matching evidence")
        if artifact.exists() or verification_path.exists() or temporary.exists():
            raise SystemExit(
                f"activation batch {index} has incomplete output; preserve and inspect it"
            )

        collect_command = [
            sys.executable,
            str(Path(__file__).with_name("collect_activation_stats.py")),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--local-model-provenance",
            str(args.local_model_provenance),
            "--plan",
            str(args.plan),
            "--input",
            str(args.input),
            "--output",
            str(artifact),
            "--ledger",
            str(args.ledger),
            "--gpus",
            str(args.gpus),
            "--reserved-gpu-hours",
            str(args.reserved_gpu_hours),
            "--max-length",
            str(batch["maximum_sample_tokens"]),
            "--start-sample",
            str(batch["start_sample"]),
            "--max-samples",
            str(batch["samples"]),
            "--use-fla-kernel",
            "--gpu-weight-memory-gib",
            str(gpu_weight_memory_gib),
            "--cpu-offload-memory-gib",
            str(args.cpu_offload_memory_gib),
            "--mtp-device",
            args.mtp_device,
            "--prefill-chunk-tokens",
            str(args.prefill_chunk_tokens),
        ]
        print(
            f"batch={index} status=collect-start samples={batch['samples']} "
            f"tokens={batch['sequence_tokens']} gpu_weight_memory_gib={gpu_weight_memory_gib}",
            flush=True,
        )
        subprocess.run(collect_command, check=True, env=environment)
        verify_command = [
            sys.executable,
            str(Path(__file__).with_name("verify_activation_stats.py")),
            "--artifact",
            str(artifact),
            "--plan",
            str(args.plan),
            "--batch-plan",
            str(args.batch_plan),
            "--input",
            str(args.input),
            "--batch-index",
            str(index),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--local-model-provenance-sha256",
            provenance_sha256,
            "--output",
            str(verification_path),
        ]
        subprocess.run(verify_command, check=True)
        print(f"batch={index} status=verified", flush=True)


if __name__ == "__main__":
    main()
