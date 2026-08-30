#!/usr/bin/env python3
"""Run and summarize the exact 10K BF16 teacher probe on physical GPU1+2."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from recovery_io import atomic_json


FORMAT = "ctox.teacher-throughput-probe-result.v1"
PROBE_SELECTION_FORMAT = "ctox.teacher-throughput-probe-selection.v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    result = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        result.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
                "memory_used_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
                "driver_version": fields[6],
            }
        )
    if {record["index"] for record in result} < {0, 1, 2}:
        raise ValueError("probe host does not expose physical GPU0, GPU1, and GPU2")
    return result


def ledger_entries(path: Path, cache_paths: set[str]) -> list[dict[str, Any]]:
    matched = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            command = [str(value) for value in record.get("command", [])]
            output = None
            if "--output" in command:
                position = command.index("--output")
                if position + 1 < len(command):
                    output = str(Path(command[position + 1]).resolve())
            if record.get("stage") == "teacher-cache" and output in cache_paths:
                matched.append(record)
    if not matched or any(not record.get("success") for record in matched):
        raise ValueError("probe teacher ledger is absent or contains a failed run")
    return matched


def summarize(
    selection: dict[str, Any],
    batch_plan: dict[str, Any],
    output_root: Path,
    prefix: str,
    ledger: Path,
) -> dict[str, Any]:
    batches = batch_plan.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("probe batch plan has no batches")
    if int(batch_plan.get("summary", {}).get("samples", -1)) != 10_000:
        raise ValueError("probe batch plan is not exactly 10,000 samples")
    if int(selection.get("records", -1)) != 10_000:
        raise ValueError("probe selection is not exactly 10,000 samples")
    verification_records = []
    cache_paths = set()
    maximum_allocated = [0, 0]
    maximum_reserved = [0, 0]
    artifact_bytes = 0
    verified_samples = 0
    for batch in batches:
        index = int(batch["batch_index"])
        cache = output_root / f"{prefix}-batch-{index:03d}-v1"
        verification_path = (
            output_root / f"{prefix}-batch-{index:03d}-v1-verification-v1.json"
        )
        run_path = cache / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if verification.get("status") != "passed":
            raise ValueError(f"probe batch {index} did not pass verification")
        if int(verification.get("samples", -1)) != int(batch["samples"]):
            raise ValueError(f"probe batch {index} sample count differs")
        memory = run.get("cuda_memory")
        if not isinstance(memory, list) or len(memory) != 2:
            raise ValueError(f"probe batch {index} has no two-GPU CUDA memory evidence")
        for device in memory:
            logical = int(device["index"])
            if logical not in (0, 1):
                raise ValueError("probe run used a logical CUDA device outside GPU1+2")
            maximum_allocated[logical] = max(
                maximum_allocated[logical], int(device["peak_allocated_bytes"])
            )
            maximum_reserved[logical] = max(
                maximum_reserved[logical], int(device["peak_reserved_bytes"])
            )
        verified_samples += int(verification["samples"])
        artifact_bytes += int(verification["artifact_bytes"])
        cache_paths.add(str(cache.resolve()))
        verification_records.append(
            {
                "batch_index": index,
                "run_sha256": sha256_path(run_path),
                "verification_sha256": sha256_path(verification_path),
                "samples": int(verification["samples"]),
                "artifact_bytes": int(verification["artifact_bytes"]),
            }
        )
    if verified_samples != 10_000:
        raise ValueError("verified probe cache does not contain exactly 10,000 samples")
    entries = ledger_entries(ledger, cache_paths)
    elapsed_seconds = sum(float(record["elapsed_seconds"]) for record in entries)
    gpu_hours = sum(float(record["gpu_hours"]) for record in entries)
    if elapsed_seconds <= 0:
        raise ValueError("probe elapsed time is not positive")
    sequence_tokens = int(selection["sequence_tokens"])
    scale = 1_000_000 / verified_samples
    return {
        "verified_samples": verified_samples,
        "sequence_tokens": sequence_tokens,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": verified_samples / elapsed_seconds,
        "sequence_tokens_per_second": sequence_tokens / elapsed_seconds,
        "gpu_hours": gpu_hours,
        "artifact_bytes": artifact_bytes,
        "bytes_per_sample": artifact_bytes / verified_samples,
        "peak_allocated_bytes_by_logical_device": maximum_allocated,
        "peak_reserved_bytes_by_logical_device": maximum_reserved,
        "projected_million": {
            "elapsed_seconds": elapsed_seconds * scale,
            "gpu_hours": gpu_hours * scale,
            "artifact_bytes": int(artifact_bytes * scale),
            "linear_projection_only": True,
        },
        "batches": verification_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        selection_bytes = args.selection_evidence.read_bytes()
        selection = json.loads(selection_bytes)
        if selection.get("format") != PROBE_SELECTION_FORMAT:
            raise ValueError("unsupported teacher probe selection")
        materialized = Path(selection["materialized"])
        if sha256_path(materialized) != selection["materialized_sha256"]:
            raise ValueError("teacher probe materialized input changed")
        batch_plan_bytes = args.batch_plan.read_bytes()
        batch_plan = json.loads(batch_plan_bytes)
        before = gpu_inventory()
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_teacher_batches.py")),
            "--batch-plan",
            str(args.batch_plan),
            "--input",
            str(materialized),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--local-model-provenance",
            str(args.local_model_provenance),
            "--teacher-provenance-sha256",
            args.teacher_provenance_sha256,
            "--output-root",
            str(args.output_root),
            "--output-prefix",
            args.output_prefix,
            "--ledger",
            str(args.ledger),
            "--hf-home",
            str(args.hf_home),
            "--gpus",
            "2",
            "--physical-gpus",
            "1,2",
            "--reserved-gpu-hours",
            str(args.reserved_gpu_hours),
            "--gpu-weight-memory-gib",
            "14",
            "--long-context-gpu-weight-memory-gib",
            "10",
            "--long-context-threshold-tokens",
            "65536",
            "--cpu-offload-memory-gib",
            "96",
            "--mtp-device",
            "cuda:1",
            "--prefill-chunk-tokens",
            "512",
            "--start-batch",
            "0",
            "--end-batch",
            str(len(batch_plan["batches"])),
        ]
        if args.resume_incomplete:
            command.append("--resume-incomplete")
        started = time.time()
        subprocess.run(command, check=True)
        ended = time.time()
        after = gpu_inventory()
        measurement = summarize(
            selection, batch_plan, args.output_root, args.output_prefix, args.ledger
        )
        document = {
            "format": FORMAT,
            "status": "passed",
            "selection_evidence": str(args.selection_evidence.resolve()),
            "selection_evidence_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "batch_plan": str(args.batch_plan.resolve()),
            "batch_plan_sha256": hashlib.sha256(batch_plan_bytes).hexdigest(),
            "physical_gpus": [1, 2],
            "reserved_for_greppy": [0],
            "cuda_visible_devices": "1,2",
            "gpu_weight_budget_bytes_each": 14 * 1024**3,
            "mtp_device": "cuda:1",
            "invocation_started_unix": started,
            "invocation_ended_unix": ended,
            "gpu_inventory_before": before,
            "gpu_inventory_after": after,
            "measurement": measurement,
        }
        atomic_json(args.output, document)
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
