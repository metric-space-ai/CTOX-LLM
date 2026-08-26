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

from run_teacher_batches import cache_environment


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
    parser.add_argument("--cpu-offload-memory-gib", type=int, default=96)
    parser.add_argument("--mtp-device", default="cuda:1")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    args = parser.parse_args()

    batch_plan_bytes = args.batch_plan.read_bytes()
    batch_plan = json.loads(batch_plan_bytes)
    if batch_plan.get("format") != "ctox.activation-batch-plan.v1":
        raise SystemExit("unsupported activation batch-plan format")
    if sha256(args.input) != batch_plan.get("input_sha256"):
        raise SystemExit("activation input does not match the batch plan")
    quant_plan_sha256 = sha256(args.plan)
    provenance_sha256 = sha256(args.local_model_provenance)
    batch_plan_sha256 = hashlib.sha256(batch_plan_bytes).hexdigest()
    batches = batch_plan["batches"]
    end_batch = len(batches) if args.end_batch is None else args.end_batch
    if not 0 <= args.start_batch < end_batch <= len(batches):
        raise SystemExit("invalid activation batch range")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = cache_environment(os.environ, args.hf_home)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for batch in batches[args.start_batch:end_batch]:
        index = int(batch["batch_index"])
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
            str(args.gpu_weight_memory_gib),
            "--cpu-offload-memory-gib",
            str(args.cpu_offload_memory_gib),
            "--mtp-device",
            args.mtp_device,
            "--prefill-chunk-tokens",
            str(args.prefill_chunk_tokens),
        ]
        print(
            f"batch={index} status=collect-start samples={batch['samples']} "
            f"tokens={batch['sequence_tokens']}",
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
