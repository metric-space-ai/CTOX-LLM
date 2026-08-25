#!/usr/bin/env python3
"""Run and verify immutable teacher-cache batches from an admitted plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def completed_batch_matches(
    run: dict[str, Any],
    verification: dict[str, Any],
    batch: dict[str, Any],
    revision: str,
    provenance_sha256: str,
) -> bool:
    return (
        verification.get("status") == "passed"
        and verification.get("teacher_revision") == revision
        and verification.get("teacher_provenance_sha256") == provenance_sha256
        and int(verification.get("samples", -1)) == int(batch["samples"])
        and run.get("teacher_revision") == revision
        and run.get("local_model_provenance_sha256") == provenance_sha256
        and int(run.get("start_sample", -1)) == int(batch["start_sample"])
        and int(run.get("selected_samples", -1)) == int(batch["samples"])
        and int(run.get("written_samples", -1)) == int(batch["samples"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--reserved-gpu-hours", type=float, default=4.0)
    parser.add_argument("--gpu-weight-memory-gib", type=int, default=16)
    parser.add_argument("--cpu-offload-memory-gib", type=int, default=96)
    parser.add_argument("--mtp-device", default="cuda:2")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    args = parser.parse_args()

    batch_document = json.loads(args.batch_plan.read_text(encoding="utf-8"))
    cache_plan_path = Path(batch_document["cache_plan"])
    cache_plan_bytes = cache_plan_path.read_bytes()
    cache_plan = json.loads(cache_plan_bytes)
    if hashlib.sha256(cache_plan_bytes).hexdigest() != batch_document.get("cache_plan_sha256"):
        raise SystemExit("batch plan cache-plan binding does not match")
    settings = cache_plan["settings"]
    batches = batch_document["batches"]
    end_batch = len(batches) if args.end_batch is None else args.end_batch
    if not 0 <= args.start_batch < end_batch <= len(batches):
        raise SystemExit("invalid batch range")
    args.output_root.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for batch in batches[args.start_batch:end_batch]:
        index = int(batch["batch_index"])
        cache = args.output_root / f"{args.output_prefix}-batch-{index:03d}-v1"
        verification_path = args.output_root / f"{args.output_prefix}-batch-{index:03d}-v1-verification-v1.json"
        run_path = cache / "run.json"
        if verification_path.exists() and run_path.exists():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            if completed_batch_matches(
                run,
                verification,
                batch,
                args.revision,
                args.teacher_provenance_sha256,
            ):
                print(f"batch={index} status=verified-skip", flush=True)
                continue
            raise SystemExit(f"batch {index} has non-matching existing evidence")
        if cache.exists() or verification_path.exists():
            raise SystemExit(f"batch {index} has incomplete existing output; preserve and inspect it")

        cache_command = [
            sys.executable,
            str(Path(__file__).with_name("cache_teacher.py")),
            "--model", args.model,
            "--revision", args.revision,
            "--local-model-provenance", str(args.local_model_provenance),
            "--input", str(args.input),
            "--output", str(cache),
            "--ledger", str(args.ledger),
            "--gpus", str(args.gpus),
            "--reserved-gpu-hours", str(args.reserved_gpu_hours),
            "--top-k", str(settings["top_k"]),
            "--max-length", str(batch["maximum_sample_tokens"]),
            "--hidden-layers", ",".join(str(layer) for layer in settings["hidden_layers"]),
            "--target-mode", "assistant",
            "--marker-window", str(settings["marker_window"]),
            "--uniform-hidden-positions", str(settings["uniform_hidden_positions"]),
            "--assistant-hidden-positions", str(settings["assistant_hidden_positions"]),
            "--start-sample", str(batch["start_sample"]),
            "--max-samples", str(batch["samples"]),
            "--use-fla-kernel",
            "--gpu-weight-memory-gib", str(args.gpu_weight_memory_gib),
            "--cpu-offload-memory-gib", str(args.cpu_offload_memory_gib),
            "--mtp-device", args.mtp_device,
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
        ]
        print(
            f"batch={index} status=cache-start samples={batch['samples']} "
            f"tokens={batch['sequence_tokens']}",
            flush=True,
        )
        subprocess.run(cache_command, check=True, env=environment)
        verify_command = [
            sys.executable,
            str(Path(__file__).with_name("verify_teacher_cache.py")),
            "--cache", str(cache),
            "--input", str(args.input),
            "--teacher-revision", args.revision,
            "--teacher-provenance-sha256", args.teacher_provenance_sha256,
            "--hidden-size", str(settings["hidden_size"]),
            "--require-mtp",
            "--output", str(verification_path),
        ]
        subprocess.run(verify_command, check=True)
        print(f"batch={index} status=verified", flush=True)


if __name__ == "__main__":
    main()
