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


def cache_environment(
    inherited: dict[str, str], hf_home: Path | None
) -> dict[str, str]:
    environment = inherited.copy()
    if hf_home is None:
        return environment
    resolved = hf_home.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"Hugging Face cache root is not a directory: {resolved}")
    if not os.access(resolved, os.W_OK | os.X_OK):
        raise ValueError(f"Hugging Face cache root is not writable: {resolved}")
    environment["HF_HOME"] = str(resolved)
    environment["HF_HUB_CACHE"] = str(resolved / "hub")
    return environment


def bind_physical_gpus(
    environment: dict[str, str],
    physical_gpus: str | None,
    gpu_count: int,
    mtp_device: str,
) -> dict[str, str]:
    if gpu_count <= 0:
        raise ValueError("GPU count must be positive")
    if physical_gpus is None:
        return environment
    try:
        devices = [int(value) for value in physical_gpus.split(",") if value != ""]
    except ValueError as error:
        raise ValueError("physical GPUs must be comma-separated integer indices") from error
    if len(devices) != gpu_count or len(set(devices)) != len(devices):
        raise ValueError("physical GPU list must contain exactly --gpus unique indices")
    if any(device <= 0 for device in devices):
        raise ValueError("physical GPU 0 is reserved for Greppy")
    if not mtp_device.startswith("cuda:"):
        raise ValueError("MTP device must be a logical CUDA device")
    try:
        logical_mtp = int(mtp_device.split(":", 1)[1])
    except ValueError as error:
        raise ValueError("MTP device has an invalid logical CUDA index") from error
    if not 0 <= logical_mtp < gpu_count:
        raise ValueError("MTP logical CUDA device is outside the isolated GPU set")
    bound = environment.copy()
    bound["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in devices)
    return bound


def gpu_weight_memory_for_batch(
    default_gib: int,
    long_context_gib: int | None,
    long_context_threshold_tokens: int,
    maximum_sample_tokens: int,
) -> int:
    """Select one deterministic weight-placement tier before model load."""
    if default_gib <= 0:
        raise ValueError("default GPU weight memory must be positive")
    if long_context_threshold_tokens <= 0:
        raise ValueError("long-context threshold must be positive")
    if maximum_sample_tokens <= 0:
        raise ValueError("maximum sample tokens must be positive")
    if long_context_gib is None:
        return default_gib
    if long_context_gib <= 0 or long_context_gib > default_gib:
        raise ValueError(
            "long-context GPU weight memory must be positive and no larger than the default"
        )
    if maximum_sample_tokens >= long_context_threshold_tokens:
        return long_context_gib
    return default_gib


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
    parser.add_argument(
        "--hf-home",
        type=Path,
        help="validated cache root inherited by the pinned kernel/model loaders",
    )
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument(
        "--physical-gpus",
        help="ordered physical CUDA devices; GPU 0 is rejected (release profile: 1,2)",
    )
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
    parser.add_argument("--mtp-device", default="cuda:2")
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="resume an exact incomplete cache prefix instead of rejecting it",
    )
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

    try:
        environment = bind_physical_gpus(
            cache_environment(os.environ, args.hf_home),
            args.physical_gpus,
            args.gpus,
            args.mtp_device,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for batch in batches[args.start_batch:end_batch]:
        index = int(batch["batch_index"])
        gpu_weight_memory_gib = gpu_weight_memory_for_batch(
            args.gpu_weight_memory_gib,
            args.long_context_gpu_weight_memory_gib,
            args.long_context_threshold_tokens,
            int(batch["maximum_sample_tokens"]),
        )
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
        resume = cache.exists() and not verification_path.exists() and args.resume_incomplete
        if (cache.exists() or verification_path.exists()) and not resume:
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
            "--gpu-weight-memory-gib", str(gpu_weight_memory_gib),
            "--cpu-offload-memory-gib", str(args.cpu_offload_memory_gib),
            "--mtp-device", args.mtp_device,
            "--prefill-chunk-tokens", str(args.prefill_chunk_tokens),
        ]
        if resume:
            cache_command.append("--resume")
        print(
            f"batch={index} status={'cache-resume' if resume else 'cache-start'} samples={batch['samples']} "
            f"tokens={batch['sequence_tokens']} gpu_weight_memory_gib={gpu_weight_memory_gib}",
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
