#!/usr/bin/env python3
"""Run one fixed-code recovery smoke against the exact verified cache bytes."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from fanout_recovery import QWEN38_FANOUT_POLICY
from teacher_cache_dataset import VerifiedTeacherCache


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: argparse.Namespace) -> list[str]:
    cache_sha256 = sha256_path(args.teacher_cache_set)
    cache = VerifiedTeacherCache.from_manifest(args.teacher_cache_set, cache_sha256)
    if args.sample_id not in {str(item["id"]) for item in cache.artifacts}:
        raise ValueError("recovery smoke sample is absent from the verified teacher cache")
    return [
        str(Path(sys.executable).resolve()),
        str(Path(__file__).with_name("train_recovery.py").resolve()),
        "--artifact",
        str(args.artifact.resolve()),
        "--model-source",
        str(args.model_source.resolve()),
        "--revision",
        args.revision,
        "--local-model-provenance",
        str(args.local_model_provenance.resolve()),
        "--teacher-cache-set",
        str(args.teacher_cache_set.resolve()),
        "--teacher-cache-set-sha256",
        cache_sha256,
        "--output-scales",
        str(args.output_scales.resolve()),
        "--output-report",
        str(args.output_report.resolve()),
        "--output-evidence",
        str(args.output_evidence.resolve()),
        "--checkpoint-dir",
        str(args.checkpoint_dir.resolve()),
        "--ledger",
        str(args.ledger.resolve()),
        "--reserved-gpu-hours",
        str(args.reserved_gpu_hours),
        "--gpus",
        "1",
        "--device",
        args.device,
        "--compute-dtype",
        "bfloat16",
        "--epochs",
        "1",
        "--max-optimizer-steps",
        "1",
        "--sample-id",
        args.sample_id,
        "--max-sequence-tokens",
        str(args.max_sequence_tokens),
        "--prefill-chunk-tokens",
        str(args.prefill_chunk_tokens),
        "--oversize-policy",
        "fail",
        "--checkpoint-every",
        "1",
        "--use-fla-kernel",
        "--fanout-s-in-policy",
        QWEN38_FANOUT_POLICY,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-cache-set", type=Path, required=True)
    parser.add_argument("--output-scales", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--reserved-gpu-hours", type=float, default=4.0)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.reserved_gpu_hours <= 0:
        parser.error("--reserved-gpu-hours must be positive")
    if args.max_sequence_tokens <= 0 or args.prefill_chunk_tokens <= 0:
        parser.error("sequence and chunk token counts must be positive")
    for output in (args.output_scales, args.output_report, args.output_evidence):
        if output.exists():
            parser.error(f"refusing to overwrite {output}")
    return args


def main() -> None:
    args = parse_args()
    try:
        subprocess.run(command(args), check=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
