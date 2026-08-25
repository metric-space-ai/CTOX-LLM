#!/usr/bin/env python3
"""Split a teacher-cache plan into contiguous, independently verifiable batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def batches(
    samples: list[dict[str, Any]],
    max_samples: int,
    max_sequence_tokens: int,
    max_projected_bytes: int,
) -> list[dict[str, Any]]:
    result = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_tokens, current_bytes
        if not current:
            return
        result.append(
            {
                "batch_index": len(result),
                "start_sample": int(current[0]["source_line"]) - 1,
                "samples": len(current),
                "sequence_tokens": current_tokens,
                "projected_cache_bytes": current_bytes,
                "maximum_sample_tokens": max(int(sample["sequence_tokens"]) for sample in current),
                "first_id": str(current[0]["id"]),
                "last_id": str(current[-1]["id"]),
            }
        )
        current = []
        current_tokens = 0
        current_bytes = 0

    for expected_line, sample in enumerate(samples, 1):
        if int(sample["source_line"]) != expected_line:
            raise ValueError("cache plan samples are not a contiguous source slice")
        sample_tokens = int(sample["sequence_tokens"])
        sample_bytes = int(sample["projected_file_bytes"])
        if sample_tokens > max_sequence_tokens or sample_bytes > max_projected_bytes:
            raise ValueError(f"sample {sample['id']} exceeds an individual batch limit")
        if current and (
            len(current) >= max_samples
            or current_tokens + sample_tokens > max_sequence_tokens
            or current_bytes + sample_bytes > max_projected_bytes
        ):
            flush()
        current.append(sample)
        current_tokens += sample_tokens
        current_bytes += sample_bytes
    flush()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-sequence-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-projected-bytes", type=int, default=2 * 1024**3)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if min(args.max_samples, args.max_sequence_tokens, args.max_projected_bytes) <= 0:
        raise SystemExit("batch limits must be positive")
    try:
        encoded = args.cache_plan.read_bytes()
        source = json.loads(encoded)
        planned = batches(
            source["samples"],
            args.max_samples,
            args.max_sequence_tokens,
            args.max_projected_bytes,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if sum(batch["samples"] for batch in planned) != source["summary"]["samples"]:
        raise SystemExit("batch sample count does not match cache plan")
    if sum(batch["projected_cache_bytes"] for batch in planned) != source["summary"]["projected_cache_bytes"]:
        raise SystemExit("batch bytes do not match cache plan")
    document = {
        "format": "ctox.teacher-cache-batch-plan.v1",
        "cache_plan": str(args.cache_plan.resolve()),
        "cache_plan_sha256": hashlib.sha256(encoded).hexdigest(),
        "limits": {
            "max_samples": args.max_samples,
            "max_sequence_tokens": args.max_sequence_tokens,
            "max_projected_bytes": args.max_projected_bytes,
        },
        "summary": {
            "batches": len(planned),
            "samples": sum(batch["samples"] for batch in planned),
            "sequence_tokens": sum(batch["sequence_tokens"] for batch in planned),
            "projected_cache_bytes": sum(batch["projected_cache_bytes"] for batch in planned),
        },
        "batches": planned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
