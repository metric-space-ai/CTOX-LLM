#!/usr/bin/env python3
"""Plan contiguous, independently recoverable activation-statistics batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from select_activation_calibration import load_jsonl, load_token_counts, write_bytes_atomic


def activation_batches(
    records: list[dict[str, Any]],
    token_counts: dict[str, int],
    maximum_samples: int,
    maximum_batch_tokens: int,
    maximum_sequence_tokens: int,
) -> list[dict[str, Any]]:
    if min(maximum_samples, maximum_batch_tokens, maximum_sequence_tokens) <= 0:
        raise ValueError("activation batch limits must be positive")
    result: list[dict[str, Any]] = []
    current: list[tuple[int, str, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        result.append(
            {
                "batch_index": len(result),
                "start_sample": current[0][0],
                "samples": len(current),
                "sequence_tokens": current_tokens,
                "maximum_sample_tokens": max(tokens for _line, _id, tokens in current),
                "first_id": current[0][1],
                "last_id": current[-1][1],
            }
        )
        current = []
        current_tokens = 0

    for source_line, record in enumerate(records):
        sample_id = str(record["id"])
        tokens = int(token_counts[sample_id])
        if tokens > maximum_sequence_tokens:
            raise ValueError(
                f"sample {sample_id} has {tokens} tokens, above the sequence limit"
            )
        if tokens > maximum_batch_tokens:
            raise ValueError(
                f"sample {sample_id} has {tokens} tokens, above the batch-token limit"
            )
        if current and (
            len(current) >= maximum_samples
            or current_tokens + tokens > maximum_batch_tokens
        ):
            flush()
        current.append((source_line, sample_id, tokens))
        current_tokens += tokens
    flush()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cache-plan", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=196_608)
    parser.add_argument("--max-sequence-tokens", type=int, default=131_072)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        input_bytes = args.input.read_bytes()
        records = load_jsonl(args.input)
        record_ids = {str(record["id"]) for record in records}
        if len(record_ids) != len(records):
            raise ValueError("activation input contains duplicate sample ids")
        token_counts = load_token_counts(args.cache_plan, record_ids)
        planned = activation_batches(
            records,
            token_counts,
            args.max_samples,
            args.max_batch_tokens,
            args.max_sequence_tokens,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.activation-batch-plan.v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "cache_plans": [
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.cache_plan
        ],
        "limits": {
            "max_samples": args.max_samples,
            "max_batch_tokens": args.max_batch_tokens,
            "max_sequence_tokens": args.max_sequence_tokens,
        },
        "summary": {
            "batches": len(planned),
            "samples": sum(batch["samples"] for batch in planned),
            "sequence_tokens": sum(batch["sequence_tokens"] for batch in planned),
        },
        "batches": planned,
    }
    write_bytes_atomic(
        args.output,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


if __name__ == "__main__":
    main()
