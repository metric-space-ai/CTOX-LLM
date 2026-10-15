#!/usr/bin/env python3
"""Calculate the exact tensor payload and a conservative teacher-cache size.

The planner executes the pinned tokenizer and the same target-position rules as
``cache_teacher.py`` but never loads the BF16 teacher.  It is therefore the
required disk-admission check before a release-size GPU cache run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from cache_teacher import (
    assistant_prefix,
    mtp_target_positions,
    position_sets,
    validate_local_model_provenance,
)
from prompt_format import render_record


I32_BYTES = 4
U8_BYTES = 1
BF16_BYTES = 2
F32_BYTES = 4
DEFAULT_FILE_OVERHEAD_BYTES = 16 * 1024


def sample_tensor_bytes(
    sequence_tokens: int,
    logit_targets: int,
    hidden_targets: int,
    mtp_targets: int,
    mtp_hidden_targets: int,
    hidden_size: int,
    hidden_layer_count: int,
    top_k: int,
) -> dict[str, int]:
    """Return bytes for every tensor family written by ``cache_teacher.py``."""

    values = {
        "input_ids": sequence_tokens * I32_BYTES,
        "attention_mask": sequence_tokens * U8_BYTES,
        "logit_positions": logit_targets * I32_BYTES,
        "hidden_positions": hidden_targets * I32_BYTES,
        "topk_indices": logit_targets * top_k * I32_BYTES,
        "topk_logprobs": logit_targets * top_k * BF16_BYTES,
        "residual_probability": logit_targets * F32_BYTES,
        "hidden_layers": hidden_layer_count * hidden_targets * hidden_size * BF16_BYTES,
        "mtp_positions": mtp_targets * I32_BYTES,
        "mtp_hidden_positions": mtp_hidden_targets * I32_BYTES,
        "mtp_hidden": mtp_hidden_targets * hidden_size * BF16_BYTES,
        "mtp_topk_indices": mtp_targets * top_k * I32_BYTES,
        "mtp_topk_logprobs": mtp_targets * top_k * BF16_BYTES,
        "mtp_residual_probability": mtp_targets * F32_BYTES,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("teacher-cache tensor sizes cannot be negative")
    return values


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def plan(
    args: argparse.Namespace,
    tokenizer: Any,
    local_provenance: dict[str, Any] | None,
    local_provenance_sha256: str | None,
) -> dict[str, Any]:
    samples = []
    tensor_totals: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    projected_files = []
    sequence_lengths = []
    logit_target_counts = []
    hidden_target_counts = []

    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            rendered = render_record(tokenizer, record)
            sequence_tokens = len(
                tokenizer(rendered, add_special_tokens=False).input_ids
            )
            prefix = assistant_prefix(tokenizer, record, rendered)
            prefix_tokens = len(
                tokenizer(prefix, add_special_tokens=False).input_ids
            )
            logit_positions, hidden_positions = position_sets(
                sequence_tokens,
                "assistant",
                prefix_tokens,
                [int(offset) for offset in record.get("marker_token_offsets", [])],
                args.marker_window,
                args.uniform_hidden_positions,
                args.assistant_hidden_positions,
            )
            mtp_positions = mtp_target_positions(sequence_tokens, logit_positions)
            mtp_set = set(mtp_positions)
            mtp_hidden_positions = [
                position for position in hidden_positions if position in mtp_set
            ]
            tensors = sample_tensor_bytes(
                sequence_tokens,
                len(logit_positions),
                len(hidden_positions),
                len(mtp_positions),
                len(mtp_hidden_positions),
                args.hidden_size,
                len(args.hidden_layers),
                args.top_k,
            )
            tensor_bytes = sum(tensors.values())
            projected_file_bytes = tensor_bytes + args.file_overhead_bytes
            tensor_totals.update(tensors)
            projected_files.append(projected_file_bytes)
            sequence_lengths.append(sequence_tokens)
            logit_target_counts.append(len(logit_positions))
            hidden_target_counts.append(len(hidden_positions))
            counts[f"category:{record.get('category', 'unknown')}"] += 1
            counts[f"language:{record.get('language', 'unknown')}"] += 1
            samples.append(
                {
                    "id": str(record["id"]),
                    "source_line": line_number,
                    "category": record.get("category"),
                    "language": record.get("language"),
                    "sequence_tokens": sequence_tokens,
                    "assistant_prefix_tokens": prefix_tokens,
                    "logit_targets": len(logit_positions),
                    "hidden_targets": len(hidden_positions),
                    "mtp_targets": len(mtp_positions),
                    "mtp_hidden_targets": len(mtp_hidden_positions),
                    "tensor_bytes": tensor_bytes,
                    "projected_file_bytes": projected_file_bytes,
                }
            )

    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    tensor_payload_bytes = sum(tensor_totals.values())
    projected_cache_bytes = sum(projected_files)
    largest = max(samples, key=lambda sample: sample["projected_file_bytes"], default=None)
    return {
        "format": "ctox.teacher-cache-plan.v1",
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "local_model_provenance_sha256": local_provenance_sha256,
        "local_model_root_sha256": (
            local_provenance["root_sha256"] if local_provenance is not None else None
        ),
        "settings": {
            "target_mode": "assistant",
            "mtp_targets": True,
            "hidden_size": args.hidden_size,
            "hidden_layers": args.hidden_layers,
            "top_k": args.top_k,
            "marker_window": args.marker_window,
            "uniform_hidden_positions": args.uniform_hidden_positions,
            "assistant_hidden_positions": args.assistant_hidden_positions,
            "conservative_file_overhead_bytes": args.file_overhead_bytes,
        },
        "summary": {
            "samples": len(samples),
            "tensor_payload_bytes": tensor_payload_bytes,
            "projected_cache_bytes": projected_cache_bytes,
            "projected_cache_gib": projected_cache_bytes / (1024**3),
            "file_bytes_p50": percentile(projected_files, 0.50),
            "file_bytes_p95": percentile(projected_files, 0.95),
            "file_bytes_max": max(projected_files, default=0),
            "sequence_tokens_p50": percentile(sequence_lengths, 0.50),
            "sequence_tokens_p95": percentile(sequence_lengths, 0.95),
            "sequence_tokens_max": max(sequence_lengths, default=0),
            "logit_targets_p50": percentile(logit_target_counts, 0.50),
            "logit_targets_p95": percentile(logit_target_counts, 0.95),
            "logit_targets_max": max(logit_target_counts, default=0),
            "hidden_targets_p50": percentile(hidden_target_counts, 0.50),
            "hidden_targets_p95": percentile(hidden_target_counts, 0.95),
            "hidden_targets_max": max(hidden_target_counts, default=0),
            "largest_sample": largest,
            "counts": dict(sorted(counts.items())),
            "tensor_bytes": dict(sorted(tensor_totals.items())),
        },
        "samples": samples,
    }


def write_sample_index(path: Path, samples: list[dict[str, Any]]) -> None:
    """Durably write the ordered, memory-bounded downstream token sidecar."""

    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for sample in samples:
                temporary.write(
                    json.dumps(
                        {
                            "id": sample["id"],
                            "sequence_tokens": sample["sequence_tokens"],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample-index",
        type=Path,
        help="ordered JSONL id/sequence-token sidecar for million-corpus evidence",
    )
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--hidden-layers", default="0,15,31,47,63")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--marker-window", type=int, default=32)
    parser.add_argument("--uniform-hidden-positions", type=int, default=64)
    parser.add_argument("--assistant-hidden-positions", type=int, default=64)
    parser.add_argument("--file-overhead-bytes", type=int, default=DEFAULT_FILE_OVERHEAD_BYTES)
    args = parser.parse_args()
    args.hidden_layers = [
        int(layer) for layer in args.hidden_layers.split(",") if layer
    ]
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.sample_index is not None and args.sample_index.exists():
        raise SystemExit(f"refusing to overwrite {args.sample_index}")
    if args.hidden_size <= 0 or not args.hidden_layers or args.top_k <= 0:
        raise SystemExit("hidden size, hidden layers, and top-k must be positive")
    if min(
        args.marker_window,
        args.uniform_hidden_positions,
        args.assistant_hidden_positions,
        args.file_overhead_bytes,
    ) < 0:
        raise SystemExit("position counts and file overhead must be non-negative")
    if args.assistant_hidden_positions == 0:
        raise SystemExit("assistant hidden positions must be positive")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before planning teacher cache") from error

    try:
        local_provenance, local_provenance_sha256 = validate_local_model_provenance(
            Path(args.tokenizer),
            args.tokenizer_revision,
            args.local_model_provenance,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
    )
    document = plan(
        args,
        tokenizer,
        local_provenance,
        local_provenance_sha256,
    )
    if args.sample_index is not None:
        try:
            write_sample_index(args.sample_index, document["samples"])
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
