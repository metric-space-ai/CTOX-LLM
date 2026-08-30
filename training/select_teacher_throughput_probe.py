#!/usr/bin/env python3
"""Select the exact, context-stratified 10K BF16 teacher throughput probe."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from build_million_corpus_evidence import context_bucket
from build_recovery_run_plan import validate_million_corpus_audit


FORMAT = "ctox.teacher-throughput-probe-selection.v1"
QUOTAS = {
    "up_to_2k": 6_500,
    "2k_to_8k": 2_500,
    "8k_to_32k": 800,
    "32k_to_64k": 150,
    "64k_to_128k": 50,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_lines(path: Path, lines: list[bytes]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for line in lines:
                output.write(line.rstrip(b"\n") + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select(
    materialized: Path,
    token_counts: Path,
    seed: str,
) -> tuple[list[bytes], list[bytes], dict[str, int]]:
    # Each heap retains the largest negative hash at its root, allowing a
    # deterministic lowest-hash sample without keeping one million payloads.
    heaps: dict[str, list[tuple[int, int, bytes, bytes]]] = {
        bucket: [] for bucket in QUOTAS
    }
    seen = {bucket: 0 for bucket in QUOTAS}
    with materialized.open("rb") as samples, token_counts.open("rb") as tokens:
        row = 0
        while True:
            sample_line = samples.readline()
            token_line = tokens.readline()
            if not sample_line and not token_line:
                break
            row += 1
            if not sample_line or not token_line:
                raise ValueError(f"probe inputs differ in length at row {row}")
            sample = json.loads(sample_line)
            token = json.loads(token_line)
            sample_id = str(sample.get("id", ""))
            if not sample_id or str(token.get("id", "")) != sample_id:
                raise ValueError(f"probe inputs are out of order at row {row}")
            sequence_tokens = token.get("sequence_tokens")
            if isinstance(sequence_tokens, bool) or not isinstance(sequence_tokens, int):
                raise ValueError(f"sample {sample_id} has invalid sequence_tokens")
            bucket = context_bucket(sequence_tokens)
            seen[bucket] += 1
            rank = int.from_bytes(
                hashlib.sha256(f"{seed}\0{sample_id}".encode()).digest(), "big"
            )
            candidate = (-rank, -row, sample_line, token_line)
            heap = heaps[bucket]
            if len(heap) < QUOTAS[bucket]:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    gaps = {bucket: QUOTAS[bucket] - len(heaps[bucket]) for bucket in QUOTAS}
    if any(gaps.values()):
        raise ValueError(f"training partition cannot fill probe quotas: {gaps}")
    selected = []
    for bucket, heap in heaps.items():
        for _negative_rank, negative_row, sample_line, token_line in heap:
            selected.append((-negative_row, bucket, sample_line, token_line))
    selected.sort()
    return (
        [record[2] for record in selected],
        [record[3] for record in selected],
        seen,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    audit, evidence = validate_million_corpus_audit(
        args.million_corpus_audit, args.million_corpus_audit_sha256
    )
    train = evidence["partitions"]["train"]
    materialized = Path(train["binding_paths"]["materialized_sha256"])
    token_counts = Path(train["binding_paths"]["token_plan_sha256"])
    if sha256_path(materialized) != train["materialized_sha256"]:
        raise ValueError("admitted training materialized file changed")
    if sha256_path(token_counts) != train["token_plan_sha256"]:
        raise ValueError("admitted training token sidecar changed")
    sample_lines, token_lines, available = select(materialized, token_counts, args.seed)
    atomic_lines(args.output, sample_lines)
    try:
        atomic_lines(args.output_token_counts, token_lines)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    selected_counts = {bucket: 0 for bucket in QUOTAS}
    selected_tokens = 0
    for line in token_lines:
        record = json.loads(line)
        tokens = int(record["sequence_tokens"])
        selected_counts[context_bucket(tokens)] += 1
        selected_tokens += tokens
    return {
        "format": FORMAT,
        "status": "selected",
        "million_corpus_audit": str(args.million_corpus_audit.resolve()),
        "million_corpus_audit_sha256": args.million_corpus_audit_sha256,
        "million_corpus_evidence": audit["evidence"],
        "million_corpus_evidence_sha256": audit["evidence_sha256"],
        "source_materialized_sha256": train["materialized_sha256"],
        "source_token_counts_sha256": train["token_plan_sha256"],
        "seed": args.seed,
        "records": len(sample_lines),
        "sequence_tokens": selected_tokens,
        "context_quotas": QUOTAS,
        "context_selected": selected_counts,
        "context_available": available,
        "materialized": str(args.output.resolve()),
        "materialized_sha256": sha256_path(args.output),
        "token_counts": str(args.output_token_counts.resolve()),
        "token_counts_sha256": sha256_path(args.output_token_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--million-corpus-audit", type=Path, required=True)
    parser.add_argument("--million-corpus-audit-sha256", required=True)
    parser.add_argument("--seed", default="ctox-qwen38-teacher-probe-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-token-counts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    if args.evidence.exists():
        raise SystemExit(f"refusing to overwrite {args.evidence}")
    try:
        document = build(args)
        atomic_lines(
            args.evidence,
            [json.dumps(document, sort_keys=True, separators=(",", ":")).encode()],
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
