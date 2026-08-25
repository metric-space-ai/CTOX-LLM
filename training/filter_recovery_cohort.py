#!/usr/bin/env python3
"""Remove unusable or leaking records from a materialized recovery cohort.

The filter is deliberately independent of dataset-specific schemas.  It checks
the exact normalized payload consumed by recovery, keeps the domain-tag stream
in lockstep, and can reject payloads already present in another partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_manifest import canonical_text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def payload_hash(record: dict[str, Any]) -> str:
    actual = hashlib.sha256(canonical_text(record).encode("utf-8")).hexdigest()
    declared = record.get("prompt_sha256")
    if declared is not None and declared != actual:
        raise ValueError(f"record {record.get('id')} has a changed recovery payload")
    return actual


def has_content(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def has_conditioning_signal(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") == "assistant":
                continue
            if has_content(message.get("content")) or has_content(message.get("tool_calls")):
                return True
        return False
    return any(has_content(record.get(key)) for key in ("prompt", "text"))


def has_target(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return True
    last = messages[-1]
    return (
        isinstance(last, dict)
        and last.get("role") == "assistant"
        and (has_content(last.get("content")) or has_content(last.get("tool_calls")))
    )


def filter_records(
    records: Iterable[dict[str, Any]],
    tags: Iterable[dict[str, Any]],
    denied_payload_hashes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    tag_by_id = {str(tag["id"]): tag for tag in tags}
    denied = denied_payload_hashes or set()
    seen_ids: set[str] = set()
    seen_payloads: set[str] = set()
    kept: list[dict[str, Any]] = []
    kept_tags: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()

    for record in records:
        sample_id = str(record["id"])
        if sample_id in seen_ids:
            removed["duplicate_id"] += 1
            continue
        seen_ids.add(sample_id)
        tag = tag_by_id.get(sample_id)
        if tag is None:
            raise ValueError(f"record {sample_id} has no domain tag")
        digest = payload_hash(record)
        if not has_conditioning_signal(record):
            removed["empty_conditioning"] += 1
            continue
        if not has_target(record):
            removed["empty_or_missing_target"] += 1
            continue
        if digest in denied:
            removed["cross_partition_payload"] += 1
            continue
        if digest in seen_payloads:
            removed["duplicate_payload"] += 1
            continue
        seen_payloads.add(digest)
        kept.append(record)
        kept_tags.append(tag)

    extra_tags = set(tag_by_id) - seen_ids
    if extra_tags:
        raise ValueError(f"tag stream has {len(extra_tags)} records absent from the cohort")
    return kept, kept_tags, removed


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
            for row in rows:
                temporary.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--deny-input", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tags", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output, args.output_tags, args.report):
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")

    denied = {
        payload_hash(record)
        for path in args.deny_input
        for record in read_jsonl(path)
    }
    try:
        kept, kept_tags, removed = filter_records(
            read_jsonl(args.input), read_jsonl(args.tags), denied
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    write_jsonl_atomic(args.output, kept)
    write_jsonl_atomic(args.output_tags, kept_tags)
    report = {
        "format": "ctox.recovery-cohort-filter.v1",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "tags": str(args.tags),
        "tags_sha256": sha256_file(args.tags),
        "deny_inputs": [str(path) for path in args.deny_input],
        "deny_payload_hashes": len(denied),
        "input_records": len(kept) + sum(removed.values()),
        "kept_records": len(kept),
        "removed": dict(sorted(removed.items())),
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
        "output_tags": str(args.output_tags),
        "output_tags_bytes": args.output_tags.stat().st_size,
        "output_tags_sha256": sha256_file(args.output_tags),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
