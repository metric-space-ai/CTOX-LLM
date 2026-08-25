#!/usr/bin/env python3
"""Merge independently classified tag shards in exact materialized order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from classify_domains import quota_gaps, validate_rubric


def merge_ordered_tags(
    records: list[dict[str, Any]], tag_shards: list[list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for shard in tag_shards:
        for tag in shard:
            sample_id = str(tag["id"])
            previous = by_id.get(sample_id)
            if previous is not None and previous != tag:
                raise ValueError(f"conflicting domain tag {sample_id}")
            if previous is not None:
                raise ValueError(f"duplicate domain tag {sample_id}")
            by_id[sample_id] = tag
    record_ids = [str(record["id"]) for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("materialized candidate set contains duplicate ids")
    if set(by_id) != set(record_ids):
        missing = sorted(set(record_ids) - set(by_id))[:5]
        extra = sorted(set(by_id) - set(record_ids))[:5]
        raise ValueError(f"tag shards differ from materialized set: missing={missing}, extra={extra}")
    return [by_id[sample_id] for sample_id in record_ids]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
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
            for record in records:
                encoded = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                temporary.write(encoded.decode("utf-8"))
                digest.update(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tag-shard", type=Path, action="append", required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.summary.exists():
        raise SystemExit(f"refusing to overwrite {args.summary}")
    try:
        records = read_jsonl(args.input)
        shards = [read_jsonl(path) for path in args.tag_shard]
        rubric_bytes = args.rubric.read_bytes()
        rubric = json.loads(rubric_bytes)
        validate_rubric(rubric)
        merged = merge_ordered_tags(records, shards)
        output_sha256 = write_jsonl_atomic(args.output, merged)
        label_counts: Counter[str] = Counter()
        primary_counts: Counter[str] = Counter()
        fallback = 0
        for tag in merged:
            label_counts.update(tag["labels"])
            primary_counts[str(tag["primary_label"])] += 1
            fallback += int(bool(tag.get("below_threshold_fallback")))
        multi_gaps, primary_gaps = quota_gaps(
            label_counts, primary_counts, rubric, args.partition
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.recovery-merged-domain-tags.v1",
        "partition": args.partition,
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "tag_shards": [str(path.resolve()) for path in args.tag_shard],
        "tag_shard_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.tag_shard],
        "rubric_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
        "records": len(merged),
        "below_threshold_fallback_records": fallback,
        "domain_counts": dict(sorted(label_counts.items())),
        "primary_domain_counts": dict(sorted(primary_counts.items())),
        "quota_gaps": multi_gaps,
        "primary_quota_gaps": primary_gaps,
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": output_sha256,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
