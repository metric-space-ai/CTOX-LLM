#!/usr/bin/env python3
"""Apply hard source-provenance primary labels to frozen NLI tag scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from classify_domains import deterministic_primary_label, quota_gaps, validate_rubric
from merge_domain_tags import merge_ordered_tags, write_jsonl_atomic


def apply_overrides(
    records: list[dict[str, Any]], tags: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    ordered = merge_ordered_tags(records, [tags])
    counts: Counter[str] = Counter()
    output = []
    for record, original in zip(records, ordered, strict=True):
        tag = dict(original)
        classifier_primary = str(
            tag.get("classifier_primary_label")
            or max(tag["scores"], key=tag["scores"].get)
        )
        source_primary = deterministic_primary_label(record)
        tag["classifier_primary_label"] = classifier_primary
        if source_primary is not None:
            tag["primary_label"] = source_primary
            tag["primary_confidence"] = 1.0
            tag["primary_source"] = "source_fact"
            labels = set(tag["labels"])
            labels.add(source_primary)
            tag["labels"] = sorted(labels)
            counts[source_primary] += 1
        else:
            tag["primary_label"] = classifier_primary
            tag["primary_confidence"] = float(tag["scores"][classifier_primary])
            tag["primary_source"] = "classifier"
        output.append(tag)
    return output, counts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.summary.exists():
        raise SystemExit(f"refusing to overwrite {args.summary}")
    try:
        records = read_jsonl(args.input)
        original_tags = read_jsonl(args.tags)
        rubric_bytes = args.rubric.read_bytes()
        rubric = json.loads(rubric_bytes)
        validate_rubric(rubric)
        corrected, override_counts = apply_overrides(records, original_tags)
        output_sha256 = write_jsonl_atomic(args.output, corrected)
        label_counts: Counter[str] = Counter()
        primary_counts: Counter[str] = Counter()
        source_facts = 0
        for tag in corrected:
            label_counts.update(tag["labels"])
            primary_counts[str(tag["primary_label"])] += 1
            source_facts += int(tag["primary_source"] == "source_fact")
        multi_gaps, primary_gaps = quota_gaps(
            label_counts, primary_counts, rubric, args.partition
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.recovery-primary-source-overrides.v1",
        "partition": args.partition,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "input_tags_sha256": hashlib.sha256(args.tags.read_bytes()).hexdigest(),
        "rubric_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
        "records": len(corrected),
        "source_fact_primary_records": source_facts,
        "source_fact_counts": dict(sorted(override_counts.items())),
        "domain_counts": dict(sorted(label_counts.items())),
        "primary_domain_counts": dict(sorted(primary_counts.items())),
        "quota_gaps": multi_gaps,
        "primary_quota_gaps": primary_gaps,
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
