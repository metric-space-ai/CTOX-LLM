#!/usr/bin/env python3
"""Select a cheap teacher-cache smoke spanning every domain and language."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def by_id(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        sample_id = str(record["id"])
        if sample_id in result:
            raise ValueError(f"duplicate {label} sample {sample_id}")
        result[sample_id] = record
    return result


def select_ids(
    records: list[dict[str, Any]],
    tags: dict[str, dict[str, Any]],
    plans: dict[str, dict[str, Any]],
    required_domains: list[str],
    required_languages: list[str],
    max_sequence_tokens: int,
) -> tuple[set[str], dict[str, str], dict[str, str]]:
    candidates = [
        record
        for record in records
        if int(plans[str(record["id"])]["sequence_tokens"]) <= max_sequence_tokens
    ]
    if not candidates:
        raise ValueError("no smoke candidates fit the sequence limit")

    def rank(record: dict[str, Any]) -> tuple[int, int, str]:
        plan = plans[str(record["id"])]
        return (
            int(plan["projected_file_bytes"]),
            int(plan["sequence_tokens"]),
            str(record["id"]),
        )

    selected = set()
    domain_samples = {}
    for domain in required_domains:
        matching = [
            record
            for record in candidates
            if tags[str(record["id"])].get("primary_label") == domain
        ]
        if not matching:
            raise ValueError(f"no primary-domain smoke candidate for {domain}")
        sample_id = str(min(matching, key=rank)["id"])
        selected.add(sample_id)
        domain_samples[domain] = sample_id

    language_samples = {}
    for language in required_languages:
        existing = [
            record
            for record in candidates
            if str(record["id"]) in selected and record.get("language") == language
        ]
        if existing:
            sample_id = str(min(existing, key=rank)["id"])
        else:
            matching = [
                record for record in candidates if record.get("language") == language
            ]
            if not matching:
                raise ValueError(f"no language smoke candidate for {language}")
            sample_id = str(min(matching, key=rank)["id"])
            selected.add(sample_id)
        language_samples[language] = sample_id
    return selected, domain_samples, language_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--domain-tags", type=Path, required=True)
    parser.add_argument("--cache-plan", type=Path, required=True)
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite smoke output or report")
    if args.max_sequence_tokens <= 0:
        raise SystemExit("--max-sequence-tokens must be positive")
    try:
        records = load_jsonl(args.input)
        tags = by_id(load_jsonl(args.domain_tags), "domain-tag")
        plan_document = json.loads(args.cache_plan.read_text(encoding="utf-8"))
        plans = by_id(plan_document["samples"], "cache-plan")
        rubric = json.loads(args.domain_rubric.read_text(encoding="utf-8"))
        record_ids = {str(record["id"]) for record in records}
        if set(tags) != record_ids or set(plans) != record_ids:
            raise ValueError("input, domain tags, and cache plan do not contain identical samples")
        required_domains = sorted(rubric["domains"])
        required_languages = sorted({str(record["language"]) for record in records})
        selected, domain_samples, language_samples = select_ids(
            records,
            tags,
            plans,
            required_domains,
            required_languages,
            args.max_sequence_tokens,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    selected_records = [record for record in records if str(record["id"]) in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in selected_records:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    selected_bytes = args.output.read_bytes()
    report = {
        "format": "ctox.teacher-cache-smoke-selection.v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "domain_tags_sha256": hashlib.sha256(args.domain_tags.read_bytes()).hexdigest(),
        "cache_plan_sha256": hashlib.sha256(args.cache_plan.read_bytes()).hexdigest(),
        "domain_rubric_sha256": hashlib.sha256(args.domain_rubric.read_bytes()).hexdigest(),
        "max_sequence_tokens": args.max_sequence_tokens,
        "samples": len(selected_records),
        "output": str(args.output.resolve()),
        "output_bytes": len(selected_bytes),
        "output_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "projected_cache_bytes": sum(
            int(plans[str(record["id"])]["projected_file_bytes"])
            for record in selected_records
        ),
        "sequence_tokens": sum(
            int(plans[str(record["id"])]["sequence_tokens"])
            for record in selected_records
        ),
        "assistant_logit_targets": sum(
            int(plans[str(record["id"])]["logit_targets"])
            for record in selected_records
        ),
        "required_domains": required_domains,
        "required_languages": required_languages,
        "domain_samples": domain_samples,
        "language_samples": language_samples,
        "selected_ids": [str(record["id"]) for record in selected_records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
