#!/usr/bin/env python3
"""Select high-confidence disjoint samples that close primary-domain gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from prompt_format import render_record


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_supplement(
    records: list[dict[str, Any]],
    tags: dict[str, dict[str, Any]],
    gaps: dict[str, dict[str, int]],
    token_counts: dict[str, int],
    margin: int,
    minimum_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    selected_ids = set()
    domain_samples = {}
    for domain, gap in sorted(gaps.items()):
        needed = int(gap["required"]) - int(gap["observed"]) + margin
        candidates = []
        for record in records:
            sample_id = str(record["id"])
            tag = tags[sample_id]
            score = float(tag["scores"].get(domain, 0.0))
            if tag.get("primary_label") == domain and score >= minimum_confidence:
                candidates.append((record, score))
        candidates.sort(
            key=lambda item: (
                -item[1],
                token_counts[str(item[0]["id"])],
                str(item[0]["id"]),
            )
        )
        if len(candidates) < needed:
            raise ValueError(
                f"primary domain {domain} has {len(candidates)} high-confidence candidates, "
                f"requires {needed}"
            )
        chosen = [str(record["id"]) for record, _score in candidates[:needed]]
        domain_samples[domain] = chosen
        selected_ids.update(chosen)
    return (
        [record for record in records if str(record["id"]) in selected_ids],
        domain_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate-tags", type=Path, required=True)
    parser.add_argument("--baseline-gate", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--minimum-confidence", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite supplement output or report")
    if args.margin < 0 or not 0 < args.minimum_confidence <= 1:
        raise SystemExit("invalid supplement margin or confidence")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before supplement selection") from error
    try:
        records = load_jsonl(args.input)
        tag_records = load_jsonl(args.candidate_tags)
        tags = {str(tag["id"]): tag for tag in tag_records}
        if len(tags) != len(tag_records):
            raise ValueError("candidate domain tags contain duplicate ids")
        record_ids = {str(record["id"]) for record in records}
        if set(tags) != record_ids:
            raise ValueError("candidate records and domain tags differ")
        baseline = json.loads(args.baseline_gate.read_text(encoding="utf-8"))
        gaps = baseline["primary_quota_gaps"]
        if not gaps:
            raise ValueError("baseline primary gate has no gaps")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision
        )
        token_counts = {
            str(record["id"]): len(
                tokenizer(render_record(tokenizer, record), add_special_tokens=False).input_ids
            )
            for record in records
        }
        selected, domain_samples = select_supplement(
            records,
            tags,
            gaps,
            token_counts,
            args.margin,
            args.minimum_confidence,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in selected:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    encoded = args.output.read_bytes()
    document = {
        "format": "ctox.primary-domain-supplement-selection.v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "candidate_tags_sha256": hashlib.sha256(args.candidate_tags.read_bytes()).hexdigest(),
        "baseline_gate_sha256": hashlib.sha256(args.baseline_gate.read_bytes()).hexdigest(),
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "minimum_confidence": args.minimum_confidence,
        "margin": args.margin,
        "samples": len(selected),
        "tokens": sum(token_counts[str(record["id"])] for record in selected),
        "output_bytes": len(encoded),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "domain_samples": domain_samples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
