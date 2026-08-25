#!/usr/bin/env python3
"""Fail closed on joint language and semantic-domain selection coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classify_domains import validate_rubric


def coverage_report(
    records: list[dict[str, Any]],
    tags: dict[str, dict[str, Any]],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
    partition: str,
) -> dict[str, Any]:
    validate_rubric(domain_rubric)
    record_ids = {str(record["id"]) for record in records}
    if len(record_ids) != len(records):
        raise ValueError("materialized selection contains duplicate ids")
    if set(tags) != record_ids:
        missing = sorted(record_ids - set(tags))[:5]
        extra = sorted(set(tags) - record_ids)[:5]
        raise ValueError(f"domain tags differ from selection: missing={missing}, extra={extra}")

    domains = domain_rubric["domains"]
    expected_languages = language_rubric["languages"]
    translation_domain = language_rubric["translation_domain"]
    language_counts: Counter[str] = Counter()
    non_translation_counts: Counter[str] = Counter()
    primary_domains: dict[str, set[str]] = defaultdict(set)
    non_english_families: Counter[str] = Counter()
    for record in records:
        sample_id = str(record["id"])
        language = str(record["language"])
        if language not in expected_languages:
            raise ValueError(f"selection contains undeclared language {language}")
        primary = str(tags[sample_id]["primary_label"])
        if primary not in domains:
            raise ValueError(f"sample {sample_id} has unknown primary domain {primary}")
        language_counts[language] += 1
        primary_domains[language].add(primary)
        if primary != translation_domain:
            non_translation_counts[language] += 1
        if language != "en":
            non_english_families[str(domains[primary]["family"])] += 1

    language_gaps = {}
    diversity_gaps = {}
    non_translation_gaps = {}
    for language, requirements in expected_languages.items():
        minimum = int(requirements[f"minimum_{partition}"])
        minimum_domains = int(requirements[f"minimum_primary_domains_{partition}"])
        minimum_non_translation = int(
            requirements[f"minimum_non_translation_{partition}"]
        )
        if language_counts[language] < minimum:
            language_gaps[language] = {
                "observed": language_counts[language],
                "required": minimum,
            }
        if len(primary_domains[language]) < minimum_domains:
            diversity_gaps[language] = {
                "observed": len(primary_domains[language]),
                "required": minimum_domains,
            }
        if non_translation_counts[language] < minimum_non_translation:
            non_translation_gaps[language] = {
                "observed": non_translation_counts[language],
                "required": minimum_non_translation,
            }

    family_gaps = {}
    for family, minima in language_rubric["aggregate_non_english_family_minima"].items():
        required = int(minima[partition])
        if non_english_families[family] < required:
            family_gaps[family] = {
                "observed": non_english_families[family],
                "required": required,
            }
    passed = not any(
        (language_gaps, diversity_gaps, non_translation_gaps, family_gaps)
    )
    return {
        "records": len(records),
        "language_counts": dict(sorted(language_counts.items())),
        "primary_domain_diversity": {
            language: len(primary_domains[language]) for language in sorted(expected_languages)
        },
        "non_translation_counts": dict(sorted(non_translation_counts.items())),
        "aggregate_non_english_family_counts": dict(sorted(non_english_families.items())),
        "language_quota_gaps": language_gaps,
        "primary_domain_diversity_gaps": diversity_gaps,
        "non_translation_gaps": non_translation_gaps,
        "aggregate_non_english_family_gaps": family_gaps,
        "status": "passed" if passed else "supplement_required",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
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
            json.dump(document, temporary, indent=2, sort_keys=True)
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
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--language-rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        records = read_jsonl(args.input)
        tag_records = read_jsonl(args.tags)
        tags = {str(tag["id"]): tag for tag in tag_records}
        if len(tags) != len(tag_records):
            raise ValueError("domain tag file contains duplicate ids")
        domain_bytes = args.domain_rubric.read_bytes()
        language_bytes = args.language_rubric.read_bytes()
        report = coverage_report(
            records,
            tags,
            json.loads(domain_bytes),
            json.loads(language_bytes),
            args.partition,
        )
        document = {
            "format": "ctox.recovery-joint-selection-audit.v1",
            "partition": args.partition,
            "input": str(args.input.resolve()),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "tags": str(args.tags.resolve()),
            "tags_sha256": hashlib.sha256(args.tags.read_bytes()).hexdigest(),
            "domain_rubric_sha256": hashlib.sha256(domain_bytes).hexdigest(),
            "language_rubric_sha256": hashlib.sha256(language_bytes).hexdigest(),
            **report,
        }
        write_json_atomic(args.output, document)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
