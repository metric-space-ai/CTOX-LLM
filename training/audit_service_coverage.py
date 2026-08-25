#!/usr/bin/env python3
"""Audit semantic-domain by service-mode coverage for one frozen partition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from classify_domains import validate_rubric as validate_domain_rubric
from classify_service_modes import validate_rubric as validate_service_rubric
from classify_domains import write_json_atomic


def service_coverage_report(
    records: list[dict[str, Any]],
    domain_tags: dict[str, dict[str, Any]],
    service_tags: dict[str, dict[str, Any]],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
    service_rubric: dict[str, Any],
    partition: str,
) -> dict[str, Any]:
    validate_domain_rubric(domain_rubric)
    validate_service_rubric(service_rubric, domain_rubric, language_rubric)
    record_ids = {str(record["id"]) for record in records}
    if len(record_ids) != len(records):
        raise ValueError("materialized selection contains duplicate ids")
    for name, tags in (("domain", domain_tags), ("service", service_tags)):
        if set(tags) != record_ids:
            missing = sorted(record_ids - set(tags))[:5]
            extra = sorted(set(tags) - record_ids)[:5]
            raise ValueError(f"{name} tags differ from selection: missing={missing}, extra={extra}")

    domains = domain_rubric["domains"]
    modes = service_rubric["modes"]
    declared_languages = set(language_rubric["languages"])
    mode_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    domain_modes: dict[str, Counter[str]] = defaultdict(Counter)
    family_modes: dict[str, set[str]] = defaultdict(set)
    language_modes: dict[str, set[str]] = defaultdict(set)

    for record in records:
        sample_id = str(record["id"])
        domain = str(domain_tags[sample_id]["primary_label"])
        if domain not in domains:
            raise ValueError(f"sample {sample_id} has unknown primary domain {domain}")
        language = str(record["language"])
        if language not in declared_languages:
            raise ValueError(f"sample {sample_id} has undeclared language {language}")
        labels = {str(label) for label in service_tags[sample_id].get("labels", [])}
        if not labels:
            raise ValueError(f"sample {sample_id} has no service-mode labels")
        unknown = labels - set(modes)
        if unknown:
            raise ValueError(f"sample {sample_id} has unknown service modes {sorted(unknown)}")
        family = str(domains[domain]["family"])
        domain_counts[domain] += 1
        mode_counts.update(labels)
        domain_modes[domain].update(labels)
        family_modes[family].update(labels)
        language_modes[language].update(labels)

    domain_presence_gaps = {}
    if service_rubric["policy"]["all_declared_domains_required"]:
        domain_presence_gaps = {
            domain: {"observed": 0, "required": 1}
            for domain in domains
            if domain_counts[domain] == 0
        }
    mode_quota_gaps = {}
    for mode, requirements in modes.items():
        required = int(requirements[f"minimum_{partition}"])
        if mode_counts[mode] < required:
            mode_quota_gaps[mode] = {
                "observed": mode_counts[mode],
                "required": required,
            }

    domain_diversity_minimum = int(
        service_rubric["policy"][f"minimum_distinct_modes_per_domain_{partition}"]
    )
    domain_diversity_gaps = {
        domain: {
            "observed": len(domain_modes[domain]),
            "required": domain_diversity_minimum,
        }
        for domain in domains
        if len(domain_modes[domain]) < domain_diversity_minimum
    }
    family_diversity_minimum = int(
        service_rubric["policy"][f"minimum_distinct_modes_per_family_{partition}"]
    )
    required_families = domain_rubric["policy"]["required_families"]
    family_diversity_gaps = {
        family: {
            "observed": len(family_modes[family]),
            "required": family_diversity_minimum,
        }
        for family in required_families
        if len(family_modes[family]) < family_diversity_minimum
    }
    language_diversity_gaps = {}
    for language, minima in service_rubric["language_minimum_distinct_modes"].items():
        required = int(minima[partition])
        if len(language_modes[language]) < required:
            language_diversity_gaps[language] = {
                "observed": len(language_modes[language]),
                "required": required,
            }

    pair_gaps: dict[str, dict[str, dict[str, int]]] = {}
    for domain, required_modes in service_rubric["required_domain_mode_pairs"].items():
        for mode, minima in required_modes.items():
            required = int(minima[partition])
            observed = domain_modes[domain][mode]
            if observed < required:
                pair_gaps.setdefault(domain, {})[mode] = {
                    "observed": observed,
                    "required": required,
                }

    gaps = (
        domain_presence_gaps,
        mode_quota_gaps,
        domain_diversity_gaps,
        family_diversity_gaps,
        language_diversity_gaps,
        pair_gaps,
    )
    return {
        "records": len(records),
        "mode_counts": dict(sorted(mode_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "domain_mode_counts": {
            domain: dict(sorted(domain_modes[domain].items())) for domain in sorted(domains)
        },
        "domain_mode_diversity": {
            domain: len(domain_modes[domain]) for domain in sorted(domains)
        },
        "family_mode_diversity": {
            family: len(family_modes[family]) for family in sorted(required_families)
        },
        "language_mode_diversity": {
            language: len(language_modes[language]) for language in sorted(declared_languages)
        },
        "domain_presence_gaps": domain_presence_gaps,
        "mode_quota_gaps": mode_quota_gaps,
        "domain_mode_diversity_gaps": domain_diversity_gaps,
        "family_mode_diversity_gaps": family_diversity_gaps,
        "language_mode_diversity_gaps": language_diversity_gaps,
        "required_domain_mode_pair_gaps": pair_gaps,
        "status": "passed" if not any(gaps) else "supplement_required",
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def indexed_tags(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    tags = {str(record["id"]): record for record in records}
    if len(tags) != len(records):
        raise ValueError(f"{path} contains duplicate ids")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--domain-tags", type=Path, required=True)
    parser.add_argument("--service-tags", type=Path, required=True)
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--language-rubric", type=Path, required=True)
    parser.add_argument("--service-rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = {
            "domain_rubric": args.domain_rubric.read_bytes(),
            "language_rubric": args.language_rubric.read_bytes(),
            "service_rubric": args.service_rubric.read_bytes(),
        }
        report = service_coverage_report(
            read_jsonl(args.input),
            indexed_tags(args.domain_tags),
            indexed_tags(args.service_tags),
            json.loads(inputs["domain_rubric"]),
            json.loads(inputs["language_rubric"]),
            json.loads(inputs["service_rubric"]),
            args.partition,
        )
        write_json_atomic(
            args.output,
            {
                "format": "ctox.recovery-service-coverage-audit.v1",
                "partition": args.partition,
                "input": str(args.input.resolve()),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "domain_tags_sha256": hashlib.sha256(args.domain_tags.read_bytes()).hexdigest(),
                "service_tags_sha256": hashlib.sha256(args.service_tags.read_bytes()).hexdigest(),
                **{
                    f"{name}_sha256": hashlib.sha256(data).hexdigest()
                    for name, data in inputs.items()
                },
                **report,
            },
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
