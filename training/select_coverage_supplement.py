#!/usr/bin/env python3
"""Select a deterministic supplement that closes joint domain/language gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_selection_coverage import coverage_report, validate_language_rubric
from classify_domains import quota_gaps, validate_rubric
from merge_domain_tags import write_jsonl_atomic
from prompt_format import render_record


def remaining_requirements(
    baseline_records: list[dict[str, Any]],
    baseline_tags: dict[str, dict[str, Any]],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
    partition: str,
    domain_margin: int,
) -> tuple[dict[str, Counter[str]], dict[str, set[str]]]:
    label_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    non_translation_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    language_domains: dict[str, set[str]] = defaultdict(set)
    domains = domain_rubric["domains"]
    translation = language_rubric["translation_domain"]
    for record in baseline_records:
        tag = baseline_tags[str(record["id"])]
        language = str(record["language"])
        primary = str(tag["primary_label"])
        label_counts.update(tag["labels"])
        primary_counts[primary] += 1
        language_counts[language] += 1
        language_domains[language].add(primary)
        if primary != translation:
            non_translation_counts[language] += 1
        if language != "en":
            family_counts[str(domains[primary]["family"])] += 1

    label_gaps, primary_gaps = quota_gaps(
        label_counts, primary_counts, domain_rubric, partition
    )
    requirements: dict[str, Counter[str]] = {
        "domain_label": Counter(
            {
                domain: int(gap["required"]) - int(gap["observed"])
                for domain, gap in label_gaps.items()
            }
        ),
        "domain_primary": Counter(
            {
                domain: int(gap["required"])
                - int(gap["observed"])
                + domain_margin
                for domain, gap in primary_gaps.items()
            }
        ),
        "language": Counter(),
        "non_translation": Counter(),
        "language_diversity": Counter(),
        "non_english_family": Counter(),
    }
    for language, minima in language_rubric["languages"].items():
        requirements["language"][language] = max(
            0, int(minima[f"minimum_{partition}"]) - language_counts[language]
        )
        requirements["non_translation"][language] = max(
            0,
            int(minima[f"minimum_non_translation_{partition}"])
            - non_translation_counts[language],
        )
        requirements["language_diversity"][language] = max(
            0,
            int(minima[f"minimum_primary_domains_{partition}"])
            - len(language_domains[language]),
        )
    for family, minima in language_rubric[
        "aggregate_non_english_family_minima"
    ].items():
        requirements["non_english_family"][family] = max(
            0, int(minima[partition]) - family_counts[family]
        )
    return requirements, language_domains


def candidate_coverage(
    record: dict[str, Any],
    tag: dict[str, Any],
    assigned_primary: str | None,
    requirements: dict[str, Counter[str]],
    language_domains: dict[str, set[str]],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
) -> list[tuple[str, str]]:
    primary = assigned_primary
    language = str(record["language"])
    covered = []
    for label in tag["labels"]:
        if requirements["domain_label"][str(label)] > 0:
            covered.append(("domain_label", str(label)))
    if assigned_primary is not None and requirements["domain_primary"][primary] > 0:
        covered.append(("domain_primary", primary))
    if requirements["language"][language] > 0:
        covered.append(("language", language))
    if (
        primary != language_rubric["translation_domain"]
        and requirements["non_translation"][language] > 0
    ):
        covered.append(("non_translation", language))
    if (
        requirements["language_diversity"][language] > 0
        and primary not in language_domains[language]
    ):
        covered.append(("language_diversity", language))
    if language != "en":
        family = str(domain_rubric["domains"][primary]["family"])
        if requirements["non_english_family"][family] > 0:
            covered.append(("non_english_family", family))
    return covered


def primary_assignment(
    tag: dict[str, Any],
    requirements: dict[str, Counter[str]],
    minimum_confidence: float,
    tie_tolerance: float,
) -> str | None:
    original = str(tag["primary_label"])
    scores = {str(name): float(score) for name, score in tag["scores"].items()}
    maximum = max(scores.values())
    candidates = []
    for domain, needed in requirements["domain_primary"].items():
        if needed <= 0:
            continue
        confidence = float(
            tag.get("primary_confidence", scores.get(domain, 0.0))
            if domain == original
            else scores.get(domain, 0.0)
        )
        if confidence < minimum_confidence:
            continue
        distance = (
            0.0
            if domain == original and tag.get("primary_source") == "source_fact"
            else maximum - scores.get(domain, 0.0)
        )
        if domain != original and distance > tie_tolerance:
            continue
        candidates.append(
            (
                0 if domain == original else 1,
                distance,
                -confidence,
                -needed,
                domain,
            )
        )
    return min(candidates)[-1] if candidates else None


def assigned_tag(tag: dict[str, Any], assignment: str | None) -> dict[str, Any]:
    effective = dict(tag)
    if assignment is None or assignment == str(tag["primary_label"]):
        return effective
    scores = {str(name): float(score) for name, score in tag["scores"].items()}
    effective["pre_assignment_primary_label"] = str(tag["primary_label"])
    effective["primary_label"] = assignment
    effective["primary_confidence"] = scores[assignment]
    effective["primary_source"] = "near_tie_coverage_assignment"
    effective["primary_margin_from_classifier_max"] = max(scores.values()) - scores[assignment]
    return effective


def select_joint_supplement(
    baseline_records: list[dict[str, Any]],
    baseline_tags: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_tags: dict[str, dict[str, Any]],
    token_counts: dict[str, int],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
    partition: str,
    domain_margin: int,
    minimum_confidence: float,
    primary_tie_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_rubric(domain_rubric)
    validate_language_rubric(language_rubric, domain_rubric)
    requirements, language_domains = remaining_requirements(
        baseline_records,
        baseline_tags,
        domain_rubric,
        language_rubric,
        partition,
        domain_margin,
    )
    baseline_ids = {str(record["id"]) for record in baseline_records}
    if baseline_ids & {str(record["id"]) for record in candidates}:
        raise ValueError("supplement candidates overlap the baseline")
    weights = {
        "domain_label": 2,
        "domain_primary": 4,
        "language": 4,
        "non_translation": 2,
        "language_diversity": 3,
        "non_english_family": 3,
    }
    eligible = []
    for record in candidates:
        sample_id = str(record["id"])
        tag = candidate_tags[sample_id]
        confidence = float(
            tag.get(
                "primary_confidence",
                max(float(score) for score in tag["scores"].values()),
            )
        )
        eligible.append((record, tag, confidence))

    selected = []
    selected_tags = []
    selected_coverage = []
    while any(value > 0 for counter in requirements.values() for value in counter.values()):
        ranked = []
        for record, tag, confidence in eligible:
            assignment = primary_assignment(
                tag, requirements, minimum_confidence, primary_tie_tolerance
            )
            effective = assigned_tag(tag, assignment)
            coverage = candidate_coverage(
                record,
                effective,
                assignment,
                requirements,
                language_domains,
                domain_rubric,
                language_rubric,
            )
            if coverage:
                score = sum(weights[kind] for kind, _name in coverage)
                sample_id = str(record["id"])
                effective_confidence = float(
                    effective.get("primary_confidence", confidence)
                )
                ranked.append(
                    (
                        -score,
                        -effective_confidence,
                        token_counts[sample_id],
                        sample_id,
                        record,
                        effective,
                        coverage,
                    )
                )
        if not ranked:
            unresolved = {
                kind: {name: count for name, count in counter.items() if count > 0}
                for kind, counter in requirements.items()
                if any(count > 0 for count in counter.values())
            }
            raise ValueError(f"candidate pool cannot close coverage requirements: {unresolved}")
        ranked.sort(key=lambda item: item[:4])
        _score, _confidence, _tokens, sample_id, record, tag, coverage = ranked[0]
        selected.append(record)
        selected_tags.append(tag)
        selected_coverage.append({"id": sample_id, "coverage": coverage})
        eligible = [item for item in eligible if str(item[0]["id"]) != sample_id]
        primary = str(tag["primary_label"])
        language = str(record["language"])
        was_new_domain = primary not in language_domains[language]
        language_domains[language].add(primary)
        for kind, name in coverage:
            if kind == "language_diversity" and not was_new_domain:
                continue
            requirements[kind][name] = max(0, requirements[kind][name] - 1)
    return selected, selected_tags, {
        "selected_coverage": selected_coverage,
        "remaining_requirements": {
            kind: {name: count for name, count in counter.items() if count > 0}
            for kind, counter in requirements.items()
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tags_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(record["id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("tag file contains duplicate ids")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-tags", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-tags", type=Path, required=True)
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--language-rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--domain-margin", type=int, default=4)
    parser.add_argument("--minimum-confidence", type=float, default=0.8)
    parser.add_argument(
        "--primary-tie-tolerance",
        type=float,
        default=0.02,
        help="maximum classifier-score distance for an auditable sibling-domain assignment",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tags", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_tags.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite supplement output or report")
    if not 0 <= args.primary_tie_tolerance <= 1:
        raise SystemExit("--primary-tie-tolerance must be in [0, 1]")
    try:
        from transformers import AutoTokenizer

        baseline = read_jsonl(args.baseline)
        baseline_tags = tags_by_id(read_jsonl(args.baseline_tags))
        candidates = read_jsonl(args.candidates)
        candidate_tags = tags_by_id(read_jsonl(args.candidate_tags))
        if set(baseline_tags) != {str(record["id"]) for record in baseline}:
            raise ValueError("baseline records and tags differ")
        if set(candidate_tags) != {str(record["id"]) for record in candidates}:
            raise ValueError("candidate records and tags differ")
        domain_bytes = args.domain_rubric.read_bytes()
        language_bytes = args.language_rubric.read_bytes()
        domain_rubric = json.loads(domain_bytes)
        language_rubric = json.loads(language_bytes)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision
        )
        token_counts = {
            str(record["id"]): len(
                tokenizer(render_record(tokenizer, record), add_special_tokens=False).input_ids
            )
            for record in candidates
        }
        selected, selected_tags, selection = select_joint_supplement(
            baseline,
            baseline_tags,
            candidates,
            candidate_tags,
            token_counts,
            domain_rubric,
            language_rubric,
            args.partition,
            args.domain_margin,
            args.minimum_confidence,
            args.primary_tie_tolerance,
        )
        combined = baseline + selected
        combined_tags = dict(baseline_tags)
        combined_tags.update({str(tag["id"]): tag for tag in selected_tags})
        joint = coverage_report(
            combined,
            combined_tags,
            domain_rubric,
            language_rubric,
            args.partition,
        )
        label_counts: Counter[str] = Counter()
        primary_counts: Counter[str] = Counter()
        for tag in combined_tags.values():
            label_counts.update(tag["labels"])
            primary_counts[str(tag["primary_label"])] += 1
        multi_gaps, primary_gaps = quota_gaps(
            label_counts, primary_counts, domain_rubric, args.partition
        )
        if joint["status"] != "passed" or multi_gaps or primary_gaps:
            raise ValueError("selected supplement did not close all release coverage gates")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in selected:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    output_bytes = args.output.read_bytes()
    output_tags_sha256 = write_jsonl_atomic(args.output_tags, selected_tags)
    document = {
        "format": "ctox.joint-coverage-supplement-selection.v1",
        "partition": args.partition,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "baseline_tags_sha256": hashlib.sha256(args.baseline_tags.read_bytes()).hexdigest(),
        "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "candidate_tags_sha256": hashlib.sha256(args.candidate_tags.read_bytes()).hexdigest(),
        "domain_rubric_sha256": hashlib.sha256(domain_bytes).hexdigest(),
        "language_rubric_sha256": hashlib.sha256(language_bytes).hexdigest(),
        "minimum_confidence": args.minimum_confidence,
        "primary_tie_tolerance": args.primary_tie_tolerance,
        "domain_margin": args.domain_margin,
        "selected_samples": len(selected),
        "selected_tokens": sum(token_counts[str(record["id"])] for record in selected),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "output_bytes": len(output_bytes),
        "output_tags": str(args.output_tags.resolve()),
        "output_tags_bytes": args.output_tags.stat().st_size,
        "output_tags_sha256": output_tags_sha256,
        "joint_gate": joint,
        "domain_multi_label_gaps": multi_gaps,
        "domain_primary_gaps": primary_gaps,
        **selection,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
