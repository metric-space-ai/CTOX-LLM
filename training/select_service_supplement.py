#!/usr/bin/env python3
"""Select a deterministic disjoint supplement for service-mode coverage gaps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_service_coverage import indexed_tags, read_jsonl, service_coverage_report
from classify_domains import write_json_atomic
from classify_service_modes import validate_rubric
from merge_domain_tags import write_jsonl_atomic


def build_state(
    records: list[dict[str, Any]],
    domain_tags: dict[str, dict[str, Any]],
    service_tags: dict[str, dict[str, Any]],
    domain_rubric: dict[str, Any],
    service_rubric: dict[str, Any],
    partition: str,
) -> dict[str, Any]:
    modes = service_rubric["modes"]
    mode_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    domain_modes: dict[str, set[str]] = defaultdict(set)
    family_modes: dict[str, set[str]] = defaultdict(set)
    language_modes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sample_id = str(record["id"])
        domain = str(domain_tags[sample_id]["primary_label"])
        family = str(domain_rubric["domains"][domain]["family"])
        language = str(record["language"])
        labels = {str(label) for label in service_tags[sample_id]["labels"]}
        mode_counts.update(labels)
        domain_modes[domain].update(labels)
        family_modes[family].update(labels)
        language_modes[language].update(labels)
        pair_counts.update((domain, label) for label in labels)

    mode_remaining = Counter(
        {
            mode: max(0, int(requirements[f"minimum_{partition}"]) - mode_counts[mode])
            for mode, requirements in modes.items()
        }
    )
    domain_minimum = int(
        service_rubric["policy"][f"minimum_distinct_modes_per_domain_{partition}"]
    )
    domain_remaining = Counter(
        {
            domain: max(0, domain_minimum - len(domain_modes[domain]))
            for domain in domain_rubric["domains"]
        }
    )
    family_minimum = int(
        service_rubric["policy"][f"minimum_distinct_modes_per_family_{partition}"]
    )
    family_remaining = Counter(
        {
            family: max(0, family_minimum - len(family_modes[family]))
            for family in domain_rubric["policy"]["required_families"]
        }
    )
    language_remaining = Counter(
        {
            language: max(
                0,
                int(minima[partition]) - len(language_modes[language]),
            )
            for language, minima in service_rubric[
                "language_minimum_distinct_modes"
            ].items()
        }
    )
    pair_remaining: Counter[tuple[str, str]] = Counter()
    for domain, pairs in service_rubric["required_domain_mode_pairs"].items():
        for mode, minima in pairs.items():
            pair_remaining[(domain, mode)] = max(
                0,
                int(minima[partition]) - pair_counts[(domain, mode)],
            )
    return {
        "mode_remaining": mode_remaining,
        "domain_remaining": domain_remaining,
        "family_remaining": family_remaining,
        "language_remaining": language_remaining,
        "pair_remaining": pair_remaining,
        "domain_modes": domain_modes,
        "family_modes": family_modes,
        "language_modes": language_modes,
    }


def candidate_contributions(
    record: dict[str, Any],
    domain_tag: dict[str, Any],
    service_tag: dict[str, Any],
    state: dict[str, Any],
    domain_rubric: dict[str, Any],
) -> list[tuple[str, Any]]:
    domain = str(domain_tag["primary_label"])
    family = str(domain_rubric["domains"][domain]["family"])
    language = str(record["language"])
    labels = sorted({str(label) for label in service_tag["labels"]})
    contributions: list[tuple[str, Any]] = []
    for label in labels:
        if state["mode_remaining"][label] > 0:
            contributions.append(("mode", label))
        if state["pair_remaining"][(domain, label)] > 0:
            contributions.append(("pair", (domain, label)))
    new_domain = [label for label in labels if label not in state["domain_modes"][domain]]
    new_family = [label for label in labels if label not in state["family_modes"][family]]
    new_language = [label for label in labels if label not in state["language_modes"][language]]
    contributions.extend(
        ("domain", (domain, label))
        for label in new_domain[: state["domain_remaining"][domain]]
    )
    contributions.extend(
        ("family", (family, label))
        for label in new_family[: state["family_remaining"][family]]
    )
    contributions.extend(
        ("language", (language, label))
        for label in new_language[: state["language_remaining"][language]]
    )
    return contributions


def consume(
    record: dict[str, Any],
    domain_tag: dict[str, Any],
    service_tag: dict[str, Any],
    state: dict[str, Any],
    domain_rubric: dict[str, Any],
) -> None:
    domain = str(domain_tag["primary_label"])
    family = str(domain_rubric["domains"][domain]["family"])
    language = str(record["language"])
    labels = {str(label) for label in service_tag["labels"]}
    for label in labels:
        if state["mode_remaining"][label] > 0:
            state["mode_remaining"][label] -= 1
        pair = (domain, label)
        if state["pair_remaining"][pair] > 0:
            state["pair_remaining"][pair] -= 1
    for key, owner in (
        ("domain", domain),
        ("family", family),
        ("language", language),
    ):
        mode_set = state[f"{key}_modes"][owner]
        added = len(labels - mode_set)
        mode_set.update(labels)
        state[f"{key}_remaining"][owner] = max(
            0, state[f"{key}_remaining"][owner] - added
        )


def unresolved(state: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            ("/".join(key) if isinstance(key, tuple) else key): value
            for key, value in values.items()
            if value > 0
        }
        for name, values in state.items()
        if name.endswith("_remaining")
    }


def select_supplement(
    baseline: list[dict[str, Any]],
    baseline_domain_tags: dict[str, dict[str, Any]],
    baseline_service_tags: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    candidate_domain_tags: dict[str, dict[str, Any]],
    candidate_service_tags: dict[str, dict[str, Any]],
    token_costs: dict[str, int],
    domain_rubric: dict[str, Any],
    language_rubric: dict[str, Any],
    service_rubric: dict[str, Any],
    partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_rubric(service_rubric, domain_rubric, language_rubric)
    baseline_ids = {str(record["id"]) for record in baseline}
    candidate_ids = {str(record["id"]) for record in candidates}
    if len(baseline_ids) != len(baseline) or len(candidate_ids) != len(candidates):
        raise ValueError("baseline or candidate pool contains duplicate ids")
    overlap = baseline_ids & candidate_ids
    if overlap:
        raise ValueError(f"candidate pool overlaps baseline ids: {sorted(overlap)[:5]}")
    baseline_payloads = {
        str(record["prompt_sha256"])
        for record in baseline
        if record.get("prompt_sha256")
    }
    candidate_payloads = {
        str(record["prompt_sha256"])
        for record in candidates
        if record.get("prompt_sha256")
    }
    payload_overlap = baseline_payloads & candidate_payloads
    if payload_overlap:
        raise ValueError(
            f"candidate pool overlaps baseline payloads: {sorted(payload_overlap)[:5]}"
        )
    expected_tag_sets = (
        ("baseline domain", set(baseline_domain_tags), baseline_ids),
        ("baseline service", set(baseline_service_tags), baseline_ids),
        ("candidate domain", set(candidate_domain_tags), candidate_ids),
        ("candidate service", set(candidate_service_tags), candidate_ids),
        ("token cost", set(token_costs), candidate_ids),
    )
    for name, observed, expected in expected_tag_sets:
        if observed != expected:
            raise ValueError(f"{name} ids differ from its materialized cohort")
    declared_domains = set(domain_rubric["domains"])
    declared_languages = set(language_rubric["languages"])
    declared_modes = set(service_rubric["modes"])
    for record in baseline + candidates:
        sample_id = str(record["id"])
        domain_tag = (
            baseline_domain_tags[sample_id]
            if sample_id in baseline_ids
            else candidate_domain_tags[sample_id]
        )
        service_tag = (
            baseline_service_tags[sample_id]
            if sample_id in baseline_ids
            else candidate_service_tags[sample_id]
        )
        if str(record["language"]) not in declared_languages:
            raise ValueError(f"sample {sample_id} has an undeclared language")
        if str(domain_tag["primary_label"]) not in declared_domains:
            raise ValueError(f"sample {sample_id} has an unknown primary domain")
        labels = {str(label) for label in service_tag.get("labels", [])}
        if not labels or not labels <= declared_modes:
            raise ValueError(f"sample {sample_id} has invalid service-mode labels")
    if any(cost <= 0 for cost in token_costs.values()):
        raise ValueError("candidate token costs must be positive")
    state = build_state(
        baseline,
        baseline_domain_tags,
        baseline_service_tags,
        domain_rubric,
        service_rubric,
        partition,
    )
    weights = {"pair": 8, "mode": 4, "domain": 4, "family": 2, "language": 2}
    remaining = {str(record["id"]): record for record in candidates}
    selected: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    while any(unresolved(state).values()):
        ranked = []
        for sample_id, record in remaining.items():
            contributions = candidate_contributions(
                record,
                candidate_domain_tags[sample_id],
                candidate_service_tags[sample_id],
                state,
                domain_rubric,
            )
            score = sum(weights[kind] for kind, _ in contributions)
            if score:
                ranked.append(
                    (-score, int(token_costs[sample_id]), sample_id, contributions, record)
                )
        if not ranked:
            raise ValueError(f"candidate pool cannot close service gaps: {unresolved(state)}")
        _, token_cost, sample_id, contributions, record = min(ranked)
        selected.append(record)
        trace.append(
            {
                "id": sample_id,
                "tokens": token_cost,
                "contributions": [
                    [kind, "/".join(key) if isinstance(key, tuple) else key]
                    for kind, key in contributions
                ],
            }
        )
        consume(
            record,
            candidate_domain_tags[sample_id],
            candidate_service_tags[sample_id],
            state,
            domain_rubric,
        )
        del remaining[sample_id]
    return selected, {"trace": trace, "remaining_requirements": unresolved(state)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-domain-tags", type=Path, required=True)
    parser.add_argument("--baseline-service-tags", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-domain-tags", type=Path, required=True)
    parser.add_argument("--candidate-service-tags", type=Path, required=True)
    parser.add_argument("--token-costs", type=Path, required=True)
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--language-rubric", type=Path, required=True)
    parser.add_argument("--service-rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    try:
        baseline = read_jsonl(args.baseline)
        candidates = read_jsonl(args.candidates)
        domain_rubric = json.loads(args.domain_rubric.read_bytes())
        language_rubric = json.loads(args.language_rubric.read_bytes())
        service_rubric = json.loads(args.service_rubric.read_bytes())
        validate_rubric(service_rubric, domain_rubric, language_rubric)
        baseline_domain_tags = indexed_tags(args.baseline_domain_tags)
        baseline_service_tags = indexed_tags(args.baseline_service_tags)
        candidate_domain_tags = indexed_tags(args.candidate_domain_tags)
        candidate_service_tags = indexed_tags(args.candidate_service_tags)
        token_costs = {
            str(key): int(value)
            for key, value in json.loads(args.token_costs.read_bytes()).items()
        }
        selected, evidence = select_supplement(
            baseline,
            baseline_domain_tags,
            baseline_service_tags,
            candidates,
            candidate_domain_tags,
            candidate_service_tags,
            token_costs,
            domain_rubric,
            language_rubric,
            service_rubric,
            args.partition,
        )
        selected_sha256 = write_jsonl_atomic(args.output, selected)
        selected_ids = {str(record["id"]) for record in selected}
        combined = baseline + selected
        combined_domain_tags = dict(baseline_domain_tags)
        combined_domain_tags.update(
            {sample_id: candidate_domain_tags[sample_id] for sample_id in selected_ids}
        )
        combined_service_tags = dict(baseline_service_tags)
        combined_service_tags.update(
            {sample_id: candidate_service_tags[sample_id] for sample_id in selected_ids}
        )
        final_report = service_coverage_report(
            combined,
            combined_domain_tags,
            combined_service_tags,
            domain_rubric,
            language_rubric,
            service_rubric,
            args.partition,
        )
        if final_report["status"] != "passed":
            raise ValueError("selected supplement did not pass the complete service audit")
        write_json_atomic(
            args.evidence,
            {
                "format": "ctox.recovery-service-supplement.v1",
                "partition": args.partition,
                "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                "candidates_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
                "selected_records": len(selected),
                "selected_tokens": sum(token_costs[str(record["id"])] for record in selected),
                "selected_sha256": selected_sha256,
                **evidence,
                "final_report": final_report,
            },
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
