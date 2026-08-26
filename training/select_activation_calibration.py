#!/usr/bin/env python3
"""Select a deterministic, release-cohort-only activation calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CATEGORY_MINIMA = {
    "agentic": 40,
    "chat": 64,
    "code": 40,
    "long_context": 12,
    "math": 40,
}
DEFAULT_LENGTH_MINIMA = {
    "4k_16k": 16,
    "16k_32k": 8,
    "32k_64k": 4,
    "over_96k": 3,
}
FEATURE_WEIGHTS = {
    "domain": 4.0,
    "language": 4.0,
    "service": 2.0,
    "category": 3.0,
    "length": 3.0,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(record["id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"{label} contains duplicate sample ids")
    return result


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def sequence_bucket(tokens: int) -> str:
    if tokens <= 4_096:
        return "up_to_4k"
    if tokens <= 16_384:
        return "4k_16k"
    if tokens <= 32_768:
        return "16k_32k"
    if tokens <= 65_536:
        return "32k_64k"
    if tokens <= 98_304:
        return "64k_96k"
    return "over_96k"


def sample_features(
    record: dict[str, Any],
    domain_tag: dict[str, Any],
    service_tag: dict[str, Any],
    tokens: int,
) -> set[tuple[str, str]]:
    return {
        ("domain", str(domain_tag["primary_label"])),
        ("language", str(record["language"])),
        ("category", str(record["category"])),
        ("length", sequence_bucket(tokens)),
        *{("service", str(label)) for label in service_tag["labels"]},
    }


def select_calibration(
    records: list[dict[str, Any]],
    domain_tags: dict[str, dict[str, Any]],
    service_tags: dict[str, dict[str, Any]],
    token_counts: dict[str, int],
    requirements: dict[tuple[str, str], int],
    maximum_samples: int,
    maximum_sequence_tokens: int,
) -> tuple[list[str], Counter[tuple[str, str]]]:
    if maximum_samples <= 0 or maximum_sequence_tokens <= 0:
        raise ValueError("calibration sample and sequence limits must be positive")
    record_ids = {str(record["id"]) for record in records}
    for label, values in (
        ("domain tags", domain_tags),
        ("service tags", service_tags),
        ("token counts", token_counts),
    ):
        if set(values) != record_ids:
            raise ValueError(f"{label} differ from the release cohort")
    if any(required <= 0 for required in requirements.values()):
        raise ValueError("calibration requirements must be positive")

    candidates = []
    for record in records:
        sample_id = str(record["id"])
        tokens = int(token_counts[sample_id])
        if tokens <= maximum_sequence_tokens:
            candidates.append(
                (
                    sample_id,
                    tokens,
                    sample_features(
                        record,
                        domain_tags[sample_id],
                        service_tags[sample_id],
                        tokens,
                    ),
                )
            )
    if not candidates:
        raise ValueError("no calibration samples fit the sequence limit")
    available = Counter(feature for _, _, features in candidates for feature in features)
    impossible = {
        feature: {"available": available[feature], "required": required}
        for feature, required in requirements.items()
        if available[feature] < required
    }
    if impossible:
        raise ValueError(f"release cohort cannot satisfy calibration requirements: {impossible}")

    counts: Counter[tuple[str, str]] = Counter()
    selected: list[str] = []
    remaining = {sample_id: (tokens, features) for sample_id, tokens, features in candidates}
    while any(counts[feature] < required for feature, required in requirements.items()):
        ranked = []
        for sample_id, (tokens, features) in remaining.items():
            gain = sum(
                FEATURE_WEIGHTS[kind]
                for kind, name in features
                if (kind, name) in requirements
                and counts[(kind, name)] < requirements[(kind, name)]
            )
            if gain > 0:
                ranked.append((-gain, tokens, sample_id, features))
        if not ranked:
            unresolved = {
                feature: required - counts[feature]
                for feature, required in requirements.items()
                if counts[feature] < required
            }
            raise ValueError(f"calibration selection cannot close requirements: {unresolved}")
        _negative_gain, _tokens, sample_id, features = min(ranked)
        selected.append(sample_id)
        counts.update(features)
        del remaining[sample_id]
        if len(selected) > maximum_samples:
            raise ValueError("calibration requirements exceed the sample budget")

    while len(selected) < maximum_samples and remaining:
        ranked = []
        for sample_id, (tokens, features) in remaining.items():
            diversity = sum(
                FEATURE_WEIGHTS[kind] / (counts[(kind, name)] + 1)
                for kind, name in features
            )
            cost_adjusted = diversity / math.sqrt(max(tokens, 1))
            ranked.append((-cost_adjusted, tokens, sample_id, features))
        _negative_score, _tokens, sample_id, features = min(ranked)
        selected.append(sample_id)
        counts.update(features)
        del remaining[sample_id]
    if len(selected) != maximum_samples:
        raise ValueError("release cohort has fewer eligible calibration samples than requested")
    return selected, counts


def load_token_counts(paths: list[Path], expected_ids: set[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for sample in document["samples"]:
            sample_id = str(sample["id"])
            if sample_id not in expected_ids:
                continue
            tokens = int(sample["sequence_tokens"])
            if sample_id in result and result[sample_id] != tokens:
                raise ValueError(f"cache plans disagree on tokens for {sample_id}")
            result[sample_id] = tokens
    if set(result) != expected_ids:
        raise ValueError("cache plans do not cover the complete release cohort")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--domain-tags", type=Path, required=True)
    parser.add_argument("--service-tags", type=Path, required=True)
    parser.add_argument("--cache-plan", type=Path, action="append", required=True)
    parser.add_argument("--domain-rubric", type=Path, required=True)
    parser.add_argument("--language-rubric", type=Path, required=True)
    parser.add_argument("--service-rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=131_072)
    parser.add_argument("--minimum-per-domain", type=int, default=4)
    parser.add_argument("--minimum-per-language", type=int, default=6)
    parser.add_argument("--minimum-per-service", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite calibration output or report")
    try:
        records = load_jsonl(args.input)
        domain_tags = indexed(load_jsonl(args.domain_tags), "domain tags")
        service_tags = indexed(load_jsonl(args.service_tags), "service tags")
        record_ids = {str(record["id"]) for record in records}
        if len(record_ids) != len(records):
            raise ValueError("release cohort contains duplicate sample ids")
        token_counts = load_token_counts(args.cache_plan, record_ids)
        domain_rubric = json.loads(args.domain_rubric.read_text(encoding="utf-8"))
        language_rubric = json.loads(args.language_rubric.read_text(encoding="utf-8"))
        service_rubric = json.loads(args.service_rubric.read_text(encoding="utf-8"))
        requirements = {
            **{
                ("domain", str(name)): args.minimum_per_domain
                for name in domain_rubric["domains"]
            },
            **{
                ("language", str(name)): args.minimum_per_language
                for name in language_rubric["languages"]
            },
            **{
                ("service", str(name)): args.minimum_per_service
                for name in service_rubric["modes"]
            },
            **{
                ("category", name): minimum
                for name, minimum in DEFAULT_CATEGORY_MINIMA.items()
            },
            **{
                ("length", name): minimum
                for name, minimum in DEFAULT_LENGTH_MINIMA.items()
            },
        }
        selected_ids, feature_counts = select_calibration(
            records,
            domain_tags,
            service_tags,
            token_counts,
            requirements,
            args.samples,
            args.max_sequence_tokens,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    selected = set(selected_ids)
    selected_records = [record for record in records if str(record["id"]) in selected]
    encoded = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in selected_records
    ).encode("utf-8")
    write_bytes_atomic(args.output, encoded)
    report = {
        "format": "ctox.activation-calibration-selection.v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "domain_tags_sha256": hashlib.sha256(args.domain_tags.read_bytes()).hexdigest(),
        "service_tags_sha256": hashlib.sha256(args.service_tags.read_bytes()).hexdigest(),
        "cache_plan_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in args.cache_plan
        ],
        "rubric_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("domain", args.domain_rubric),
                ("language", args.language_rubric),
                ("service", args.service_rubric),
            )
        },
        "samples": len(selected_records),
        "sequence_tokens": sum(token_counts[str(record["id"])] for record in selected_records),
        "maximum_sequence_tokens": max(
            token_counts[str(record["id"])] for record in selected_records
        ),
        "output": str(args.output.resolve()),
        "output_bytes": len(encoded),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "requirements": {
            f"{kind}:{name}": required
            for (kind, name), required in sorted(requirements.items())
        },
        "feature_counts": {
            f"{kind}:{name}": feature_counts[(kind, name)]
            for kind, name in sorted(feature_counts)
        },
        "selected_ids": [str(record["id"]) for record in selected_records],
    }
    write_bytes_atomic(
        args.report,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


if __name__ == "__main__":
    main()
