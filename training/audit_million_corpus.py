#!/usr/bin/env python3
"""Validate the immutable million-sample recovery corpus evidence.

This is the final, cheap admission gate before teacher-cache planning.  The
input is deliberately an evidence document produced from materialized records,
domain/service tags, token-count plans, provenance manifests, and the frozen
semantic-dedup assignment.  Counts without their content hashes are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from recovery_io import atomic_json


POLICY_FORMAT = "ctox.recovery-million-corpus-policy.v1"
EVIDENCE_FORMAT = "ctox.recovery-million-corpus-evidence.v1"
AUDIT_FORMAT = "ctox.recovery-million-corpus-audit.v1"
PARTITIONS = ("train", "calibration", "held_out")
REQUIRED_BINDINGS = (
    "materialized_sha256",
    "domain_tags_sha256",
    "service_tags_sha256",
    "token_plan_sha256",
    "provenance_manifest_sha256",
    "semantic_dedup_sha256",
    "content_root_sha256",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: Any, field: str) -> str:
    encoded = str(value)
    if len(encoded) != 64 or any(character not in "0123456789abcdef" for character in encoded):
        raise ValueError(f"{field} is not a lowercase SHA-256")
    return encoded


def require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def validate_fraction_map(values: dict[str, Any], field: str, total: float | None = None) -> None:
    observed = 0.0
    if not values:
        raise ValueError(f"{field} cannot be empty")
    for name, raw in values.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{field}.{name} must lie in [0, 1]")
        observed += value
    if total is not None and not math.isclose(observed, total, abs_tol=1e-12):
        raise ValueError(f"{field} fractions must sum to {total}, got {observed}")


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("format") != POLICY_FORMAT:
        raise ValueError("million-corpus policy has the wrong format")
    if set(policy.get("partitions", {})) != set(PARTITIONS):
        raise ValueError("million-corpus policy must declare train/calibration/held_out")
    for partition in PARTITIONS:
        minimum = require_nonnegative_int(
            policy["partitions"][partition].get("minimum_records"),
            f"partitions.{partition}.minimum_records",
        )
        if minimum == 0:
            raise ValueError(f"partitions.{partition}.minimum_records must be positive")
    validate_fraction_map(policy["primary_mix_minimum_fractions"], "primary_mix", 1.0)
    validate_fraction_map(policy["context_mix_minimum_fractions"], "context_mix", 1.0)
    language = policy["language_policy"]
    validate_fraction_map(language["minimum_fractions"], "language minimum")
    validate_fraction_map(language["maximum_fractions"], "language maximum")
    other = float(language["other_languages_minimum_fraction"])
    if not 0.0 <= other <= 1.0:
        raise ValueError("other language minimum must lie in [0, 1]")
    if set(language["minimum_fractions"]) & set(language["maximum_fractions"]):
        raise ValueError("a language cannot have both a minimum and maximum policy")
    for name, expected in policy["hard_gates"].items():
        require_nonnegative_int(expected, f"hard_gates.{name}")


def count_map(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return {
        str(name): require_nonnegative_int(count, f"{field}.{name}")
        for name, count in value.items()
    }


def fraction_gaps(
    counts: dict[str, int],
    records: int,
    minima: dict[str, Any],
    field: str,
) -> dict[str, dict[str, int | float]]:
    gaps: dict[str, dict[str, int | float]] = {}
    for name, raw_fraction in minima.items():
        fraction = float(raw_fraction)
        required = math.ceil(records * fraction)
        observed = counts.get(name, 0)
        if observed < required:
            gaps[name] = {
                "observed": observed,
                "required": required,
                "minimum_fraction": fraction,
            }
    return gaps


def audit(policy: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    validate_policy(policy)
    if evidence.get("format") != EVIDENCE_FORMAT:
        raise ValueError("million-corpus evidence has the wrong format")
    if set(evidence.get("partitions", {})) != set(PARTITIONS):
        raise ValueError("million-corpus evidence must contain exactly three partitions")

    hard_gate_gaps: dict[str, dict[str, int]] = {}
    for name, raw_expected in policy["hard_gates"].items():
        expected = require_nonnegative_int(raw_expected, f"hard_gates.{name}")
        observed = require_nonnegative_int(evidence.get("hard_gates", {}).get(name), name)
        if observed != expected:
            hard_gate_gaps[name] = {"observed": observed, "required": expected}

    partition_reports: dict[str, Any] = {}
    for partition in PARTITIONS:
        current = evidence["partitions"][partition]
        records = require_nonnegative_int(current.get("records"), f"{partition}.records")
        unique_ids = require_nonnegative_int(current.get("unique_ids"), f"{partition}.unique_ids")
        unique_payloads = require_nonnegative_int(
            current.get("unique_payloads"), f"{partition}.unique_payloads"
        )
        unique_semantic_clusters = require_nonnegative_int(
            current.get("unique_semantic_clusters"),
            f"{partition}.unique_semantic_clusters",
        )
        for binding in REQUIRED_BINDINGS:
            require_sha256(current.get(binding), f"{partition}.{binding}")
        binding_paths = current.get("binding_paths")
        if not isinstance(binding_paths, dict) or set(binding_paths) != set(REQUIRED_BINDINGS[:-1]):
            raise ValueError(
                f"{partition}.binding_paths must name every source binding except content_root"
            )
        for name, path in binding_paths.items():
            if not isinstance(path, str) or not Path(path).is_absolute():
                raise ValueError(f"{partition}.binding_paths.{name} must be absolute")

        minimum_records = int(policy["partitions"][partition]["minimum_records"])
        cardinality_gaps = {}
        if records < minimum_records:
            cardinality_gaps["records"] = {
                "observed": records,
                "required": minimum_records,
            }
        for name, observed in (
            ("unique_ids", unique_ids),
            ("unique_payloads", unique_payloads),
            ("unique_semantic_clusters", unique_semantic_clusters),
        ):
            if observed != records:
                cardinality_gaps[name] = {"observed": observed, "required": records}

        primary = count_map(current.get("primary_mix"), f"{partition}.primary_mix")
        context = count_map(current.get("context_mix"), f"{partition}.context_mix")
        languages = count_map(current.get("languages"), f"{partition}.languages")
        if sum(primary.values()) != records:
            raise ValueError(f"{partition}.primary_mix does not sum to records")
        if sum(context.values()) != records:
            raise ValueError(f"{partition}.context_mix does not sum to records")
        if sum(languages.values()) != records:
            raise ValueError(f"{partition}.languages does not sum to records")

        primary_gaps = fraction_gaps(
            primary,
            records,
            policy["primary_mix_minimum_fractions"],
            f"{partition}.primary_mix",
        )
        context_gaps = fraction_gaps(
            context,
            records,
            policy["context_mix_minimum_fractions"],
            f"{partition}.context_mix",
        )
        language_policy = policy["language_policy"]
        language_gaps = fraction_gaps(
            languages,
            records,
            language_policy["minimum_fractions"],
            f"{partition}.languages",
        )
        for language, raw_fraction in language_policy["maximum_fractions"].items():
            maximum_fraction = float(raw_fraction)
            allowed = math.floor(records * maximum_fraction)
            observed = languages.get(language, 0)
            if observed > allowed:
                language_gaps[language] = {
                    "observed": observed,
                    "maximum": allowed,
                    "maximum_fraction": maximum_fraction,
                }
        declared = set(language_policy["minimum_fractions"]) | set(
            language_policy["maximum_fractions"]
        )
        other_observed = sum(count for name, count in languages.items() if name not in declared)
        other_required = math.ceil(
            records * float(language_policy["other_languages_minimum_fraction"])
        )
        if other_observed < other_required:
            language_gaps["__other__"] = {
                "observed": other_observed,
                "required": other_required,
                "minimum_fraction": float(
                    language_policy["other_languages_minimum_fraction"]
                ),
            }

        all_gaps = cardinality_gaps, primary_gaps, context_gaps, language_gaps
        partition_reports[partition] = {
            "records": records,
            "cardinality_gaps": cardinality_gaps,
            "primary_mix_gaps": primary_gaps,
            "context_mix_gaps": context_gaps,
            "language_gaps": language_gaps,
            "status": "passed" if not any(all_gaps) else "failed",
        }

    status = "passed"
    if hard_gate_gaps or any(
        report["status"] != "passed" for report in partition_reports.values()
    ):
        status = "failed"
    return {
        "format": AUDIT_FORMAT,
        "status": status,
        "partitions": partition_reports,
        "hard_gate_gaps": hard_gate_gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        policy_bytes = args.policy.read_bytes()
        evidence_bytes = args.evidence.read_bytes()
        report = audit(json.loads(policy_bytes), json.loads(evidence_bytes))
        document = {
            **report,
            "policy": str(args.policy.resolve()),
            "policy_sha256": sha256_bytes(policy_bytes),
            "evidence": str(args.evidence.resolve()),
            "evidence_sha256": sha256_bytes(evidence_bytes),
        }
        if report["status"] != "passed":
            raise ValueError(json.dumps(document, sort_keys=True))
        atomic_json(args.output, document)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
