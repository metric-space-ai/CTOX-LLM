#!/usr/bin/env python3
"""Compare direct and recovered fixed-qcode held-out evaluation reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from recovery_io import atomic_json


GAP_WEIGHTS = {
    "kl": 1.0,
    "hidden": 1.0,
    "mtp_kl": 0.5,
    "mtp_hidden": 0.5,
}


def distillation_gap(losses: dict[str, Any]) -> float:
    missing = set(GAP_WEIGHTS) - set(losses)
    if missing:
        raise ValueError(f"evaluation losses omit {sorted(missing)}")
    gap = sum(float(losses[name]) * weight for name, weight in GAP_WEIGHTS.items())
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("evaluation distillation gap is invalid")
    return gap


def compare_aggregate(direct: dict[str, Any], recovered: dict[str, Any]) -> dict[str, Any]:
    for key in ("records", "sequence_tokens", "target_counts"):
        if direct.get(key) != recovered.get(key):
            raise ValueError(f"evaluation aggregate {key} differs")
    direct_losses = direct["target_weighted_mean_losses"]
    recovered_losses = recovered["target_weighted_mean_losses"]
    if set(direct_losses) != set(recovered_losses):
        raise ValueError("evaluation loss families differ")
    direct_gap = distillation_gap(direct_losses)
    recovered_gap = distillation_gap(recovered_losses)
    closure = (direct_gap - recovered_gap) / direct_gap if direct_gap > 0 else 0.0
    return {
        "records": direct["records"],
        "sequence_tokens": direct["sequence_tokens"],
        "direct_distillation_gap": direct_gap,
        "recovered_distillation_gap": recovered_gap,
        "gap_closed_fraction": closure,
        "gap_closed_ppm": round(closure * 1_000_000),
        "target_weighted_loss_delta": {
            name: float(recovered_losses[name]) - float(direct_losses[name])
            for name in sorted(direct_losses)
        },
    }


def compare_reports(
    direct: dict[str, Any],
    recovered: dict[str, Any],
    minimum_gap_closed: float,
) -> dict[str, Any]:
    if not 0 <= minimum_gap_closed <= 1:
        raise ValueError("minimum gap closure must lie in [0, 1]")
    for label, report in (("direct", direct), ("recovered", recovered)):
        if report.get("format") != "ctox.recovery.heldout-evaluation.v1":
            raise ValueError(f"{label} evaluation format is unsupported")
        if report.get("status") != "complete":
            raise ValueError(f"{label} evaluation is not complete")
    identity_keys = (
        "model",
        "revision",
        "local_model_provenance_sha256",
        "logical_qcode_root_sha256",
        "teacher_cache_set_sha256",
        "teacher_artifact_root_sha256",
        "materialized_sha256",
        "domain_tags_sha256",
        "service_tags_sha256",
        "prefill_chunk_tokens",
        "compute_dtype",
    )
    for key in identity_keys:
        if direct.get(key) != recovered.get(key):
            raise ValueError(f"evaluation identity {key} differs")
    direct_ids = [sample["id"] for sample in direct.get("samples", [])]
    recovered_ids = [sample["id"] for sample in recovered.get("samples", [])]
    if not direct_ids or direct_ids != recovered_ids:
        raise ValueError("evaluation sample order or identity differs")

    direct_aggregates = direct["aggregates"]
    recovered_aggregates = recovered["aggregates"]
    overall = compare_aggregate(
        direct_aggregates["overall"], recovered_aggregates["overall"]
    )
    group_comparisons = {}
    regressions = []
    for family, direct_groups in direct_aggregates["groups"].items():
        recovered_groups = recovered_aggregates["groups"].get(family)
        if recovered_groups is None or set(direct_groups) != set(recovered_groups):
            raise ValueError(f"evaluation {family} groups differ")
        group_comparisons[family] = {}
        for name, direct_group in direct_groups.items():
            comparison = compare_aggregate(direct_group, recovered_groups[name])
            group_comparisons[family][name] = comparison
            if comparison["gap_closed_fraction"] < 0:
                regressions.append(
                    {
                        "family": family,
                        "name": name,
                        "gap_closed_fraction": comparison["gap_closed_fraction"],
                    }
                )
    passed = overall["gap_closed_fraction"] >= minimum_gap_closed and not regressions
    return {
        "format": "ctox.recovery.heldout-comparison.v1",
        "status": "passed" if passed else "failed",
        "minimum_gap_closed_fraction": minimum_gap_closed,
        "logical_qcode_root_sha256": direct["logical_qcode_root_sha256"],
        "teacher_cache_set_sha256": direct["teacher_cache_set_sha256"],
        "direct_artifact_sha256": direct["artifact_sha256"],
        "recovered_artifact_sha256": recovered["artifact_sha256"],
        "overall": overall,
        "group_regressions": regressions,
        "groups": group_comparisons,
        "scope": (
            "BF16 distillation KL/hidden gap closure; task accuracy, generation, "
            "tool execution, and 128K retrieval gates remain separate"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gap-closed", type=float, default=0.30)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        direct = json.loads(args.direct.read_text(encoding="utf-8"))
        recovered = json.loads(args.recovered.read_text(encoding="utf-8"))
        result = compare_reports(direct, recovered, args.minimum_gap_closed)
        atomic_json(args.output, result)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if result["status"] != "passed":
        raise SystemExit("held-out recovery comparison failed")


if __name__ == "__main__":
    main()
