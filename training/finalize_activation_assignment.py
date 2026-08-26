#!/usr/bin/env python3
"""Merge verified activation batches and derive the release Q2/Q4 assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_activation_batches import completed_batch_matches


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_artifacts(
    output_root: Path,
    output_prefix: str,
    batches: list[dict[str, Any]],
    batch_plan_sha256: str,
    quant_plan_sha256: str,
    model: str,
    revision: str,
    provenance_sha256: str,
) -> list[Path]:
    artifacts: list[Path] = []
    for batch in batches:
        index = int(batch["batch_index"])
        artifact = output_root / f"{output_prefix}-batch-{index:03d}-v1.safetensors"
        verification_path = (
            output_root
            / f"{output_prefix}-batch-{index:03d}-v1-verification-v1.json"
        )
        if not artifact.is_file() or not verification_path.is_file():
            raise ValueError(f"activation batch {index} is not complete")
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if not completed_batch_matches(
            artifact,
            verification,
            batch,
            batch_plan_sha256,
            quant_plan_sha256,
            model,
            revision,
            provenance_sha256,
        ):
            raise ValueError(f"activation batch {index} does not match its verification")
        artifacts.append(artifact)
    return artifacts


def merged_artifact_matches(
    path: Path,
    artifacts: list[Path],
    batches: list[dict[str, Any]],
    plan_sha256: str,
    provenance_sha256: str,
    safe_open: Any,
) -> bool:
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        return (
            metadata.get("format") == "ctox.activation-diagonal.v1"
            and metadata.get("quant_plan_sha256") == plan_sha256
            and metadata.get("local_model_provenance_sha256") == provenance_sha256
            and json.loads(metadata.get("input_sha256", "[]"))
            == [sha256(artifact) for artifact in artifacts]
            and int(metadata.get("merged_batches", -1)) == len(artifacts)
            and int(metadata.get("samples", -1))
            == sum(int(batch["samples"]) for batch in batches)
            and int(metadata.get("tokens", -1))
            == sum(int(batch["sequence_tokens"]) for batch in batches)
        )


def sensitivity_matches(
    path: Path,
    stats: Path,
    plan_sha256: str,
    provenance_sha256: str,
) -> bool:
    document = json.loads(path.read_text(encoding="utf-8"))
    return (
        document.get("format") == "ctox.q2q4.sensitivity.v1"
        and document.get("quant_plan_sha256") == plan_sha256
        and document.get("local_model_provenance_sha256") == provenance_sha256
        and document.get("activation_stats_sha256") == sha256(stats)
        and bool(document.get("candidates"))
        and all(candidate.get("observed") for candidate in document["candidates"])
    )


def assignment_matches(
    path: Path,
    sensitivity: Path,
    plan_sha256: str,
    budget_bytes: int,
) -> bool:
    document = json.loads(path.read_text(encoding="utf-8"))
    return (
        document.get("format") == "ctox.q2q4.assignment.v2"
        and document.get("plan_sha256") == plan_sha256
        and document.get("sensitivity_sha256") == sha256(sensitivity)
        and int(document.get("budget_bytes", -1)) == budget_bytes
        and int(document.get("bytes_used", budget_bytes + 1)) <= budget_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--merged-stats", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--row-group-size", type=int, default=256)
    args = parser.parse_args()
    batch_plan_bytes = args.batch_plan.read_bytes()
    batch_plan = json.loads(batch_plan_bytes)
    if batch_plan.get("format") != "ctox.activation-batch-plan.v1":
        raise SystemExit("unsupported activation batch-plan format")
    if sha256(args.input) != batch_plan.get("input_sha256"):
        raise SystemExit("activation input does not match the batch plan")
    plan_sha256 = sha256(args.plan)
    provenance_sha256 = sha256(args.local_model_provenance)
    try:
        artifacts = verified_artifacts(
            args.artifact_root,
            args.artifact_prefix,
            batch_plan["batches"],
            hashlib.sha256(batch_plan_bytes).hexdigest(),
            plan_sha256,
            str(args.checkpoint),
            args.revision,
            provenance_sha256,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before finalization") from error
    if args.merged_stats.exists():
        if not merged_artifact_matches(
            args.merged_stats,
            artifacts,
            batch_plan["batches"],
            plan_sha256,
            provenance_sha256,
            safe_open,
        ):
            raise SystemExit("existing merged activation statistics do not match")
        print("stage=merge status=verified-skip", flush=True)
    else:
        merge_command = [
            sys.executable,
            str(Path(__file__).with_name("merge_activation_stats.py")),
        ]
        for artifact in artifacts:
            merge_command.extend(("--input", str(artifact)))
        merge_command.extend(("--output", str(args.merged_stats)))
        subprocess.run(merge_command, check=True)

    if args.sensitivity.exists():
        if not sensitivity_matches(
            args.sensitivity,
            args.merged_stats,
            plan_sha256,
            provenance_sha256,
        ):
            raise SystemExit("existing sensitivity does not match")
        print("stage=sensitivity status=verified-skip", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("score_quant_sensitivity.py")),
                "--checkpoint",
                str(args.checkpoint),
                "--revision",
                args.revision,
                "--local-model-provenance",
                str(args.local_model_provenance),
                "--plan",
                str(args.plan),
                "--stats",
                str(args.merged_stats),
                "--output",
                str(args.sensitivity),
                "--ledger",
                str(args.ledger),
                "--reserved-gpu-hours",
                str(args.reserved_gpu_hours),
                "--device",
                args.device,
                "--rows-per-chunk",
                str(args.rows_per_chunk),
                "--row-group-size",
                str(args.row_group_size),
            ],
            check=True,
        )
    if args.assignment.exists():
        if not assignment_matches(
            args.assignment,
            args.sensitivity,
            plan_sha256,
            args.budget_bytes,
        ):
            raise SystemExit("existing Q2/Q4 assignment does not match")
        print("stage=assignment status=verified-skip", flush=True)
    else:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("optimize_q4_budget.py")),
                "--sensitivity",
                str(args.sensitivity),
                "--plan",
                str(args.plan),
                "--budget-bytes",
                str(args.budget_bytes),
                "--output",
                str(args.assignment),
            ],
            check=True,
        )
    assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "complete",
                "activation_batches": len(artifacts),
                "merged_stats_sha256": sha256(args.merged_stats),
                "sensitivity_sha256": sha256(args.sensitivity),
                "assignment_sha256": sha256(args.assignment),
                "assignment_bytes_used": assignment["bytes_used"],
                "assignment_budget_bytes": assignment["budget_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
