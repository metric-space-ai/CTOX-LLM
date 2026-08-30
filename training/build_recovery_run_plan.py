#!/usr/bin/env python3
"""Admit one complete fixed-Q2/Q4 recovery, pack, and held-out run.

The plan is intentionally host-specific and contains argv arrays, not shell
snippets.  It re-verifies every teacher artifact and every CTOX tensor before
spending the remaining GPU-hour budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from cache_teacher import validate_local_model_provenance
from ctox_artifact import CtoxArtifact
from evaluate_recovery import load_indexed_jsonl, require_exact_ids
from fanout_recovery import QWEN38_FANOUT_POLICY
from recovery_io import atomic_json
from run_ledger import total_gpu_hours
from teacher_cache_dataset import VerifiedTeacherCache


MODEL = "Qwen/Qwen3.8-27B"
FOLD_PACKAGE_LIMIT_BYTES = 8_375_186_227
REQUIRED_LOSSES = frozenset(
    {"kl", "ce", "hidden", "mtp_kl", "mtp_ce", "mtp_hidden"}
)
QUANT_DTYPES = frozenset({"q2_b64", "q4_b64", "mixed_q2_q4_b64"})
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{label} is not a JSON object")
    return document


def require_sha256(value: Any, label: str) -> str:
    encoded = str(value)
    if not SHA256.fullmatch(encoded):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return encoded


def resolved(path: Path) -> str:
    return str(path.resolve())


def safetensors_metadata(path: Path) -> dict[str, str]:
    """Read only the bounded safetensors JSON header, without importing torch."""

    with path.open("rb") as source:
        encoded_length = source.read(8)
        if len(encoded_length) != 8:
            raise ValueError("activation statistics lack a safetensors header")
        (header_length,) = struct.unpack("<Q", encoded_length)
        if header_length <= 2 or header_length > 128 * 1024 * 1024:
            raise ValueError("activation-statistics header length is invalid")
        encoded = source.read(header_length)
        if len(encoded) != header_length:
            raise ValueError("activation-statistics header is truncated")
    header = json.loads(encoded)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValueError("activation statistics have no string metadata contract")
    return metadata


def validate_cache(
    path: Path,
    expected_sha256: str,
    expected_samples: int,
    revision: str,
    provenance_sha256: str,
    expected_input_sha256: str,
) -> VerifiedTeacherCache:
    encoded_manifest = path.read_bytes()
    raw_manifest = json.loads(encoded_manifest)
    expected_input = raw_manifest.get("expected_input")
    if not isinstance(expected_input, dict):
        raise ValueError("teacher cache-set does not bind its materialized input")
    if expected_input.get("sha256") != expected_input_sha256:
        raise ValueError("teacher cache-set materialized input differs from million-corpus evidence")
    cache = VerifiedTeacherCache.from_manifest(path, expected_sha256)
    if len(cache.artifacts) != expected_samples:
        raise ValueError(
            f"teacher cache has {len(cache.artifacts)} samples, expected {expected_samples}"
        )
    if cache.teacher_revision != revision:
        raise ValueError("teacher cache revision differs from the recovery revision")
    if cache.teacher_provenance_sha256 != provenance_sha256:
        raise ValueError("teacher cache provenance differs from the local BF16 model")
    if not bool(cache.settings.get("mtp_targets")):
        raise ValueError("teacher cache does not contain MTP targets")
    if int(cache.settings.get("top_k", -1)) != 64:
        raise ValueError("teacher cache is not the release top-64 contract")
    # The cache-set manifest re-verifies batch documents.  Admission additionally
    # rehashes the actual payloads so a stale path cannot pass on metadata alone.
    for index in range(len(cache.artifacts)):
        cache.verified_artifact_path(index)
    return cache


def validate_million_corpus_audit(
    path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_sha256(expected_sha256, "million-corpus audit hash")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("million-corpus audit hash differs from contract")
    audit = json.loads(encoded)
    if audit.get("format") != "ctox.recovery-million-corpus-audit.v1":
        raise ValueError("unsupported million-corpus audit")
    if audit.get("status") != "passed" or audit.get("hard_gate_gaps") != {}:
        raise ValueError("million-corpus audit did not pass")
    partitions = audit.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        "train",
        "calibration",
        "held_out",
    }:
        raise ValueError("million-corpus audit has the wrong partitions")
    minima = {"train": 1_000_000, "calibration": 50_000, "held_out": 50_000}
    for partition, minimum in minima.items():
        record = partitions[partition]
        if record.get("status") != "passed" or int(record.get("records", -1)) < minimum:
            raise ValueError(f"million-corpus {partition} partition is not admitted")
    evidence_path = Path(str(audit.get("evidence", "")))
    evidence_bytes = evidence_path.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != audit.get("evidence_sha256"):
        raise ValueError("million-corpus evidence changed after audit")
    evidence = json.loads(evidence_bytes)
    if evidence.get("format") != "ctox.recovery-million-corpus-evidence.v1":
        raise ValueError("unsupported million-corpus evidence")
    if any(int(value) != 0 for value in evidence.get("hard_gates", {}).values()):
        raise ValueError("million-corpus evidence contains a failed hard gate")
    return audit, evidence


def validate_activation_assignment(
    activation_stats: Path,
    sensitivity_path: Path,
    assignment_path: Path,
    plan_path: Path,
    revision: str,
    provenance_sha256: str,
    calibration_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_bytes = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    plan = json.loads(plan_bytes)
    if plan.get("format") != "ctox.q2q4.quant-plan.v2":
        raise ValueError("final recovery requires a measured v2 Q2/Q4 plan")
    if plan.get("model") != MODEL or plan.get("revision") != revision:
        raise ValueError("quant plan model or revision differs")
    if plan.get("local_model_provenance_sha256") != provenance_sha256:
        raise ValueError("quant plan BF16 provenance differs")
    if plan.get("mtp") != "resident" or plan.get("vision") != "separate":
        raise ValueError("quant plan does not keep MTP resident and vision separate")
    if not bool(plan.get("fits_fold_limit")):
        raise ValueError("quant plan does not fit the Fold resident limit")
    if int(plan.get("total_bytes", -1)) > FOLD_PACKAGE_LIMIT_BYTES:
        raise ValueError("quant plan exceeds the complete text+MTP byte ceiling")

    stats_sha256 = sha256_path(activation_stats)
    metadata = safetensors_metadata(activation_stats)
    if metadata.get("format") != "ctox.activation-diagonal.v1":
        raise ValueError("unsupported activation-statistics format")
    if metadata.get("revision") != revision:
        raise ValueError("activation-statistics revision differs")
    if metadata.get("local_model_provenance_sha256") != provenance_sha256:
        raise ValueError("activation-statistics BF16 provenance differs")
    if json.loads(metadata.get("unobserved_tensors", "null")) != []:
        raise ValueError("activation statistics leave model tensors unobserved")
    activation_ids = [str(value) for value in json.loads(metadata["sample_ids"])]
    if len(activation_ids) != len(set(activation_ids)):
        raise ValueError("activation statistics contain duplicate sample IDs")
    if set(activation_ids) != calibration_ids:
        raise ValueError(
            "final Q2/Q4 assignment was not measured over the admitted calibration cohort"
        )
    if int(metadata.get("samples", -1)) != len(calibration_ids):
        raise ValueError("activation-statistics sample count differs")

    sensitivity = read_json(sensitivity_path, "sensitivity report")
    sensitivity_sha256 = sha256_path(sensitivity_path)
    source_plan_sha256 = require_sha256(
        sensitivity.get("quant_plan_sha256"), "sensitivity source-plan hash"
    )
    if sensitivity.get("format") != "ctox.q2q4.sensitivity.v1":
        raise ValueError("unsupported Q2/Q4 sensitivity report")
    if sensitivity.get("model") != MODEL or sensitivity.get("revision") != revision:
        raise ValueError("sensitivity model or revision differs")
    if sensitivity.get("local_model_provenance_sha256") != provenance_sha256:
        raise ValueError("sensitivity BF16 provenance differs")
    if sensitivity.get("activation_stats_sha256") != stats_sha256:
        raise ValueError("sensitivity does not bind the activation statistics")
    if metadata.get("quant_plan_sha256") != source_plan_sha256:
        raise ValueError("activation statistics and sensitivity source plan differ")
    candidates = sensitivity.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("sensitivity report contains no candidates")
    if any(not bool(candidate.get("observed")) for candidate in candidates):
        raise ValueError("sensitivity report contains unobserved matrices")
    candidate_names = [str(candidate.get("name", "")) for candidate in candidates]
    if not all(candidate_names) or len(candidate_names) != len(set(candidate_names)):
        raise ValueError("sensitivity candidate identities are empty or duplicated")

    assignment = read_json(assignment_path, "Q2/Q4 assignment")
    assignment_sha256 = sha256_path(assignment_path)
    if assignment.get("format") != "ctox.q2q4.assignment.v2":
        raise ValueError("final recovery requires a measured v2 assignment")
    if assignment.get("plan_sha256") != source_plan_sha256:
        raise ValueError("Q2/Q4 assignment source plan differs")
    if assignment.get("sensitivity_sha256") != sensitivity_sha256:
        raise ValueError("Q2/Q4 assignment does not bind the sensitivity report")
    if int(assignment.get("bytes_used", -1)) != int(plan["total_bytes"]):
        raise ValueError("assignment and rebuilt quant plan byte totals differ")
    if int(assignment.get("bytes_used", -1)) > int(assignment.get("budget_bytes", -2)):
        raise ValueError("Q2/Q4 assignment exceeds its declared budget")
    plan_assignment = plan.get("assignment")
    if not isinstance(plan_assignment, dict) or plan_assignment.get("sha256") != assignment_sha256:
        raise ValueError("rebuilt quant plan does not bind the Q2/Q4 assignment")

    quantized_names = {
        str(tensor["name"])
        for tensor in plan.get("tensors", [])
        if tensor.get("dtype") in QUANT_DTYPES
    }
    if set(candidate_names) != quantized_names:
        raise ValueError("sensitivity candidates do not cover every quantized matrix")
    return plan, {
        "activation_stats_sha256": stats_sha256,
        "activation_samples": len(activation_ids),
        "source_quant_plan_sha256": source_plan_sha256,
        "sensitivity_sha256": sensitivity_sha256,
        "sensitivity_candidates": len(candidate_names),
        "assignment_sha256": assignment_sha256,
        "final_quant_plan_sha256": plan_sha256,
    }


def validate_artifact(
    artifact_path: Path,
    plan: dict[str, Any],
    plan_sha256: str,
    revision: str,
) -> dict[str, Any]:
    artifact_sha256 = sha256_path(artifact_path)
    if artifact_path.stat().st_size > FOLD_PACKAGE_LIMIT_BYTES:
        raise ValueError("initializer artifact exceeds the text+MTP package ceiling")
    with CtoxArtifact(artifact_path, verify_tensors=True) as artifact:
        manifest = artifact.manifest
        if manifest.get("model") != MODEL or manifest.get("revision") != revision:
            raise ValueError("initializer artifact model or revision differs")
        recovery = manifest.get("recovery")
        if not isinstance(recovery, dict) or not recovery.get("fixed_logical_qcodes"):
            raise ValueError("initializer artifact does not bind fixed logical qcodes")
        if recovery.get("plan_sha256") != plan_sha256:
            raise ValueError("initializer artifact and final quant plan differ")
        expected = plan.get("tensors", [])
        actual = manifest.get("tensors", [])
        if len(expected) != len(actual):
            raise ValueError("initializer artifact and quant plan tensor counts differ")
        descriptor_keys = ("name", "dtype", "shape", "offset", "length", "segments")
        for planned, packed in zip(expected, actual):
            for key in descriptor_keys:
                if planned.get(key, []) != packed.get(key, []):
                    raise ValueError(
                        f"initializer tensor {planned.get('name')} differs in {key}"
                    )
    return {
        "path": resolved(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "sha256": artifact_sha256,
        "fixed_logical_qcodes": True,
        "mtp": "resident",
    }


def validate_smoke(
    path: Path,
    artifact_sha256: str,
    cache_sha256: str,
    revision: str,
    provenance_sha256: str,
    prefill_chunk_tokens: int,
) -> dict[str, Any]:
    report = read_json(path, "recovery smoke report")
    if report.get("format") != "ctox.recovery.training-run.v1":
        raise ValueError("unsupported recovery smoke report")
    if report.get("status") != "bounded_run_complete":
        raise ValueError("recovery smoke did not finish its bounded run")
    required = {
        "model": MODEL,
        "revision": revision,
        "local_model_provenance_sha256": provenance_sha256,
        "artifact_sha256": artifact_sha256,
        "teacher_cache_set_sha256": cache_sha256,
        "max_optimizer_steps": 1,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "oversize_policy": "fail",
        "gradient_checkpointing": False,
        "fixed_logical_qcodes": True,
        "fanout_s_in_policy": QWEN38_FANOUT_POLICY,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(
                f"recovery smoke {key} is {report.get(key)!r}, expected {expected!r}"
            )
    cursor = report.get("cursor", {})
    if int(cursor.get("optimizer_steps", -1)) != 1 or int(
        cursor.get("samples_seen", -1)
    ) < 1:
        raise ValueError("recovery smoke did not execute exactly one optimizer step")
    if report.get("skipped_oversize_samples"):
        raise ValueError("recovery smoke skipped an oversize sample")
    losses = report.get("recent_mean_losses")
    if not isinstance(losses, dict) or not REQUIRED_LOSSES.issubset(losses):
        raise ValueError("recovery smoke did not exercise every release loss")
    if not isinstance(report.get("base_graph"), dict) or not isinstance(
        report.get("mtp_graph"), dict
    ):
        raise ValueError("recovery smoke lacks target or MTP graph evidence")
    return {
        "path": resolved(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "optimizer_steps": 1,
        "samples_seen": int(cursor["samples_seen"]),
        "losses": sorted(REQUIRED_LOSSES),
        "prefill_chunk_tokens": prefill_chunk_tokens,
    }


def argv(script: Path, python: str, values: list[tuple[str, Any]]) -> list[str]:
    command = [python, resolved(script)]
    for flag, value in values:
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    return command


def output_contract(output_root: Path) -> dict[str, Path]:
    return {
        "scales": output_root / "recovery-scales.safetensors",
        "training_report": output_root / "training-report.json",
        "training_evidence": output_root / "training-evidence.json",
        "checkpoints": output_root / "checkpoints",
        "recovered_artifact": output_root / "qwen38-27b-q2q4-recovered.ctoxq",
        "direct_evaluation": output_root / "heldout-direct.json",
        "recovered_evaluation": output_root / "heldout-recovered.json",
        "comparison": output_root / "heldout-comparison.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--activation-stats", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--quant-plan", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--million-corpus-audit", type=Path, required=True)
    parser.add_argument("--million-corpus-audit-sha256", required=True)
    parser.add_argument("--train-cache-set", type=Path, required=True)
    parser.add_argument("--train-cache-set-sha256", required=True)
    parser.add_argument("--evaluation-cache-set", type=Path, required=True)
    parser.add_argument("--evaluation-cache-set-sha256", required=True)
    parser.add_argument("--evaluation-materialized", type=Path, required=True)
    parser.add_argument("--evaluation-domain-tags", type=Path, required=True)
    parser.add_argument("--evaluation-service-tags", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--training-root", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--physical-gpu",
        type=int,
        required=True,
        help="single physical GPU exposed as cuda:0; GPU 0 is reserved for Greppy",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--training-reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--packing-reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--evaluation-reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--gpu-hour-ceiling", type=float, default=240.0)
    args = parser.parse_args()

    try:
        if min(
            args.epochs,
            args.gradient_accumulation,
            args.prefill_chunk_tokens,
            args.training_reserved_gpu_hours,
            args.packing_reserved_gpu_hours,
            args.evaluation_reserved_gpu_hours,
            args.gpu_hour_ceiling,
        ) <= 0:
            raise ValueError("counts, chunks, reserves, and budget must be positive")
        if args.physical_gpu <= 0:
            raise ValueError("physical GPU 0 is reserved and cannot run Qwen recovery")
        if args.device != "cuda:0":
            raise ValueError("the isolated recovery device must be logical cuda:0")
        if not args.python.is_file():
            raise ValueError(f"Python interpreter is missing: {args.python}")
        if args.output.exists():
            raise ValueError(f"refusing to overwrite {args.output}")
        outputs = output_contract(args.output_root)
        collisions = [path for path in outputs.values() if path.exists()]
        if collisions:
            raise ValueError(f"recovery output already exists: {collisions[0]}")
        require_sha256(args.train_cache_set_sha256, "training cache-set hash")
        require_sha256(
            args.evaluation_cache_set_sha256, "evaluation cache-set hash"
        )
        _provenance, provenance_sha256 = validate_local_model_provenance(
            args.model_source, args.revision, args.local_model_provenance
        )
        if provenance_sha256 is None:
            raise ValueError("recovery planning requires a verified local BF16 model")

        corpus_audit, corpus_evidence = validate_million_corpus_audit(
            args.million_corpus_audit,
            args.million_corpus_audit_sha256,
        )
        corpus_partitions = corpus_evidence["partitions"]
        expected_train_samples = int(corpus_audit["partitions"]["train"]["records"])
        expected_evaluation_samples = int(
            corpus_audit["partitions"]["held_out"]["records"]
        )

        train_cache = validate_cache(
            args.train_cache_set,
            args.train_cache_set_sha256,
            expected_train_samples,
            args.revision,
            provenance_sha256,
            corpus_partitions["train"]["materialized_sha256"],
        )
        evaluation_cache = validate_cache(
            args.evaluation_cache_set,
            args.evaluation_cache_set_sha256,
            expected_evaluation_samples,
            args.revision,
            provenance_sha256,
            corpus_partitions["held_out"]["materialized_sha256"],
        )
        if train_cache.settings != evaluation_cache.settings:
            raise ValueError("training and held-out teacher settings differ")
        train_ids = {str(item["id"]) for item in train_cache.artifacts}
        evaluation_ids = {str(item["id"]) for item in evaluation_cache.artifacts}
        overlap = train_ids.intersection(evaluation_ids)
        if overlap:
            raise ValueError(f"training and held-out caches overlap at {min(overlap)}")

        for path, label in (
            (args.evaluation_materialized, "held-out materialized cohort"),
            (args.evaluation_domain_tags, "held-out domain tags"),
            (args.evaluation_service_tags, "held-out service tags"),
        ):
            require_exact_ids(evaluation_ids, load_indexed_jsonl(path, label), label)

        calibration_binding = corpus_partitions["calibration"]
        calibration_path = Path(
            calibration_binding["binding_paths"]["materialized_sha256"]
        )
        if sha256_path(calibration_path) != calibration_binding["materialized_sha256"]:
            raise ValueError("calibration materialized cohort changed after corpus admission")
        calibration_ids = {
            str(record["id"])
            for record in load_indexed_jsonl(
                calibration_path, "calibration materialized cohort"
            )
        }
        if len(calibration_ids) != int(
            corpus_audit["partitions"]["calibration"]["records"]
        ):
            raise ValueError("calibration identities differ from million-corpus audit")
        if calibration_ids & train_ids or calibration_ids & evaluation_ids:
            raise ValueError("calibration identities overlap training or held-out cache")

        plan, assignment_evidence = validate_activation_assignment(
            args.activation_stats,
            args.sensitivity,
            args.assignment,
            args.quant_plan,
            args.revision,
            provenance_sha256,
            calibration_ids,
        )
        artifact = validate_artifact(
            args.artifact,
            plan,
            assignment_evidence["final_quant_plan_sha256"],
            args.revision,
        )
        smoke = validate_smoke(
            args.smoke_report,
            artifact["sha256"],
            args.train_cache_set_sha256,
            args.revision,
            provenance_sha256,
            args.prefill_chunk_tokens,
        )

        consumed = total_gpu_hours(args.ledger)
        reservations = {
            "training": args.training_reserved_gpu_hours,
            "packing": args.packing_reserved_gpu_hours,
            "direct_evaluation": args.evaluation_reserved_gpu_hours,
            "recovered_evaluation": args.evaluation_reserved_gpu_hours,
        }
        reserved = sum(reservations.values())
        projected = consumed + reserved
        if projected > args.gpu_hour_ceiling:
            raise ValueError(
                f"complete recovery would use {projected:.3f} GPU-hours, above "
                f"the {args.gpu_hour_ceiling:.3f} ceiling"
            )

        scripts = {
            name: args.training_root / name
            for name in (
                "train_recovery.py",
                "pack_checkpoint.py",
                "evaluate_recovery.py",
                "compare_recovery_evaluations.py",
            )
        }
        for script in scripts.values():
            if not script.is_file():
                raise ValueError(f"recovery stage script is missing: {script}")
        python_command = resolved(args.python)
        stage_environment = {"CUDA_VISIBLE_DEVICES": str(args.physical_gpu)}
        common_model = [
            ("--model-source", resolved(args.model_source)),
            ("--revision", args.revision),
            ("--local-model-provenance", resolved(args.local_model_provenance)),
        ]
        loss_weights = json.dumps(
            {
                "ce": 1.0,
                "hidden": 1.0,
                "kl": 1.0,
                "mtp_ce": 0.5,
                "mtp_hidden": 0.5,
                "mtp_kl": 0.5,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        training_command = argv(
            scripts["train_recovery.py"],
            python_command,
            [
                ("--artifact", resolved(args.artifact)),
                *common_model,
                ("--teacher-cache-set", resolved(args.train_cache_set)),
                ("--teacher-cache-set-sha256", args.train_cache_set_sha256),
                ("--output-scales", resolved(outputs["scales"])),
                ("--output-report", resolved(outputs["training_report"])),
                ("--output-evidence", resolved(outputs["training_evidence"])),
                ("--checkpoint-dir", resolved(outputs["checkpoints"])),
                ("--ledger", resolved(args.ledger)),
                ("--reserved-gpu-hours", args.training_reserved_gpu_hours),
                ("--gpus", 1),
                ("--device", args.device),
                ("--compute-dtype", "bfloat16"),
                ("--epochs", args.epochs),
                ("--max-sequence-tokens", 0),
                ("--prefill-chunk-tokens", args.prefill_chunk_tokens),
                ("--oversize-policy", "fail"),
                ("--learning-rate", 0.0002),
                ("--gradient-accumulation", args.gradient_accumulation),
                ("--gradient-clip", 1.0),
                ("--scale-regularization", 0.0001),
                ("--rows-per-chunk", 128),
                ("--logit-chunk", 16),
                ("--checkpoint-every", 25),
                ("--seed", 38),
                ("--loss-weights", loss_weights),
                ("--use-fla-kernel", True),
                ("--fanout-s-in-policy", QWEN38_FANOUT_POLICY),
            ],
        )
        pack_command = argv(
            scripts["pack_checkpoint.py"],
            python_command,
            [
                ("--checkpoint", resolved(args.model_source)),
                ("--revision", args.revision),
                ("--local-model-provenance", resolved(args.local_model_provenance)),
                ("--plan", resolved(args.quant_plan)),
                ("--output", resolved(outputs["recovered_artifact"])),
                ("--recovery-scales", resolved(outputs["scales"])),
                ("--ledger", resolved(args.ledger)),
                ("--reserved-gpu-hours", args.packing_reserved_gpu_hours),
                ("--device", args.device),
                ("--rows-per-chunk", 256),
            ],
        )

        def evaluation_command(artifact_path: Path, output: Path) -> list[str]:
            return argv(
                scripts["evaluate_recovery.py"],
                python_command,
                [
                    ("--artifact", resolved(artifact_path)),
                    *common_model,
                    ("--teacher-cache-set", resolved(args.evaluation_cache_set)),
                    (
                        "--teacher-cache-set-sha256",
                        args.evaluation_cache_set_sha256,
                    ),
                    ("--materialized", resolved(args.evaluation_materialized)),
                    ("--domain-tags", resolved(args.evaluation_domain_tags)),
                    ("--service-tags", resolved(args.evaluation_service_tags)),
                    ("--output", resolved(output)),
                    ("--ledger", resolved(args.ledger)),
                    (
                        "--reserved-gpu-hours",
                        args.evaluation_reserved_gpu_hours,
                    ),
                    ("--gpus", 1),
                    ("--device", args.device),
                    ("--compute-dtype", "bfloat16"),
                    ("--rows-per-chunk", 128),
                    ("--logit-chunk", 16),
                    ("--prefill-chunk-tokens", args.prefill_chunk_tokens),
                ],
            )

        stages = [
            {
                "name": "train_recovery",
                "requires": ["admission"],
                "environment": stage_environment,
                "gpu_count": 1,
                "reserved_gpu_hours": reservations["training"],
                "argv": training_command,
                "outputs": [
                    resolved(outputs["scales"]),
                    resolved(outputs["training_report"]),
                    resolved(outputs["training_evidence"]),
                ],
            },
            {
                "name": "pack_recovered_artifact",
                "requires": ["train_recovery:status=complete"],
                "environment": stage_environment,
                "gpu_count": 1,
                "reserved_gpu_hours": reservations["packing"],
                "argv": pack_command,
                "outputs": [resolved(outputs["recovered_artifact"])],
            },
            {
                "name": "evaluate_direct",
                "requires": ["pack_recovered_artifact"],
                "environment": stage_environment,
                "gpu_count": 1,
                "reserved_gpu_hours": reservations["direct_evaluation"],
                "argv": evaluation_command(
                    args.artifact, outputs["direct_evaluation"]
                ),
                "outputs": [resolved(outputs["direct_evaluation"])],
            },
            {
                "name": "evaluate_recovered",
                "requires": ["evaluate_direct"],
                "environment": stage_environment,
                "gpu_count": 1,
                "reserved_gpu_hours": reservations["recovered_evaluation"],
                "argv": evaluation_command(
                    outputs["recovered_artifact"], outputs["recovered_evaluation"]
                ),
                "outputs": [resolved(outputs["recovered_evaluation"])],
            },
            {
                "name": "compare_heldout",
                "requires": ["evaluate_direct", "evaluate_recovered"],
                "environment": {},
                "gpu_count": 0,
                "reserved_gpu_hours": 0.0,
                "argv": argv(
                    scripts["compare_recovery_evaluations.py"],
                    python_command,
                    [
                        ("--direct", resolved(outputs["direct_evaluation"])),
                        ("--recovered", resolved(outputs["recovered_evaluation"])),
                        ("--output", resolved(outputs["comparison"])),
                        ("--minimum-gap-closed", 0.50),
                    ],
                ),
                "outputs": [resolved(outputs["comparison"])],
            },
        ]
        document = {
            "format": "ctox.recovery.execution-plan.v1",
            "status": "admitted",
            "model": MODEL,
            "revision": args.revision,
            "local_model_provenance": {
                "path": resolved(args.local_model_provenance),
                "sha256": provenance_sha256,
            },
            "million_corpus": {
                "audit": resolved(args.million_corpus_audit),
                "audit_sha256": args.million_corpus_audit_sha256,
                "evidence": corpus_audit["evidence"],
                "evidence_sha256": corpus_audit["evidence_sha256"],
                "training_samples": expected_train_samples,
                "calibration_samples": len(calibration_ids),
                "held_out_samples": expected_evaluation_samples,
            },
            "initializer_artifact": artifact,
            "assignment_evidence": assignment_evidence,
            "teacher_caches": {
                "training": {
                    "path": resolved(args.train_cache_set),
                    "sha256": args.train_cache_set_sha256,
                    "samples": len(train_ids),
                    "artifact_root_sha256": train_cache.manifest()[
                        "artifact_root_sha256"
                    ],
                },
                "heldout": {
                    "path": resolved(args.evaluation_cache_set),
                    "sha256": args.evaluation_cache_set_sha256,
                    "samples": len(evaluation_ids),
                    "artifact_root_sha256": evaluation_cache.manifest()[
                        "artifact_root_sha256"
                    ],
                },
                "disjoint": True,
            },
            "smoke": smoke,
            "implementation": {
                "python": python_command,
                "scripts": {
                    name: {
                        "path": resolved(path),
                        "sha256": sha256_path(path),
                    }
                    for name, path in sorted(scripts.items())
                },
            },
            "training": {
                "epochs": args.epochs,
                "gradient_accumulation": args.gradient_accumulation,
                "expected_optimizer_steps": math.ceil(
                    len(train_ids) / args.gradient_accumulation
                )
                * args.epochs,
                "fixed_logical_qcodes": True,
                "prefill_chunk_tokens": args.prefill_chunk_tokens,
                "oversize_policy": "fail",
                "resume": {
                    "checkpoint_dir": resolved(outputs["checkpoints"]),
                    "required_flag": "--resume-checkpoint",
                    "contract": "exact run_contract_sha256",
                },
            },
            "gpu_budget": {
                "ledger": resolved(args.ledger),
                "ceiling_gpu_hours": args.gpu_hour_ceiling,
                "consumed_gpu_hours": consumed,
                "reserved_gpu_hours": reservations,
                "projected_gpu_hours": projected,
                "headroom_gpu_hours": args.gpu_hour_ceiling - projected,
            },
            "execution_order": "serial",
            "stages": stages,
            "release_gates_after_distillation": [
                "weighted_benchmark_at_least_95_percent_of_bf16",
                "no_primary_category_below_90_percent_of_bf16",
                "agentic_tool_calling_german_code_mtp_golden_tests",
                "128k_retrieval_at_least_90_percent_of_bf16",
                "backend_memory_roofline_and_unload_gates",
            ],
        }
        atomic_json(args.output, document)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
