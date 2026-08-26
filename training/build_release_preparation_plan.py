#!/usr/bin/env python3
"""Build the serial, resumable GPU1+2 plan through the recovery smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from recovery_io import atomic_json


FORMAT = "ctox.recovery.preparation-plan.v1"
EXECUTION_FORMAT = "ctox.recovery.execution-plan.v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
FOLD_PACKAGE_LIMIT_BYTES = 8_373_089_075
TRAIN_SAMPLES = 2_328
EVALUATION_SAMPLES = 642


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def batch_verifications(plan_path: Path, root: Path, prefix: str) -> list[Path]:
    document = json.loads(plan_path.read_text(encoding="utf-8"))
    batches = document.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"batch plan has no batches: {plan_path}")
    indices = [int(batch["batch_index"]) for batch in batches]
    if indices != list(range(len(indices))):
        raise ValueError(f"batch plan is not contiguous from zero: {plan_path}")
    if int(document.get("summary", {}).get("batches", -1)) != len(indices):
        raise ValueError(f"batch plan summary differs: {plan_path}")
    return [
        root / f"{prefix}-batch-{index:03d}-v1-verification-v1.json"
        for index in indices
    ]


def jsonl_ids(path: Path, expected: int, label: str) -> set[str]:
    identities = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            document = json.loads(line)
            sample_id = str(document.get("id", ""))
            if not sample_id:
                raise ValueError(f"{label} line {line_number} has no identity")
            identities.append(sample_id)
    if len(identities) != expected or len(set(identities)) != expected:
        raise ValueError(f"{label} does not contain exactly {expected} unique samples")
    return set(identities)


def validate_source_snapshot(root: Path, commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not COMMIT.fullmatch(commit):
        raise ValueError("source commit is not a full lowercase Git commit")
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source snapshot is absent or is a symlink")
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise ValueError("source snapshot root is writable")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source snapshot contains a symlink: {path}")
        if path.is_dir() and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f"source snapshot contains a writable directory: {path}")
        if path.is_file():
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise ValueError(f"source snapshot contains a writable file: {path}")
            relative = path.relative_to(root).as_posix()
            files.append((relative, path.stat().st_size, sha256_path(path)))
    if not files:
        raise ValueError("source snapshot is empty")
    digest = hashlib.sha256()
    for relative, size, file_sha256 in files:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(file_sha256.encode())
        digest.update(b"\n")
    return {
        "path": str(root),
        "commit": commit,
        "files": len(files),
        "tree_sha256": digest.hexdigest(),
        "read_only": True,
    }


def stage(
    name: str,
    requires: list[str],
    argv: list[str],
    outputs: list[Path],
    physical_gpus: str | None = None,
    resume_policy: str = "none",
) -> dict[str, Any]:
    gpu_count = 0 if physical_gpus is None else len(physical_gpus.split(","))
    return {
        "name": name,
        "requires": requires,
        "environment": (
            {} if physical_gpus is None else {"CUDA_VISIBLE_DEVICES": physical_gpus}
        ),
        "gpu_count": gpu_count,
        "resume_policy": resume_policy,
        "argv": argv,
        "outputs": [resolved(path) for path in outputs],
    }


def script_argv(python: Path, source_root: Path, name: str, *values: object) -> list[str]:
    script = source_root / "training" / name
    if not script.is_file():
        raise ValueError(f"source snapshot lacks required script: {script}")
    return [
        resolved(python),
        resolved(script),
        *(str(value) for value in values),
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = validate_source_snapshot(args.source_root, args.source_commit)
    source_root = Path(source["path"])
    python = args.python.expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("recovery Python is absent or not executable")
    if not SHA256.fullmatch(args.teacher_provenance_sha256):
        raise ValueError("teacher provenance is not a lowercase SHA-256")

    immutable_inputs = {
        "local_model_provenance": args.local_model_provenance,
        "train_input": args.train_input,
        "evaluation_input": args.evaluation_input,
        "train_missing_batch_plan": args.train_missing_batch_plan,
        "evaluation_batch_plan": args.evaluation_batch_plan,
        "activation_batch_plan": args.activation_batch_plan,
        "base_quant_plan": args.base_quant_plan,
    }
    input_bindings = {}
    for name, raw_path in immutable_inputs.items():
        path = raw_path.expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"immutable preparation input is absent or symlinked: {path}")
        input_bindings[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    if not args.model_source.expanduser().resolve().is_dir():
        raise ValueError("local BF16 model source is absent")
    if not args.hf_home.expanduser().resolve().is_dir():
        raise ValueError("Hugging Face cache root is absent")
    train_ids = jsonl_ids(args.train_input, TRAIN_SAMPLES, "training cohort")
    evaluation_ids = jsonl_ids(
        args.evaluation_input, EVALUATION_SAMPLES, "evaluation cohort"
    )
    if train_ids.intersection(evaluation_ids):
        raise ValueError("training and evaluation cohorts overlap")
    if args.smoke_sample_id not in train_ids:
        raise ValueError("smoke sample is absent from the complete training cohort")

    data = args.data_root.expanduser().resolve()
    teacher_root = data / "teacher-cache"
    train_cache = teacher_root / "release-recovery-train-2328-v1-cache-set.json"
    evaluation_cache = teacher_root / "release-recovery-evaluation-642-v1-cache-set.json"
    train_missing_plan = args.train_missing_batch_plan.expanduser().resolve()
    evaluation_plan = args.evaluation_batch_plan.expanduser().resolve()
    activation_plan = args.activation_batch_plan.expanduser().resolve()
    train_missing_verifications = batch_verifications(
        train_missing_plan, teacher_root, args.train_missing_prefix
    )
    absent = [path for path in train_missing_verifications if not path.is_file()]
    if absent:
        raise ValueError(
            f"teacher cache is not terminal; {len(absent)} batch verifications are absent"
        )
    existing = [path.expanduser().resolve() for path in args.train_existing_verification]
    if len(existing) != 5 or any(not path.is_file() for path in existing):
        raise ValueError("release training cache requires exactly five existing verifications")
    if len(set(existing)) != len(existing):
        raise ValueError("existing teacher verification is repeated")

    train_cache_argv = script_argv(
        python,
        source_root,
        "build_teacher_cache_set.py",
    )
    for verification in existing:
        train_cache_argv.extend(
            ["--bound-verification", resolved(verification), sha256_path(verification)]
        )
    train_cache_argv.extend(
        [
            "--bound-batch-group",
            resolved(train_missing_plan),
            sha256_path(train_missing_plan),
            resolved(teacher_root),
            args.train_missing_prefix,
            "--expected-input",
            resolved(args.train_input),
            "--teacher-revision",
            args.revision,
            "--teacher-provenance-sha256",
            args.teacher_provenance_sha256,
            "--output",
            resolved(train_cache),
        ]
    )

    evaluation_verifications = batch_verifications(
        evaluation_plan, teacher_root, args.evaluation_prefix
    )
    activation_verifications = batch_verifications(
        activation_plan, data / "activation-stats", args.activation_prefix
    )
    common_teacher = [
        "--model",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--teacher-provenance-sha256",
        args.teacher_provenance_sha256,
        "--ledger",
        resolved(args.ledger),
        "--hf-home",
        resolved(args.hf_home),
        "--gpus",
        "2",
        "--reserved-gpu-hours",
        "24",
        "--gpu-weight-memory-gib",
        "16",
        "--long-context-gpu-weight-memory-gib",
        "10",
        "--long-context-threshold-tokens",
        "65536",
        "--cpu-offload-memory-gib",
        "96",
        "--mtp-device",
        "cuda:1",
        "--prefill-chunk-tokens",
        "512",
    ]
    evaluation_argv = script_argv(
        python,
        source_root,
        "run_teacher_batches.py",
        "--batch-plan",
        resolved(evaluation_plan),
        "--input",
        resolved(args.evaluation_input),
        "--output-root",
        resolved(teacher_root),
        "--output-prefix",
        args.evaluation_prefix,
        *common_teacher,
        "--start-batch",
        "0",
        "--end-batch",
        str(len(evaluation_verifications)),
        "--resume-incomplete",
    )
    evaluation_cache_argv = script_argv(
        python,
        source_root,
        "build_teacher_cache_set.py",
        "--bound-batch-group",
        resolved(evaluation_plan),
        sha256_path(evaluation_plan),
        resolved(teacher_root),
        args.evaluation_prefix,
        "--expected-input",
        resolved(args.evaluation_input),
        "--teacher-revision",
        args.revision,
        "--teacher-provenance-sha256",
        args.teacher_provenance_sha256,
        "--output",
        resolved(evaluation_cache),
    )

    activation_argv = script_argv(
        python,
        source_root,
        "run_activation_batches.py",
        "--batch-plan",
        resolved(activation_plan),
        "--plan",
        resolved(args.base_quant_plan),
        "--input",
        resolved(args.train_input),
        "--model",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--output-root",
        resolved(data / "activation-stats"),
        "--output-prefix",
        args.activation_prefix,
        "--ledger",
        resolved(args.ledger),
        "--hf-home",
        resolved(args.hf_home),
        "--gpus",
        "2",
        "--reserved-gpu-hours",
        "24",
        "--gpu-weight-memory-gib",
        "16",
        "--long-context-gpu-weight-memory-gib",
        "10",
        "--long-context-threshold-tokens",
        "65536",
        "--cpu-offload-memory-gib",
        "96",
        "--mtp-device",
        "cuda:1",
        "--prefill-chunk-tokens",
        "512",
    )

    merged = data / "activation-stats/release-activation-full-train-2328-merged-v1.safetensors"
    sensitivity = data / "sensitivity/release-activation-full-train-2328-all506-v1.json"
    assignment = data / "assignments/release-activation-full-train-2328-all506-v1.json"
    final_quant_plan = data / "plans/release-activation-full-train-2328-all506-v1.json"
    scale_fit = (
        data
        / "recovery/release-activation-full-train-2328-all506-scale-fit-v1.safetensors"
    )
    scale_report = data / "recovery/release-activation-full-train-2328-all506-scale-fit-v1.json"
    artifact = data / "packs/release-activation-full-train-2328-all506-scale-fit-v1.ctoxq"
    smoke_base = data / "recovery/release-recovery-e2e-full-train-smoke-1step-v1"
    smoke_outputs = [
        smoke_base.with_suffix(".safetensors"),
        smoke_base.with_suffix(".json"),
        smoke_base.with_name(f"{smoke_base.name}-evidence.json"),
    ]

    finalize_argv = script_argv(
        python,
        source_root,
        "finalize_activation_assignment.py",
        "--batch-plan",
        resolved(activation_plan),
        "--plan",
        resolved(args.base_quant_plan),
        "--input",
        resolved(args.train_input),
        "--artifact-root",
        resolved(data / "activation-stats"),
        "--artifact-prefix",
        args.activation_prefix,
        "--checkpoint",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--merged-stats",
        resolved(merged),
        "--sensitivity",
        resolved(sensitivity),
        "--assignment",
        resolved(assignment),
        "--ledger",
        resolved(args.ledger),
        "--budget-bytes",
        str(FOLD_PACKAGE_LIMIT_BYTES),
        "--reserved-gpu-hours",
        "12",
        "--device",
        "cuda:0",
        "--rows-per-chunk",
        "128",
        "--row-group-size",
        "256",
    )
    quant_plan_argv = script_argv(
        python,
        source_root,
        "build_quant_plan.py",
        "--checkpoint",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--assignment",
        resolved(assignment),
        "--assignment-source-plan",
        resolved(args.base_quant_plan),
        "--output",
        resolved(final_quant_plan),
    )
    fit_argv = script_argv(
        python,
        source_root,
        "fit_recovery_scales.py",
        "--checkpoint",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--plan",
        resolved(final_quant_plan),
        "--stats",
        resolved(merged),
        "--output",
        resolved(scale_fit),
        "--report",
        resolved(scale_report),
        "--ledger",
        resolved(args.ledger),
        "--reserved-gpu-hours",
        "12",
        "--device",
        "cuda:0",
        "--gpus",
        "1",
        "--iterations",
        "6",
        "--rows-per-chunk",
        "128",
        "--scale-min",
        "0.25",
        "--scale-max",
        "4.0",
    )
    pack_argv = script_argv(
        python,
        source_root,
        "pack_checkpoint.py",
        "--checkpoint",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--plan",
        resolved(final_quant_plan),
        "--recovery-scales",
        resolved(scale_fit),
        "--output",
        resolved(artifact),
        "--ledger",
        resolved(args.ledger),
        "--reserved-gpu-hours",
        "6",
        "--device",
        "cuda:0",
        "--rows-per-chunk",
        "256",
    )
    smoke_argv = script_argv(
        python,
        source_root,
        "run_bound_recovery_smoke.py",
        "--artifact",
        resolved(artifact),
        "--model-source",
        resolved(args.model_source),
        "--revision",
        args.revision,
        "--local-model-provenance",
        resolved(args.local_model_provenance),
        "--teacher-cache-set",
        resolved(train_cache),
        "--output-scales",
        resolved(smoke_outputs[0]),
        "--output-report",
        resolved(smoke_outputs[1]),
        "--output-evidence",
        resolved(smoke_outputs[2]),
        "--checkpoint-dir",
        resolved(data / "recovery/checkpoints/release-recovery-e2e-full-train-smoke-1step-v1"),
        "--ledger",
        resolved(args.ledger),
        "--sample-id",
        args.smoke_sample_id,
        "--device",
        "cuda:0",
        "--prefill-chunk-tokens",
        "512",
    )

    scripts = {}
    for path in sorted((source_root / "training").glob("*.py")):
        scripts[path.name] = {"path": resolved(path), "sha256": sha256_path(path)}
    stages = [
        stage("build_train_cache_set", ["admission"], train_cache_argv, [train_cache]),
        stage(
            "cache_evaluation_teacher",
            ["build_train_cache_set"],
            evaluation_argv,
            evaluation_verifications,
            "1,2",
            "application",
        ),
        stage(
            "build_evaluation_cache_set",
            ["cache_evaluation_teacher"],
            evaluation_cache_argv,
            [evaluation_cache],
        ),
        stage(
            "collect_full_activation_stats",
            ["build_evaluation_cache_set"],
            activation_argv,
            activation_verifications,
            "1,2",
            "application",
        ),
        stage(
            "finalize_q2q4_assignment",
            ["collect_full_activation_stats"],
            finalize_argv,
            [merged, sensitivity, assignment],
            "1",
        ),
        stage(
            "build_final_quant_plan",
            ["finalize_q2q4_assignment"],
            quant_plan_argv,
            [final_quant_plan],
        ),
        stage(
            "fit_recovery_scales",
            ["build_final_quant_plan"],
            fit_argv,
            [scale_fit, scale_report],
            "1",
        ),
        stage("pack_initializer", ["fit_recovery_scales"], pack_argv, [artifact], "1"),
        stage("smoke_recovery", ["pack_initializer"], smoke_argv, smoke_outputs, "1"),
    ]
    return {
        "format": EXECUTION_FORMAT,
        "plan_kind": FORMAT,
        "status": "admitted",
        "execution_order": "serial",
        "source_snapshot": source,
        "immutable_inputs": input_bindings,
        "cohorts": {
            "training_samples": len(train_ids),
            "evaluation_samples": len(evaluation_ids),
            "overlap": 0,
            "smoke_sample_id": args.smoke_sample_id,
        },
        "immutable_inputs": input_bindings,
        "implementation": {
            "python": resolved(python),
            "scripts": scripts,
        },
        "model": "Qwen/Qwen3.8-27B",
        "revision": args.revision,
        "physical_gpu_policy": {
            "reserved_for_greppy": [0],
            "ctox": [1, 2],
            "cpu_fallback": False,
        },
        "stages": stages,
        "next_gate": "build_recovery_run_plan.py after smoke report verification",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-existing-verification", type=Path, action="append", default=[])
    parser.add_argument("--train-missing-batch-plan", type=Path, required=True)
    parser.add_argument("--train-missing-prefix", required=True)
    parser.add_argument("--evaluation-input", type=Path, required=True)
    parser.add_argument("--evaluation-batch-plan", type=Path, required=True)
    parser.add_argument("--evaluation-prefix", required=True)
    parser.add_argument("--activation-batch-plan", type=Path, required=True)
    parser.add_argument("--activation-prefix", required=True)
    parser.add_argument("--base-quant-plan", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--smoke-sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    return args


def main() -> None:
    args = parse_args()
    try:
        atomic_json(args.output, build(args))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
