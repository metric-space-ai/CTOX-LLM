#!/usr/bin/env python3
"""Evaluate one packed fixed-qcode checkpoint on the disjoint teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from cache_teacher import validate_local_model_provenance
from ctox_artifact import CtoxArtifact
from recovery_io import atomic_json
from run_ledger import GpuRun, require_budget
from teacher_cache_dataset import VerifiedTeacherCache


LOSS_TARGETS = {
    "kl": "logit_targets",
    "ce": "logit_targets",
    "hidden": "hidden_targets",
    "mtp_kl": "mtp_targets",
    "mtp_ce": "mtp_targets",
    "mtp_hidden": "mtp_hidden_targets",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_indexed_jsonl(path: Path, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", ""))
            if not sample_id:
                raise ValueError(f"{label} line {line_number} has no sample id")
            if sample_id in indexed:
                raise ValueError(f"{label} contains duplicate sample {sample_id}")
            indexed[sample_id] = record
    if not indexed:
        raise ValueError(f"{label} is empty")
    return indexed


def require_exact_ids(
    expected: set[str],
    observed: dict[str, Any],
    label: str,
) -> None:
    actual = set(observed)
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(f"{label} ids differ: missing={missing}, extra={extra}")


def logical_qcode_root(manifest: dict[str, Any]) -> str:
    """Bind only immutable quant codes, independent of trained scale tensors."""

    quantized = [
        tensor
        for tensor in manifest.get("tensors", [])
        if tensor.get("dtype") in {"q2_b64", "q4_b64", "mixed_q2_q4_b64"}
    ]
    if not quantized:
        raise ValueError("artifact contains no logical Q2/Q4 code tensors")
    digest = hashlib.sha256()
    for tensor in sorted(quantized, key=lambda item: str(item["name"])):
        descriptor = json.dumps(
            {
                "name": tensor["name"],
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "segments": tensor.get("segments", []),
                "sha256": tensor["sha256"],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
    return digest.hexdigest()


class MetricAggregate:
    def __init__(self) -> None:
        self.records = 0
        self.sequence_tokens = 0
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.weighted_sums: defaultdict[str, float] = defaultdict(float)
        self.target_counts: defaultdict[str, int] = defaultdict(int)

    def add(self, sample: dict[str, Any]) -> None:
        self.records += 1
        self.sequence_tokens += int(sample["sequence_tokens"])
        for name, value in sample["losses"].items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"non-finite evaluation loss {name}")
            targets = int(sample[LOSS_TARGETS[name]])
            if targets <= 0:
                raise ValueError(f"evaluation sample has no {LOSS_TARGETS[name]}")
            self.sums[name] += value
            self.weighted_sums[name] += value * targets
            self.target_counts[name] += targets

    def report(self) -> dict[str, Any]:
        if self.records == 0:
            raise ValueError("cannot report an empty metric aggregate")
        return {
            "records": self.records,
            "sequence_tokens": self.sequence_tokens,
            "sample_mean_losses": {
                name: self.sums[name] / self.records for name in sorted(self.sums)
            },
            "target_weighted_mean_losses": {
                name: self.weighted_sums[name] / self.target_counts[name]
                for name in sorted(self.sums)
            },
            "target_counts": dict(sorted(self.target_counts.items())),
        }


def aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    overall = MetricAggregate()
    groups: dict[str, defaultdict[str, MetricAggregate]] = {
        family: defaultdict(MetricAggregate)
        for family in (
            "category",
            "language",
            "primary_domain",
            "domain",
            "service_mode",
            "source",
        )
    }
    for sample in samples:
        overall.add(sample)
        labels = {
            "category": [sample["category"]],
            "language": [sample["language"]],
            "primary_domain": [sample["primary_domain"]],
            "domain": sample["domains"],
            "service_mode": sample["service_modes"],
            "source": [sample["source"]],
        }
        for family, values in labels.items():
            for value in values:
                groups[family][str(value)].add(sample)
    return {
        "overall": overall.report(),
        "groups": {
            family: {
                name: aggregate.report()
                for name, aggregate in sorted(values.items())
            }
            for family, values in groups.items()
        },
    }


def evaluate_losses(
    runtime: Any,
    teacher: dict[str, Any],
    sequence_tokens: int,
    prefill_chunk_tokens: int,
    torch: Any,
) -> tuple[dict[str, float], int]:
    chunked = prefill_chunk_tokens > 0 and sequence_tokens > prefill_chunk_tokens
    steps = (
        runtime.loss_chunks(teacher, prefill_chunk_tokens)
        if chunked
        else (runtime.losses(teacher),)
    )
    totals: defaultdict[str, float] = defaultdict(float)
    chunks = 0
    with torch.no_grad():
        for chunks, (_total, losses) in enumerate(steps, 1):
            for name, value in losses.items():
                measured = float(value.detach().float().cpu())
                if not math.isfinite(measured):
                    raise ValueError(f"non-finite evaluation loss {name}")
                totals[name] += measured
    if chunks == 0 or set(totals) != set(LOSS_TARGETS):
        raise ValueError("evaluation did not produce every loss family")
    return dict(sorted(totals.items())), chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--teacher-cache-set", type=Path, required=True)
    parser.add_argument("--teacher-cache-set-sha256", required=True)
    parser.add_argument("--materialized", type=Path, required=True)
    parser.add_argument("--domain-tags", type=Path, required=True)
    parser.add_argument("--service-tags", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--rows-per-chunk", type=int, default=128)
    parser.add_argument("--logit-chunk", type=int, default=16)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=512)
    parser.add_argument("--sample-limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if min(
        args.gpus,
        args.rows_per_chunk,
        args.logit_chunk,
        args.prefill_chunk_tokens,
    ) <= 0:
        raise SystemExit("GPU, row, logit, and prefill chunk counts must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise SystemExit("--sample-limit must be positive")
    require_budget(args.ledger, args.reserved_gpu_hours)

    try:
        import torch
        from end_to_end_recovery import build_packed_student, load_teacher_file
    except ImportError as error:
        raise SystemExit("install training/requirements.in before evaluation") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA evaluation requested but unavailable")

    try:
        cache = VerifiedTeacherCache.from_manifest(
            args.teacher_cache_set, args.teacher_cache_set_sha256
        )
        expected_ids = {str(item["id"]) for item in cache.artifacts}
        materialized = load_indexed_jsonl(args.materialized, "materialized cohort")
        domain_tags = load_indexed_jsonl(args.domain_tags, "domain tags")
        service_tags = load_indexed_jsonl(args.service_tags, "service tags")
        for label, records in (
            ("materialized cohort", materialized),
            ("domain tags", domain_tags),
            ("service tags", service_tags),
        ):
            require_exact_ids(expected_ids, records, label)
        _provenance, provenance_sha256 = validate_local_model_provenance(
            Path(args.model_source), args.revision, args.local_model_provenance
        )
        if provenance_sha256 != cache.teacher_provenance_sha256:
            raise ValueError("evaluation model provenance differs from teacher cache")

        started = time.monotonic()
        samples = []
        with GpuRun(args.ledger, "heldout-recovery-evaluation", args.gpus, sys.argv):
            with CtoxArtifact(args.artifact, verify_tensors=True) as artifact:
                if artifact.manifest.get("revision") != args.revision:
                    raise ValueError("evaluation artifact revision differs")
                recovery = artifact.manifest.get("recovery")
                if not isinstance(recovery, dict) or not recovery.get(
                    "fixed_logical_qcodes"
                ):
                    raise ValueError("evaluation artifact does not bind fixed qcodes")
                qcode_root = logical_qcode_root(artifact.manifest)
                runtime, base, mtp, fanout = build_packed_student(
                    args.model_source,
                    args.revision,
                    artifact,
                    torch.device(args.device),
                    getattr(torch, args.compute_dtype),
                    args.rows_per_chunk,
                    [int(value) for value in cache.settings["hidden_layers"]],
                    int(cache.settings["top_k"]),
                    args.logit_chunk,
                    False,
                    str(recovery.get("fanout_s_in_policy", "independent")),
                    torch,
                )
                selected = cache.artifacts[: args.sample_limit]
                for index, descriptor in enumerate(selected):
                    sample_id = str(descriptor["id"])
                    teacher = load_teacher_file(cache.verified_artifact_path(index), torch)
                    sequence_tokens = int(teacher["input_ids"].shape[1])
                    losses, chunks = evaluate_losses(
                        runtime,
                        teacher,
                        sequence_tokens,
                        args.prefill_chunk_tokens,
                        torch,
                    )
                    record = materialized[sample_id]
                    domain = domain_tags[sample_id]
                    service = service_tags[sample_id]
                    samples.append(
                        {
                            "id": sample_id,
                            "sequence_tokens": sequence_tokens,
                            "logit_targets": int(teacher["logit_positions"].numel()),
                            "hidden_targets": int(teacher["hidden_positions"].numel()),
                            "mtp_targets": int(teacher["mtp_positions"].numel()),
                            "mtp_hidden_targets": int(
                                teacher["mtp_hidden_positions"].numel()
                            ),
                            "prefill_chunks": chunks,
                            "category": str(record["category"]),
                            "language": str(record["language"]),
                            "source": str(record["source_repo"]),
                            "primary_domain": str(domain["primary_label"]),
                            "domains": sorted(str(value) for value in domain["labels"]),
                            "service_modes": sorted(
                                str(value) for value in service["labels"]
                            ),
                            "losses": losses,
                        }
                    )
                    del teacher
                    print(
                        json.dumps(
                            {
                                "sample": sample_id,
                                "position": index + 1,
                                "total": len(selected),
                                "tokens": sequence_tokens,
                                "chunks": chunks,
                                **losses,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        document = {
            "format": "ctox.recovery.heldout-evaluation.v1",
            "status": (
                "complete"
                if args.sample_limit is None
                else "subset_evaluation_complete"
            ),
            "model": "Qwen/Qwen3.8-27B",
            "revision": args.revision,
            "local_model_provenance_sha256": provenance_sha256,
            "artifact": str(args.artifact.resolve()),
            "artifact_sha256": sha256_path(args.artifact),
            "logical_qcode_root_sha256": qcode_root,
            "recovery": recovery,
            "teacher_cache_set": str(args.teacher_cache_set.resolve()),
            "teacher_cache_set_sha256": args.teacher_cache_set_sha256,
            "teacher_artifact_root_sha256": cache.manifest()[
                "artifact_root_sha256"
            ],
            "materialized_sha256": sha256_path(args.materialized),
            "domain_tags_sha256": sha256_path(args.domain_tags),
            "service_tags_sha256": sha256_path(args.service_tags),
            "prefill_chunk_tokens": args.prefill_chunk_tokens,
            "compute_dtype": args.compute_dtype,
            "base_graph": base,
            "mtp_graph": mtp,
            "fanout_s_in": fanout,
            "elapsed_seconds": time.monotonic() - started,
            "aggregates": aggregate_samples(samples),
            "samples": samples,
        }
        atomic_json(args.output, document)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
