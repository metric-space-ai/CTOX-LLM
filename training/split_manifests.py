#!/usr/bin/env python3
"""Create deterministic, disjoint recovery-training and evaluation cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def rank(seed: str, partition: str, sample_id: str) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{partition}\0{sample_id}".encode("utf-8")
    ).digest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or len(sample_id) != 64:
                raise ValueError(f"{path}:{line_number} has no SHA-256 sample id")
            previous = seen.get(sample_id)
            if previous is not None:
                if previous != record:
                    raise ValueError(f"{path} contains conflicting duplicate {sample_id}")
                continue
            seen[sample_id] = record
            records.append(record)
    return records


def split(
    paths: list[Path],
    train_per_manifest: int,
    evaluation_per_manifest: int,
    seed: str,
    require_release_eligible: bool = True,
    stratify_field: str | None = None,
    excluded_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_per_manifest <= 0 or evaluation_per_manifest <= 0:
        raise ValueError("partition sizes must be positive")
    excluded_ids = excluded_ids or set()
    train: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    globally_seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        records = read_manifest(path)
        if require_release_eligible:
            ineligible = [record["id"] for record in records if not record.get("release_eligible")]
            if ineligible:
                raise ValueError(
                    f"{path} contains {len(ineligible)} release-ineligible records"
                )
        unique = []
        for record in records:
            if record["id"] in excluded_ids:
                continue
            previous = globally_seen.get(record["id"])
            if previous is not None:
                if previous != record:
                    raise ValueError(f"conflicting duplicate sample id {record['id']}")
                continue
            globally_seen[record["id"]] = record
            unique.append(record)
        strata: dict[str, list[dict[str, Any]]] = {}
        for record in unique:
            if stratify_field is None:
                stratum = path.name
            else:
                value = record.get(stratify_field)
                if value is None or not str(value):
                    raise ValueError(f"{path} record {record['id']} lacks {stratify_field}")
                stratum = str(value)
            strata.setdefault(stratum, []).append(record)

        for stratum, stratum_records in sorted(strata.items()):
            required = train_per_manifest + evaluation_per_manifest
            if len(stratum_records) < required:
                raise ValueError(
                    f"{path} stratum {stratum!r} has only {len(stratum_records)} unique "
                    f"records; requested {required}"
                )

            # Training is selected first. Evaluation uses an independent
            # ranking over the remainder, making overlap impossible.
            ranked_train = sorted(
                stratum_records,
                key=lambda record: (
                    rank(
                        seed,
                        "train" if stratify_field is None else f"train:{stratum}",
                        record["id"],
                    ),
                    record["id"],
                ),
            )
            selected_train = ranked_train[:train_per_manifest]
            train_ids = {record["id"] for record in selected_train}
            remaining = [
                record for record in stratum_records if record["id"] not in train_ids
            ]
            selected_evaluation = sorted(
                remaining,
                key=lambda record: (
                    rank(
                        seed,
                        "evaluation"
                        if stratify_field is None
                        else f"evaluation:{stratum}",
                        record["id"],
                    ),
                    record["id"],
                ),
            )[:evaluation_per_manifest]
            train.extend(selected_train)
            evaluation.extend(selected_evaluation)

    train.sort(key=lambda record: (record["category"], record["source_repo"], record["id"]))
    evaluation.sort(
        key=lambda record: (record["category"], record["source_repo"], record["id"])
    )
    if {record["id"] for record in train} & {record["id"] for record in evaluation}:
        raise AssertionError("training and evaluation partitions overlap")
    return train, evaluation


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            payload = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            output.write(payload.decode("utf-8"))
            digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--train-per-manifest", type=int, required=True)
    parser.add_argument("--evaluation-per-manifest", type=int, required=True)
    parser.add_argument("--seed", default="ctox-qwen38-recovery-split-v1")
    parser.add_argument(
        "--stratify-field",
        help="apply train/evaluation quotas independently to each value of this field",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="exclude every sample identity listed here from both partitions",
    )
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    parser.add_argument(
        "--allow-quarantined",
        action="store_true",
        help="research-only: permit records not approved for the public checkpoint",
    )
    args = parser.parse_args()
    if args.train_output == args.evaluation_output:
        raise SystemExit("training and evaluation outputs must differ")
    try:
        excluded_ids = {
            record["id"]
            for path in args.exclude_manifest
            for record in read_manifest(path)
        }
        train, evaluation = split(
            args.manifest,
            args.train_per_manifest,
            args.evaluation_per_manifest,
            args.seed,
            require_release_eligible=not args.allow_quarantined,
            stratify_field=args.stratify_field,
            excluded_ids=excluded_ids,
        )
        train_sha256 = write_jsonl(args.train_output, train)
        evaluation_sha256 = write_jsonl(args.evaluation_output, evaluation)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "format": "ctox.recovery-split.v1",
                "seed": args.seed,
                "source_manifests": len(args.manifest),
                "stratify_field": args.stratify_field,
                "excluded_records": len(excluded_ids),
                "train_records": len(train),
                "train_sha256": train_sha256,
                "evaluation_records": len(evaluation),
                "evaluation_sha256": evaluation_sha256,
                "overlap": 0,
                "release_eligible_required": not args.allow_quarantined,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
