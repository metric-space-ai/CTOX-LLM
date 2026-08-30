#!/usr/bin/env python3
"""Build content-bound evidence for the million-sample recovery gate.

All per-sample files must have the same order as the materialized JSONL.  This
keeps the pass memory bounded.  A temporary SQLite uniqueness index proves
cross-partition ID, payload, and semantic-cluster disjointness without holding
millions of strings in Python objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, TextIO

from build_manifest import canonical_text
from recovery_io import atomic_json


FORMAT = "ctox.recovery-million-corpus-evidence.v1"
PARTITIONS = ("train", "calibration", "held_out")
GROUP_FORMAT = "ctox.recovery-million-primary-groups.v1"
SEMANTIC_REQUIRED = (
    "cluster_id",
    "embedding_model",
    "embedding_revision",
    "distance_threshold",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(source: TextIO, name: str) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{name}:{line_number} is not a JSON object")
        yield value


def context_bucket(tokens: int) -> str:
    if tokens <= 0:
        raise ValueError("sequence token count must be positive")
    if tokens <= 2_048:
        return "up_to_2k"
    if tokens <= 8_192:
        return "2k_to_8k"
    if tokens <= 32_768:
        return "8k_to_32k"
    if tokens <= 65_536:
        return "32k_to_64k"
    if tokens <= 131_072:
        return "64k_to_128k"
    raise ValueError(f"sequence length {tokens} exceeds the 128K recovery policy")


def primary_group(
    domain: str,
    service_labels: set[str],
    groups: dict[str, Any],
) -> str:
    for mode, group in groups["priority_service_modes"].items():
        if mode in service_labels:
            return str(group)
    try:
        return str(groups["domains"][domain])
    except KeyError as error:
        raise ValueError(f"domain {domain} has no million-corpus primary group") from error


def validate_groups(groups: dict[str, Any]) -> None:
    if groups.get("format") != GROUP_FORMAT:
        raise ValueError("primary-group map has the wrong format")
    declared = set(groups.get("domains", {}).values()) | set(
        groups.get("priority_service_modes", {}).values()
    )
    expected = {
        "general_dialogue_knowledge",
        "coding_software",
        "agentic_tools",
        "mathematics_stem",
        "professional",
        "language_humanities_creative",
        "long_context",
    }
    if declared != expected:
        raise ValueError(f"primary-group map differs from policy: {sorted(declared ^ expected)}")


def initialize_uniqueness(database: sqlite3.Connection) -> None:
    database.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE identities (
            kind TEXT NOT NULL,
            value TEXT NOT NULL,
            partition_name TEXT NOT NULL,
            PRIMARY KEY (kind, value, partition_name)
        ) WITHOUT ROWID;
        CREATE INDEX identities_lookup ON identities(kind, value);
        """
    )


def record_identity(
    database: sqlite3.Connection,
    kind: str,
    value: str,
    partition: str,
    hard_gates: Counter[str],
) -> bool:
    existing = database.execute(
        "SELECT partition_name FROM identities WHERE kind = ? AND value = ?",
        (kind, value),
    ).fetchall()
    prior_partitions = {str(row[0]) for row in existing}
    if partition in prior_partitions:
        hard_gates[
            "semantic_duplicate_records" if kind == "semantic" else "exact_duplicate_records"
        ] += 1
        return False
    if prior_partitions:
        gate = {
            "id": "cross_partition_id_overlap",
            "payload": "cross_partition_payload_overlap",
            "semantic": "cross_partition_semantic_cluster_overlap",
        }[kind]
        hard_gates[gate] += 1
    database.execute(
        "INSERT INTO identities(kind, value, partition_name) VALUES (?, ?, ?)",
        (kind, value, partition),
    )
    return True


def checked_id(record: dict[str, Any], expected: str, source: str) -> None:
    observed = str(record.get("id", ""))
    if observed != expected:
        raise ValueError(f"{source} is out of order: expected {expected}, got {observed}")


def collect_partition(
    partition: str,
    paths: dict[str, Path],
    groups: dict[str, Any],
    database: sqlite3.Connection,
    hard_gates: Counter[str],
    semantic_contract: dict[str, Any],
) -> dict[str, Any]:
    primary_counts: Counter[str] = Counter()
    context_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    records = 0
    content_root = hashlib.sha256()

    with ExitStack() as stack:
        opened = {
            name: stack.enter_context(path.open(encoding="utf-8"))
            for name, path in paths.items()
        }
        streams = {
            name: jsonl(source, str(paths[name])) for name, source in opened.items()
        }
        for row_number, rows in enumerate(
            zip_longest(*(streams[name] for name in paths), fillvalue=None), 1
        ):
            joined = dict(zip(paths, rows))
            missing = [name for name, row in joined.items() if row is None]
            if missing:
                raise ValueError(
                    f"{partition} sidecars differ in length at row {row_number}: {missing}"
                )
            materialized = joined["materialized"]
            sample_id = str(materialized.get("id", ""))
            if not sample_id:
                raise ValueError(f"{partition} row {row_number} has no sample id")
            for name in paths:
                if name != "materialized":
                    checked_id(joined[name], sample_id, str(paths[name]))

            payload = hashlib.sha256(
                canonical_text(materialized).encode("utf-8")
            ).hexdigest()
            if materialized.get("prompt_sha256") != payload:
                raise ValueError(f"sample {sample_id} has a changed recovery payload")
            domain = str(joined["domain_tags"].get("primary_label", ""))
            services = {str(label) for label in joined["service_tags"].get("labels", [])}
            if not services:
                raise ValueError(f"sample {sample_id} has no service-mode labels")
            token_count = joined["token_counts"].get("sequence_tokens")
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError(f"sample {sample_id} has an invalid sequence token count")
            provenance = joined["provenance"]
            if not provenance.get("release_eligible", False):
                hard_gates["release_ineligible_records"] += 1
            if not str(provenance.get("license", "")).strip():
                hard_gates["unresolved_license_records"] += 1
            required_provenance = (
                "source_repo",
                "source_revision",
                "source_id",
                "prompt_sha256",
            )
            if any(not str(provenance.get(name, "")).strip() for name in required_provenance):
                hard_gates["missing_provenance_records"] += 1
            if provenance.get("prompt_sha256") != payload:
                raise ValueError(f"sample {sample_id} provenance binds another payload")

            semantic = joined["semantic_dedup"]
            for name in SEMANTIC_REQUIRED:
                if semantic.get(name) in (None, ""):
                    raise ValueError(f"sample {sample_id} lacks semantic field {name}")
            contract = {name: semantic[name] for name in SEMANTIC_REQUIRED[1:]}
            if not semantic_contract:
                semantic_contract.update(contract)
            elif semantic_contract != contract:
                raise ValueError("semantic-dedup model/revision/threshold changed inside corpus")
            cluster = str(semantic["cluster_id"])

            unique_counts["id"] += int(
                record_identity(database, "id", sample_id, partition, hard_gates)
            )
            unique_counts["payload"] += int(
                record_identity(database, "payload", payload, partition, hard_gates)
            )
            unique_counts["semantic"] += int(
                record_identity(database, "semantic", cluster, partition, hard_gates)
            )
            primary_counts[primary_group(domain, services, groups)] += 1
            context_counts[context_bucket(token_count)] += 1
            language_counts[str(materialized.get("language", "und"))] += 1
            bound = {
                "id": sample_id,
                "payload_sha256": payload,
                "domain": domain,
                "service_modes": sorted(services),
                "sequence_tokens": token_count,
                "semantic_cluster": cluster,
                "source_repo": provenance.get("source_repo"),
                "source_revision": provenance.get("source_revision"),
            }
            content_root.update(
                json.dumps(bound, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
                    "utf-8"
                )
            )
            content_root.update(b"\n")
            records += 1

    binding_names = {
        "materialized": "materialized_sha256",
        "domain_tags": "domain_tags_sha256",
        "service_tags": "service_tags_sha256",
        "token_counts": "token_plan_sha256",
        "provenance": "provenance_manifest_sha256",
        "semantic_dedup": "semantic_dedup_sha256",
    }
    bindings = {binding_names[name]: sha256_file(path) for name, path in paths.items()}
    return {
        "records": records,
        "unique_ids": unique_counts["id"],
        "unique_payloads": unique_counts["payload"],
        "unique_semantic_clusters": unique_counts["semantic"],
        **bindings,
        "content_root_sha256": content_root.hexdigest(),
        "binding_paths": {
            binding_names[name]: str(path.resolve()) for name, path in paths.items()
        },
        "primary_mix": dict(sorted(primary_counts.items())),
        "context_mix": dict(sorted(context_counts.items())),
        "languages": dict(sorted(language_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument(
        "--partition",
        action="append",
        nargs=7,
        metavar=(
            "NAME",
            "MATERIALIZED",
            "DOMAIN_TAGS",
            "SERVICE_TAGS",
            "TOKEN_COUNTS",
            "PROVENANCE",
            "SEMANTIC_DEDUP",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        groups_bytes = args.groups.read_bytes()
        groups = json.loads(groups_bytes)
        validate_groups(groups)
        provided = {values[0]: values[1:] for values in args.partition}
        if set(provided) != set(PARTITIONS) or len(provided) != len(args.partition):
            raise ValueError("provide train, calibration, and held_out exactly once")
        hard_gates: Counter[str] = Counter()
        semantic_contract: dict[str, Any] = {}
        with tempfile.TemporaryDirectory(prefix="ctox-million-corpus-") as temporary:
            database = sqlite3.connect(str(Path(temporary) / "uniqueness.sqlite3"))
            try:
                initialize_uniqueness(database)
                partitions = {}
                keys = (
                    "materialized",
                    "domain_tags",
                    "service_tags",
                    "token_counts",
                    "provenance",
                    "semantic_dedup",
                )
                for partition in PARTITIONS:
                    paths = {
                        key: Path(value) for key, value in zip(keys, provided[partition])
                    }
                    partitions[partition] = collect_partition(
                        partition,
                        paths,
                        groups,
                        database,
                        hard_gates,
                        semantic_contract,
                    )
                database.commit()
            finally:
                database.close()
        gate_names = (
            "exact_duplicate_records",
            "semantic_duplicate_records",
            "cross_partition_id_overlap",
            "cross_partition_payload_overlap",
            "cross_partition_semantic_cluster_overlap",
            "release_ineligible_records",
            "unresolved_license_records",
            "missing_provenance_records",
        )
        document = {
            "format": FORMAT,
            "groups": str(args.groups.resolve()),
            "groups_sha256": hashlib.sha256(groups_bytes).hexdigest(),
            "semantic_contract": semantic_contract,
            "partitions": partitions,
            "hard_gates": {name: hard_gates[name] for name in gate_names},
        }
        atomic_json(args.output, document)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, sqlite3.Error) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
