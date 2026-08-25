#!/usr/bin/env python3
"""Materialize pinned recovery records into an access-controlled JSONL cache.

The committed manifest contains provenance and hashes only. This script streams
the exact pinned source revisions, verifies every payload hash, and writes only
the requested records. Materialized source text must never be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_manifest import canonical_text, recovery_payload, source_id_for


SourceKey = tuple[str, str, str, str]
RecordKey = tuple[str, str]


def load_manifests(
    paths: list[Path], allow_quarantined: bool
) -> dict[SourceKey, dict[RecordKey, dict[str, Any]]]:
    groups: dict[SourceKey, dict[RecordKey, dict[str, Any]]] = defaultdict(dict)
    seen_ids: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("release_eligible", False) and not allow_quarantined:
                    raise ValueError(
                        f"{path}:{line_number}: quarantined sample {record['id']}; "
                        "pass --allow-quarantined only for isolated research"
                    )
                previous = seen_ids.get(record["id"])
                if previous is not None:
                    if previous != record:
                        raise ValueError(f"conflicting duplicate sample id {record['id']}")
                    continue
                seen_ids[record["id"]] = record
                source_key = (
                    record["source_repo"],
                    record["source_revision"],
                    record["subset"],
                    record["split"],
                )
                record_key = (record["source_id"], record["prompt_sha256"])
                if record_key in groups[source_key]:
                    raise ValueError(f"ambiguous source coordinate {source_key!r} {record_key!r}")
                groups[source_key][record_key] = record
    return groups


def materialize(groups: dict[SourceKey, dict[RecordKey, dict[str, Any]]], output: Any) -> int:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("install training/requirements.in before materializing prompts") from error

    written = 0
    for (repo, revision, subset, split), wanted in sorted(groups.items()):
        remaining = dict(wanted)
        dataset = load_dataset(repo, subset, split=split, revision=revision, streaming=True)
        for index, row in enumerate(dataset):
            source_id = source_id_for(row, index)
            payload_sha = hashlib.sha256(canonical_text(row).encode("utf-8")).hexdigest()
            record = remaining.pop((source_id, payload_sha), None)
            if record is None:
                continue
            materialized = dict(record)
            materialized.update(recovery_payload(row))
            output.write(json.dumps(materialized, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
            if not remaining:
                break
        if remaining:
            examples = ", ".join(source_id for source_id, _ in list(remaining)[:5])
            raise RuntimeError(
                f"source revision no longer yielded {len(remaining)} requested records "
                f"for {repo}/{subset}/{split}: {examples}"
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-quarantined", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    groups = load_manifests(args.manifest, args.allow_quarantined)
    if not groups:
        raise SystemExit("manifests contain no records")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            count = materialize(groups, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        if args.output.exists():
            raise RuntimeError(f"output appeared while materializing: {args.output}")
        temporary_path.rename(args.output)
        temporary_path = None
        print(json.dumps({"output": str(args.output), "records": count}, sort_keys=True))
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
    # See build_manifest.py: avoid a known Xet/Arrow finalizer hang only after
    # the output has been flushed, fsynced, and atomically renamed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
