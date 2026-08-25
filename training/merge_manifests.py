#!/usr/bin/env python3
"""Merge immutable recovery manifests with duplicate and policy checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def merge(
    paths: list[Path], require_release_eligible: bool = True
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record.get("id")
                if not isinstance(sample_id, str) or len(sample_id) != 64:
                    raise ValueError(f"{path}:{line_number} has no SHA-256 sample id")
                if require_release_eligible and not record.get("release_eligible"):
                    raise ValueError(f"{path}:{line_number} is release-ineligible")
                previous = records.get(sample_id)
                if previous is not None and previous != record:
                    raise ValueError(f"conflicting duplicate sample id {sample_id}")
                records[sample_id] = record
    return sorted(
        records.values(),
        key=lambda record: (
            record.get("category", ""),
            record.get("language", ""),
            record.get("source_repo", ""),
            record["id"],
        ),
    )


def write(path: Path, records: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as output:
        for record in records:
            encoded = (
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            output.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--allow-quarantined", action="store_true")
    args = parser.parse_args()
    try:
        records = merge(
            args.manifest,
            require_release_eligible=not args.allow_quarantined,
        )
        if args.expected_records is not None and len(records) != args.expected_records:
            raise ValueError(
                f"merged {len(records)} unique records; expected {args.expected_records}"
            )
        sha256 = write(args.output, records)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "format": "ctox.recovery-merged-manifest.v1",
                "input_manifests": len(args.manifest),
                "records": len(records),
                "sha256": sha256,
                "release_eligible_required": not args.allow_quarantined,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
