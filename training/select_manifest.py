#!/usr/bin/env python3
"""Select a deterministic balanced subset from provenance manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def score(seed: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{sample_id}".encode("utf-8")).digest()


def select(paths: list[Path], per_manifest: int, seed: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for path in paths:
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                previous = seen.get(record["id"])
                if previous is not None:
                    if previous != record:
                        raise ValueError(f"conflicting duplicate sample id {record['id']}")
                    continue
                seen[record["id"]] = record
                records.append(record)
        if len(records) < per_manifest:
            raise ValueError(f"{path} has only {len(records)} records; requested {per_manifest}")
        records.sort(key=lambda record: (score(seed, record["id"]), record["id"]))
        selected.extend(records[:per_manifest])
    selected.sort(key=lambda record: (record["category"], record["id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--per-manifest", type=int, required=True)
    parser.add_argument("--seed", default="ctox-qwen38-recovery-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_manifest <= 0:
        raise SystemExit("--per-manifest must be positive")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    records = select(args.manifest, args.per_manifest, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
