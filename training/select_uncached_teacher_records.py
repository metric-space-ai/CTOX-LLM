#!/usr/bin/env python3
"""Select the exact final-corpus identities absent from verified teacher caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from teacher_cache_dataset import VerifiedTeacherCache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_missing(
    records: Iterable[dict[str, Any]], cached_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = list(records)
    by_id: dict[str, dict[str, Any]] = {}
    for record in ordered:
        sample_id = str(record["id"])
        if sample_id in by_id:
            raise ValueError(f"final recovery cohort duplicates {sample_id}")
        by_id[sample_id] = record
    extra = cached_ids - set(by_id)
    if extra:
        raise ValueError(
            f"verified teacher cache contains {len(extra)} identities outside final cohort"
        )
    reused = [record for record in ordered if str(record["id"]) in cached_ids]
    missing = [record for record in ordered if str(record["id"]) not in cached_ids]
    if len(reused) != len(cached_ids) or len(reused) + len(missing) != len(ordered):
        raise AssertionError("teacher cache partition is not exact")
    return reused, missing


def distribution(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record.get(key, "unknown")) for record in records).items()))


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                temporary.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--verification", type=Path, action="append", required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite uncached cohort output or report")
    try:
        records = read_jsonl(args.input)
        cache = VerifiedTeacherCache(
            args.verification,
            args.teacher_revision,
            args.teacher_provenance_sha256,
        )
        cached_ids = {str(artifact["id"]) for artifact in cache.artifacts}
        reused, missing = select_missing(records, cached_ids)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error

    write_jsonl_atomic(args.output, missing)
    verification_evidence = []
    for path in args.verification:
        verification_evidence.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    report = {
        "format": "ctox.teacher-cache-missing-selection.v1",
        "input": str(args.input.resolve()),
        "input_bytes": args.input.stat().st_size,
        "input_sha256": sha256_file(args.input),
        "final_records": len(records),
        "teacher_revision": args.teacher_revision,
        "teacher_provenance_sha256": args.teacher_provenance_sha256,
        "teacher_settings": cache.settings,
        "verification_evidence": verification_evidence,
        "reused_records": len(reused),
        "missing_records": len(missing),
        "reused": {
            "categories": distribution(reused, "category"),
            "languages": distribution(reused, "language"),
            "sources": distribution(reused, "source_repo"),
        },
        "missing": {
            "categories": distribution(missing, "category"),
            "languages": distribution(missing, "language"),
            "sources": distribution(missing, "source_repo"),
        },
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256_file(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
