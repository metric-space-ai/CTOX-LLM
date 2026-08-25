#!/usr/bin/env python3
"""Bind all verified immutable batches into one recovery cache-set manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from teacher_cache_dataset import VerifiedTeacherCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--verification", type=Path, action="append", default=[])
    parser.add_argument("--expected-input", type=Path)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-artifact-rehash",
        action="store_true",
        help="development-only: omit the final full artifact byte/hash pass",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        plan_arguments = (args.batch_plan, args.verification_root, args.prefix)
        if args.verification:
            if any(value is not None for value in plan_arguments):
                raise ValueError("explicit verifications cannot be combined with a batch plan")
            verification_paths = args.verification
            batch_plan_bytes = None
            batch_plan = None
        else:
            if any(value is None for value in plan_arguments):
                raise ValueError(
                    "provide either --verification entries or batch plan, root, and prefix"
                )
            batch_plan_bytes = args.batch_plan.read_bytes()
            batch_plan = json.loads(batch_plan_bytes)
            verification_paths = [
                args.verification_root
                / f"{args.prefix}-batch-{int(batch['batch_index']):03d}-v1-verification-v1.json"
                for batch in batch_plan["batches"]
            ]
        cache = VerifiedTeacherCache(
            verification_paths,
            args.teacher_revision,
            args.teacher_provenance_sha256,
        )
        if batch_plan is not None:
            if len(cache.batches) != int(batch_plan["summary"]["batches"]):
                raise ValueError("verified cache batch count differs from plan")
            if len(cache.artifacts) != int(batch_plan["summary"]["samples"]):
                raise ValueError("verified cache sample count differs from plan")
        expected_input = None
        if args.expected_input is not None:
            expected_bytes = args.expected_input.read_bytes()
            expected_records = [
                json.loads(line)
                for line in expected_bytes.decode("utf-8").splitlines()
                if line.strip()
            ]
            expected_ids = [str(record["id"]) for record in expected_records]
            if len(expected_ids) != len(set(expected_ids)):
                raise ValueError("expected teacher-cache input contains duplicate identities")
            cached_ids = [str(artifact["id"]) for artifact in cache.artifacts]
            if set(cached_ids) != set(expected_ids):
                missing = len(set(expected_ids) - set(cached_ids))
                extra = len(set(cached_ids) - set(expected_ids))
                raise ValueError(
                    f"verified teacher cache differs from expected input: {missing} missing, {extra} extra"
                )
            expected_input = {
                "path": str(args.expected_input.resolve()),
                "bytes": len(expected_bytes),
                "sha256": hashlib.sha256(expected_bytes).hexdigest(),
                "records": len(expected_ids),
            }
        if not args.skip_artifact_rehash:
            for index in range(len(cache.artifacts)):
                cache.verified_artifact_path(index)
        document = cache.manifest()
        document.update(
            {
                "batch_plan": (
                    str(args.batch_plan.resolve()) if args.batch_plan is not None else None
                ),
                "batch_plan_sha256": (
                    hashlib.sha256(batch_plan_bytes).hexdigest()
                    if batch_plan_bytes is not None
                    else None
                ),
                "verification_root": (
                    str(args.verification_root.resolve())
                    if args.verification_root is not None
                    else None
                ),
                "prefix": args.prefix,
                "expected_input": expected_input,
                "all_artifacts_rehashed": not args.skip_artifact_rehash,
            }
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
