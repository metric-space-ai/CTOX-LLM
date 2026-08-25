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
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--verification-root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
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
        if len(cache.batches) != int(batch_plan["summary"]["batches"]):
            raise ValueError("verified cache batch count differs from plan")
        if len(cache.artifacts) != int(batch_plan["summary"]["samples"]):
            raise ValueError("verified cache sample count differs from plan")
        if not args.skip_artifact_rehash:
            for index in range(len(cache.artifacts)):
                cache.verified_artifact_path(index)
        document = cache.manifest()
        document.update(
            {
                "batch_plan": str(args.batch_plan.resolve()),
                "batch_plan_sha256": hashlib.sha256(batch_plan_bytes).hexdigest(),
                "verification_root": str(args.verification_root.resolve()),
                "prefix": args.prefix,
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
