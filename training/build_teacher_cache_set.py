#!/usr/bin/env python3
"""Bind all verified immutable batches into one recovery cache-set manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from select_activation_calibration import write_bytes_atomic
from teacher_cache_dataset import VerifiedTeacherCache


SHA256 = re.compile(r"^[0-9a-f]{64}$")


def batch_group(
    plan_path: Path,
    verification_root: Path,
    prefix: str,
) -> tuple[list[Path], dict[str, object], int]:
    """Resolve and bind one immutable batch plan to its verification files."""

    encoded = plan_path.read_bytes()
    plan = json.loads(encoded)
    batches = plan["batches"]
    expected_batches = int(plan["summary"]["batches"])
    expected_samples = int(plan["summary"]["samples"])
    if len(batches) != expected_batches:
        raise ValueError(f"batch count differs from summary in {plan_path}")
    paths = [
        verification_root
        / f"{prefix}-batch-{int(batch['batch_index']):03d}-v1-verification-v1.json"
        for batch in batches
    ]
    indices = [int(batch["batch_index"]) for batch in batches]
    if indices != list(range(len(indices))):
        raise ValueError(f"batch indices are not contiguous from zero in {plan_path}")
    return (
        paths,
        {
            "batch_plan": str(plan_path.resolve()),
            "batch_plan_bytes": len(encoded),
            "batch_plan_sha256": hashlib.sha256(encoded).hexdigest(),
            "verification_root": str(verification_root.resolve()),
            "prefix": prefix,
            "batches": expected_batches,
            "samples": expected_samples,
        },
        expected_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-plan", type=Path)
    parser.add_argument("--verification-root", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument(
        "--batch-group",
        nargs=3,
        action="append",
        default=[],
        metavar=("PLAN", "VERIFICATION_ROOT", "PREFIX"),
        help="repeatable batch-plan/root/prefix group for a combined final cache set",
    )
    parser.add_argument(
        "--bound-batch-group",
        nargs=4,
        action="append",
        default=[],
        metavar=("PLAN", "SHA256", "VERIFICATION_ROOT", "PREFIX"),
        help="repeatable batch group whose exact plan bytes are frozen by SHA-256",
    )
    parser.add_argument("--verification", type=Path, action="append", default=[])
    parser.add_argument(
        "--bound-verification",
        nargs=2,
        action="append",
        default=[],
        metavar=("PATH", "SHA256"),
        help="repeatable explicit verification whose exact bytes are frozen by SHA-256",
    )
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
        if args.expected_input is not None:
            if args.verification:
                raise ValueError(
                    "an expected-input final cache set requires bound explicit verifications"
                )
            if args.batch_group or any(
                value is not None
                for value in (args.batch_plan, args.verification_root, args.prefix)
            ):
                raise ValueError(
                    "an expected-input final cache set requires bound batch groups"
                )
        plan_arguments = (args.batch_plan, args.verification_root, args.prefix)
        has_legacy_group = any(value is not None for value in plan_arguments)
        if has_legacy_group and (
            any(value is None for value in plan_arguments)
            or args.verification
            or args.bound_verification
            or args.batch_group
            or args.bound_batch_group
        ):
            raise ValueError(
                "legacy batch-plan arguments must be complete and cannot be combined"
            )
        if (
            not has_legacy_group
            and not args.verification
            and not args.bound_verification
            and not args.batch_group
            and not args.bound_batch_group
        ):
            raise ValueError(
                "provide verifications, batch groups, or batch plan, root, and prefix"
            )
        if has_legacy_group:
            raw_groups = [(str(args.batch_plan), str(args.verification_root), args.prefix)]
        else:
            raw_groups = args.batch_group
        bound_raw_groups = args.bound_batch_group
        verification_paths = [path.expanduser().resolve() for path in args.verification]
        bound_verifications = []
        for raw_path, expected_sha256 in args.bound_verification:
            path = Path(raw_path).expanduser().resolve()
            if not SHA256.fullmatch(expected_sha256):
                raise ValueError(f"bound verification SHA-256 is invalid: {expected_sha256}")
            encoded = path.read_bytes()
            actual_sha256 = hashlib.sha256(encoded).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"bound verification changed: {path} is {actual_sha256}, "
                    f"expected {expected_sha256}"
                )
            verification_paths.append(path)
            bound_verifications.append(
                {
                    "path": str(path),
                    "bytes": len(encoded),
                    "sha256": actual_sha256,
                }
            )
        if len(verification_paths) != len(set(verification_paths)):
            raise ValueError("explicit teacher-cache verification path occurs more than once")
        explicit_batch_count = len(verification_paths)
        if raw_groups or bound_raw_groups:
            batch_groups = []
            group_ranges = []
            batch_offset = explicit_batch_count
            for raw_plan, raw_root, prefix in raw_groups:
                paths, record, samples = batch_group(Path(raw_plan), Path(raw_root), prefix)
                verification_paths.extend(paths)
                batch_groups.append(record)
                group_ranges.append((batch_offset, len(paths), samples))
                batch_offset += len(paths)
            for raw_plan, expected_sha256, raw_root, prefix in bound_raw_groups:
                if not SHA256.fullmatch(expected_sha256):
                    raise ValueError(
                        f"bound batch-plan SHA-256 is invalid: {expected_sha256}"
                    )
                paths, record, samples = batch_group(
                    Path(raw_plan), Path(raw_root), prefix
                )
                actual_sha256 = str(record["batch_plan_sha256"])
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        f"bound batch plan changed: {Path(raw_plan).resolve()} is "
                        f"{actual_sha256}, expected {expected_sha256}"
                    )
                record["expected_batch_plan_sha256"] = expected_sha256
                verification_paths.extend(paths)
                batch_groups.append(record)
                group_ranges.append((batch_offset, len(paths), samples))
                batch_offset += len(paths)
        else:
            batch_groups = None
            group_ranges = []
        cache = VerifiedTeacherCache(
            verification_paths,
            args.teacher_revision,
            args.teacher_provenance_sha256,
        )
        if batch_groups is not None:
            for offset, batches, samples in group_ranges:
                actual = cache.batches[offset : offset + batches]
                actual_samples = sum(int(batch["samples"]) for batch in actual)
                if len(actual) != batches or actual_samples != samples:
                    raise ValueError("verified cache group differs from its batch plan")
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
                    batch_groups[0]["batch_plan_sha256"]
                    if batch_groups is not None and len(batch_groups) == 1
                    else None
                ),
                "verification_root": (
                    str(args.verification_root.resolve())
                    if args.verification_root is not None
                    else None
                ),
                "prefix": args.prefix,
                "bound_verifications": bound_verifications,
                "batch_groups": batch_groups,
                "expected_input": expected_input,
                "all_artifacts_rehashed": not args.skip_artifact_rehash,
            }
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_bytes_atomic(
        args.output,
        (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


if __name__ == "__main__":
    main()
