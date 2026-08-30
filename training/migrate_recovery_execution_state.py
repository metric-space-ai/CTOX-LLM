#!/usr/bin/env python3
"""Create a new recovery execution state from a verified completed prefix.

Execution plans are immutable.  When a new plan fixes a non-semantic runtime
binding (for example the Python executable), already completed prefix stages
must not be rerun or copied by hand.  This tool rehashes every completed output
and admits only an ordered prefix whose declared outputs are identical in both
plans.  The new state records both source hashes and is written create-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PLAN_FORMAT = "ctox.recovery.execution-plan.v1"
STATE_FORMAT = "ctox.recovery.execution-state.v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return document, hashlib.sha256(payload).hexdigest()


def atomic_create_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite recovery state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def output_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("completed stage has no output evidence")
    result = {}
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("completed output evidence is not an object")
        path = Path(str(output.get("path", ""))).resolve()
        if str(path) in result:
            raise ValueError(f"completed output is repeated: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"completed output is absent: {path}")
        digest = sha256_path(path)
        if digest != output.get("sha256") or path.stat().st_size != output.get("bytes"):
            raise ValueError(f"completed output evidence changed: {path}")
        result[str(path)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return result


def migrate(
    source_state_path: Path,
    target_plan_path: Path,
    runner_path: Path,
    output_path: Path,
    completed_count: int,
) -> None:
    source_state, source_state_sha256 = read_json(source_state_path)
    target_plan, target_plan_sha256 = read_json(target_plan_path)
    if source_state.get("format") != STATE_FORMAT:
        raise ValueError("source recovery state has the wrong format")
    if target_plan.get("format") != PLAN_FORMAT or target_plan.get("status") != "admitted":
        raise ValueError("target recovery plan is not admitted")
    stages = target_plan.get("stages")
    completed = source_state.get("completed")
    if not isinstance(stages, list) or not isinstance(completed, list):
        raise ValueError("recovery plan/state has invalid stage lists")
    if completed_count <= 0 or completed_count > len(stages) or completed_count > len(completed):
        raise ValueError("completed prefix length is invalid")
    migrated = []
    for index in range(completed_count):
        stage = stages[index]
        record = completed[index]
        if stage.get("name") != record.get("name") or record.get("status") != "complete":
            raise ValueError(f"completed stage {index} is not the target-plan prefix")
        declared = {str(Path(value).resolve()) for value in stage.get("outputs", [])}
        evidence = output_map(record)
        if declared != set(evidence):
            raise ValueError(f"completed stage outputs differ for {stage.get('name')}")
        migrated.append({**record, "outputs": [evidence[path] for path in sorted(evidence)]})
    runner = runner_path.resolve()
    # The execution runner is normally invoked through the plan-bound Python
    # interpreter and therefore need not carry an executable mode bit.
    if not runner.is_file():
        raise ValueError(f"target runner is absent: {runner}")
    document = {
        "format": STATE_FORMAT,
        "plan": str(target_plan_path.resolve()),
        "plan_sha256": target_plan_sha256,
        "runner": str(runner),
        "runner_sha256": sha256_path(runner),
        "status": "running",
        "completed": migrated,
        "active_stage": None,
        "failure": None,
        "migration": {
            "source_state": str(source_state_path.resolve()),
            "source_state_sha256": source_state_sha256,
            "migrated_stages": [record["name"] for record in migrated],
        },
    }
    atomic_create_json(output_path.resolve(), document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-count", type=int, required=True)
    args = parser.parse_args()
    migrate(
        args.source_state,
        args.target_plan,
        args.runner,
        args.output,
        args.completed_count,
    )


if __name__ == "__main__":
    main()
