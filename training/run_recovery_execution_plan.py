#!/usr/bin/env python3
"""Execute one admitted recovery plan serially with fail-closed resume state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


FORMAT = "ctox.recovery.execution-plan.v1"
STATE_FORMAT = "ctox.recovery.execution-state.v1"
CHECKPOINT_PATTERN = re.compile(r"^recovery-step-(\d{6})\.safetensors$")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_plan(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if document.get("format") != FORMAT or document.get("status") != "admitted":
        raise ValueError("recovery execution plan is not admitted v1")
    if document.get("execution_order") != "serial":
        raise ValueError("recovery execution plan is not serial")
    return document, hashlib.sha256(payload).hexdigest()


def validate_implementation(document: dict[str, Any]) -> None:
    implementation = document.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("recovery plan lacks implementation binding")
    python = Path(str(implementation.get("python", "")))
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"recovery Python is absent or not executable: {python}")
    scripts = implementation.get("scripts")
    if not isinstance(scripts, dict) or not scripts:
        raise ValueError("recovery plan lacks script hashes")
    for name, binding in scripts.items():
        if not isinstance(binding, dict):
            raise ValueError(f"invalid script binding for {name}")
        path = Path(str(binding.get("path", "")))
        expected = str(binding.get("sha256", ""))
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"recovery script changed after admission: {path}")


def canonical_requirement(requirement: str) -> str | None:
    if requirement == "admission":
        return None
    suffix = ":status=complete"
    if requirement.endswith(suffix):
        return requirement[: -len(suffix)]
    return requirement


def validate_stages(document: dict[str, Any]) -> list[dict[str, Any]]:
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("recovery plan has no stages")
    seen: set[str] = set()
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("recovery stage is not an object")
        name = str(stage.get("name", ""))
        if not name or name in seen:
            raise ValueError(f"duplicate or empty recovery stage: {name!r}")
        requirements = stage.get("requires")
        if not isinstance(requirements, list):
            raise ValueError(f"stage {name} lacks requirements")
        for raw in requirements:
            requirement = canonical_requirement(str(raw))
            if requirement is not None and requirement not in seen:
                raise ValueError(f"stage {name} has unsatisfied dependency {raw}")
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
            raise ValueError(f"stage {name} has invalid argv")
        outputs = stage.get("outputs")
        if not isinstance(outputs, list) or not outputs or not all(isinstance(v, str) for v in outputs):
            raise ValueError(f"stage {name} has invalid outputs")
        gpu_count = int(stage.get("gpu_count", -1))
        environment = stage.get("environment")
        if not isinstance(environment, dict):
            raise ValueError(f"stage {name} has invalid environment")
        if set(environment) - {"CUDA_VISIBLE_DEVICES"}:
            raise ValueError(f"stage {name} requests unadmitted environment variables")
        if gpu_count == 1:
            physical_gpu = str(environment.get("CUDA_VISIBLE_DEVICES", ""))
            if not physical_gpu.isdigit() or int(physical_gpu) == 0:
                raise ValueError(f"stage {name} does not bind one non-Greppy physical GPU")
            if "--device" not in argv:
                raise ValueError(f"stage {name} does not use logical cuda:0")
            device_index = argv.index("--device")
            if device_index + 1 >= len(argv) or argv[device_index + 1] != "cuda:0":
                raise ValueError(f"stage {name} does not use logical cuda:0")
        elif gpu_count == 0:
            if environment:
                raise ValueError(f"CPU stage {name} unexpectedly binds CUDA")
        else:
            raise ValueError(f"stage {name} requests unsupported GPU count {gpu_count}")
        seen.add(name)
    return stages


def new_state(plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    return {
        "format": STATE_FORMAT,
        "plan": str(plan_path.resolve()),
        "plan_sha256": plan_sha256,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_path(Path(__file__)),
        "status": "running",
        "completed": [],
        "active_stage": None,
        "failure": None,
        "updated_unix": time.time(),
    }


def load_state(path: Path, plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        state.get("format") != STATE_FORMAT
        or state.get("plan") != str(plan_path.resolve())
        or state.get("plan_sha256") != plan_sha256
        or state.get("runner") != str(Path(__file__).resolve())
        or state.get("runner_sha256") != sha256_path(Path(__file__))
    ):
        raise ValueError("recovery resume state does not bind this exact plan")
    if not isinstance(state.get("completed"), list):
        raise ValueError("recovery resume state has invalid completion records")
    return state


def validate_completed_outputs(record: dict[str, Any]) -> None:
    outputs = record.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("completed recovery stage lacks output evidence")
    for output in outputs:
        path = Path(str(output.get("path", "")))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"completed recovery output is absent: {path}")
        if sha256_path(path) != output.get("sha256"):
            raise ValueError(f"completed recovery output changed: {path}")


def latest_checkpoint(argv: list[str]) -> Path | None:
    if "--checkpoint-dir" not in argv:
        return None
    directory = Path(argv[argv.index("--checkpoint-dir") + 1])
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def output_evidence(paths: list[str]) -> list[dict[str, Any]]:
    evidence = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"recovery stage did not produce a nonempty file: {path}")
        evidence.append(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_path(path)}
        )
    return evidence


def run(plan_path: Path, state_path: Path, resume: bool, dry_run: bool) -> None:
    document, plan_sha256 = read_plan(plan_path)
    validate_implementation(document)
    stages = validate_stages(document)
    if state_path.exists():
        if not resume:
            raise ValueError(f"recovery state already exists; pass --resume: {state_path}")
        state = load_state(state_path, plan_path, plan_sha256)
    else:
        state = new_state(plan_path, plan_sha256)
        if not dry_run:
            atomic_json(state_path, state)

    completed_records = {
        str(record["name"]): record for record in state["completed"]
    }
    for record in completed_records.values():
        validate_completed_outputs(record)

    for stage in stages:
        name = stage["name"]
        if name in completed_records:
            print(f"stage={name} status=verified-skip", flush=True)
            continue
        for raw in stage["requires"]:
            requirement = canonical_requirement(raw)
            if requirement is not None and requirement not in completed_records:
                raise ValueError(f"stage {name} dependency is not complete: {raw}")
        existing = [Path(path) for path in stage["outputs"] if Path(path).exists()]
        if existing:
            raise ValueError(
                f"incomplete stage {name} has existing outputs; inspect before resume: {existing}"
            )
        argv = list(stage["argv"])
        checkpoint = latest_checkpoint(argv) if resume else None
        if checkpoint is not None:
            if name != "train_recovery" or "--resume-checkpoint" in argv:
                raise ValueError(f"unexpected checkpoint resume contract for stage {name}")
            argv.extend(["--resume-checkpoint", str(checkpoint.resolve())])
        print(
            f"stage={name} status={'dry-run' if dry_run else 'start'}"
            + (f" checkpoint={checkpoint}" if checkpoint else ""),
            flush=True,
        )
        if dry_run:
            continue
        validate_implementation(document)
        state.update(active_stage=name, failure=None, updated_unix=time.time())
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(stage["environment"])
        environment["PYTHONUNBUFFERED"] = "1"
        started = time.time()
        try:
            subprocess.run(argv, check=True, env=environment)
            evidence = output_evidence(stage["outputs"])
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            state.update(
                status="failed",
                active_stage=name,
                failure={"stage": name, "error": str(error), "failed_unix": time.time()},
                updated_unix=time.time(),
            )
            atomic_json(state_path, state)
            raise
        record = {
            "name": name,
            "status": "complete",
            "started_unix": started,
            "ended_unix": time.time(),
            "argv": argv,
            "environment": stage["environment"],
            "outputs": evidence,
        }
        state["completed"].append(record)
        completed_records[name] = record
        state.update(status="running", active_stage=None, failure=None, updated_unix=time.time())
        atomic_json(state_path, state)
        print(f"stage={name} status=complete", flush=True)

    if not dry_run:
        state.update(status="complete", active_stage=None, failure=None, updated_unix=time.time())
        atomic_json(state_path, state)
    print(f"recovery-plan status={'validated' if dry_run else 'complete'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state = args.state or args.plan.with_name(f"{args.plan.stem}-state-v1.json")
    try:
        run(args.plan, state, args.resume, args.dry_run)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
