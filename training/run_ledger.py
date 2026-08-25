"""Append-only GPU-hour ledger used by every recovery stage."""

from __future__ import annotations

import json
import os
import time
from contextlib import AbstractContextManager
from pathlib import Path


class GpuRun(AbstractContextManager["GpuRun"]):
    def __init__(self, ledger: Path, stage: str, gpu_count: int, command: list[str]) -> None:
        if gpu_count < 1:
            raise ValueError("gpu_count must be positive")
        self.ledger = ledger
        self.stage = stage
        self.gpu_count = gpu_count
        self.command = command
        self.started = time.time()

    def __enter__(self) -> "GpuRun":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        ended = time.time()
        entry = {
            "stage": self.stage,
            "pid": os.getpid(),
            "started_unix": self.started,
            "ended_unix": ended,
            "elapsed_seconds": ended - self.started,
            "gpu_count": self.gpu_count,
            "gpu_hours": (ended - self.started) * self.gpu_count / 3600.0,
            "command": self.command,
            "success": exc_type is None,
        }
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, sort_keys=True) + "\n")


def total_gpu_hours(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    with ledger.open(encoding="utf-8") as source:
        return sum(json.loads(line)["gpu_hours"] for line in source if line.strip())


def require_budget(ledger: Path, requested_gpu_hours: float, ceiling: float = 240.0) -> None:
    projected = total_gpu_hours(ledger) + requested_gpu_hours
    if projected > ceiling:
        raise RuntimeError(f"GPU-hour ceiling exceeded: projected {projected:.2f}, limit {ceiling:.2f}")
