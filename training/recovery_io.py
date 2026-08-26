"""Durable output transactions shared by Qwen recovery tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def durable_replace(temporary: Path, destination: Path) -> None:
    """Commit one prepared file and durably persist its directory entry."""
    with temporary.open("rb") as source:
        os.fsync(source.fileno())
    os.replace(temporary, destination)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory = os.open(destination.parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        destination.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    durable_replace(temporary, path)


def prepare_output_transaction(
    output_scales: Path,
    output_report: Path,
    output_evidence: Path,
    resume_checkpoint: Path | None,
) -> None:
    """Recover an output set only when its commit-marker evidence is absent."""
    if output_evidence.exists():
        raise ValueError(f"refusing to overwrite committed {output_evidence}")
    partial = [path for path in (output_scales, output_report) if path.exists()]
    if partial and resume_checkpoint is None:
        encoded = ", ".join(str(path) for path in partial)
        raise ValueError(
            f"incomplete recovery output transaction exists ({encoded}); "
            "resume from its final checkpoint"
        )
    for path in partial:
        path.unlink()
    for output in (output_scales, output_report, output_evidence):
        temporary = output.with_name(f".{output.name}.tmp")
        if temporary.exists():
            temporary.unlink()
