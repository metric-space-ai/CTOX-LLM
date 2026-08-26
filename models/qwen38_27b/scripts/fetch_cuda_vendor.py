#!/usr/bin/env python3
"""Fetch byte-identical CUDA reference sources declared in UPSTREAM.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parents[1] / "vendor/cuda/UPSTREAM.json",
    )
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository = document["upstream"]["repository"]
    revision = document["upstream"]["revision"]
    if repository != "https://github.com/ggml-org/llama.cpp":
        raise SystemExit(f"unsupported vendor repository: {repository}")
    requested = set(args.path)
    entries = [
        entry
        for entry in document["files"]
        if not requested or entry["path"] in requested
    ]
    if requested != {entry["path"] for entry in entries}:
        missing = sorted(requested - {entry["path"] for entry in entries})
        raise SystemExit(f"paths are not declared by the manifest: {missing}")

    for entry in entries:
        relative = Path(entry["path"])
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise SystemExit(f"vendor path escapes manifest root: {relative}")
        url = (
            "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
            f"{revision}/{entry['upstream_path']}"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(
                f"digest mismatch for {relative}: {digest} != {entry['sha256']}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        print(f"fetched {relative} sha256={digest} bytes={len(payload)}")


if __name__ == "__main__":
    main()
