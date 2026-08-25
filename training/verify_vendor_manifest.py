#!/usr/bin/env python3
"""Verify every local file digest declared by a vendored-source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    seen: set[str] = set()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("vendor manifest contains no files")
    for entry in files:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("vendor entry requires string path and sha256")
        if relative in seen:
            raise ValueError(f"duplicate vendor path {relative}")
        seen.add(relative)
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"vendor path escapes manifest directory: {relative}")
        if not path.is_file():
            raise ValueError(f"missing vendored file {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"digest mismatch for {relative}: {actual} != {expected}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    count = verify(args.manifest)
    print(json.dumps({"manifest": str(args.manifest), "verified_files": count}, sort_keys=True))


if __name__ == "__main__":
    main()
