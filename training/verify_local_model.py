#!/usr/bin/env python3
"""Verify a staged local model byte-for-byte against a pinned HF revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def root_digest(files: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda item: item["name"]):
        digest.update(
            f"{entry['name']}\0{entry['bytes']}\0{entry['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def verify(model: Path, repo: str, revision: str, api: Any, download: Any) -> dict[str, Any]:
    info = api.model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub resolved {revision} to unexpected commit {info.sha}")
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    required = sorted(model.glob("model-*.safetensors"))
    if not required:
        raise RuntimeError(f"{model} contains no sharded safetensors")
    required.extend(
        model / name
        for name in (
            "model.safetensors.index.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    )
    files = []
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"required staged file is missing: {path}")
        sibling = remote.get(path.name)
        if sibling is None:
            raise RuntimeError(f"{path.name} is absent from {repo}@{revision}")
        actual_sha = sha256_file(path)
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            expected_size = int(lfs.size)
            expected_sha = str(lfs.sha256)
        else:
            remote_path = Path(
                download(repo_id=repo, filename=path.name, revision=revision)
            )
            expected_size = remote_path.stat().st_size
            expected_sha = sha256_file(remote_path)
        if path.stat().st_size != expected_size or actual_sha != expected_sha:
            raise RuntimeError(
                f"staged file differs from {repo}@{revision}: {path.name}"
            )
        files.append(
            {"name": path.name, "bytes": expected_size, "sha256": expected_sha}
        )
    return {
        "format": "ctox.verified-local-model.v1",
        "model": repo,
        "revision": revision,
        "local_root": str(model.resolve()),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "root_sha256": root_digest(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise SystemExit("install training/requirements.in before verification") from error
    try:
        document = verify(args.model, args.repo, args.revision, HfApi(), hf_hub_download)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(document, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(args.output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(json.dumps(document, sort_keys=True))


if __name__ == "__main__":
    main()
