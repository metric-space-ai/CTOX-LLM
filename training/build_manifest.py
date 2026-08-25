#!/usr/bin/env python3
"""Stream recovery sources into a provenance-only JSONL manifest.

The script intentionally does not copy prompt text into the manifest. It emits
content hashes and source coordinates so release eligibility can be audited
without redistributing source records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SOURCES = {
    "nemotron-v1": {
        "repo": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "release_eligible": True,
        "quarantine_reason": None,
    },
    "nemotron-v2": {
        "repo": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "release_eligible": False,
        "quarantine_reason": "gated generated-data terms require release review",
    },
    "nemotron-agentic-v1": {
        "repo": "nvidia/Nemotron-Agentic-v1",
        "release_eligible": True,
        "quarantine_reason": None,
    },
    "nemotron-sft-agentic-v2": {
        "repo": "nvidia/Nemotron-SFT-Agentic-v2",
        "release_eligible": True,
        "quarantine_reason": None,
    },
}

CATEGORY_HINTS = {
    "code": "code",
    "math": "math",
    "tool": "agentic",
    "agent": "agentic",
    "chat": "chat",
    "stem": "math",
    "structured": "structured",
    "long": "long_context",
}


@dataclass(frozen=True)
class Record:
    id: str
    source_repo: str
    source_revision: str
    subset: str
    split: str
    source_id: str
    license: str
    generator: str | None
    category: str
    language: str
    prompt_sha256: str
    release_eligible: bool
    quarantine_reason: str | None


def canonical_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "input", "question", "messages", "conversation", "text"):
        value = row.get(key)
        if value:
            return json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def category_for(subset: str, split: str, row: dict[str, Any]) -> str:
    explicit = str(row.get("category", "")).lower()
    lowered = f"{subset} {split} {explicit}".lower()
    return next((category for hint, category in CATEGORY_HINTS.items() if hint in lowered), "chat")


def records(args: argparse.Namespace) -> Iterable[Record]:
    try:
        from datasets import get_dataset_config_names, load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("install training/requirements.in before building a manifest") from error

    source = SOURCES[args.source]
    repo = source["repo"]
    revision = HfApi().dataset_info(repo, revision=args.revision).sha
    subsets = [args.subset] if args.subset else get_dataset_config_names(repo, revision=revision)
    emitted = 0
    for subset in subsets:
        dataset = load_dataset(repo, subset, split=args.split, revision=revision, streaming=True)
        for index, row in enumerate(dataset):
            text = canonical_text(row)
            prompt_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_id = str(row.get("id", row.get("uuid", index)))
            identity = "\0".join((repo, revision, subset, args.split, source_id, prompt_sha))
            license_name = str(row.get("license", row.get("source_license", "dataset-card")))
            language = str(row.get("language", row.get("lang", "und")))
            generator = row.get("generator") or row.get("model_name")
            yield Record(
                id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                source_repo=repo,
                source_revision=revision,
                subset=subset,
                split=args.split,
                source_id=source_id,
                license=license_name,
                generator=str(generator) if generator else None,
                category=category_for(subset, args.split, row),
                language=language,
                prompt_sha256=prompt_sha,
                release_eligible=bool(source["release_eligible"]),
                quarantine_reason=source["quarantine_reason"],
            )
            emitted += 1
            if args.limit and emitted >= args.limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--subset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        for record in records(args):
            output.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
