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
import os
import sys
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
    "german-instruct": {
        "repo": "Beko2210/German-Instruct-Dataset",
        "release_eligible": True,
        "quarantine_reason": None,
    },
}

GERMAN_INSTRUCT_REPO = "Beko2210/German-Instruct-Dataset"

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


def recovery_payload(
    row: dict[str, Any], source_repo: str | None = None
) -> dict[str, Any]:
    """Return the complete payload consumed by recovery and teacher caching.

    Hashing only a prompt would leave a changed reference answer undetected.
    Conversational records therefore retain all turns; paired prompt/answer
    records are normalized to chat messages.
    """

    for key in ("messages", "conversation", "conversations"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return {"messages": value}

    prompt = next(
        (row[key] for key in ("prompt", "input", "question", "instruction") if row.get(key)),
        None,
    )
    answer = next(
        (
            row[key]
            for key in ("response", "output", "answer", "completion")
            if row.get(key)
        ),
        None,
    )
    if source_repo == GERMAN_INSTRUCT_REPO and prompt is not None:
        context = row.get("context")
        if context:
            prompt = f"{prompt}\n\nKontext:\n{context}"
    if prompt is not None and answer is not None:
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }
    if prompt is not None:
        return {"prompt": prompt if isinstance(prompt, str) else stable_json(prompt)}
    if row.get("text") is not None:
        text = row["text"]
        return {"prompt": text if isinstance(text, str) else stable_json(text)}
    return {"prompt": stable_json(row)}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_text(row: dict[str, Any], source_repo: str | None = None) -> str:
    return stable_json(recovery_payload(row, source_repo))


def source_id_for(row: dict[str, Any], index: int) -> str:
    return str(row.get("id", row.get("uuid", index)))


def category_for(subset: str, split: str, row: dict[str, Any]) -> str:
    explicit = str(row.get("category", "")).lower()
    if explicit == "coding":
        return "code"
    if explicit == "rag":
        return "long_context"
    if explicit in {"business", "german_pro", "bureaucracy", "hard_prompts", "safety"}:
        return "chat"
    lowered = f"{subset} {split} {explicit}".lower()
    return next((category for hint, category in CATEGORY_HINTS.items() if hint in lowered), "chat")


def nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


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
            text = canonical_text(row, repo)
            prompt_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            source_id = source_id_for(row, index)
            identity = "\0".join((repo, revision, subset, args.split, source_id, prompt_sha))
            metadata = nested_metadata(row)
            license_name = str(
                row.get("license", row.get("source_license", metadata.get("license", "dataset-card")))
            )
            language = str(row.get("language", row.get("lang", "und")))
            generator = row.get("generator") or row.get("model_name") or metadata.get("annotator")
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
    # Hugging Face streaming leaves Xet/Arrow worker threads alive on some
    # Linux builds and may hang or abort during interpreter finalization. At
    # this point the exclusive output context has closed and flushed its Python
    # buffers. Bypass only the broken third-party finalizers after a successful
    # main; exceptions still take the normal non-zero path above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
