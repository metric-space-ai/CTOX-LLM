#!/usr/bin/env python3
"""Audit materialized recovery cohorts before expensive teacher caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_manifest import canonical_text
from prompt_format import normalize_messages, render_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def distribution(values: list[int]) -> dict[str, int]:
    return {
        "minimum": min(values) if values else 0,
        "median": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else 0,
        "total": sum(values),
    }


def is_structured(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith(("{", "[", "<", "```"))


def audit(path: Path, tokenizer: Any) -> tuple[dict[str, Any], set[str], set[str]]:
    ids: set[str] = set()
    payload_hashes: set[str] = set()
    categories: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    prompt_tokens = []
    assistant_tokens = []
    multi_turn = 0
    tool_records = 0
    tool_definitions = 0
    structured_answers = 0
    assistant_ready = 0
    records = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record["id"]
            if sample_id in ids:
                raise ValueError(f"{path}:{line_number} duplicates {sample_id}")
            ids.add(sample_id)
            actual_payload = hashlib.sha256(canonical_text(record).encode("utf-8")).hexdigest()
            if actual_payload != record["prompt_sha256"]:
                raise ValueError(f"{path}:{line_number} has a changed recovery payload")
            if actual_payload in payload_hashes:
                raise ValueError(f"{path}:{line_number} duplicates a recovery payload")
            payload_hashes.add(actual_payload)
            messages = normalize_messages(record.get("messages", []))
            conditioning = [
                message
                for message in messages
                if message.get("role") != "assistant"
                and (
                    str(message.get("content") or "").strip()
                    or message.get("tool_calls")
                )
            ]
            if messages and not conditioning:
                raise ValueError(f"{path}:{line_number} has no conditioning content")
            rendered = render_record(tokenizer, record)
            prompt_tokens.append(
                len(tokenizer(rendered, add_special_tokens=False).input_ids)
            )
            if len(messages) > 2:
                multi_turn += 1
            tools = record.get("tools") or []
            if tools:
                tool_records += 1
                tool_definitions += len(tools)
            if messages and messages[-1]["role"] == "assistant":
                assistant_ready += 1
                answer = messages[-1].get("content") or ""
                assistant_tokens.append(
                    len(tokenizer(answer, add_special_tokens=False).input_ids)
                )
                structured_answers += int(is_structured(answer))
            categories[record["category"]] += 1
            languages[record["language"]] += 1
            sources[record["source_repo"]] += 1
            records += 1
    return (
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "records": records,
            "unique_ids": len(ids),
            "payload_hashes_verified": records,
            "categories": dict(sorted(categories.items())),
            "languages": dict(sorted(languages.items())),
            "sources": dict(sorted(sources.items())),
            "prompt_tokens": distribution(prompt_tokens),
            "assistant_target_tokens": distribution(assistant_tokens),
            "assistant_target_ready_records": assistant_ready,
            "multi_turn_records": multi_turn,
            "tool_records": tool_records,
            "tool_definitions": tool_definitions,
            "structured_final_answers": structured_answers,
        },
        ids,
        payload_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before auditing") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision
    )
    try:
        train, train_ids, train_payloads = audit(args.train, tokenizer)
        evaluation, evaluation_ids, evaluation_payloads = audit(args.evaluation, tokenizer)
        overlap = train_ids & evaluation_ids
        if overlap:
            raise ValueError(f"train/evaluation overlap contains {len(overlap)} records")
        payload_overlap = train_payloads & evaluation_payloads
        if payload_overlap:
            raise ValueError(
                f"train/evaluation payload overlap contains {len(payload_overlap)} records"
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.recovery-corpus-audit.v1",
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "train": train,
        "evaluation": evaluation,
        "train_evaluation_overlap": 0,
        "train_evaluation_payload_overlap": 0,
    }
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
