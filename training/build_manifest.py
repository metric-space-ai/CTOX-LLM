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
from urllib.parse import quote
from urllib.request import Request, urlopen


SOURCES = {
    "nemotron-v1": {
        "repo": "nvidia/Nemotron-Post-Training-Dataset-v1",
        "reviewed_revision": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
        "default_splits": ("chat", "code", "math", "stem", "tool_calling"),
        "default_language": "en",
        "allowed_licenses": ("cc-by-4.0",),
        "release_eligible": True,
        "quarantine_reason": None,
    },
    "nemotron-v2": {
        "repo": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "reviewed_revision": "5c89e01dd720ae0f4058445ed49c5fb68a03c76e",
        "default_splits": (
            "chat",
            "code",
            "math",
            "stem",
            "multilingual_de",
        ),
        "default_language": "und",
        "allowed_licenses": ("cc-by-4.0",),
        "release_eligible": False,
        "quarantine_reason": "gated access and derivative-use review are not yet documented",
    },
    "nemotron-agentic-v1": {
        "repo": "nvidia/Nemotron-Agentic-v1",
        "reviewed_revision": "650d590978ca35c8f1ecea2faf136e5fac421b62",
        "default_splits": ("interactive_agent", "tool_calling"),
        "default_language": "en",
        "allowed_licenses": ("cc-by-4.0",),
        "release_eligible": True,
        "quarantine_reason": None,
        "raw_jsonl": True,
    },
    "nemotron-sft-agentic-v2": {
        "repo": "nvidia/Nemotron-SFT-Agentic-v2",
        "reviewed_revision": "7c804833427f633ccd53b582dbf02525fd680f78",
        "default_splits": ("interactive_agent", "search", "tool_calling"),
        "default_language": "en",
        "allowed_licenses": ("apache-2.0", "cc-by-4.0", "mit"),
        "release_eligible": True,
        "quarantine_reason": None,
        "raw_jsonl": True,
    },
    "german-instruct": {
        "repo": "Beko2210/German-Instruct-Dataset",
        "reviewed_revision": "4456bdf1b82f906a70fb9e5431530d2e9d1c565b",
        "default_splits": ("train",),
        "default_language": "de",
        "allowed_licenses": ("cc-by-4.0",),
        "release_eligible": True,
        "quarantine_reason": None,
    },
    "aya": {
        "repo": "CohereLabs/aya_dataset",
        "reviewed_revision": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
        "default_subsets": ("default",),
        "default_splits": ("train",),
        "default_language": "und",
        "allowed_licenses": ("apache-2.0",),
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
    "search": "agentic",
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
    dataset_card_licenses: list[str]
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
            payload = {"messages": value}
            tools = row.get("tools")
            if isinstance(tools, list) and tools:
                payload["tools"] = tools
            return payload

    prompt = next(
        (
            row[key]
            for key in ("prompt", "input", "inputs", "question", "instruction")
            if row.get(key)
        ),
        None,
    )
    answer = next(
        (
            row[key]
            for key in ("response", "output", "targets", "answer", "completion")
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
    # Generated/materialized records retain both their immutable sample `id`
    # and the source coordinate used to construct that identity. Prefer the
    # explicit coordinate so re-materialization looks up the same manifest key.
    for candidate in (row.get("source_id"), row.get("id"), row.get("uuid")):
        if candidate is not None and str(candidate):
            return str(candidate)
    metadata = nested_metadata(row)
    for key in ("uuid", "id", "alt_id"):
        candidate = metadata.get(key)
        if candidate is not None and str(candidate):
            return str(candidate)
    return str(index)


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


def language_for(row: dict[str, Any], default: str) -> str:
    """Return a stable language identifier, preferring ISO-like source codes."""

    for key in ("language_code", "language", "lang"):
        value = row.get(key)
        if value is not None and str(value).strip():
            normalized = str(value).strip().lower()
            if normalized not in {"und", "unknown", "none", "n/a"}:
                return normalized
    return default


def nested_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            decoded = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def license_ids(value: Any) -> list[str]:
    """Normalize dataset-card and per-record license identifiers.

    Empty placeholders deliberately do not become licenses; callers must fall
    back to the pinned dataset card or fail closed.
    """

    values = value if isinstance(value, list) else [value]
    normalized = []
    aliases = {
        "cc by 4.0": "cc-by-4.0",
        "cc-by-4": "cc-by-4.0",
        "apache 2.0": "apache-2.0",
        "apache-2": "apache-2.0",
    }
    for item in values:
        if item is None:
            continue
        license_id = str(item).strip().lower()
        if license_id in {"", "dataset-card", "unknown", "none", "n/a"}:
            continue
        license_id = aliases.get(license_id, license_id)
        if license_id not in normalized:
            normalized.append(license_id)
    return sorted(normalized)


def card_license_ids(info: Any) -> list[str]:
    card = getattr(info, "card_data", None)
    if card is None:
        return []
    if isinstance(card, dict):
        return license_ids(card.get("license"))
    return license_ids(getattr(card, "license", None))


def raw_jsonl_rows(repo: str, revision: str, split: str) -> Iterable[dict[str, Any]]:
    """Stream a pinned raw split without Arrow schema coercion.

    NVIDIA's Agentic tool schemas are intentionally heterogeneous. The
    high-level datasets reader attempts to cast those JSON objects into one
    Arrow struct and can fail before yielding otherwise valid records.
    """

    url = (
        "https://huggingface.co/datasets/"
        f"{quote(repo, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"data/{quote(split, safe='')}.jsonl"
    )
    request = Request(url, headers={"User-Agent": "ctox-llm-recovery-manifest/1"})
    with urlopen(request, timeout=120) as response:
        for line_number, line in enumerate(response, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid raw JSONL at {repo}@{revision}/{split}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"raw JSONL row is not an object at "
                    f"{repo}@{revision}/{split}:{line_number}"
                )
            yield row


def source_uses_raw_jsonl(repo: str) -> bool:
    matches = [source for source in SOURCES.values() if source["repo"] == repo]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one recovery source policy for {repo}")
    return bool(matches[0].get("raw_jsonl"))


def records(args: argparse.Namespace) -> Iterable[Record]:
    try:
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit("install training/requirements.in before building a manifest") from error

    source = SOURCES[args.source]
    repo = source["repo"]
    requested_revision = args.revision or source["reviewed_revision"]
    info = HfApi().dataset_info(repo, revision=requested_revision)
    revision = info.sha
    dataset_card_licenses = card_license_ids(info)
    allowed_licenses = set(source["allowed_licenses"])
    if not dataset_card_licenses:
        raise RuntimeError(f"{repo}@{revision} has no machine-readable dataset-card license")
    unexpected_card_licenses = set(dataset_card_licenses) - allowed_licenses
    if unexpected_card_licenses and source["release_eligible"]:
        raise RuntimeError(
            f"{repo}@{revision} contains unreviewed dataset-card licenses: "
            + ", ".join(sorted(unexpected_card_licenses))
        )
    raw_jsonl = bool(source.get("raw_jsonl"))
    if raw_jsonl and args.subset not in {None, "default"}:
        raise RuntimeError(f"{repo} raw JSONL source supports only the default subset")
    subsets = (
        ["default"]
        if raw_jsonl
        else [args.subset]
        if args.subset
        else list(
            source.get("default_subsets")
            or get_dataset_config_names(repo, revision=revision)
        )
    )
    requested_splits = tuple(args.split or source["default_splits"])
    emitted = 0
    language_emitted = {language: 0 for language in (args.language or [])}
    for subset in subsets:
        available_splits = (
            set(source["default_splits"])
            if raw_jsonl
            else set(get_dataset_split_names(repo, subset, revision=revision))
        )
        unavailable = set(requested_splits) - available_splits
        if unavailable:
            raise RuntimeError(
                f"{repo}/{subset}@{revision} has no splits: {', '.join(sorted(unavailable))}; "
                f"available: {', '.join(sorted(available_splits))}"
            )
        for split in requested_splits:
            dataset = (
                raw_jsonl_rows(repo, revision, split)
                if raw_jsonl
                else load_dataset(repo, subset, split=split, revision=revision, streaming=True)
            )
            for index, row in enumerate(dataset):
                text = canonical_text(row, repo)
                prompt_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                source_id = source_id_for(row, index)
                identity = "\0".join((repo, revision, subset, split, source_id, prompt_sha))
                metadata = nested_metadata(row)
                record_licenses = license_ids(
                    row.get("license", row.get("source_license", metadata.get("license")))
                )
                effective_licenses = record_licenses or dataset_card_licenses
                unexpected_licenses = set(effective_licenses) - allowed_licenses
                release_eligible = bool(source["release_eligible"] and not unexpected_licenses)
                quarantine_reason = source["quarantine_reason"]
                if unexpected_licenses:
                    quarantine_reason = (
                        "record contains unreviewed licenses: "
                        + ", ".join(sorted(unexpected_licenses))
                    )
                language = language_for(row, source["default_language"])
                if args.language and language not in args.language:
                    continue
                if (
                    args.per_language_limit
                    and language_emitted[language] >= args.per_language_limit
                ):
                    continue
                generator = (
                    row.get("generator")
                    or row.get("model_name")
                    or row.get("model")
                    or metadata.get("annotator")
                )
                yield Record(
                    id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                    source_repo=repo,
                    source_revision=revision,
                    subset=subset,
                    split=split,
                    source_id=source_id,
                    license=",".join(effective_licenses),
                    dataset_card_licenses=dataset_card_licenses,
                    generator=str(generator) if generator else None,
                    category=category_for(subset, split, row),
                    language=language,
                    prompt_sha256=prompt_sha,
                    release_eligible=release_eligible,
                    quarantine_reason=quarantine_reason,
                )
                emitted += 1
                if language in language_emitted:
                    language_emitted[language] += 1
                if args.limit and emitted >= args.limit:
                    return
                if args.per_language_limit and all(
                    count >= args.per_language_limit
                    for count in language_emitted.values()
                ):
                    return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--subset")
    parser.add_argument(
        "--split",
        action="append",
        help="source split to include; repeat for multiple splits (defaults are source-specific)",
    )
    parser.add_argument("--revision")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--language",
        action="append",
        help="emit only this normalized language code; repeat for multiple languages",
    )
    parser.add_argument(
        "--per-language-limit",
        type=int,
        help="emit this many records for every --language stratum",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_language_limit and not args.language:
        parser.error("--per-language-limit requires at least one --language")
    if args.per_language_limit is not None and args.per_language_limit <= 0:
        parser.error("--per-language-limit must be positive")
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
