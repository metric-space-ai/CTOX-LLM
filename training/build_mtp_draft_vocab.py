#!/usr/bin/env python3
"""Build a release-bound restricted MTP vocabulary from model output tokens.

The selected IDs accelerate draft LM-head evaluation only. They never limit
the target LM head: every proposal is verified against full-vocabulary target
logits, so a missing ID becomes a normal speculative rejection rather than a
semantic model change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from prompt_format import normalize_messages, render_record


ONE_MILLION = 1_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(record.get("id", "")) for record in records]
    if not records or any(not sample_id for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path} must contain unique non-empty sample ids")
    return records


def domain_labels(path: Path, expected_ids: set[str]) -> dict[str, str]:
    tags = read_jsonl(path)
    labels = {}
    for tag in tags:
        label = str(tag.get("primary_label", ""))
        if not label:
            raise ValueError(f"domain tag {tag['id']} has no primary label")
        labels[str(tag["id"])] = label
    if set(labels) != expected_ids:
        raise ValueError("domain tags and materialized recovery records differ")
    return labels


def validate_teacher_cache_set(path: Path, input_sha256: str, samples: int) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_input = document.get("expected_input") or {}
    settings = document.get("settings") or {}
    if (
        document.get("format") != "ctox.teacher-cache-set.v1"
        or int(document.get("samples", -1)) != samples
        or expected_input.get("sha256") != input_sha256
        or int(expected_input.get("records", -1)) != samples
        or document.get("all_artifacts_rehashed") is not True
        or settings.get("mtp_targets") is not True
    ):
        raise ValueError("teacher cache set does not bind the exact rehashed MTP recovery input")
    return document


def assistant_output_ids(tokenizer: Any, record: dict[str, Any]) -> list[int]:
    messages = normalize_messages(record.get("messages", []))
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"record {record.get('id')} has no final assistant message")
    template_args: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if record.get("tools"):
        template_args["tools"] = record["tools"]
    prefix = tokenizer.apply_chat_template(messages[:-1], **template_args)
    rendered = render_record(tokenizer, record)
    if not rendered.startswith(prefix):
        raise ValueError(f"record {record['id']} assistant prefix differs from rendered prompt")
    full_ids = list(tokenizer(rendered, add_special_tokens=False).input_ids)
    prefix_ids = list(tokenizer(prefix, add_special_tokens=False).input_ids)
    if full_ids[: len(prefix_ids)] != prefix_ids or len(full_ids) <= len(prefix_ids):
        raise ValueError(f"record {record['id']} has no exact assistant token suffix")
    return [int(token) for token in full_ids[len(prefix_ids) :]]


def normalized_token_scores(
    overall: Counter[int],
    code: Counter[int],
    domains: dict[str, Counter[int]],
    languages: dict[str, Counter[int]],
) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)

    def add(counter: Counter[int], weight: float) -> None:
        total = sum(counter.values())
        if total:
            for token, count in counter.items():
                scores[token] += weight * count / total

    add(overall, 1.0)
    add(code, 1.0)
    for counter in domains.values():
        add(counter, 1.0 / len(domains))
    for counter in languages.values():
        add(counter, 1.0 / len(languages))
    return scores


def select_tokens(
    overall: Counter[int],
    code: Counter[int],
    domains: dict[str, Counter[int]],
    languages: dict[str, Counter[int]],
    token_count: int,
    required_ids: Iterable[int],
) -> list[int]:
    if token_count <= 0:
        raise ValueError("token count must be positive")
    required = set(required_ids)
    if any(token < 0 for token in required) or len(required) > token_count:
        raise ValueError("required token IDs do not fit the draft vocabulary")
    scores = normalized_token_scores(overall, code, domains, languages)
    candidates = sorted(scores, key=lambda token: (-scores[token], token))
    selected = set(required)
    for token in candidates:
        if len(selected) == token_count:
            break
        selected.add(token)
    if len(selected) != token_count:
        raise ValueError(
            f"only {len(selected)} distinct output/required tokens exist for {token_count} slots"
        )
    return sorted(selected)


def coverage_ppm(counter: Counter[int], selected: set[int]) -> int:
    total = sum(counter.values())
    if total <= 0:
        raise ValueError("coverage group contains no output tokens")
    covered = sum(count for token, count in counter.items() if token in selected)
    return round(covered * ONE_MILLION / total)


def write_atomic(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--domain-tags", type=Path, required=True)
    parser.add_argument("--teacher-cache-set", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--token-count", type=int, default=40_000)
    parser.add_argument("--required-token-ids", default="")
    parser.add_argument("--output-ids", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    parser.add_argument("--minimum-overall-coverage-ppm", type=int, default=950_000)
    parser.add_argument("--minimum-code-coverage-ppm", type=int, default=900_000)
    parser.add_argument("--minimum-domain-coverage-ppm", type=int, default=800_000)
    parser.add_argument("--minimum-language-coverage-ppm", type=int, default=800_000)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before building draft vocab") from error

    records = read_jsonl(args.input)
    input_sha256 = sha256_file(args.input)
    validate_teacher_cache_set(args.teacher_cache_set, input_sha256, len(records))
    labels = domain_labels(args.domain_tags, {str(record["id"]) for record in records})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    overall: Counter[int] = Counter()
    code: Counter[int] = Counter()
    domains: dict[str, Counter[int]] = defaultdict(Counter)
    languages: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        sample_id = str(record["id"])
        language = str(record.get("language", ""))
        category = str(record.get("category", ""))
        if not language or not category:
            raise ValueError(f"record {sample_id} lacks language or category")
        tokens = assistant_output_ids(tokenizer, record)
        if any(token < 0 or token >= len(tokenizer) for token in tokens):
            raise ValueError(f"record {sample_id} contains an out-of-range token")
        overall.update(tokens)
        domains[labels[sample_id]].update(tokens)
        languages[language].update(tokens)
        if category == "code":
            code.update(tokens)
    if not code:
        raise ValueError("recovery data contains no coding output tokens")

    requested_required = {
        int(value) for value in args.required_token_ids.split(",") if value.strip()
    }
    required_ids = requested_required | {int(token) for token in tokenizer.all_special_ids}
    selected_ids = select_tokens(
        overall, code, dict(domains), dict(languages), args.token_count, required_ids
    )
    selected = set(selected_ids)
    domain_coverage = {
        name: coverage_ppm(counter, selected) for name, counter in sorted(domains.items())
    }
    language_coverage = {
        name: coverage_ppm(counter, selected) for name, counter in sorted(languages.items())
    }
    metrics = {
        "overall": coverage_ppm(overall, selected),
        "code": coverage_ppm(code, selected),
        "minimum_domain": min(domain_coverage.values()),
        "minimum_language": min(language_coverage.values()),
    }
    thresholds = {
        "overall": args.minimum_overall_coverage_ppm,
        "code": args.minimum_code_coverage_ppm,
        "minimum_domain": args.minimum_domain_coverage_ppm,
        "minimum_language": args.minimum_language_coverage_ppm,
    }
    failures = {
        name: {"observed": metrics[name], "required": threshold}
        for name, threshold in thresholds.items()
        if metrics[name] < threshold
    }
    if failures:
        raise ValueError(f"restricted MTP vocabulary misses coverage gates: {failures}")

    encoded_ids = b"".join(struct.pack("<I", token) for token in selected_ids)
    token_ids_sha256 = hashlib.sha256(encoded_ids).hexdigest()
    evidence = {
        "format": "ctox.mtp-draft-vocabulary.v1",
        "status": "passed",
        "semantic_contract": "restricted_draft_full_target_verification",
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.revision,
        "source": {
            "recovery_jsonl_sha256": input_sha256,
            "domain_tags_sha256": sha256_file(args.domain_tags),
            "teacher_cache_set_sha256": sha256_file(args.teacher_cache_set),
        },
        "selection": {
            "token_count": len(selected_ids),
            "token_ids_encoding": "strictly_increasing_u32le",
            "token_ids_bytes": len(encoded_ids),
            "token_ids_sha256": token_ids_sha256,
            "required_token_ids": sorted(required_ids),
            "score_weights": {
                "overall": 1.0,
                "code": 1.0,
                "domains_total": 1.0,
                "languages_total": 1.0,
            },
        },
        "observed_output_tokens": sum(overall.values()),
        "coverage_ppm": metrics,
        "domain_coverage_ppm": domain_coverage,
        "language_coverage_ppm": language_coverage,
        "thresholds_ppm": thresholds,
    }
    evidence_bytes = (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode()
    write_atomic(args.output_ids, encoded_ids)
    try:
        write_atomic(args.output_evidence, evidence_bytes)
    except Exception:
        args.output_ids.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
