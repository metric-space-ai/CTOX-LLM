#!/usr/bin/env python3
"""Classify cross-domain LLM service modes for a frozen recovery cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from classify_domains import (
    MODEL,
    MODEL_REVISION,
    classification_text,
    read_records,
    write_json_atomic,
    write_jsonl_atomic,
)
from prompt_format import normalize_messages


def final_answer(record: dict[str, Any]) -> str:
    messages = normalize_messages(record.get("messages", []))
    if messages and messages[-1]["role"] == "assistant":
        return str(messages[-1].get("content") or "")
    return ""


def deterministic_modes(record: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    category = str(record.get("category", "")).lower()
    split = str(record.get("split", "")).lower()
    messages = normalize_messages(record.get("messages", []))
    answer = final_answer(record).lstrip()
    if category == "agentic" or record.get("tools"):
        labels.add("tool_calling_agentic")
    if category == "code" or split == "code":
        labels.add("coding_debugging")
    if category == "math" or split == "math":
        labels.add("mathematical_reasoning")
    if category == "long_context":
        labels.add("long_context_retrieval")
    if len(messages) > 2:
        labels.add("multi_turn_dialogue")
    if answer.startswith(("{", "[", "<", "```")) or any(
        message.get("tool_calls") for message in messages
    ):
        labels.add("structured_output_constraints")
    return labels


def validate_rubric(
    rubric: dict[str, Any],
    domain_rubric: dict[str, Any] | None = None,
    language_rubric: dict[str, Any] | None = None,
) -> None:
    if rubric.get("format") != "ctox.recovery-service-mode-rubric.v1":
        raise ValueError("unsupported service-mode rubric format")
    policy = rubric["policy"]
    modes = rubric["modes"]
    if not modes:
        raise ValueError("service-mode rubric is empty")
    confidence = float(policy["minimum_confidence"])
    if not 0.0 < confidence <= 1.0:
        raise ValueError("minimum_confidence must be in (0, 1]")
    for dimension in ("domain", "family"):
        for partition in ("train", "evaluation"):
            key = f"minimum_distinct_modes_per_{dimension}_{partition}"
            minimum = int(policy[key])
            if minimum <= 0 or minimum > len(modes):
                raise ValueError(f"invalid {key}")
    for name, mode in modes.items():
        if not str(mode.get("description", "")).strip():
            raise ValueError(f"service mode {name} has no classifier description")
        if any(
            int(mode[f"minimum_{partition}"]) <= 0
            for partition in ("train", "evaluation")
        ):
            raise ValueError(f"service mode {name} has a non-positive quota")
    for language, minima in rubric["language_minimum_distinct_modes"].items():
        if any(
            int(minima[partition]) <= 0 or int(minima[partition]) > len(modes)
            for partition in ("train", "evaluation")
        ):
            raise ValueError(f"language {language} has an invalid mode-diversity quota")
    if domain_rubric is not None:
        domains = set(domain_rubric["domains"])
        for domain, pairs in rubric["required_domain_mode_pairs"].items():
            if domain not in domains:
                raise ValueError(f"service rubric names unknown domain {domain}")
            for mode, minima in pairs.items():
                if mode not in modes:
                    raise ValueError(f"service rubric names unknown mode {mode}")
                if any(
                    int(minima[partition]) <= 0
                    for partition in ("train", "evaluation")
                ):
                    raise ValueError(
                        f"domain/mode pair {domain}/{mode} has a non-positive quota"
                    )
    if language_rubric is not None and set(
        rubric["language_minimum_distinct_modes"]
    ) != set(language_rubric["languages"]):
        raise ValueError("service and language rubrics declare different languages")


def classify(
    records: list[dict[str, Any]],
    rubric: dict[str, Any],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
    batch_records: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    validate_rubric(rubric)
    modes = rubric["modes"]
    mode_names = sorted(modes)
    hypotheses = [
        "This request requires the following kind of language-model work: "
        f"{modes[name]['description']}."
        for name in mode_names
    ]
    threshold = float(rubric["policy"]["minimum_confidence"])
    entailment_id = model.config.label2id.get("entailment")
    contradiction_id = model.config.label2id.get("contradiction")
    if entailment_id is None or contradiction_id is None:
        raise ValueError("classifier config lacks entailment/contradiction labels")
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    fallback_count = 0
    for start in range(0, len(records), batch_records):
        batch = records[start : start + batch_records]
        premises: list[str] = []
        paired_hypotheses: list[str] = []
        for record in batch:
            premises.extend([classification_text(record)] * len(mode_names))
            paired_hypotheses.extend(hypotheses)
        encoded = tokenizer(
            premises,
            paired_hypotheses,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = logits[:, [contradiction_id, entailment_id]].softmax(dim=-1)[:, 1]
        probabilities = probabilities.reshape(len(batch), len(mode_names)).cpu().tolist()
        for record, scores in zip(batch, probabilities, strict=True):
            score_map = {
                name: round(float(score), 8)
                for name, score in zip(mode_names, scores, strict=True)
            }
            labels = {name for name, score in score_map.items() if score >= threshold}
            source_labels = deterministic_modes(record)
            labels.update(source_labels)
            used_fallback = False
            if not labels:
                labels.add(max(score_map, key=score_map.get))
                used_fallback = True
                fallback_count += 1
            counts.update(labels)
            output.append(
                {
                    "id": record["id"],
                    "labels": sorted(labels),
                    "deterministic_labels": sorted(source_labels),
                    "scores": score_map,
                    "below_threshold_fallback": used_fallback,
                }
            )
        del encoded, logits, probabilities
    return output, counts, fallback_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-records", type=int, default=8)
    args = parser.parse_args()
    if args.batch_records <= 0:
        raise SystemExit("--batch-records must be positive")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before classification") from error
    try:
        rubric_bytes = args.rubric.read_bytes()
        rubric = json.loads(rubric_bytes)
        validate_rubric(rubric)
        records = read_records(args.input)
        device = torch.device(args.device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device)
        model.eval()
        tags, counts, fallback_count = classify(
            records,
            rubric,
            tokenizer,
            model,
            torch,
            device,
            args.batch_records,
        )
        output_sha256 = write_jsonl_atomic(args.output, tags)
        write_json_atomic(
            args.evidence,
            {
                "format": "ctox.recovery-service-mode-classification.v1",
                "input": str(args.input.resolve()),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "rubric_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
                "classifier": args.model,
                "classifier_revision": args.revision,
                "records": len(records),
                "mode_counts": dict(sorted(counts.items())),
                "below_threshold_fallbacks": fallback_count,
                "output": str(args.output.resolve()),
                "output_sha256": output_sha256,
            },
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
