#!/usr/bin/env python3
"""Apply the frozen multilingual domain rubric to a materialized cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from prompt_format import normalize_messages


MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MODEL_REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"


def classification_text(record: dict[str, Any], character_limit: int = 12000) -> str:
    messages = normalize_messages(record.get("messages", []))
    if messages:
        selected = []
        for message in messages:
            content = message.get("content") or ""
            if message["role"] in {"user", "assistant"} and content:
                selected.append(f"{message['role']}: {content}")
        text = "\n".join(selected)
    else:
        text = str(record.get("prompt", ""))
    if len(text) <= character_limit:
        return text
    half = character_limit // 2
    return text[:half] + "\n[...middle omitted...]\n" + text[-half:]


def deterministic_labels(record: dict[str, Any], final_answer: str) -> set[str]:
    labels: set[str] = set()
    category = record.get("category")
    if category == "agentic" or record.get("tools"):
        labels.add("agentic_tools_search")
    if category == "code":
        labels.add("software_cybersecurity")
    if category == "math":
        labels.add("mathematics_logic")
    if category == "long_context":
        labels.add("data_structured_outputs")
    if final_answer.lstrip().startswith(("{", "[", "<", "```")):
        labels.add("data_structured_outputs")
    return labels


def read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_number} duplicates {record['id']}")
            seen.add(record["id"])
            records.append(record)
    return records


def classify(
    records: list[dict[str, Any]],
    rubric: dict[str, Any],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: Any,
    batch_records: int,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    domains = rubric["domains"]
    domain_names = sorted(domains)
    hypotheses = [
        f"This request belongs to the domain: {domains[name]['description']}."
        for name in domain_names
    ]
    threshold = float(rubric["policy"]["minimum_confidence"])
    entailment_id = model.config.label2id.get("entailment")
    contradiction_id = model.config.label2id.get("contradiction")
    if entailment_id is None or contradiction_id is None:
        raise ValueError("classifier config lacks entailment/contradiction labels")
    output = []
    counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    fallback_count = 0
    for start in range(0, len(records), batch_records):
        batch = records[start : start + batch_records]
        premises = []
        paired_hypotheses = []
        texts = [classification_text(record) for record in batch]
        for text in texts:
            premises.extend([text] * len(domain_names))
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
        probabilities = probabilities.reshape(len(batch), len(domain_names)).cpu().tolist()
        for record, scores in zip(batch, probabilities, strict=True):
            messages = normalize_messages(record.get("messages", []))
            final_answer = (
                messages[-1].get("content") or ""
                if messages and messages[-1]["role"] == "assistant"
                else ""
            )
            score_map = {
                name: round(float(score), 8)
                for name, score in zip(domain_names, scores, strict=True)
            }
            primary_label = max(score_map, key=score_map.get)
            primary_counts[primary_label] += 1
            labels = {name for name, score in score_map.items() if score >= threshold}
            labels.update(deterministic_labels(record, final_answer))
            used_fallback = False
            if not labels:
                labels.add(primary_label)
                used_fallback = True
                fallback_count += 1
            counts.update(labels)
            output.append(
                {
                    "id": record["id"],
                    "labels": sorted(labels),
                    "primary_label": primary_label,
                    "scores": score_map,
                    "below_threshold_fallback": used_fallback,
                }
            )
        del encoded, logits, probabilities
    return output, counts, primary_counts, fallback_count


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in records:
                payload = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                ).encode("utf-8")
                temporary.write(payload.decode("utf-8"))
                digest.update(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return digest.hexdigest()


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(document, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.rename(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def quota_gaps(
    counts: Counter[str],
    primary_counts: Counter[str],
    rubric: dict[str, Any],
    partition: str,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    minimum_key = f"minimum_{partition}"
    primary_minimum = int(rubric["policy"][f"minimum_primary_{partition}"])
    multi_label = {
        name: {"observed": counts[name], "required": int(domain[minimum_key])}
        for name, domain in rubric["domains"].items()
        if counts[name] < int(domain[minimum_key])
    }
    primary = {
        name: {"observed": primary_counts[name], "required": primary_minimum}
        for name in rubric["domains"]
        if primary_counts[name] < primary_minimum
    }
    return multi_label, primary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-records", type=int, default=4)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    args = parser.parse_args()
    if args.batch_records <= 0:
        raise SystemExit("--batch-records must be positive")
    if args.summary.exists():
        raise SystemExit(f"refusing to overwrite {args.summary}")
    rubric = json.loads(args.rubric.read_text(encoding="utf-8"))
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit("install training/requirements.in before classification") from error
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    try:
        records = read_records(args.input)
        tagged, counts, primary_counts, fallback_count = classify(
            records,
            rubric,
            tokenizer,
            model,
            torch,
            device,
            args.batch_records,
        )
        output_sha256 = write_jsonl_atomic(args.output, tagged)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    gaps, primary_gaps = quota_gaps(
        counts, primary_counts, rubric, args.partition
    )
    summary = {
        "format": "ctox.recovery-domain-audit.v1",
        "partition": args.partition,
        "input": str(args.input),
        "records": len(records),
        "classifier": args.model,
        "classifier_revision": args.revision,
        "classifier_license": "mit",
        "rubric_sha256": hashlib.sha256(args.rubric.read_bytes()).hexdigest(),
        "threshold": rubric["policy"]["minimum_confidence"],
        "domain_counts": dict(sorted(counts.items())),
        "primary_domain_counts": dict(sorted(primary_counts.items())),
        "below_threshold_fallback_records": fallback_count,
        "quota_gaps": gaps,
        "primary_quota_gaps": primary_gaps,
        "output": str(args.output),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": output_sha256,
    }
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
