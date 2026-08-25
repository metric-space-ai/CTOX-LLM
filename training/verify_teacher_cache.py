#!/usr/bin/env python3
"""Verify an immutable sparse BF16 teacher cache before recovery consumes it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_tensor_specs(
    tokens: int,
    logit_targets: int,
    hidden_targets: int,
    mtp_targets: int,
    mtp_hidden_targets: int,
    top_k: int,
    hidden_size: int,
    hidden_layers: list[int],
    require_mtp: bool,
) -> dict[str, tuple[str, list[int]]]:
    specs = {
        "input_ids": ("I32", [1, tokens]),
        "attention_mask": ("U8", [1, tokens]),
        "logit_positions": ("I32", [logit_targets]),
        "hidden_positions": ("I32", [hidden_targets]),
        "topk_indices": ("I32", [1, logit_targets, top_k]),
        "topk_logprobs": ("BF16", [1, logit_targets, top_k]),
        "residual_probability": ("F32", [1, logit_targets]),
    }
    specs.update(
        {
            f"hidden_{layer}": ("BF16", [1, hidden_targets, hidden_size])
            for layer in hidden_layers
        }
    )
    if require_mtp:
        specs.update(
            {
                "mtp_positions": ("I32", [mtp_targets]),
                "mtp_hidden_positions": ("I32", [mtp_hidden_targets]),
                "mtp_hidden": ("BF16", [1, mtp_hidden_targets, hidden_size]),
                "mtp_topk_indices": ("I32", [1, mtp_targets, top_k]),
                "mtp_topk_logprobs": ("BF16", [1, mtp_targets, top_k]),
                "mtp_residual_probability": ("F32", [1, mtp_targets]),
            }
        )
    return specs


def load_source_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ordered = []
    by_id = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["id"])
            if sample_id in by_id:
                raise ValueError(f"duplicate source sample {sample_id}")
            record["_source_line"] = line_number
            ordered.append(record)
            by_id[sample_id] = record
    return ordered, by_id


def verify(args: argparse.Namespace, safe_open: Any) -> dict[str, Any]:
    run_path = args.cache / "run.json"
    index_path = args.cache / "index.jsonl"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("teacher_revision") != args.teacher_revision:
        raise ValueError("teacher revision does not match the required revision")
    if run.get("local_model_provenance_sha256") != args.teacher_provenance_sha256:
        raise ValueError("teacher provenance does not match the required artifact")
    if run.get("target_mode") != "assistant":
        raise ValueError("release recovery requires assistant-only teacher targets")
    if bool(run.get("mtp_targets")) != args.require_mtp:
        raise ValueError("MTP target mode does not match the verifier contract")

    hidden_layers = [int(layer) for layer in run.get("hidden_layers", [])]
    top_k = int(run["top_k"])
    source_order, source_by_id = load_source_records(args.input)
    start_sample = int(run.get("start_sample", 0))
    selected_samples = int(run.get("selected_samples", 0))
    expected_source = source_order[start_sample : start_sample + selected_samples]
    expected_ids = [str(record["id"]) for record in expected_source]

    entries = []
    with index_path.open(encoding="utf-8") as index:
        for line in index:
            if line.strip():
                entries.append(json.loads(line))
    if len(entries) != selected_samples:
        raise ValueError(
            f"index has {len(entries)} samples, expected selected count {selected_samples}"
        )
    if run.get("written_samples") != selected_samples:
        raise ValueError("run manifest does not record a complete selected cache")
    if [str(entry["id"]) for entry in entries] != expected_ids:
        raise ValueError("cache index order or sample selection differs from source")

    artifact_records = []
    total_bytes = 0
    aggregate = hashlib.sha256()
    for entry in entries:
        sample_id = str(entry["id"])
        source_record = source_by_id[sample_id]
        filename = str(entry["file"])
        if Path(filename).name != filename or filename != f"{sample_id}.safetensors":
            raise ValueError(f"unsafe or non-canonical cache filename {filename}")
        path = args.cache / filename
        encoded = path.read_bytes()
        file_sha256 = hashlib.sha256(encoded).hexdigest()
        total_bytes += len(encoded)
        aggregate.update(sample_id.encode("ascii"))
        aggregate.update(bytes.fromhex(file_sha256))
        with safe_open(path, framework="pt", device="cpu") as artifact:
            metadata = artifact.metadata() or {}
            if metadata.get("sample_id") != sample_id:
                raise ValueError(f"{filename} sample metadata differs from index")
            if metadata.get("teacher_revision") != args.teacher_revision:
                raise ValueError(f"{filename} teacher revision differs from contract")
            if metadata.get("source_payload_sha256") != source_record["prompt_sha256"]:
                raise ValueError(f"{filename} source payload hash differs from input")
            fields = {
                "tokens": int(entry["tokens"]),
                "logit_targets": int(entry["logit_targets"]),
                "hidden_targets": int(entry["hidden_targets"]),
                "mtp_targets": int(entry.get("mtp_targets", 0)),
                "mtp_hidden_targets": int(entry.get("mtp_hidden_targets", 0)),
            }
            for index_name, metadata_name in (
                ("tokens", "sequence_tokens"),
                ("logit_targets", "logit_target_count"),
                ("hidden_targets", "hidden_target_count"),
                ("mtp_targets", "mtp_target_count"),
                ("mtp_hidden_targets", "mtp_hidden_target_count"),
            ):
                if int(metadata[metadata_name]) != fields[index_name]:
                    raise ValueError(f"{filename} {metadata_name} differs from index")
            specs = expected_tensor_specs(
                **fields,
                top_k=top_k,
                hidden_size=args.hidden_size,
                hidden_layers=hidden_layers,
                require_mtp=args.require_mtp,
            )
            if set(artifact.keys()) != set(specs):
                missing = sorted(set(specs) - set(artifact.keys()))
                extra = sorted(set(artifact.keys()) - set(specs))
                raise ValueError(f"{filename} tensor set differs: missing={missing}, extra={extra}")
            for name, (dtype, shape) in specs.items():
                tensor = artifact.get_slice(name)
                if tensor.get_dtype() != dtype or tensor.get_shape() != shape:
                    raise ValueError(
                        f"{filename}:{name} is {tensor.get_dtype()} {tensor.get_shape()}, "
                        f"expected {dtype} {shape}"
                    )
        artifact_records.append(
            {
                "id": sample_id,
                "file": filename,
                "bytes": len(encoded),
                "sha256": file_sha256,
            }
        )

    return {
        "format": "ctox.teacher-cache-verification.v1",
        "status": "passed",
        "cache": str(args.cache.resolve()),
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "run_sha256": sha256(run_path),
        "index_sha256": sha256(index_path),
        "teacher_revision": args.teacher_revision,
        "teacher_provenance_sha256": args.teacher_provenance_sha256,
        "hidden_size": args.hidden_size,
        "hidden_layers": hidden_layers,
        "top_k": top_k,
        "mtp_targets": args.require_mtp,
        "samples": len(entries),
        "artifact_bytes": total_bytes,
        "artifact_root_sha256": aggregate.hexdigest(),
        "artifacts": artifact_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher-revision", required=True)
    parser.add_argument("--teacher-provenance-sha256", required=True)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.hidden_size <= 0:
        raise SystemExit("--hidden-size must be positive")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before verifying teacher caches") from error
    try:
        document = verify(args, safe_open)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
