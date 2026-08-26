#!/usr/bin/env python3
"""Verify one immutable activation-statistics batch against its bound plans."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from collect_activation_stats import QUANTIZED_DTYPES
from select_activation_calibration import load_jsonl, write_bytes_atomic


EMBEDDING = "model.language_model.embed_tokens.weight"
LM_HEAD = "lm_head.weight"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantized_entries(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry["name"]): entry
        for entry in plan["tensors"]
        if entry.get("source_shard") is not None
        and entry.get("dtype") in QUANTIZED_DTYPES
    }


def expected_keys(entries: dict[str, dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for name in entries:
        if name == EMBEDDING:
            keys.add(f"{name}.row_count")
        else:
            keys.add(f"{name}.input_mean_sq")
            keys.add(f"{name}.token_count")
            if name != LM_HEAD:
                keys.add(f"{name}.output_mean_sq")
    return keys


def expected_batch(
    batch_plan_path: Path,
    input_path: Path,
    batch_index: int,
) -> tuple[dict[str, Any], list[str]]:
    batch_plan = json.loads(batch_plan_path.read_text(encoding="utf-8"))
    if batch_plan.get("format") != "ctox.activation-batch-plan.v1":
        raise ValueError("unsupported activation batch-plan format")
    if sha256(input_path) != batch_plan.get("input_sha256"):
        raise ValueError("activation input hash does not match the batch plan")
    batches = batch_plan["batches"]
    if not 0 <= batch_index < len(batches):
        raise ValueError("activation batch index is outside the plan")
    batch = batches[batch_index]
    if int(batch["batch_index"]) != batch_index:
        raise ValueError("activation batch plan is not index ordered")
    records = load_jsonl(input_path)
    start = int(batch["start_sample"])
    stop = start + int(batch["samples"])
    selected = records[start:stop]
    ids = [str(record["id"]) for record in selected]
    if len(ids) != int(batch["samples"]):
        raise ValueError("activation batch slice is incomplete")
    if ids[0] != batch["first_id"] or ids[-1] != batch["last_id"]:
        raise ValueError("activation batch boundary ids differ from the input")
    return batch, ids


def verify_artifact(
    artifact: Path,
    quant_plan_path: Path,
    batch_plan_path: Path,
    input_path: Path,
    batch_index: int,
    model: str,
    revision: str,
    provenance_sha256: str,
    torch: Any,
    safe_open: Any,
) -> dict[str, Any]:
    batch, ids = expected_batch(batch_plan_path, input_path, batch_index)
    quant_plan_bytes = quant_plan_path.read_bytes()
    plan = json.loads(quant_plan_bytes)
    entries = quantized_entries(plan)
    required_keys = expected_keys(entries)
    with safe_open(artifact, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        checks = {
            "format": metadata.get("format") == "ctox.activation-diagonal.v1",
            "model": metadata.get("model") == model,
            "revision": metadata.get("revision") == revision,
            "local_model_provenance": metadata.get("local_model_provenance_sha256")
            == provenance_sha256,
            "quant_plan": metadata.get("quant_plan_sha256")
            == hashlib.sha256(quant_plan_bytes).hexdigest(),
            "sample_ids": json.loads(metadata.get("sample_ids", "[]")) == ids,
            "samples": int(metadata.get("samples", -1)) == int(batch["samples"]),
            "tokens": int(metadata.get("tokens", -1)) == int(batch["sequence_tokens"]),
            "start_sample": int(metadata.get("start_sample", -1))
            == int(batch["start_sample"]),
            "max_samples": int(metadata.get("max_samples", -1)) == int(batch["samples"]),
            "max_length": int(metadata.get("max_length", -1))
            == int(batch["maximum_sample_tokens"]),
            "observed_modules": int(metadata.get("observed_modules", -1)) == len(entries),
            "unobserved_tensors": json.loads(metadata.get("unobserved_tensors", "null"))
            == [],
            "tensor_keys": set(source.keys()) == required_keys,
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"activation artifact contract checks failed: {failed}")

        total_tokens = int(batch["sequence_tokens"])
        mtp_tokens = total_tokens - int(batch["samples"])
        for name, entry in entries.items():
            rows, columns = (int(value) for value in entry["shape"])
            if name == EMBEDDING:
                values = source.get_tensor(f"{name}.row_count")
                if tuple(values.shape) != (rows,) or int(values.sum()) != total_tokens:
                    raise ValueError(f"invalid embedding row counts for {name}")
                continue
            input_values = source.get_tensor(f"{name}.input_mean_sq")
            count = source.get_tensor(f"{name}.token_count")
            expected_count = mtp_tokens if name.startswith("mtp.") else total_tokens
            if tuple(input_values.shape) != (columns,) or tuple(count.shape) != (1,):
                raise ValueError(f"invalid input-statistics shape for {name}")
            if int(count[0]) != expected_count:
                raise ValueError(f"invalid token count for {name}")
            if not bool(torch.isfinite(input_values).all()) or bool((input_values < 0).any()):
                raise ValueError(f"invalid input-statistics values for {name}")
            if name == LM_HEAD:
                continue
            output_values = source.get_tensor(f"{name}.output_mean_sq")
            if tuple(output_values.shape) != (rows,):
                raise ValueError(f"invalid output-statistics shape for {name}")
            if not bool(torch.isfinite(output_values).all()) or bool((output_values < 0).any()):
                raise ValueError(f"invalid output-statistics values for {name}")

    return {
        "format": "ctox.activation-statistics-verification.v1",
        "status": "passed",
        "artifact": str(artifact.resolve()),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256(artifact),
        "quant_plan_sha256": sha256(quant_plan_path),
        "batch_plan_sha256": sha256(batch_plan_path),
        "input_sha256": sha256(input_path),
        "batch_index": batch_index,
        "samples": int(batch["samples"]),
        "sequence_tokens": int(batch["sequence_tokens"]),
        "first_id": ids[0],
        "last_id": ids[-1],
        "observed_modules": len(entries),
        "tensor_keys": len(required_keys),
        "model": model,
        "revision": revision,
        "local_model_provenance_sha256": provenance_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch-plan", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        import torch
        from safetensors import safe_open

        report = verify_artifact(
            args.artifact,
            args.plan,
            args.batch_plan,
            args.input,
            args.batch_index,
            args.model,
            args.revision,
            args.local_model_provenance_sha256,
            torch,
            safe_open,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    write_bytes_atomic(
        args.output,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
