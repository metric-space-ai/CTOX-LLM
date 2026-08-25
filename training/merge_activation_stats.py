#!/usr/bin/env python3
"""Merge disjoint activation-statistics batches with exact count weighting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge(paths: list[Path], output: Path, torch: Any, safe_open: Any, save_file: Any) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    accumulated: dict[str, dict[str, Any]] = {}
    row_counts: dict[str, Any] = {}
    sample_ids: list[str] = []
    seen_ids: set[str] = set()
    reference_keys: set[str] | None = None
    reference_metadata: dict[str, str] | None = None
    total_tokens = 0
    input_hashes = []
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as source:
            metadata = source.metadata()
            if metadata.get("format") != "ctox.activation-diagonal.v1":
                raise ValueError(f"{path} is not an activation-statistics artifact")
            keys = set(source.keys())
            if reference_keys is not None and keys != reference_keys:
                raise ValueError(f"{path} has a different tensor set")
            if reference_metadata is not None:
                for field in (
                    "model",
                    "revision",
                    "target_tensors",
                    "unobserved_tensors",
                    "input_only_tensors",
                    "row_frequency_tensors",
                ):
                    if metadata.get(field) != reference_metadata.get(field):
                        raise ValueError(f"{path} metadata field {field} differs")
            reference_keys = keys
            reference_metadata = reference_metadata or metadata
            batch_ids = json.loads(metadata["sample_ids"])
            duplicates = seen_ids.intersection(batch_ids)
            if duplicates:
                raise ValueError(f"{path} repeats {len(duplicates)} sample ids")
            seen_ids.update(batch_ids)
            sample_ids.extend(batch_ids)
            total_tokens += int(metadata["tokens"])
            input_hashes.append(sha256(path))
            bases = [key.removesuffix(".token_count") for key in keys if key.endswith(".token_count")]
            for base in bases:
                count = int(source.get_tensor(f"{base}.token_count")[0])
                input_sum = source.get_tensor(f"{base}.input_mean_sq").double() * count
                output_key = f"{base}.output_mean_sq"
                output_sum = (
                    source.get_tensor(output_key).double() * count if output_key in keys else None
                )
                state = accumulated.setdefault(
                    base,
                    {
                        "input_sum": torch.zeros_like(input_sum),
                        "output_sum": torch.zeros_like(output_sum) if output_sum is not None else None,
                        "count": 0,
                    },
                )
                if (state["output_sum"] is None) != (output_sum is None):
                    raise ValueError(f"{path} changes output-statistics availability for {base}")
                state["input_sum"] += input_sum
                if output_sum is not None:
                    state["output_sum"] += output_sum
                state["count"] += count
            for key in (key for key in keys if key.endswith(".row_count")):
                base = key.removesuffix(".row_count")
                values = source.get_tensor(key).long()
                if base not in row_counts:
                    row_counts[base] = torch.zeros_like(values)
                row_counts[base] += values
    tensors = {}
    for base, state in accumulated.items():
        tensors[f"{base}.input_mean_sq"] = (state["input_sum"] / state["count"]).float()
        if state["output_sum"] is not None:
            tensors[f"{base}.output_mean_sq"] = (
                state["output_sum"] / state["count"]
            ).float()
        tensors[f"{base}.token_count"] = torch.tensor([state["count"]], dtype=torch.int64)
    for base, values in row_counts.items():
        tensors[f"{base}.row_count"] = values
    assert reference_metadata is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            **reference_metadata,
            "format": "ctox.activation-diagonal.v1",
            "sample_ids": json.dumps(sample_ids, separators=(",", ":")),
            "samples": str(len(sample_ids)),
            "tokens": str(total_tokens),
            "merged_batches": str(len(paths)),
            "input_sha256": json.dumps(input_hashes, separators=(",", ":")),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise SystemExit("install training/requirements.in before merging activations") from error
    merge(args.input, args.output, torch, safe_open, save_file)
    print(json.dumps({"output": str(args.output), "batches": len(args.input)}, sort_keys=True))


if __name__ == "__main__":
    main()
