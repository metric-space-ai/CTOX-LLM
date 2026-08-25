#!/usr/bin/env python3
"""Build a padded Q2/Q4 plan for the separately resident vision encoder."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


BLOCK = 64
ALIGNMENT = 256
BLOCK_BYTES = {"q2_b64": 18, "q4_b64": 34}


def align(value: int, alignment: int = ALIGNMENT) -> int:
    if value < 0:
        raise ValueError("value must not be negative")
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quant", choices=sorted(BLOCK_BYTES), default="q4_b64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before planning vision") from error
    weight_map = json.loads(
        (args.checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    names = sorted(name for name in weight_map if name.startswith("model.visual"))
    if not names:
        raise SystemExit("checkpoint contains no model.visual tensors")
    shard_names: dict[str, list[str]] = {}
    for name in names:
        shard_names.setdefault(weight_map[name], []).append(name)
    shapes = {}
    for shard, shard_entries in shard_names.items():
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as source:
            for name in shard_entries:
                shapes[name] = list(source.get_slice(name).get_shape())

    position = 0
    tensors = []
    dtype_bytes: Counter[str] = Counter()
    padding_values = 0
    for name in names:
        shape = shapes[name]
        elements = math.prod(shape)
        position = align(position)
        if len(shape) == 2 and elements >= 4096:
            rows, columns = shape
            if rows <= 0 or columns <= 0:
                raise SystemExit(f"invalid matrix shape for {name}: {shape}")
            storage_columns = align(columns, BLOCK)
            length = rows * (storage_columns // BLOCK) * BLOCK_BYTES[args.quant]
            entry = {
                "name": name,
                "source_shard": weight_map[name],
                "dtype": args.quant,
                "logical_shape": shape,
                "storage_shape": [rows, storage_columns],
                "offset": position,
                "length": length,
            }
            padding_values += rows * (storage_columns - columns)
            position += length
            dtype_bytes[args.quant] += length
            tensors.append(entry)
            for suffix, channels in (("s_in", columns), ("s_out", rows)):
                position = align(position)
                length = channels * 2
                tensors.append(
                    {
                        "name": f"{name}.{suffix}",
                        "source_shard": None,
                        "dtype": "f16",
                        "logical_shape": [channels],
                        "storage_shape": [channels],
                        "offset": position,
                        "length": length,
                    }
                )
                position += length
                dtype_bytes["recovery_f16"] += length
        else:
            length = elements * 2
            tensors.append(
                {
                    "name": name,
                    "source_shard": weight_map[name],
                    "dtype": "f16",
                    "logical_shape": shape,
                    "storage_shape": shape,
                    "offset": position,
                    "length": length,
                }
            )
            position += length
            dtype_bytes["f16"] += length
    total_bytes = align(position)
    plan = {
        "format": "ctox.q2q4.vision-plan.v1",
        "model": "Qwen/Qwen3.8-27B",
        "revision": args.revision,
        "alignment": ALIGNMENT,
        "block": BLOCK,
        "zero_padded_columns": True,
        "padding_values": padding_values,
        "total_bytes": total_bytes,
        "mib": total_bytes / 1024**2,
        "dtype_bytes": dict(dtype_bytes),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: plan[key] for key in ("total_bytes", "mib", "dtype_bytes")}, indent=2))


if __name__ == "__main__":
    main()
