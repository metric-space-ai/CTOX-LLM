#!/usr/bin/env python3
"""Build an exact text+MTP Q2/Q4 byte plan from BF16 safetensor metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


BLOCK = 64
Q2_BLOCK_BYTES = 18
Q4_BLOCK_BYTES = 34
ALIGNMENT = 256
FOLD_LIMIT = 8_375_186_227


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def group(name: str) -> str:
    if name.startswith("model.visual"):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.language_model") or name == "lm_head.weight":
        return "text"
    return "other"


def layer_number(name: str) -> int | None:
    marker = ".layers."
    if marker not in name:
        return None
    remainder = name.split(marker, 1)[1]
    component = remainder.split(".", 1)[0]
    return int(component) if component.isdigit() else None


def choose_dtype(
    name: str,
    shape: list[int],
    q_proj_q4_start: int,
    linear_out_q4_start: int,
    late_ffn_q4_start: int,
) -> str:
    elements = math.prod(shape)
    if name.endswith(".A_log") or name.endswith(".dt_bias"):
        return "f32"
    if len(shape) != 2 or elements < 4096 or shape[-1] % BLOCK:
        return "f16"
    layer = layer_number(name)
    if name == "lm_head.weight":
        return "q4_b64"
    if name.startswith("mtp.") and ".self_attn." in name:
        return "q4_b64"
    layer = layer_number(name)
    if ".self_attn." in name and any(
        name.endswith(suffix)
        for suffix in (".k_proj.weight", ".v_proj.weight", ".o_proj.weight")
    ):
        return "q4_b64"
    if name.endswith(".self_attn.q_proj.weight") and layer is not None and layer >= q_proj_q4_start:
        return "q4_b64"
    if (
        name.endswith(".linear_attn.out_proj.weight")
        and layer is not None
        and layer >= linear_out_q4_start
    ):
        return "q4_b64"
    if layer is not None and layer >= late_ffn_q4_start and ".mlp." in name:
        return "q4_b64"
    return "q2_b64"


def packed_bytes(dtype: str, elements: int) -> int:
    if dtype == "q2_b64":
        return math.ceil(elements / BLOCK) * Q2_BLOCK_BYTES
    if dtype == "q4_b64":
        return math.ceil(elements / BLOCK) * Q4_BLOCK_BYTES
    if dtype == "f16":
        return elements * 2
    if dtype == "f32":
        return elements * 4
    raise ValueError(dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--q-proj-q4-start", type=int, default=35)
    parser.add_argument("--linear-out-q4-start", type=int, default=64)
    parser.add_argument("--late-ffn-q4-start", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    index_path = args.checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before planning quantization") from error
    assigned_q4: set[str] | None = None
    mixed_assignment: dict[str, dict] = {}
    assignment_format: str | None = None
    if args.assignment:
        assignment = json.loads(args.assignment.read_text(encoding="utf-8"))
        assignment_format = assignment.get("format")
        if assignment_format not in {
            "ctox.q2q4.assignment.v1",
            "ctox.q2q4.assignment.v2",
        }:
            raise SystemExit("unsupported Q2/Q4 assignment")
        assigned_q4 = set(assignment["q4_tensors"])
        mixed_assignment = assignment.get("mixed_tensors", {})

    # Metadata lookup is shard-batched and never materializes tensor values.
    shard_tensors: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        shard_tensors.setdefault(shard, []).append(name)
    shapes: dict[str, list[int]] = {}
    for shard, names in shard_tensors.items():
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as source:
            for name in names:
                shapes[name] = list(source.get_slice(name).get_shape())

    tensors = []
    dtype_bytes: Counter[str] = Counter()
    logical_bytes = 0
    for name in sorted(weight_map):
        tensor_group = group(name)
        if tensor_group not in {"text", "mtp"}:
            continue
        shape = shapes[name]
        dtype = choose_dtype(
            name,
            shape,
            args.q_proj_q4_start,
            args.linear_out_q4_start,
            args.late_ffn_q4_start,
        )
        if assigned_q4 is not None and dtype in {"q2_b64", "q4_b64"}:
            dtype = "q4_b64" if name in assigned_q4 else "q2_b64"
        elements = math.prod(shape)
        segments = None
        if name in mixed_assignment:
            if len(shape) != 2 or shape[1] % BLOCK:
                raise SystemExit(f"mixed tensor {name} is not a block-aligned matrix")
            specification = mixed_assignment[name]
            group_size = int(specification["row_group_size"])
            if group_size <= 0:
                raise SystemExit(f"mixed tensor {name} has an invalid row group size")
            group_count = math.ceil(shape[0] / group_size)
            if int(specification["group_count"]) != group_count:
                raise SystemExit(f"mixed tensor {name} group count does not match its shape")
            q4_groups = set(specification["q4_groups"])
            if q4_groups - set(range(group_count)):
                raise SystemExit(f"mixed tensor {name} contains an invalid Q4 row group")
            segments = []
            segment_offset = 0
            for group_index in range(group_count):
                row_start = group_index * group_size
                row_end = min(row_start + group_size, shape[0])
                segment_dtype = "q4_b64" if group_index in q4_groups else "q2_b64"
                segment_elements = (row_end - row_start) * shape[1]
                segment_length = packed_bytes(segment_dtype, segment_elements)
                segments.append(
                    {
                        "group_index": group_index,
                        "row_start": row_start,
                        "row_end": row_end,
                        "dtype": segment_dtype,
                        "offset": segment_offset,
                        "length": segment_length,
                    }
                )
                segment_offset += segment_length
                dtype_bytes[segment_dtype] += segment_length
            dtype = "mixed_q2_q4_b64"
            size = segment_offset
        else:
            size = packed_bytes(dtype, elements)
            dtype_bytes[dtype] += size
        logical_bytes = align(logical_bytes)
        offset = logical_bytes
        logical_bytes += size
        tensor_entry = {
            "name": name,
            "source_shard": weight_map[name],
            "source_dtype": "bf16",
            "dtype": dtype,
            "shape": shape,
            "offset": offset,
            "length": size,
            "group": tensor_group,
        }
        if segments is not None:
            tensor_entry["segments"] = segments
        tensors.append(tensor_entry)
        if dtype in {"q2_b64", "q4_b64", "mixed_q2_q4_b64"}:
            # Escha-style channel scales are standalone FP16 tensors so every
            # backend can fuse them without changing the immutable qcodes.
            rows, columns = shape
            for suffix, channels in (("s_in", columns), ("s_out", rows)):
                correction_size = channels * 2
                logical_bytes = align(logical_bytes)
                tensors.append(
                    {
                        "name": f"{name}.{suffix}",
                        "source_shard": None,
                        "source_dtype": None,
                        "dtype": "f16",
                        "shape": [channels],
                        "offset": logical_bytes,
                        "length": correction_size,
                        "group": "recovery",
                    }
                )
                logical_bytes += correction_size
                dtype_bytes["recovery_f16"] += correction_size

    total_bytes = align(logical_bytes)
    quantized_names = {
        entry["name"]
        for entry in tensors
        if entry["dtype"] in {"q2_b64", "q4_b64", "mixed_q2_q4_b64"}
    }
    if assigned_q4 is not None:
        unknown = assigned_q4 - quantized_names
        if unknown:
            raise SystemExit(f"assignment contains {len(unknown)} non-quantizable tensors")
    plan = {
        "format": (
            "ctox.q2q4.quant-plan.v2"
            if assignment_format == "ctox.q2q4.assignment.v2"
            else "ctox.q2q4.quant-plan.v1"
        ),
        "model": "Qwen/Qwen3.8-27B",
        "revision": args.revision,
        "alignment": ALIGNMENT,
        "vision": "separate",
        "mtp": "resident",
        "late_ffn_q4_start": args.late_ffn_q4_start,
        "q_proj_q4_start": args.q_proj_q4_start,
        "linear_out_q4_start": args.linear_out_q4_start,
        "total_bytes": total_bytes,
        "gib": total_bytes / 1024**3,
        "fold_limit_bytes": FOLD_LIMIT,
        "fits_fold_limit": total_bytes <= FOLD_LIMIT,
        "dtype_bytes": dict(dtype_bytes),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }
    if args.assignment:
        plan["assignment"] = {
            "path": str(args.assignment),
            "sha256": hashlib.sha256(args.assignment.read_bytes()).hexdigest(),
            "q4_tensor_count": len(assigned_q4 or ()),
            "mixed_tensor_count": len(mixed_assignment),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: plan[key] for key in ("total_bytes", "gib", "fits_fold_limit", "dtype_bytes")}, indent=2))
    if total_bytes > FOLD_LIMIT:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
