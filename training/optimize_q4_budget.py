#!/usr/bin/env python3
"""Choose Q4 tensor candidates by quality gain per additional byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


QUANTIZED_DTYPES = frozenset({"q2_b64", "q4_b64", "mixed_q2_q4_b64"})


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def packed_bytes(dtype: str, elements: int) -> int:
    return math.ceil(elements / 64) * (18 if dtype == "q2_b64" else 34)


def mixed_tensor_bytes(
    groups: list[dict],
    q4_groups: set[int],
) -> int:
    total = 0
    for group in groups:
        dtype = "q4_b64" if group["group_index"] in q4_groups else "q2_b64"
        total += group[f"{dtype.removesuffix('_b64')}_bytes"]
    return total


def layout_bytes(
    plan: dict,
    q4_tensors: set[str],
    mixed_groups: dict[str, list[dict]] | None = None,
    q4_row_groups: dict[str, set[int]] | None = None,
) -> int:
    mixed_groups = mixed_groups or {}
    q4_row_groups = q4_row_groups or {}
    position = 0
    for entry in plan["tensors"]:
        position = align(position, plan["alignment"])
        if entry["source_shard"] is not None and entry["dtype"] in QUANTIZED_DTYPES:
            if entry["name"] in mixed_groups:
                position += mixed_tensor_bytes(
                    mixed_groups[entry["name"]],
                    q4_row_groups.get(entry["name"], set()),
                )
            else:
                elements = math.prod(entry["shape"])
                dtype = "q4_b64" if entry["name"] in q4_tensors else "q2_b64"
                position += packed_bytes(dtype, elements)
        else:
            position += entry["length"]
    return align(position, plan["alignment"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_selections(
    candidates: list[dict],
    mixed_groups: dict[str, list[dict]],
) -> tuple[set[str], dict[str, set[int]]]:
    selected = {
        item["name"]
        for item in candidates
        if item.get("fixed_q4") and item["name"] not in mixed_groups
    }
    selected_row_groups: dict[str, set[int]] = {}
    for item in candidates:
        if item["name"] not in mixed_groups:
            continue
        groups = mixed_groups[item["name"]]
        if item.get("fixed_q4"):
            selected_row_groups[item["name"]] = {
                group["group_index"] for group in groups
            }
        else:
            selected_row_groups[item["name"]] = {
                group["group_index"] for group in groups if group.get("fixed_q4")
            }
    return selected, selected_row_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sensitivity = json.loads(args.sensitivity.read_text(encoding="utf-8"))
    candidates = sensitivity.get("candidates", sensitivity) if isinstance(sensitivity, dict) else sensitivity
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    mixed_groups = {
        item["name"]: item["row_groups"]
        for item in candidates
        if item.get("row_groups")
    }
    selected, selected_row_groups = initial_selections(candidates, mixed_groups)
    base_bytes = layout_bytes(plan, selected, mixed_groups, selected_row_groups)
    if base_bytes > args.budget_bytes:
        raise SystemExit(f"fixed Q4 policy requires {base_bytes} bytes, above budget")
    optional = []
    for item in candidates:
        if item["name"] in mixed_groups:
            for group in item["row_groups"]:
                if group.get("fixed_q4"):
                    continue
                extra = group["q4_bytes"] - group["q2_bytes"]
                if extra <= 0 or group["quality_gain"] <= 0:
                    continue
                optional.append(
                    (
                        group["quality_gain"] / extra,
                        "row_group",
                        item["name"],
                        group["group_index"],
                    )
                )
            continue
        if item.get("fixed_q4"):
            continue
        extra = item["q4_bytes"] - item["q2_bytes"]
        if extra <= 0 or item["quality_gain"] <= 0:
            continue
        optional.append((item["quality_gain"] / extra, "tensor", item["name"], -1))
    for _, kind, name, group_index in sorted(
        optional,
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    ):
        trial_tensors = selected
        trial_rows = selected_row_groups
        if kind == "tensor":
            trial_tensors = selected | {name}
        else:
            trial_rows = {key: set(value) for key, value in selected_row_groups.items()}
            trial_rows.setdefault(name, set()).add(group_index)
        if layout_bytes(plan, trial_tensors, mixed_groups, trial_rows) <= args.budget_bytes:
            selected = trial_tensors
            selected_row_groups = trial_rows
    bytes_used = layout_bytes(plan, selected, mixed_groups, selected_row_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": "ctox.q2q4.assignment.v2",
                "budget_bytes": args.budget_bytes,
                "base_bytes": base_bytes,
                "bytes_used": bytes_used,
                "plan_sha256": sha256(args.plan),
                "sensitivity_sha256": sha256(args.sensitivity),
                "q4_tensors": sorted(selected),
                "mixed_tensors": {
                    name: {
                        "row_group_size": next(
                            item["row_group_size"] for item in candidates if item["name"] == name
                        ),
                        "group_count": len(groups),
                        "q4_groups": sorted(selected_row_groups.get(name, set())),
                    }
                    for name, groups in sorted(mixed_groups.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
