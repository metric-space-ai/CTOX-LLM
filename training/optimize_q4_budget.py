#!/usr/bin/env python3
"""Choose Q4 tensor candidates by quality gain per additional byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from select_activation_calibration import write_bytes_atomic


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


def validate_sensitivity_contract(sensitivity: dict, plan: dict, plan_sha256: str) -> None:
    if sensitivity.get("format") != "ctox.q2q4.sensitivity.v1":
        raise ValueError("unsupported Q2/Q4 sensitivity format")
    if sensitivity.get("model") != plan.get("model"):
        raise ValueError("sensitivity model does not match the quant plan")
    if sensitivity.get("revision") != plan.get("revision"):
        raise ValueError("sensitivity revision does not match the quant plan")
    if sensitivity.get("quant_plan_sha256") != plan_sha256:
        raise ValueError("sensitivity is not bound to the quant plan")
    unobserved = [item["name"] for item in sensitivity["candidates"] if not item.get("observed")]
    if unobserved:
        raise ValueError(f"sensitivity contains {len(unobserved)} unobserved candidates")


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


def optimized_selections(
    plan: dict,
    candidates: list[dict],
    mixed_groups: dict[str, list[dict]],
    selected: set[str],
    selected_row_groups: dict[str, set[int]],
    budget_bytes: int,
) -> tuple[set[str], dict[str, set[int]], list[dict]]:
    alignment = int(plan["alignment"])
    candidate_by_name = {item["name"]: item for item in candidates}
    optional: list[tuple[str, str, int, float, int]] = []
    for item in candidates:
        if item["name"] in mixed_groups:
            for group in item["row_groups"]:
                extra = int(group["q4_bytes"]) - int(group["q2_bytes"])
                if (
                    group.get("fixed_q4")
                    or group["quality_gain"] <= 0
                    or extra <= 0
                ):
                    continue
                optional.append(
                    (
                        "row_group",
                        item["name"],
                        int(group["group_index"]),
                        float(group["quality_gain"]),
                        extra,
                    )
                )
            continue
        extra = int(item["q4_bytes"]) - int(item["q2_bytes"])
        if item.get("fixed_q4") or item["quality_gain"] <= 0 or extra <= 0:
            continue
        optional.append(
            ("tensor", item["name"], -1, float(item["quality_gain"]), extra)
        )

    remaining = optional
    decisions: list[dict] = []
    current_bytes = layout_bytes(
        plan,
        selected,
        mixed_groups,
        selected_row_groups,
    )
    mixed_payload_bytes = {
        name: mixed_tensor_bytes(groups, selected_row_groups.get(name, set()))
        for name, groups in mixed_groups.items()
    }
    while remaining:
        best: tuple[tuple[float, float, str, str, int], tuple, int] | None = None
        for candidate in remaining:
            kind, name, group_index, quality_gain, extra = candidate
            if kind == "tensor":
                item = candidate_by_name[name]
                marginal_bytes = align(int(item["q4_bytes"]), alignment) - align(
                    int(item["q2_bytes"]), alignment
                )
            else:
                current_payload = mixed_payload_bytes[name]
                marginal_bytes = align(current_payload + extra, alignment) - align(
                    current_payload, alignment
                )
            if marginal_bytes < 0:
                raise ValueError("Q4 selection unexpectedly reduces packed layout bytes")
            trial_bytes = current_bytes + marginal_bytes
            if trial_bytes > budget_bytes:
                continue
            gain_per_byte = (
                math.inf if marginal_bytes == 0 else quality_gain / marginal_bytes
            )
            # `min` over the negative score is deterministic and keeps the
            # stable semantic identity as the final tie-breaker.
            key = (-gain_per_byte, -quality_gain, kind, name, group_index)
            if best is None or key < best[0]:
                best = (key, candidate, trial_bytes)
        if best is None:
            break
        _, candidate, trial_bytes = best
        kind, name, group_index, quality_gain, extra = candidate
        marginal_bytes = trial_bytes - current_bytes
        if kind == "tensor":
            selected = selected | {name}
        else:
            selected_row_groups = {
                key: set(value) for key, value in selected_row_groups.items()
            }
            selected_row_groups.setdefault(name, set()).add(group_index)
            mixed_payload_bytes[name] += extra
        decisions.append(
            {
                "rank": len(decisions) + 1,
                "kind": kind,
                "name": name,
                "group_index": None if group_index < 0 else group_index,
                "quality_gain": quality_gain,
                "marginal_layout_bytes": marginal_bytes,
                "quality_gain_per_marginal_byte": (
                    None if marginal_bytes == 0 else quality_gain / marginal_bytes
                ),
                "layout_bytes_after": trial_bytes,
            }
        )
        current_bytes = trial_bytes
        remaining.remove(candidate)
    exact_bytes = layout_bytes(plan, selected, mixed_groups, selected_row_groups)
    if exact_bytes != current_bytes:
        raise ValueError(
            f"incremental layout accounting produced {current_bytes}, exact layout is {exact_bytes}"
        )
    return selected, selected_row_groups, decisions


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
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    try:
        validate_sensitivity_contract(
            sensitivity,
            plan,
            hashlib.sha256(plan_bytes).hexdigest(),
        )
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    candidates = sensitivity["candidates"]
    mixed_groups = {
        item["name"]: item["row_groups"]
        for item in candidates
        if item.get("row_groups")
    }
    selected, selected_row_groups = initial_selections(candidates, mixed_groups)
    base_bytes = layout_bytes(plan, selected, mixed_groups, selected_row_groups)
    if base_bytes > args.budget_bytes:
        raise SystemExit(f"fixed Q4 policy requires {base_bytes} bytes, above budget")
    selected, selected_row_groups, decisions = optimized_selections(
        plan,
        candidates,
        mixed_groups,
        selected,
        selected_row_groups,
        args.budget_bytes,
    )
    bytes_used = layout_bytes(plan, selected, mixed_groups, selected_row_groups)
    write_bytes_atomic(
        args.output,
        (
            json.dumps(
                {
                    "format": "ctox.q2q4.assignment.v2",
                    "selection_policy": "activation-weighted Q2-to-Q4 quality gain per added packed byte",
                    "selection_decisions": decisions,
                    "budget_bytes": args.budget_bytes,
                    "base_bytes": base_bytes,
                    "bytes_used": bytes_used,
                    "plan_sha256": sha256(args.plan),
                    "sensitivity_sha256": sha256(args.sensitivity),
                    "q4_tensors": sorted(selected),
                    "mixed_tensors": {
                        name: {
                            "row_group_size": next(
                                item["row_group_size"]
                                for item in candidates
                                if item["name"] == name
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
            + "\n"
        ).encode("utf-8"),
    )


if __name__ == "__main__":
    main()
