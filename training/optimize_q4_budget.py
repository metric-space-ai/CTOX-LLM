#!/usr/bin/env python3
"""Choose Q4 tensor candidates by quality gain per additional byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def packed_bytes(dtype: str, elements: int) -> int:
    return math.ceil(elements / 64) * (18 if dtype == "q2_b64" else 34)


def layout_bytes(plan: dict, q4_tensors: set[str]) -> int:
    position = 0
    for entry in plan["tensors"]:
        position = align(position, plan["alignment"])
        if entry["source_shard"] is not None and entry["dtype"] in {"q2_b64", "q4_b64"}:
            elements = math.prod(entry["shape"])
            dtype = "q4_b64" if entry["name"] in q4_tensors else "q2_b64"
            position += packed_bytes(dtype, elements)
        else:
            position += entry["length"]
    return align(position, plan["alignment"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    selected = {item["name"] for item in candidates if item.get("fixed_q4")}
    base_bytes = layout_bytes(plan, selected)
    if base_bytes > args.budget_bytes:
        raise SystemExit(f"fixed Q4 policy requires {base_bytes} bytes, above budget")
    optional = []
    for item in candidates:
        if item.get("fixed_q4"):
            continue
        extra = item["q4_bytes"] - item["q2_bytes"]
        if extra <= 0 or item["quality_gain"] <= 0:
            continue
        optional.append((item["quality_gain"] / extra, extra, item["name"]))
    for _, _, name in sorted(optional, key=lambda item: (-item[0], item[2])):
        trial = selected | {name}
        if layout_bytes(plan, trial) <= args.budget_bytes:
            selected = trial
    bytes_used = layout_bytes(plan, selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": "ctox.q2q4.assignment.v1",
                "budget_bytes": args.budget_bytes,
                "base_bytes": base_bytes,
                "bytes_used": bytes_used,
                "plan_sha256": sha256(args.plan),
                "sensitivity_sha256": sha256(args.sensitivity),
                "q4_tensors": sorted(selected),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
