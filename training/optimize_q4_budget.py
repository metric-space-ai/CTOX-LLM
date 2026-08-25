#!/usr/bin/env python3
"""Choose Q4 tensor candidates by quality gain per additional byte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    sensitivity = json.loads(args.sensitivity.read_text(encoding="utf-8"))
    candidates = sensitivity.get("candidates", sensitivity) if isinstance(sensitivity, dict) else sensitivity
    selected = []
    bytes_used = sum(item["q4_bytes"] if item.get("fixed_q4") else item["q2_bytes"] for item in candidates)
    optional = []
    for item in candidates:
        if item.get("fixed_q4"):
            selected.append(item["name"])
            continue
        extra = item["q4_bytes"] - item["q2_bytes"]
        if extra <= 0:
            continue
        optional.append((item["quality_gain"] / extra, extra, item["name"]))
    for _, extra, name in sorted(optional, reverse=True):
        if bytes_used + extra <= args.budget_bytes:
            bytes_used += extra
            selected.append(name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "format": "ctox.q2q4.assignment.v1",
                "budget_bytes": args.budget_bytes,
                "bytes_used": bytes_used,
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
