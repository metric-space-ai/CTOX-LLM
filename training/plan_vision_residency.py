#!/usr/bin/env python3
"""Choose the minimum whole weight bundles to evict for transient vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def source_bundles(plan: dict, group: str) -> list[dict]:
    """Return source tensors with their adjacent recovery/alignment span."""
    tensors = plan["tensors"]
    source_indices = [
        index
        for index, entry in enumerate(tensors)
        if entry["source_shard"] is not None and entry["group"] == group
    ]
    if not source_indices:
        return []
    bundles = []
    for position, index in enumerate(source_indices):
        entry = tensors[index]
        if position + 1 < len(source_indices):
            end = tensors[source_indices[position + 1]]["offset"]
        else:
            later = [item["offset"] for item in tensors[index + 1 :] if item["group"] != group]
            end = min(later) if later else plan["total_bytes"]
        span = end - entry["offset"]
        if span < entry["length"]:
            raise ValueError(f"invalid tensor span for {entry['name']}")
        bundles.append(
            {
                "name": entry["name"],
                "offset": entry["offset"],
                "bytes": span,
            }
        )
    return bundles


def choose_evictions(
    text: dict,
    vision_bytes: int,
    steady_total_bytes: int,
    target_bytes: int,
) -> dict:
    if min(vision_bytes, steady_total_bytes, target_bytes) < 0:
        raise ValueError("memory sizes must not be negative")
    if text.get("format") != "ctox.q2q4.quant-plan.v1":
        raise ValueError("unsupported text plan")
    required = max(0, steady_total_bytes + vision_bytes - target_bytes)
    mtp_bundles = source_bundles(text, "mtp")
    if not mtp_bundles:
        raise ValueError("text plan has no resident MTP package")
    mtp_start = min(bundle["offset"] for bundle in mtp_bundles)
    mtp_bytes = text["total_bytes"] - mtp_start
    selected = [{"name": "resident-mtp-package", "offset": mtp_start, "bytes": mtp_bytes}]
    evicted = mtp_bytes
    remaining = max(0, required - evicted)
    if remaining:
        candidates = [
            bundle
            for bundle in source_bundles(text, "text")
            if bundle["bytes"] >= remaining
            and bundle["name"] not in {"lm_head.weight", "model.language_model.embed_tokens.weight"}
        ]
        if not candidates:
            raise ValueError(f"no single text bundle can satisfy remaining {remaining} bytes")
        candidate = min(candidates, key=lambda item: (item["bytes"], item["name"]))
        selected.append(candidate)
        evicted += candidate["bytes"]
    projected = steady_total_bytes - evicted + vision_bytes
    if projected > target_bytes:
        raise ValueError("residency selection does not fit target")
    return {
        "required_eviction_bytes": required,
        "evicted_bytes": evicted,
        "selected": selected,
        "projected_bytes": projected,
        "headroom_bytes": target_bytes - projected,
        "mtp_source_bundles": len(mtp_bundles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-plan", type=Path, required=True)
    parser.add_argument("--vision-plan", type=Path, required=True)
    parser.add_argument("--steady-total-bytes", type=int, required=True)
    parser.add_argument("--target-bytes", type=int, default=10_415_295_693)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    text = json.loads(args.text_plan.read_text(encoding="utf-8"))
    vision = json.loads(args.vision_plan.read_text(encoding="utf-8"))
    if vision.get("format") != "ctox.q2q4.vision-plan.v1":
        raise SystemExit("unsupported vision plan")
    try:
        selection = choose_evictions(
            text,
            vision["total_bytes"],
            args.steady_total_bytes,
            args.target_bytes,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.vision-residency.v1",
        "steady_total_bytes": args.steady_total_bytes,
        "vision_bytes": vision["total_bytes"],
        "target_bytes": args.target_bytes,
        "workspace_policy": "reuse_decoder_arena",
        "os_swap_required": False,
        **selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
