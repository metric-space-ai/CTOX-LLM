#!/usr/bin/env python3
"""Re-evaluate frozen domain tags when selection-gate policy changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from classify_domains import quota_gaps, validate_rubric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    try:
        rubric_bytes = args.rubric.read_bytes()
        rubric = json.loads(rubric_bytes)
        validate_rubric(rubric)
        counts: Counter[str] = Counter()
        primary_counts: Counter[str] = Counter()
        fallback = 0
        records = 0
        seen = set()
        with args.tags.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                tag = json.loads(line)
                sample_id = str(tag["id"])
                if sample_id in seen:
                    raise ValueError(f"duplicate domain tag {sample_id}")
                seen.add(sample_id)
                counts.update(tag["labels"])
                primary_counts[str(tag["primary_label"])] += 1
                fallback += int(bool(tag.get("below_threshold_fallback")))
                records += 1
        unknown = (set(counts) | set(primary_counts)) - set(rubric["domains"])
        if unknown:
            raise ValueError(f"domain tags contain unknown labels: {sorted(unknown)}")
        gaps, primary_gaps = quota_gaps(
            counts, primary_counts, rubric, args.partition
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    document = {
        "format": "ctox.recovery-domain-selection-gate.v1",
        "partition": args.partition,
        "tags": str(args.tags.resolve()),
        "tags_sha256": hashlib.sha256(args.tags.read_bytes()).hexdigest(),
        "rubric": str(args.rubric.resolve()),
        "rubric_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
        "records": records,
        "below_threshold_fallback_records": fallback,
        "domain_counts": dict(sorted(counts.items())),
        "primary_domain_counts": dict(sorted(primary_counts.items())),
        "quota_gaps": gaps,
        "primary_quota_gaps": primary_gaps,
        "multi_label_gate": "passed" if not gaps else "failed",
        "primary_gate": "passed" if not primary_gaps else "supplement_required",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
