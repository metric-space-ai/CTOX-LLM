#!/usr/bin/env python3
"""Verify the pinned TensorRT-LLM MTP kernel extraction byte-for-byte."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verify_vendor_manifest import verify


EXPECTED_REVISION = "390f92d76377afe0159e8d7ffd2312483cf2a358"
EXPECTED_SYMBOL = "ctox_trtllm_mtp_accept_draft_token_sm86"
UPSTREAM_START_LINE = 250
UPSTREAM_END_LINE = 312


def body_from_first_brace(text: str) -> str:
    lines = text.splitlines(keepends=True)
    try:
        first = next(index for index, line in enumerate(lines) if line.rstrip("\r\n") == "{")
    except StopIteration as exc:
        raise ValueError("kernel source lacks an opening body brace") from exc
    return "".join(lines[first:])


def verify_extraction(manifest_path: Path) -> dict[str, object]:
    verified_files = verify(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upstream", {}).get("revision") != EXPECTED_REVISION:
        raise ValueError("unexpected TensorRT-LLM revision")
    root = manifest_path.parent.resolve()
    upstream_path = root / "upstream/mtpKernels.cu"
    extraction_path = root / "mtp_accept_draft_token_sm86.cu"
    upstream_lines = upstream_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(upstream_lines) < UPSTREAM_END_LINE:
        raise ValueError("upstream MTP source is shorter than the pinned line range")
    upstream_kernel = "".join(upstream_lines[UPSTREAM_START_LINE - 1 : UPSTREAM_END_LINE])
    extraction = extraction_path.read_text(encoding="utf-8")
    if body_from_first_brace(upstream_kernel) != body_from_first_brace(extraction):
        raise ValueError("standalone MTP kernel body differs from upstream lines 250-312")
    if extraction.count(EXPECTED_SYMBOL) != 1:
        raise ValueError("standalone MTP kernel must export exactly one named entry point")
    if not extraction.lstrip().startswith("/*") or "Licensed under the Apache License" not in extraction:
        raise ValueError("standalone MTP kernel lacks its upstream license header")
    if "#include" in extraction:
        raise ValueError("standalone MTP extraction unexpectedly depends on an include")
    if 'extern "C" __global__ void ' + EXPECTED_SYMBOL not in extraction:
        raise ValueError("standalone MTP kernel lacks C linkage")
    return {
        "manifest": str(manifest_path),
        "verified_files": verified_files,
        "revision": EXPECTED_REVISION,
        "symbol": EXPECTED_SYMBOL,
        "upstream_lines": [UPSTREAM_START_LINE, UPSTREAM_END_LINE],
        "body_byte_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_extraction(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
