#!/usr/bin/env python3
"""Stream BF16 shards into a checksummed CTOX Q2/Q4 artifact.

The packer keeps safetensor files memory-mapped, slices matrices by rows, and
quantizes chunks on one GPU. It never materializes the full checkpoint or full
output in RAM. Recovery tensors are initialized to one and replaced by the
trained scales in the final pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

from quantization import quantize_components
from run_ledger import GpuRun, require_budget


MAGIC = b"CTOXQ2Q4"
ENDIAN_MARKER = 0x01020304
HEADER_BYTES = 64
Q2_BLOCK_BYTES = 18
Q4_BLOCK_BYTES = 34


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def tensor_bytes(tensor) -> bytes:
    import torch

    return tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def quantize_blocks(values, dtype: str) -> bytes:
    import torch

    blocks = values.float().reshape(-1, 64)
    scales, codes = quantize_components(torch, values, dtype)
    if dtype == "q2_b64":
        grouped = codes.reshape(-1, 16, 4)
        packed = (
            grouped[:, :, 0]
            | (grouped[:, :, 1] << 2)
            | (grouped[:, :, 2] << 4)
            | (grouped[:, :, 3] << 6)
        )
        output = torch.empty((blocks.shape[0], Q2_BLOCK_BYTES), dtype=torch.uint8, device=values.device)
        output[:, :2] = scales.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
        output[:, 2:] = packed
    elif dtype == "q4_b64":
        grouped = codes.reshape(-1, 32, 2)
        packed = grouped[:, :, 0] | (grouped[:, :, 1] << 4)
        output = torch.empty((blocks.shape[0], Q4_BLOCK_BYTES), dtype=torch.uint8, device=values.device)
        output[:, :2] = scales.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, 2)
        output[:, 2:] = packed
    else:
        raise ValueError(dtype)
    return tensor_bytes(output)


def write_source_tensor(
    output: BinaryIO,
    source,
    entry: dict,
    device: str,
    rows_per_chunk: int,
) -> str:
    import torch

    digest = hashlib.sha256()
    written = 0
    dtype = entry["dtype"]
    shape = entry["shape"]
    name = entry["name"]
    if dtype == "mixed_q2_q4_b64":
        if len(shape) != 2 or shape[1] % 64:
            raise RuntimeError(f"mixed tensor {name} is not a block-aligned matrix")
        tensor_slice = source.get_slice(name)
        expected_row = 0
        for segment in entry.get("segments", []):
            if segment["row_start"] != expected_row or segment["row_end"] <= expected_row:
                raise RuntimeError(f"mixed tensor {name} has non-contiguous row segments")
            segment_written = 0
            for row_start in range(segment["row_start"], segment["row_end"], rows_per_chunk):
                row_end = min(segment["row_end"], row_start + rows_per_chunk)
                values = tensor_slice[row_start:row_end].to(device=device, dtype=torch.float32)
                payload = quantize_blocks(values, segment["dtype"])
                output.write(payload)
                digest.update(payload)
                written += len(payload)
                segment_written += len(payload)
                del values
            if segment_written != segment["length"]:
                raise RuntimeError(
                    f"mixed tensor {name} segment {segment['group_index']} wrote "
                    f"{segment_written} bytes, expected {segment['length']}"
                )
            expected_row = segment["row_end"]
        if expected_row != shape[0]:
            raise RuntimeError(f"mixed tensor {name} segments do not cover every row")
    elif dtype in {"q2_b64", "q4_b64"}:
        if len(shape) != 2 or shape[1] % 64:
            raise RuntimeError(f"quantized tensor {name} is not a block-aligned matrix")
        tensor_slice = source.get_slice(name)
        for row_start in range(0, shape[0], rows_per_chunk):
            row_end = min(shape[0], row_start + rows_per_chunk)
            values = tensor_slice[row_start:row_end].to(device=device, dtype=torch.float32)
            payload = quantize_blocks(values, dtype)
            output.write(payload)
            digest.update(payload)
            written += len(payload)
            del values
    else:
        target_dtype = torch.float16 if dtype == "f16" else torch.float32
        payload = tensor_bytes(source.get_tensor(name).to(dtype=target_dtype))
        output.write(payload)
        digest.update(payload)
        written = len(payload)
    if written != entry["length"]:
        raise RuntimeError(f"packed {name} into {written} bytes, plan requires {entry['length']}")
    return digest.hexdigest()


def write_recovery_tensor(output: BinaryIO, entry: dict) -> str:
    import torch

    if entry["dtype"] != "f16" or len(entry["shape"]) != 1:
        raise RuntimeError(f"invalid generated recovery tensor {entry['name']}")
    payload = tensor_bytes(torch.ones(entry["shape"], dtype=torch.float16))
    if len(payload) != entry["length"]:
        raise RuntimeError(f"recovery tensor {entry['name']} length mismatch")
    output.write(payload)
    return hashlib.sha256(payload).hexdigest()


def assemble_artifact(output_path: Path, data_path: Path, plan: dict, tensor_hashes: dict[str, str]) -> None:
    manifest_tensors = []
    for entry in plan["tensors"]:
        tensor = {
            "name": entry["name"],
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "offset": entry["offset"],
            "length": entry["length"],
            "sha256": tensor_hashes[entry["name"]],
        }
        if "segments" in entry:
            tensor["segments"] = entry["segments"]
        manifest_tensors.append(tensor)
    version = 2 if plan["format"] == "ctox.q2q4.quant-plan.v2" else 1
    manifest = {
        "format": f"ctox.q2q4.v{version}",
        "model": plan["model"],
        "revision": plan["revision"],
        "alignment": plan["alignment"],
        "target": "canonical-b64",
        "tensors": manifest_tensors,
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    data_offset = align(HEADER_BYTES + len(manifest_bytes), plan["alignment"])
    header = struct.pack(
        "<8sIIQQII24x",
        MAGIC,
        version,
        ENDIAN_MARKER,
        len(manifest_bytes),
        data_offset,
        len(manifest_tensors),
        plan["alignment"],
    )
    with output_path.open("xb") as output, data_path.open("rb") as data:
        output.write(header)
        output.write(manifest_bytes)
        output.write(b"\0" * (data_offset - HEADER_BYTES - len(manifest_bytes)))
        shutil.copyfileobj(data, output, length=16 * 1024 * 1024)
        output.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserved-gpu-hours", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-chunk", type=int, default=256)
    args = parser.parse_args()
    require_budget(args.ledger, args.reserved_gpu_hours)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.rows_per_chunk < 1:
        raise SystemExit("rows-per-chunk must be positive")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["format"] not in {
        "ctox.q2q4.quant-plan.v1",
        "ctox.q2q4.quant-plan.v2",
    } or not plan["fits_fold_limit"]:
        raise SystemExit("plan is unsupported or exceeds the Fold limit")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before packing") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")

    shards = sorted(
        {entry["source_shard"] for entry in plan["tensors"] if entry["source_shard"] is not None}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with GpuRun(args.ledger, "direct-q2q4-pack", 1, sys.argv), tempfile.TemporaryDirectory(
        prefix="ctox-q2q4-pack-", dir=args.output.parent
    ) as temporary, ExitStack() as stack:
        sources = {
            shard: stack.enter_context(
                safe_open(args.checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in shards
        }
        data_path = Path(temporary) / "tensor-data.bin"
        tensor_hashes: dict[str, str] = {}
        with data_path.open("xb") as data:
            for index, entry in enumerate(plan["tensors"], 1):
                data.seek(entry["offset"])
                if entry["source_shard"] is None:
                    digest = write_recovery_tensor(data, entry)
                else:
                    digest = write_source_tensor(
                        data,
                        sources[entry["source_shard"]],
                        entry,
                        args.device,
                        args.rows_per_chunk,
                    )
                tensor_hashes[entry["name"]] = digest
                print(f"[{index}/{len(plan['tensors'])}] {entry['dtype']} {entry['name']}", flush=True)
            data.truncate(plan["total_bytes"])
            data.flush()
        assemble_artifact(args.output, data_path, plan, tensor_hashes)
    artifact_hash = hashlib.sha256()
    with args.output.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(16 * 1024 * 1024), b""):
            artifact_hash.update(chunk)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": artifact_hash.hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
