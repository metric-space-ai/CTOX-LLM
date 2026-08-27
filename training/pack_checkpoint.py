#!/usr/bin/env python3
"""Stream BF16 shards into a checksummed CTOX Q2/Q4 artifact.

The packer keeps safetensor files memory-mapped, slices matrices by rows, and
quantizes chunks on one GPU. It never materializes the full checkpoint or full
output in RAM. Direct baselines use explicit identity recovery; release
candidates consume a complete, plan-bound recovery safetensors artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

from fanout_recovery import (
    POLICIES,
    QWEN38_FANOUT_POLICY,
    fanout_group_sha256,
    qwen38_fanout_groups,
)
from cache_teacher import validate_local_model_provenance
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recovery_entries(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in plan["tensors"] if entry.get("group") == "recovery"]


def validate_recovery_source(
    plan: dict[str, Any],
    plan_sha256: str,
    recovery: Any,
    *,
    allow_bounded_verifier: bool = False,
) -> dict[str, Any]:
    """Validate trained scales before expensive model packing.

    Complete recovery produces a release-eligible ``trained`` descriptor. A
    bounded run is accepted only behind the explicit verifier switch and is
    encoded as ``verifier`` so native release validation cannot promote it.
    """

    metadata = recovery.metadata() or {}
    status = metadata.get("status")
    if status == "complete":
        recovery_mode = "trained"
    elif status == "bounded_run_complete" and allow_bounded_verifier:
        recovery_mode = "verifier"
    else:
        expected = "'complete' or explicitly admitted 'bounded_run_complete'"
        raise RuntimeError(
            f"recovery metadata status is {status!r}, expected {expected}"
        )
    required_metadata = {
        "format": "ctox.recovery.channel-scales.v2",
        "model": plan["model"],
        "revision": plan["revision"],
        "plan_sha256": plan_sha256,
        "fixed_logical_qcodes": "true",
    }
    if plan.get("local_model_provenance_sha256"):
        required_metadata["local_model_provenance_sha256"] = plan[
            "local_model_provenance_sha256"
        ]
    for key, expected in required_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise RuntimeError(
                f"recovery metadata {key} is {actual!r}, expected {expected!r}"
            )

    expected = {entry["name"]: entry for entry in recovery_entries(plan)}
    actual = set(recovery.keys())
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise RuntimeError(
            f"recovery tensor set mismatch: {len(missing)} missing, {len(extra)} extra"
        )
    for name, entry in expected.items():
        tensor = recovery.get_tensor(name)
        if tuple(tensor.shape) != tuple(entry["shape"]):
            raise RuntimeError(f"recovery tensor {name} shape mismatch")
        if str(tensor.dtype) != "torch.float16":
            raise RuntimeError(f"recovery tensor {name} must be FP16")

    descriptor = {
        "mode": recovery_mode,
        "format": metadata["format"],
        "plan_sha256": metadata["plan_sha256"],
        "activation_stats_sha256": metadata.get("activation_stats_sha256", ""),
        "report_sha256": metadata.get("report_sha256", ""),
        "fixed_logical_qcodes": True,
    }
    for key in ("activation_stats_sha256", "report_sha256"):
        value = descriptor[key]
        if len(value) != 64 or not all(character in "0123456789abcdef" for character in value):
            raise RuntimeError(f"recovery metadata {key} is not a lowercase SHA-256")
    policy = metadata.get("fanout_s_in_policy")
    if policy is not None:
        if policy not in POLICIES:
            raise RuntimeError(f"unsupported recovery fanout_s_in_policy {policy}")
        group_digest = metadata.get("fanout_group_sha256", "")
        if len(group_digest) != 64 or not all(
            character in "0123456789abcdef" for character in group_digest
        ):
            raise RuntimeError("recovery fanout_group_sha256 is not a lowercase SHA-256")
        groups = (
            qwen38_fanout_groups(
                (
                    entry["name"]
                    for entry in plan["tensors"]
                    if entry.get("dtype")
                    in {"q2_b64", "q4_b64", "mixed_q2_q4_b64"}
                )
            )
            if policy == QWEN38_FANOUT_POLICY
            else []
        )
        if fanout_group_sha256(groups) != group_digest:
            raise RuntimeError("recovery fanout group digest differs from the quant plan")
        for group in groups:
            tensors = [recovery.get_tensor(name) for name in group["scale_names"]]
            reference = tensors[0]
            if any(not reference.equal(tensor) for tensor in tensors[1:]):
                raise RuntimeError(
                    f"recovery fanout scales differ at {group['prefix']}"
                )
        descriptor.update(
            {
                "fanout_s_in_policy": policy,
                "fanout_group_sha256": group_digest,
                "fanout_group_count": len(groups),
                "fanout_logical_s_in_tensors": sum(
                    len(group["scale_names"]) for group in groups
                ),
            }
        )
    elif metadata.get("fanout_group_sha256") is not None:
        raise RuntimeError("recovery fanout group digest lacks a policy")
    return descriptor


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


def write_recovery_tensor(output: BinaryIO, entry: dict, recovery: Any | None) -> str:
    import torch

    if entry["dtype"] != "f16" or len(entry["shape"]) != 1:
        raise RuntimeError(f"invalid generated recovery tensor {entry['name']}")
    tensor = (
        torch.ones(entry["shape"], dtype=torch.float16)
        if recovery is None
        else recovery.get_tensor(entry["name"])
    )
    if not bool(torch.isfinite(tensor).all()) or not bool((tensor > 0).all()):
        raise RuntimeError(f"recovery tensor {entry['name']} must be finite and positive")
    payload = tensor_bytes(tensor)
    if len(payload) != entry["length"]:
        raise RuntimeError(f"recovery tensor {entry['name']} length mismatch")
    output.write(payload)
    return hashlib.sha256(payload).hexdigest()


def assemble_artifact(
    output_path: Path,
    data_path: Path,
    plan: dict,
    tensor_hashes: dict[str, str],
    recovery_descriptor: dict[str, Any],
) -> None:
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
        "recovery": recovery_descriptor,
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
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-model-provenance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recovery-scales",
        type=Path,
        help="complete plan-bound FP16 s_in/s_out safetensors; omission creates an identity baseline",
    )
    parser.add_argument(
        "--allow-bounded-verifier-recovery",
        action="store_true",
        help=(
            "pack bounded recovery only as a native verifier artifact; the "
            "manifest mode remains release-ineligible"
        ),
    )
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
        _provenance, provenance_sha256 = validate_local_model_provenance(
            args.checkpoint,
            args.revision,
            args.local_model_provenance,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    if provenance_sha256 is None:
        raise SystemExit("checkpoint packing requires verified local BF16 provenance")
    if plan.get("revision") != args.revision:
        raise SystemExit("quant plan revision does not match --revision")
    if plan.get("local_model_provenance_sha256") != provenance_sha256:
        raise SystemExit("quant plan does not match the verified local BF16 provenance")
    if plan.get("assignment") and args.recovery_scales is None:
        raise SystemExit("assigned release plans require complete recovery scales")

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise SystemExit("install training/requirements.in before packing") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but unavailable")

    plan_digest = sha256_path(args.plan)

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
        recovery = (
            stack.enter_context(safe_open(args.recovery_scales, framework="pt", device="cpu"))
            if args.recovery_scales is not None
            else None
        )
        if recovery is None:
            recovery_descriptor = {
                "mode": "identity",
                "format": "ctox.recovery.identity.v1",
                "plan_sha256": plan_digest,
                "fixed_logical_qcodes": True,
            }
        else:
            recovery_descriptor = validate_recovery_source(
                plan,
                plan_digest,
                recovery,
                allow_bounded_verifier=args.allow_bounded_verifier_recovery,
            )
            recovery_descriptor["artifact_sha256"] = sha256_path(args.recovery_scales)
        data_path = Path(temporary) / "tensor-data.bin"
        tensor_hashes: dict[str, str] = {}
        with data_path.open("xb") as data:
            for index, entry in enumerate(plan["tensors"], 1):
                data.seek(entry["offset"])
                if entry["source_shard"] is None:
                    if entry.get("group") != "recovery":
                        raise RuntimeError(
                            f"generated tensor {entry['name']} is not a recovery scale"
                        )
                    digest = write_recovery_tensor(data, entry, recovery)
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
        artifact_path = Path(temporary) / args.output.name
        assemble_artifact(
            artifact_path,
            data_path,
            plan,
            tensor_hashes,
            recovery_descriptor,
        )
        package_limit = plan.get("fold_package_limit_bytes")
        if package_limit is not None and artifact_path.stat().st_size > package_limit:
            raise RuntimeError(
                f"artifact is {artifact_path.stat().st_size} bytes, "
                f"package limit is {package_limit}"
            )
        artifact_sha256 = sha256_path(artifact_path)
        with artifact_path.open("rb") as artifact:
            os.fsync(artifact.fileno())
        artifact_path.replace(args.output)
        directory = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": artifact_sha256,
                "recovery": recovery_descriptor,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
