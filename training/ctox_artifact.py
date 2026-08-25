"""Read and verify native CTOX Q2/Q4 containers for offline recovery."""

from __future__ import annotations

import hashlib
import json
import mmap
import struct
from pathlib import Path
from typing import Any


MAGIC = b"CTOXQ2Q4"
ENDIAN_MARKER = 0x01020304
HEADER = struct.Struct("<8sIIQQII24x")
HEADER_BYTES = 64
BLOCK = 64
BLOCK_BYTES = {"q2_b64": 18, "q4_b64": 34}


def packed_bytes(dtype: str, shape: list[int], segments: list[dict[str, Any]]) -> int:
    elements = 1
    if not shape or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError("tensor shape is empty or non-positive")
    for dimension in shape:
        elements *= int(dimension)
    if dtype in BLOCK_BYTES:
        if segments:
            raise ValueError("non-mixed tensor declares segments")
        return ((elements + BLOCK - 1) // BLOCK) * BLOCK_BYTES[dtype]
    if dtype == "mixed_q2_q4_b64":
        if len(shape) != 2 or shape[1] % BLOCK or not segments:
            raise ValueError("mixed tensor must be a segmented block-aligned matrix")
        row = 0
        offset = 0
        for index, segment in enumerate(segments):
            if (
                int(segment["group_index"]) != index
                or int(segment["row_start"]) != row
                or int(segment["row_end"]) <= row
                or int(segment["offset"]) != offset
                or segment["dtype"] not in BLOCK_BYTES
            ):
                raise ValueError("mixed tensor segments are not contiguous")
            segment_elements = (int(segment["row_end"]) - row) * int(shape[1])
            length = (segment_elements // BLOCK) * BLOCK_BYTES[segment["dtype"]]
            if int(segment["length"]) != length:
                raise ValueError("mixed tensor segment length differs")
            row = int(segment["row_end"])
            offset += length
        if row != int(shape[0]):
            raise ValueError("mixed tensor segments do not cover all rows")
        return offset
    if dtype == "f16":
        return elements * 2
    if dtype == "f32":
        return elements * 4
    raise ValueError(f"unsupported CTOX tensor dtype {dtype}")


class CtoxArtifact:
    def __init__(self, path: Path, verify_tensors: bool = False) -> None:
        self.path = path
        self._file = path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(self._mmap) < HEADER_BYTES:
                raise ValueError("CTOX artifact is smaller than its header")
            magic, version, endian, manifest_len, data_offset, tensor_count, alignment = (
                HEADER.unpack_from(self._mmap)
            )
            if magic != MAGIC or version not in (1, 2) or endian != ENDIAN_MARKER:
                raise ValueError("invalid CTOX header contract")
            if alignment < 64 or alignment & (alignment - 1):
                raise ValueError("invalid CTOX tensor alignment")
            manifest_end = HEADER_BYTES + manifest_len
            if data_offset < manifest_end or data_offset > len(self._mmap):
                raise ValueError("CTOX manifest/data bounds are invalid")
            self.manifest = json.loads(self._mmap[HEADER_BYTES:manifest_end])
            if self.manifest.get("format") != f"ctox.q2q4.v{version}":
                raise ValueError("CTOX header and manifest versions differ")
            if int(self.manifest.get("alignment", 0)) != alignment:
                raise ValueError("CTOX header and manifest alignments differ")
            tensors = self.manifest.get("tensors", [])
            if len(tensors) != tensor_count:
                raise ValueError("CTOX header and manifest tensor counts differ")
            self.data_offset = int(data_offset)
            self.tensors = {}
            ranges = []
            for tensor in tensors:
                name = str(tensor["name"])
                if not name or name in self.tensors:
                    raise ValueError(f"empty or duplicate CTOX tensor {name}")
                offset = int(tensor["offset"])
                length = int(tensor["length"])
                if offset % alignment:
                    raise ValueError(f"CTOX tensor {name} is misaligned")
                expected = packed_bytes(
                    str(tensor["dtype"]),
                    [int(value) for value in tensor["shape"]],
                    tensor.get("segments", []),
                )
                if length != expected:
                    raise ValueError(f"CTOX tensor {name} byte length differs")
                end = self.data_offset + offset + length
                if end > len(self._mmap):
                    raise ValueError(f"CTOX tensor {name} exceeds the file")
                digest = str(tensor["sha256"])
                if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
                    raise ValueError(f"CTOX tensor {name} has an invalid SHA-256")
                self.tensors[name] = tensor
                ranges.append((offset, offset + length, name))
            ranges.sort()
            for left, right in zip(ranges, ranges[1:]):
                if left[1] > right[0]:
                    raise ValueError(f"CTOX tensors {left[2]} and {right[2]} overlap")
            if verify_tensors:
                self.verify_all_tensors()
        except Exception:
            self.close()
            raise

    def tensor_bytes(self, name: str) -> memoryview:
        tensor = self.tensors[name]
        start = self.data_offset + int(tensor["offset"])
        return memoryview(self._mmap)[start : start + int(tensor["length"])]

    def verify_tensor(self, name: str) -> None:
        payload = self.tensor_bytes(name)
        try:
            actual = hashlib.sha256(payload).hexdigest()
        finally:
            payload.release()
        if actual != str(self.tensors[name]["sha256"]).lower():
            raise ValueError(f"CTOX tensor {name} checksum differs")

    def verify_all_tensors(self) -> None:
        for name in self.tensors:
            self.verify_tensor(name)

    @staticmethod
    def _decode_blocks(torch: Any, payload: memoryview, dtype: str, device: str) -> Any:
        block_bytes = BLOCK_BYTES[dtype]
        if len(payload) % block_bytes:
            raise ValueError("packed CTOX block payload is truncated")
        encoded = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone().to(device)
        blocks = encoded.reshape(-1, block_bytes)
        scales = blocks[:, :2].contiguous().view(torch.float16).reshape(-1).float()
        packed = blocks[:, 2:]
        if dtype == "q2_b64":
            codes = torch.stack(
                tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)),
                dim=-1,
            ).reshape(-1, BLOCK)
            codebook = torch.tensor(
                [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0],
                device=device,
                dtype=torch.float32,
            )
            values = codebook[codes.long()]
        elif dtype == "q4_b64":
            codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(-1, BLOCK)
            values = codes.float() / 7.5 - 1.0
        else:
            raise ValueError(f"cannot decode non-quantized dtype {dtype}")
        return values * scales[:, None]

    def decode_matrix_rows(
        self,
        name: str,
        row_start: int,
        row_end: int,
        torch: Any,
        device: str,
    ) -> Any:
        """Decode only requested matrix rows; fixed packed codes remain canonical."""

        tensor = self.tensors[name]
        shape = [int(value) for value in tensor["shape"]]
        if len(shape) != 2 or shape[1] % BLOCK:
            raise ValueError(f"CTOX tensor {name} is not a block-aligned matrix")
        if not 0 <= row_start < row_end <= shape[0]:
            raise ValueError(f"invalid row range for CTOX tensor {name}")
        dtype = str(tensor["dtype"])
        segments = tensor.get("segments", [])
        if dtype in BLOCK_BYTES:
            segments = [
                {
                    "row_start": 0,
                    "row_end": shape[0],
                    "dtype": dtype,
                    "offset": 0,
                }
            ]
        elif dtype != "mixed_q2_q4_b64":
            raise ValueError(f"CTOX tensor {name} is not quantized")
        tensor_payload = self.tensor_bytes(name)
        decoded = []
        try:
            for segment in segments:
                start = max(row_start, int(segment["row_start"]))
                stop = min(row_end, int(segment["row_end"]))
                if start >= stop:
                    continue
                segment_dtype = str(segment["dtype"])
                blocks_per_row = shape[1] // BLOCK
                bytes_per_row = blocks_per_row * BLOCK_BYTES[segment_dtype]
                local_start = start - int(segment["row_start"])
                local_stop = stop - int(segment["row_start"])
                byte_start = int(segment["offset"]) + local_start * bytes_per_row
                byte_stop = int(segment["offset"]) + local_stop * bytes_per_row
                values = self._decode_blocks(
                    torch,
                    tensor_payload[byte_start:byte_stop],
                    segment_dtype,
                    device,
                ).reshape(stop - start, shape[1])
                decoded.append(values)
        finally:
            tensor_payload.release()
        if sum(value.shape[0] for value in decoded) != row_end - row_start:
            raise ValueError(f"CTOX mixed tensor {name} row coverage differs")
        return torch.cat(decoded, dim=0) if len(decoded) > 1 else decoded[0]

    def decode_float_tensor(self, name: str, torch: Any, device: str) -> Any:
        tensor = self.tensors[name]
        dtype_name = str(tensor["dtype"])
        if dtype_name not in {"f16", "f32"}:
            raise ValueError(f"CTOX tensor {name} is not a float tensor")
        payload = self.tensor_bytes(name)
        try:
            dtype = torch.float16 if dtype_name == "f16" else torch.float32
            values = torch.frombuffer(bytearray(payload), dtype=dtype).clone()
        finally:
            payload.release()
        return values.reshape(tuple(int(value) for value in tensor["shape"])).to(device)

    def close(self) -> None:
        if hasattr(self, "_mmap"):
            self._mmap.close()
        if hasattr(self, "_file"):
            self._file.close()

    def __enter__(self) -> "CtoxArtifact":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
