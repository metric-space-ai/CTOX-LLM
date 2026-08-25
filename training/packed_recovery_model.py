"""Construct trainable recovery modules directly from one native CTOX artifact."""

from __future__ import annotations

from typing import Any

from packed_recovery_ops import (
    packed_recovery_embedding_class,
    packed_recovery_linear_class,
)


QUANT_DTYPES = frozenset({"q2_b64", "q4_b64", "mixed_q2_q4_b64"})


class PackedRecoveryRegistry:
    def __init__(self, artifact: Any, torch: Any) -> None:
        self.artifact = artifact
        self.torch = torch
        self.weight_names = sorted(
            name
            for name, tensor in artifact.tensors.items()
            if tensor["dtype"] in QUANT_DTYPES and name.endswith(".weight")
        )
        if not self.weight_names:
            raise ValueError("native CTOX artifact contains no quantized weight matrices")
        for name in self.weight_names:
            rows, columns = map(int, artifact.tensors[name]["shape"])
            for suffix, expected_shape in (("s_in", [columns]), ("s_out", [rows])):
                scale_name = f"{name}.{suffix}"
                scale = artifact.tensors.get(scale_name)
                if scale is None:
                    raise ValueError(f"native CTOX artifact lacks {scale_name}")
                if scale["dtype"] != "f16" or list(map(int, scale["shape"])) != expected_shape:
                    raise ValueError(f"native CTOX recovery scale contract differs for {scale_name}")

    def make_linear(self, name: str, device: str, rows_per_chunk: int = 128) -> Any:
        if name not in self.weight_names:
            raise ValueError(f"{name} is not a registered quantized CTOX weight")
        s_in = self.artifact.decode_float_tensor(f"{name}.s_in", self.torch, device)
        s_out = self.artifact.decode_float_tensor(f"{name}.s_out", self.torch, device)
        bias_name = f"{name[:-len('.weight')]}.bias"
        bias = (
            self.artifact.decode_float_tensor(bias_name, self.torch, device)
            if bias_name in self.artifact.tensors
            else None
        )
        module_class = packed_recovery_linear_class(self.torch)
        return module_class(
            self.artifact,
            name,
            s_in,
            s_out,
            bias=bias,
            rows_per_chunk=rows_per_chunk,
        ).to(device)

    def scale_parameter_count(self) -> int:
        return sum(
            int(self.artifact.tensors[f"{name}.s_in"]["shape"][0])
            + int(self.artifact.tensors[f"{name}.s_out"]["shape"][0])
            for name in self.weight_names
        )

    def make_embedding(self, name: str, device: str) -> Any:
        if name not in self.weight_names:
            raise ValueError(f"{name} is not a registered quantized CTOX weight")
        s_in = self.artifact.decode_float_tensor(f"{name}.s_in", self.torch, device)
        s_out = self.artifact.decode_float_tensor(f"{name}.s_out", self.torch, device)
        module_class = packed_recovery_embedding_class(self.torch)
        return module_class(self.artifact, name, s_in, s_out).to(device)
