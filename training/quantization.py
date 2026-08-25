"""Canonical Q2/Q4 block quantization shared by packers and analysis."""

from __future__ import annotations

from typing import Any


def quantize_components(torch: Any, values: Any, dtype: str) -> tuple[Any, Any]:
    blocks = values.float().reshape(-1, 64)
    scales = blocks.abs().amax(dim=1)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = blocks / safe_scales[:, None]
    if dtype == "q2_b64":
        boundaries = torch.tensor([-2.0 / 3.0, 0.0, 2.0 / 3.0], device=values.device)
        codes = torch.bucketize(normalized, boundaries).to(torch.uint8)
    elif dtype == "q4_b64":
        codes = ((normalized.clamp(-1.0, 1.0) * 7.5) + 7.5).round().to(torch.uint8)
    else:
        raise ValueError(dtype)
    codes = torch.where(scales[:, None] == 0, torch.zeros_like(codes), codes)
    return scales, codes


def dequantize_components(torch: Any, scales: Any, codes: Any, dtype: str) -> Any:
    if dtype == "q2_b64":
        codebook = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=codes.device)
        normalized = codebook[codes.long()]
    elif dtype == "q4_b64":
        normalized = codes.float() / 7.5 - 1.0
    else:
        raise ValueError(dtype)
    return normalized * scales[:, None]


def quantize_dequantize(torch: Any, values: Any, dtype: str) -> Any:
    scales, codes = quantize_components(torch, values, dtype)
    # The native artifact persists block scales as FP16. Round before
    # dequantizing so sensitivity analysis matches the deployed representation.
    stored_scales = scales.to(torch.float16).float()
    return dequantize_components(torch, stored_scales, codes, dtype).reshape_as(values)
