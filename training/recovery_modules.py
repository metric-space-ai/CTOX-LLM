"""Trainable Escha-style channel corrections over frozen quantized weights."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def block_hadamard(values: Tensor, block: int = 128) -> Tensor:
    if block <= 0 or block & (block - 1):
        raise ValueError("Hadamard block must be a positive power of two")
    if values.shape[-1] % block:
        raise ValueError(f"last dimension {values.shape[-1]} is not divisible by block {block}")
    shaped = values.reshape(*values.shape[:-1], -1, block)
    result = shaped
    stride = 1
    while stride < block:
        result = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = result.unbind(dim=-2)
        result = torch.cat((left + right, left - right), dim=-1)
        stride *= 2
    return (result.reshape_as(values) / math.sqrt(block)).contiguous()


class ChannelScaleRecovery(nn.Module):
    """Frozen Q2/Q4 matrix with trainable input/output channel correction."""

    def __init__(self, dequantized_weight: Tensor, bias: Tensor | None = None, hadamard: bool = False):
        super().__init__()
        if dequantized_weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        self.register_buffer("weight", dequantized_weight.detach(), persistent=False)
        self.register_buffer("bias", bias.detach() if bias is not None else None, persistent=False)
        self.log_s_in = nn.Parameter(torch.zeros(dequantized_weight.shape[1], dtype=torch.float32))
        self.log_s_out = nn.Parameter(torch.zeros(dequantized_weight.shape[0], dtype=torch.float32))
        self.hadamard = hadamard

    def forward(self, values: Tensor) -> Tensor:
        corrected = values * self.log_s_in.exp().to(values.dtype)
        if self.hadamard:
            corrected = block_hadamard(corrected)
        output = F.linear(corrected, self.weight, self.bias)
        return output * self.log_s_out.exp().to(output.dtype)

    def correction_tensors(self) -> dict[str, Tensor]:
        return {
            "s_in": self.log_s_in.detach().exp().to(torch.float16).cpu(),
            "s_out": self.log_s_out.detach().exp().to(torch.float16).cpu(),
        }


def reconstruction_loss(student: Tensor, teacher: Tensor) -> Tensor:
    student_f32 = student.float()
    teacher_f32 = teacher.float()
    mse = F.mse_loss(student_f32, teacher_f32)
    cosine = 1.0 - F.cosine_similarity(student_f32, teacher_f32, dim=-1).mean()
    return mse + 0.1 * cosine
