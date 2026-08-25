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


def sparse_teacher_kl(
    student_logits: Tensor,
    teacher_topk_indices: Tensor,
    teacher_topk_logprobs: Tensor,
    teacher_residual_probability: Tensor,
) -> Tensor:
    """KL over exact teacher top-k classes plus one residual-mass bucket."""

    if student_logits.shape[:-1] != teacher_topk_indices.shape[:-1]:
        raise ValueError("student and teacher target dimensions differ")
    if teacher_topk_indices.shape != teacher_topk_logprobs.shape:
        raise ValueError("teacher top-k index and probability shapes differ")
    if teacher_residual_probability.shape != student_logits.shape[:-1]:
        raise ValueError("teacher residual shape differs from student targets")
    student_logprobs = F.log_softmax(student_logits.float(), dim=-1)
    selected_student_logprobs = student_logprobs.gather(
        -1, teacher_topk_indices.to(torch.long)
    )
    selected_student_probability = selected_student_logprobs.exp().sum(dim=-1)
    epsilon = torch.finfo(torch.float32).eps
    student_residual_logprob = torch.log1p(
        -selected_student_probability.clamp(max=1.0 - epsilon)
    )
    teacher_logprobs = teacher_topk_logprobs.float()
    teacher_probability = teacher_logprobs.exp()
    selected_kl = (
        teacher_probability * (teacher_logprobs - selected_student_logprobs)
    ).sum(dim=-1)
    teacher_residual = teacher_residual_probability.float().clamp(min=0.0, max=1.0)
    residual_kl = teacher_residual * (
        teacher_residual.clamp_min(torch.finfo(torch.float32).tiny).log()
        - student_residual_logprob
    )
    return (selected_kl + residual_kl).mean()


def supervised_next_token_loss(
    student_logits: Tensor,
    input_ids: Tensor,
    logit_positions: Tensor,
) -> Tensor:
    """Cross entropy for the recorded p -> token[p+1] teacher positions."""

    positions = logit_positions.to(device=input_ids.device, dtype=torch.long)
    if positions.ndim != 1 or student_logits.shape[1] != positions.shape[0]:
        raise ValueError("student logits and logit positions differ")
    if bool((positions < 0).any()) or bool((positions + 1 >= input_ids.shape[1]).any()):
        raise ValueError("logit position has no next-token target")
    targets = input_ids.index_select(1, positions + 1)
    return F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
    )


def normalized_hidden_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Scale-stable hidden reconstruction with a directional penalty."""

    if student.shape != teacher.shape:
        raise ValueError("student and teacher hidden shapes differ")
    student_f32 = student.float()
    teacher_f32 = teacher.float()
    signal = teacher_f32.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
    normalized_mse = (student_f32 - teacher_f32).square().mean() / signal
    cosine = (1.0 - F.cosine_similarity(student_f32, teacher_f32, dim=-1).mean()).clamp_min(0.0)
    return normalized_mse + 0.1 * cosine
