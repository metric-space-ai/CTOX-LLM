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
        raise ValueError(
            f"last dimension {values.shape[-1]} is not divisible by block {block}"
        )
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

    def __init__(
        self,
        dequantized_weight: Tensor,
        bias: Tensor | None = None,
        hadamard: bool = False,
    ):
        super().__init__()
        if dequantized_weight.ndim != 2:
            raise ValueError("weight must be a matrix")
        self.register_buffer("weight", dequantized_weight.detach(), persistent=False)
        self.register_buffer(
            "bias", bias.detach() if bias is not None else None, persistent=False
        )
        self.log_s_in = nn.Parameter(
            torch.zeros(dequantized_weight.shape[1], dtype=torch.float32)
        )
        self.log_s_out = nn.Parameter(
            torch.zeros(dequantized_weight.shape[0], dtype=torch.float32)
        )
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


def supervised_mtp_token_loss(
    student_logits: Tensor,
    input_ids: Tensor,
    mtp_positions: Tensor,
) -> Tensor:
    """Cross entropy for MTP base-hidden p -> draft token[p+2] positions."""

    positions = mtp_positions.to(device=input_ids.device, dtype=torch.long)
    if positions.ndim != 1 or student_logits.shape[1] != positions.shape[0]:
        raise ValueError("student MTP logits and MTP positions differ")
    if bool((positions < 0).any()) or bool((positions + 2 >= input_ids.shape[1]).any()):
        raise ValueError("MTP position has no p+2 token target")
    targets = input_ids.index_select(1, positions + 2)
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
    cosine = (
        1.0 - F.cosine_similarity(student_f32, teacher_f32, dim=-1).mean()
    ).clamp_min(0.0)
    return normalized_mse + 0.1 * cosine


def normalized_hidden_loss_contribution(
    student: Tensor,
    teacher: Tensor,
    teacher_square_sum: Tensor,
    total_vectors: int,
) -> Tensor:
    """One chunk's exact contribution to the full normalized hidden loss.

    Long-context recovery backpropagates one stateful prefill chunk at a time.
    Using the ordinary per-chunk mean would give short final chunks and sparse
    marker chunks disproportionate weight.  The caller therefore supplies the
    full teacher signal and vector count; summing every returned contribution
    is algebraically identical to :func:`normalized_hidden_loss` over the
    concatenated sequence, including its cosine term.
    """

    if student.shape != teacher.shape or student.ndim < 2:
        raise ValueError("student and teacher hidden shapes differ")
    vectors = int(student.numel() // student.shape[-1])
    if vectors <= 0 or total_vectors < vectors:
        raise ValueError("hidden contribution has an invalid global vector count")
    student_f32 = student.float()
    teacher_f32 = teacher.float()
    signal = teacher_square_sum.to(student_f32.device, dtype=torch.float32).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized_mse = (student_f32 - teacher_f32).square().sum() / signal
    cosine_sum = F.cosine_similarity(student_f32, teacher_f32, dim=-1).sum()
    cosine = (student_f32.new_tensor(float(vectors)) - cosine_sum) / total_vectors
    return normalized_mse + 0.1 * cosine


def streamed_sparse_target_losses(
    project: object,
    student_hidden: Tensor,
    teacher_topk_indices: Tensor,
    teacher_topk_logprobs: Tensor,
    teacher_residual_probability: Tensor,
    input_ids: Tensor,
    target_positions: Tensor,
    target_offset: int,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Project bounded hidden chunks while preserving exact mean KL and CE.

    A 248k-vocabulary dense logit tensor is too large for all assistant or MTP
    targets at once. This routine keeps only one vocabulary projection alive at
    a time and weights each chunk by its exact number of target rows.
    """

    if target_offset not in (1, 2):
        raise ValueError("target_offset must be one for base or two for MTP")
    if chunk_size <= 0:
        raise ValueError("logit chunk size must be positive")
    if (
        student_hidden.ndim != 3
        or target_positions.ndim != 1
        or student_hidden.shape[1] != target_positions.shape[0]
    ):
        raise ValueError("student hidden and target positions differ")
    if teacher_topk_indices.shape[:2] != student_hidden.shape[:2]:
        raise ValueError("teacher sparse targets and student hidden differ")
    if teacher_topk_logprobs.shape != teacher_topk_indices.shape:
        raise ValueError("teacher sparse index and probability shapes differ")
    if teacher_residual_probability.shape != student_hidden.shape[:2]:
        raise ValueError("teacher residual targets and student hidden differ")
    kl_total = student_hidden.new_zeros((), dtype=torch.float32)
    ce_total = student_hidden.new_zeros((), dtype=torch.float32)
    targets_total = 0
    for start in range(0, student_hidden.shape[1], chunk_size):
        stop = min(student_hidden.shape[1], start + chunk_size)
        logits = project(student_hidden[:, start:stop])
        positions = target_positions[start:stop]
        kl = sparse_teacher_kl(
            logits,
            teacher_topk_indices[:, start:stop],
            teacher_topk_logprobs[:, start:stop],
            teacher_residual_probability[:, start:stop],
        )
        ce = (
            supervised_next_token_loss(logits, input_ids, positions)
            if target_offset == 1
            else supervised_mtp_token_loss(logits, input_ids, positions)
        )
        targets = int(logits.shape[0] * logits.shape[1])
        kl_total = kl_total + kl * targets
        ce_total = ce_total + ce * targets
        targets_total += targets
        del logits, kl, ce
    if targets_total == 0:
        raise ValueError("streamed sparse loss has no target rows")
    return kl_total / targets_total, ce_total / targets_total


def recovery_loss_coefficients(
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    coefficients = {
        "kl": 1.0,
        "ce": 0.5,
        "hidden": 1.0,
        "mtp_kl": 0.5,
        "mtp_ce": 0.25,
        "mtp_hidden": 0.5,
    }
    if weights is not None:
        unknown = set(weights) - set(coefficients)
        if unknown:
            raise ValueError(f"unknown recovery loss weights: {sorted(unknown)}")
        coefficients.update(weights)
    if any(value < 0 for value in coefficients.values()) or not any(
        coefficients.values()
    ):
        raise ValueError("recovery loss weights must be non-negative and non-empty")
    return coefficients


def compose_recovery_losses(
    losses: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> Tensor:
    coefficients = recovery_loss_coefficients(weights)
    if set(losses) != set(coefficients):
        missing = sorted(set(coefficients) - set(losses))
        extra = sorted(set(losses) - set(coefficients))
        raise ValueError(
            f"recovery loss families differ: missing={missing}, extra={extra}"
        )
    return sum(coefficients[name] * losses[name] for name in coefficients)


def end_to_end_recovery_loss(
    student: dict[str, Tensor],
    teacher: dict[str, Tensor],
    hidden_layers: list[int],
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compose base and MTP distillation without weakening any target family."""

    losses = {
        "kl": sparse_teacher_kl(
            student["logits"],
            teacher["topk_indices"],
            teacher["topk_logprobs"],
            teacher["residual_probability"],
        ),
        "ce": supervised_next_token_loss(
            student["logits"], teacher["input_ids"], teacher["logit_positions"]
        ),
    }
    hidden_losses = [
        normalized_hidden_loss(student[f"hidden_{layer}"], teacher[f"hidden_{layer}"])
        for layer in hidden_layers
    ]
    if not hidden_losses:
        raise ValueError("end-to-end recovery requires hidden-state targets")
    losses["hidden"] = torch.stack(hidden_losses).mean()
    losses["mtp_kl"] = sparse_teacher_kl(
        student["mtp_logits"],
        teacher["mtp_topk_indices"],
        teacher["mtp_topk_logprobs"],
        teacher["mtp_residual_probability"],
    )
    losses["mtp_ce"] = supervised_mtp_token_loss(
        student["mtp_logits"], teacher["input_ids"], teacher["mtp_positions"]
    )
    losses["mtp_hidden"] = normalized_hidden_loss(
        student["mtp_hidden"], teacher["mtp_hidden"]
    )
    return compose_recovery_losses(losses, weights), losses
