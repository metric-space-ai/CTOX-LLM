"""Pinned accelerator helpers shared by BF16 teacher tooling."""

from __future__ import annotations

from typing import Any


FLA_KERNEL_REPOSITORY = "kernels-community/fla"
FLA_KERNEL_REVISION = "398dfa8cde0c51ff3c1770800d60ddd85cafcff7"


def install_pinned_fla_kernel() -> dict[str, str]:
    """Replace only Qwen's Gated-Delta functions with the pinned FLA build.

    Transformers' generic ``use_kernels=True`` also replaces causal-conv with
    a hub build. That build currently has no Torch 2.8/CUDA 12.8 variant on
    GPU3, while the environment already has a compatible causal-conv package.
    Selecting only FLA keeps the teacher runtime reproducible and fail-closed.
    """

    try:
        from kernels import get_kernel
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_modeling
    except ImportError as error:
        raise RuntimeError("install training/requirements.in for the pinned FLA kernel") from error

    kernel = get_kernel(FLA_KERNEL_REPOSITORY, revision=FLA_KERNEL_REVISION)
    required = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
    missing = [name for name in required if not hasattr(kernel, name)]
    if missing:
        raise RuntimeError(
            f"pinned FLA kernel {FLA_KERNEL_REVISION} lacks: {', '.join(missing)}"
        )
    qwen_modeling.torch_chunk_gated_delta_rule = kernel.chunk_gated_delta_rule
    qwen_modeling.torch_recurrent_gated_delta_rule = kernel.fused_recurrent_gated_delta_rule
    return {
        "repository": FLA_KERNEL_REPOSITORY,
        "revision": FLA_KERNEL_REVISION,
        "module_file": str(kernel.__file__),
    }


def weight_max_memory(
    gpu_count: int,
    gpu_weight_memory_gib: int | None,
    cpu_offload_memory_gib: int,
) -> dict[Any, str] | None:
    if gpu_weight_memory_gib is None:
        return None
    result: dict[Any, str] = {
        index: f"{gpu_weight_memory_gib}GiB" for index in range(gpu_count)
    }
    result["cpu"] = f"{cpu_offload_memory_gib}GiB"
    return result
