//! CUDA backend contract and SM86 kernel ABI baseline.
//!
//! The driver-API descriptor below pins the binary/kernel ABI that a future
//! vendored-derived CUDA module must satisfy on SM86 (Ampere, compute
//! capability 8.6) before it may leave [`PromotionState::Contract`]. Ground
//! truth for the packed tile techniques comes from the pinned llama.cpp
//! reference set under `models/qwen38_27b/vendor/cuda/` (see `UPSTREAM.json`
//! for revision and SHA-256 digests); anchors below use the form
//! `// ref: path:line-range` relative to that directory.
//!
//! Promotion stays fail-closed: no production kernel is authored here, no
//! scalar fallback exists, and execution is rejected until a same-device
//! verifier run and benchmark evidence exist per `docs/PROMOTION_GATES.md`.

use crate::backend::{
    Backend, BackendKind, FusedMatVec, PromotionState, RecoveredRow, RecoveredRowMatVec, ScaleSlice,
};
use crate::format::TensorDType;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

/// SM86 is Ampere consumer silicon (e.g. RTX 30-series), compute capability
/// 8.6. Upstream gates the mma path on Ampere and stream-k on Volta+, both
/// satisfied here.
// ref: ggml/src/ggml-cuda/mma.cuh:923-1022
// ref: ggml/src/ggml-cuda/mmq.cu:121-122
pub const SM86_COMPUTE_CAPABILITY: (u32, u32) = (8, 6);

/// NVIDIA warp width; all upstream dp4a/mma tiles assume 32 lanes.
// ref: ggml/src/ggml-cuda/mmq.cuh:143-144
pub const SM86_WARP_SIZE: u32 = 32;

/// Device ABI is LP64: every device pointer occupies 8 bytes in the
/// cuLaunchKernel parameter buffer.
pub const DEVICE_PTR_BYTES: u32 = 8;

/// One kernel parameter slot in the driver-API parameter buffer, in launch
/// order. `offset` is the byte offset from the start of the packed buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelParam {
    pub name: &'static str,
    pub size_bytes: u32,
    pub offset_bytes: u32,
}

/// ABI contract for one fused Q2/Q4 matvec kernel: the exact driver-API
/// symbol, the packed weight layout it consumes, and its parameter buffer.
///
/// The kernel must fuse input scale (`s_in`), output scale (`s_out`), bias,
/// and the activation in one launch; there is no unfused production path.
/// Both recovery-scale pointers address the original packed FP16 CTOXQ data;
/// a conforming module widens values in registers and must not require an f32
/// scale expansion at load time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelAbi {
    /// Exact `CUfunction` symbol the cubin/fatbin module must export.
    pub symbol: &'static str,
    /// Packed dtype this kernel consumes. Only Q2_B64 and Q4_B64 exist;
    /// there is deliberately no Q3 descriptor.
    pub dtype: TensorDType,
    /// Values per packed block (must equal `quant::BLOCK_LEN`).
    pub block_len: usize,
    /// Bytes per packed block, matching `quant` encode/decode.
    pub block_bytes: usize,
    /// Maximum resident threads per block for SM86 occupancy checks.
    pub max_threads_per_block: u32,
    pub params: &'static [KernelParam],
}

// Parameter buffer shared by both fused matvec kernels. Order and widths are
// fixed ABI; pointers are 8 bytes, scalars 4 bytes, no padding is permitted.
// ref: ggml/src/ggml-cuda/mmq.cuh:3955-4048 (launch-time argument staging)
const FUSED_MATVEC_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "weights",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "input",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "s_in",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "s_out",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "bias",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "rows",
        size_bytes: 4,
        offset_bytes: 48,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 52,
    },
    KernelParam {
        name: "activation",
        size_bytes: 4,
        offset_bytes: 56,
    },
];

/// Total bytes of the fused-matvec parameter buffer.
pub const FUSED_MATVEC_PARAM_BYTES: u32 = 60;

/// Q2_B64 fused matvec: 64 codes of 2 bits plus one f16 scale per block.
/// The integer accumulation stage follows the upstream dp4a pattern.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:18-32
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
pub const Q2_B64_FUSED_MATVEC: CudaKernelAbi = CudaKernelAbi {
    symbol: "ctox_q2_b64_fused_matvec_sm86",
    dtype: TensorDType::Q2B64,
    block_len: BLOCK_LEN,
    block_bytes: Q2_BLOCK_BYTES,
    max_threads_per_block: 256,
    params: FUSED_MATVEC_PARAMS,
};

/// Q4_B64 fused matvec: 64 codes of 4 bits plus one f16 scale per block.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:27-32
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
// ref: ggml/src/ggml-cuda/dequantize.cuh:25-38
pub const Q4_B64_FUSED_MATVEC: CudaKernelAbi = CudaKernelAbi {
    symbol: "ctox_q4_b64_fused_matvec_sm86",
    dtype: TensorDType::Q4B64,
    block_len: BLOCK_LEN,
    block_bytes: Q4_BLOCK_BYTES,
    max_threads_per_block: 256,
    params: FUSED_MATVEC_PARAMS,
};

/// Verifier-only explicit A8 activation quantizer and dp4a matvec symbols.
/// They are deliberately outside [`SM86_MODULE_ABI`]: production promotion
/// first requires an activation-quantization quality gate shared by every
/// backend, not merely successful symbol resolution.
// ref: ggml/src/ggml-cuda/vecdotq.cuh:115-137
pub const A8_QUANTIZE_SYMBOL: &str = "ctox_quantize_a8_b64_sm86";
pub const SWIGLU_A8_QUANTIZE_SYMBOL: &str = "ctox_quantize_swiglu_a8_b64_sm86";
pub const SIGMOID_GATE_A8_QUANTIZE_SYMBOL: &str = "ctox_quantize_sigmoid_gate_a8_b64_sm86";
pub const Q2_B64_A8_MATVEC_SYMBOL: &str = "ctox_q2_b64_a8_matvec_sm86";
pub const Q4_B64_A8_MATVEC_SYMBOL: &str = "ctox_q4_b64_a8_matvec_sm86";
pub const A8_BATCHED_QUANTIZE_SYMBOL: &str = "ctox_quantize_a8_b64_batched_sm86";
pub const Q2_B64_A8_BATCHED_MATMUL_SYMBOL: &str = "ctox_q2_b64_a8_batched_matmul_sm86";
pub const Q4_B64_A8_BATCHED_MATMUL_SYMBOL: &str = "ctox_q4_b64_a8_batched_matmul_sm86";
pub const Q2_B64_A8_BATCHED_MMQ_SYMBOL: &str = "ctox_q2_b64_a8_batched_mmq_sm86";
pub const Q4_B64_A8_BATCHED_MMQ_SYMBOL: &str = "ctox_q4_b64_a8_batched_mmq_sm86";
pub const Q2_B64_A8_GATHERED_MATVEC_SYMBOL: &str = "ctox_q2_b64_a8_gathered_matvec_sm86";
pub const Q4_B64_A8_GATHERED_MATVEC_SYMBOL: &str = "ctox_q4_b64_a8_gathered_matvec_sm86";
pub const ARGMAX_F32_SYMBOL: &str = "ctox_argmax_f32_sm86";
pub const TOPK_TOPP_SAMPLE_F32_SYMBOL: &str = "ctox_topk_topp_sample_f32_sm86";
pub const CUDA_SAMPLER_MAX_TOP_K: usize = 256;
pub const Q2_B64_RECOVERED_ROW_SYMBOL: &str = "ctox_q2_b64_recovered_row_sm86";
pub const Q4_B64_RECOVERED_ROW_SYMBOL: &str = "ctox_q4_b64_recovered_row_sm86";

pub const SWIGLU_A8_QUANTIZE_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "gate",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "up",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "s_in",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "q8_codes",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "q8_scales",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 40,
    },
];
pub const SWIGLU_A8_QUANTIZE_PARAM_BYTES: u32 = 44;
pub const SIGMOID_GATE_A8_QUANTIZE_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "attention",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "gate",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "s_in",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "q8_codes",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "q8_scales",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 40,
    },
];
pub const SIGMOID_GATE_A8_QUANTIZE_PARAM_BYTES: u32 = SWIGLU_A8_QUANTIZE_PARAM_BYTES;

/// Verifier-only Qwen GatedDeltaNet recurrence. The production module ABI
/// intentionally excludes this symbol until the FP16-state implementation
/// passes the scalar-oracle and same-device roofline gates.
// ref: ggml/src/ggml-cuda/gated_delta_net.cu:1-135
pub const GATED_DELTA_F16_SYMBOL: &str = "ctox_gated_delta_recurrent_f16_sm86";
pub const GATED_DELTA_PREP_F32_SYMBOL: &str = "ctox_qwen_gated_delta_prepare_f32_sm86";

/// Exact recurrent geometry in Qwen3.8-27B. Keeping it explicit prevents a
/// successfully compiled but semantically different dynamic shape from being
/// accepted by the verifier runtime.
pub const GATED_DELTA_HEADS: usize = 48;
pub const GATED_DELTA_KEY_DIM: usize = 128;
pub const GATED_DELTA_VALUE_DIM: usize = 128;
pub const GATED_DELTA_KEY_HEADS: usize = 16;
pub const GATED_DELTA_STATE_BYTES: usize =
    GATED_DELTA_HEADS * GATED_DELTA_KEY_DIM * GATED_DELTA_VALUE_DIM * 2;

/// Driver-API argument layout for `GATED_DELTA_F16_SYMBOL`. It is documented
/// independently of `SM86_MODULE_ABI` so candidate cubins are inspectable
/// without prematurely promoting the kernel.
pub const GATED_DELTA_F16_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "query",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "key",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "value",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "log_decay",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "beta",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "state",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 48,
    },
    KernelParam {
        name: "heads",
        size_bytes: 4,
        offset_bytes: 56,
    },
    KernelParam {
        name: "key_dim",
        size_bytes: 4,
        offset_bytes: 60,
    },
    KernelParam {
        name: "value_dim",
        size_bytes: 4,
        offset_bytes: 64,
    },
    KernelParam {
        name: "epsilon",
        size_bytes: 4,
        offset_bytes: 68,
    },
];
pub const GATED_DELTA_F16_PARAM_BYTES: u32 = 72;

/// Driver-API layout for the verifier-only Qwen input preparation kernel.
/// It repeats compact 16-head Q/K activations to 48 recurrence heads and
/// applies the exact model A/B transforms without a host round-trip.
pub const GATED_DELTA_PREP_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "convolved_qkv",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "raw_a",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "raw_b",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "a_log",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "dt_bias",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "query",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "key",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 48,
    },
    KernelParam {
        name: "log_decay",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 56,
    },
    KernelParam {
        name: "beta",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 64,
    },
    KernelParam {
        name: "key_heads",
        size_bytes: 4,
        offset_bytes: 72,
    },
    KernelParam {
        name: "value_heads",
        size_bytes: 4,
        offset_bytes: 76,
    },
    KernelParam {
        name: "key_dim",
        size_bytes: 4,
        offset_bytes: 80,
    },
];
pub const GATED_DELTA_PREP_F32_PARAM_BYTES: u32 = 84;

/// Remaining verifier-only operators in one Qwen linear-attention block.
// ref: ggml/src/ggml-cuda/ssm-conv.cu:1-95
// ref: ggml/src/ggml-cuda/norm.cu:1-148
pub const CAUSAL_CONV_F16_SYMBOL: &str = "ctox_causal_conv_silu_f16_sm86";
pub const GATED_RMS_NORM_F16_SYMBOL: &str = "ctox_gated_rms_norm_f16_sm86";
pub const QWEN_RMS_NORM_F16_SYMBOL: &str = "ctox_qwen_rms_norm_f16_sm86";
pub const RESIDUAL_RMS_NORM_F16_SYMBOL: &str = "ctox_qwen_residual_rms_norm_f16_sm86";
pub const PARTIAL_ROPE_F32_SYMBOL: &str = "ctox_partial_rope_f32_sm86";
pub const QUERY_GATE_NORM_ROPE_F32_SYMBOL: &str = "ctox_qwen_query_gate_norm_rope_f32_sm86";
pub const PACK_PAGED_KV_Q4_F32_SYMBOL: &str = "ctox_pack_paged_kv_q4_f32_sm86";
pub const DEMOTE_PAGED_KV_Q4_TO_Q2_SYMBOL: &str = "ctox_demote_paged_kv_q4_to_q2_sm86";
pub const PAGED_Q2Q4_GQA_F32_SYMBOL: &str = "ctox_paged_q2q4_gqa_decode_f32_sm86";
pub const PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_SYMBOL: &str =
    "ctox_paged_q2q4_gqa_split_partial_f32_sm86";
pub const PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_SYMBOL: &str =
    "ctox_paged_q2q4_gqa_split_combine_f32_sm86";
pub const PAGED_GQA_SPLIT_SEGMENTS: usize = 16;
pub const PAGED_GQA_SPLIT_MAX_QUERY_TOKENS: usize = 5;
pub const LINEAR_CONV_CHANNELS: usize = 10_240;
pub const LINEAR_CONV_KERNEL_WIDTH: usize = 4;
pub const LINEAR_CONV_STATE_BYTES: usize = LINEAR_CONV_CHANNELS * LINEAR_CONV_KERNEL_WIDTH * 2;
pub const GATED_RMS_NORM_ROWS: usize = 48;
pub const GATED_RMS_NORM_COLUMNS: usize = 128;

pub const CAUSAL_CONV_F16_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "input",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "weight",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "state",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "channels",
        size_bytes: 4,
        offset_bytes: 32,
    },
    KernelParam {
        name: "kernel_width",
        size_bytes: 4,
        offset_bytes: 36,
    },
];
pub const CAUSAL_CONV_F16_PARAM_BYTES: u32 = 40;

pub const GATED_RMS_NORM_F16_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "input",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "gate",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "weight",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "rows",
        size_bytes: 4,
        offset_bytes: 32,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 36,
    },
    KernelParam {
        name: "epsilon",
        size_bytes: 4,
        offset_bytes: 40,
    },
];
pub const GATED_RMS_NORM_F16_PARAM_BYTES: u32 = 44;

pub const QWEN_RMS_NORM_F16_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "input",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "weight",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "rows",
        size_bytes: 4,
        offset_bytes: 24,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 28,
    },
    KernelParam {
        name: "epsilon",
        size_bytes: 4,
        offset_bytes: 32,
    },
];
pub const QWEN_RMS_NORM_F16_PARAM_BYTES: u32 = 36;

pub const RESIDUAL_RMS_NORM_F16_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "residual",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "update",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "weight",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "residual_output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "normalized_output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "rows",
        size_bytes: 4,
        offset_bytes: 40,
    },
    KernelParam {
        name: "columns",
        size_bytes: 4,
        offset_bytes: 44,
    },
    KernelParam {
        name: "epsilon",
        size_bytes: 4,
        offset_bytes: 48,
    },
];
pub const RESIDUAL_RMS_NORM_F16_PARAM_BYTES: u32 = 52;

pub const PARTIAL_ROPE_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "values",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "cosine",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "sine",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "heads",
        size_bytes: 4,
        offset_bytes: 24,
    },
    KernelParam {
        name: "head_dim",
        size_bytes: 4,
        offset_bytes: 28,
    },
    KernelParam {
        name: "rotary_dim",
        size_bytes: 4,
        offset_bytes: 32,
    },
];
pub const PARTIAL_ROPE_F32_PARAM_BYTES: u32 = 36;

pub const QUERY_GATE_NORM_ROPE_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "query_gate",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "q_norm_weight",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "cosine",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "sine",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "query",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "gate",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "heads",
        size_bytes: 4,
        offset_bytes: 48,
    },
    KernelParam {
        name: "head_dim",
        size_bytes: 4,
        offset_bytes: 52,
    },
    KernelParam {
        name: "rotary_dim",
        size_bytes: 4,
        offset_bytes: 56,
    },
    KernelParam {
        name: "epsilon",
        size_bytes: 4,
        offset_bytes: 60,
    },
];
pub const QUERY_GATE_NORM_ROPE_F32_PARAM_BYTES: u32 = 64;

pub const PAGED_GQA_DESCRIPTOR_BYTES: usize = 16;
pub const PAGED_GQA_PARAMS_BYTES: usize = 48;
pub const PACK_PAGED_KV_Q4_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "q4_pages",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "key",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "value",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "physical_slot",
        size_bytes: 4,
        offset_bytes: 24,
    },
    KernelParam {
        name: "token_in_page",
        size_bytes: 4,
        offset_bytes: 28,
    },
    KernelParam {
        name: "component_values",
        size_bytes: 4,
        offset_bytes: 32,
    },
    KernelParam {
        name: "q4_token_bytes",
        size_bytes: 4,
        offset_bytes: 36,
    },
    KernelParam {
        name: "q4_page_bytes",
        size_bytes: 4,
        offset_bytes: 40,
    },
];
pub const PACK_PAGED_KV_Q4_F32_PARAM_BYTES: u32 = 44;

pub const DEMOTE_PAGED_KV_Q4_TO_Q2_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "q4_page",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "q2_page",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "tokens",
        size_bytes: 4,
        offset_bytes: 16,
    },
    KernelParam {
        name: "blocks_per_token",
        size_bytes: 4,
        offset_bytes: 20,
    },
];
pub const DEMOTE_PAGED_KV_Q4_TO_Q2_PARAM_BYTES: u32 = 24;

pub const PAGED_Q2Q4_GQA_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "query",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "q2_pages",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "q4_pages",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "descriptors",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "params",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
];
pub const PAGED_Q2Q4_GQA_F32_PARAM_BYTES: u32 = 48;

pub const PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "query",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "q2_pages",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "q4_pages",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "descriptors",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "partial_output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "partial_maximum",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 40,
    },
    KernelParam {
        name: "partial_denominator",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 48,
    },
    KernelParam {
        name: "params",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 56,
    },
    KernelParam {
        name: "query_tokens",
        size_bytes: 4,
        offset_bytes: 64,
    },
    KernelParam {
        name: "segments",
        size_bytes: 4,
        offset_bytes: 68,
    },
];
pub const PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_PARAM_BYTES: u32 = 72;

pub const PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_PARAMS: &[KernelParam] = &[
    KernelParam {
        name: "partial_output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 0,
    },
    KernelParam {
        name: "partial_maximum",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 8,
    },
    KernelParam {
        name: "partial_denominator",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 16,
    },
    KernelParam {
        name: "output",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 24,
    },
    KernelParam {
        name: "params",
        size_bytes: DEVICE_PTR_BYTES,
        offset_bytes: 32,
    },
    KernelParam {
        name: "query_tokens",
        size_bytes: 4,
        offset_bytes: 40,
    },
    KernelParam {
        name: "segments",
        size_bytes: 4,
        offset_bytes: 44,
    },
];
pub const PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_PARAM_BYTES: u32 = 48;

/// Module-level ABI contract for the SM86 kernel image: the compute
/// capability the cubin must target and every kernel it must export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaModuleAbi {
    pub compute_capability: (u32, u32),
    pub kernels: &'static [CudaKernelAbi],
}

/// The single supported module ABI. Additional architectures require their
/// own descriptor and their own verifier/benchmark evidence.
pub const SM86_MODULE_ABI: CudaModuleAbi = CudaModuleAbi {
    compute_capability: SM86_COMPUTE_CAPABILITY,
    kernels: &[Q2_B64_FUSED_MATVEC, Q4_B64_FUSED_MATVEC],
};

/// One validated launch over a homogeneous row range inside a canonical
/// `MixedQ2Q4B64` payload. Offsets always refer to the existing packed tensor;
/// validation never creates backend-specific weight codes or repacks bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaMixedRowSegment {
    pub descriptor: &'static CudaKernelAbi,
    pub row_start: u32,
    pub row_count: u32,
    pub weight_offset: usize,
}

fn validate_common_operation(operation: &FusedMatVec<'_>) -> Result<usize> {
    if operation.rows == 0 || operation.columns == 0 || !operation.columns.is_multiple_of(BLOCK_LEN)
    {
        return Err(EngineError::Shape(
            "CUDA fused matvec dimensions must be non-zero and columns divisible by 64".into(),
        ));
    }
    if operation.input.len() != operation.columns {
        return Err(EngineError::Shape(format!(
            "input has {} values, expected {}",
            operation.input.len(),
            operation.columns
        )));
    }
    for (name, scales, expected) in [
        ("s_in", operation.s_in, operation.columns),
        ("s_out", operation.s_out, operation.rows),
    ] {
        if let Some(scales) = scales {
            if !matches!(scales, ScaleSlice::F16Le(_)) {
                return Err(EngineError::UnsupportedDType(format!(
                    "CUDA {name} recovery scales must remain packed FP16"
                )));
            }
            if scales.len() != expected {
                return Err(EngineError::Shape(format!(
                    "{name} has {} values, expected {expected}",
                    scales.len()
                )));
            }
        }
    }
    if operation
        .bias
        .is_some_and(|values| values.len() != operation.rows)
    {
        return Err(EngineError::Shape("bias length differs from rows".into()));
    }
    u32::try_from(operation.rows)
        .map_err(|_| EngineError::Shape("rows exceed CUDA u32 launch limit".into()))?;
    u32::try_from(operation.columns)
        .map_err(|_| EngineError::Shape("columns exceed CUDA u32 launch limit".into()))?;
    Ok(operation.columns / BLOCK_LEN)
}

/// Validates a pure Q2_B64/Q4_B64 operation before the verifier-only CUDA
/// runtime allocates device memory. Mixed matrices use the separate validated
/// segment path below; neither path repacks or silently widens weights.
pub fn validate_operation(operation: &FusedMatVec<'_>) -> Result<&'static CudaKernelAbi> {
    let descriptor = SM86_MODULE_ABI.descriptor_for(operation.dtype)?;
    let blocks_per_row = validate_common_operation(operation)?;
    if !operation.segments.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "pure CUDA Q2/Q4 operation declares mixed row segments".into(),
        ));
    }
    let expected_weights = operation
        .rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(descriptor.block_bytes))
        .ok_or_else(|| EngineError::Shape("CUDA weight buffer size overflows usize".into()))?;
    if operation.weights.len() != expected_weights {
        return Err(EngineError::Shape(format!(
            "weight buffer has {} bytes, expected {expected_weights}",
            operation.weights.len()
        )));
    }
    Ok(descriptor)
}

/// Validates the exact manifest row groups of a mixed Q2/Q4 projection and
/// returns launch metadata into the original packed tensor. The resulting
/// segments cover every row and byte exactly once.
pub(crate) fn validate_mixed_operation(
    operation: &FusedMatVec<'_>,
) -> Result<Vec<CudaMixedRowSegment>> {
    if operation.dtype != TensorDType::MixedQ2Q4B64 {
        return Err(EngineError::UnsupportedDType(format!(
            "mixed CUDA validation requires MixedQ2Q4B64, got {:?}",
            operation.dtype
        )));
    }
    let blocks_per_row = validate_common_operation(operation)?;
    if operation.segments.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "mixed CUDA Q2/Q4 operation has no row segments".into(),
        ));
    }
    let mut launches = Vec::with_capacity(operation.segments.len());
    let mut expected_row = 0_usize;
    let mut expected_offset = 0_usize;
    for (expected_group, segment) in operation.segments.iter().enumerate() {
        let group_index = usize::try_from(segment.group_index).map_err(|_| {
            EngineError::InvalidArtifact("mixed CUDA group index overflows usize".into())
        })?;
        let row_start = usize::try_from(segment.row_start).map_err(|_| {
            EngineError::InvalidArtifact("mixed CUDA row start overflows usize".into())
        })?;
        let row_end = usize::try_from(segment.row_end).map_err(|_| {
            EngineError::InvalidArtifact("mixed CUDA row end overflows usize".into())
        })?;
        let offset = usize::try_from(segment.offset).map_err(|_| {
            EngineError::InvalidArtifact("mixed CUDA segment offset overflows usize".into())
        })?;
        let length = usize::try_from(segment.length).map_err(|_| {
            EngineError::InvalidArtifact("mixed CUDA segment length overflows usize".into())
        })?;
        if group_index != expected_group
            || row_start != expected_row
            || row_end <= row_start
            || row_end > operation.rows
            || offset != expected_offset
        {
            return Err(EngineError::InvalidArtifact(format!(
                "mixed CUDA Q2/Q4 operation has non-contiguous segment {}",
                segment.group_index
            )));
        }
        let descriptor = SM86_MODULE_ABI.descriptor_for(segment.dtype).map_err(|_| {
            EngineError::InvalidArtifact(format!(
                "mixed CUDA segment {} has invalid dtype {:?}",
                segment.group_index, segment.dtype
            ))
        })?;
        let expected_length = row_end
            .checked_sub(row_start)
            .and_then(|rows| rows.checked_mul(blocks_per_row))
            .and_then(|blocks| blocks.checked_mul(descriptor.block_bytes))
            .ok_or_else(|| EngineError::Shape("mixed CUDA segment size overflows usize".into()))?;
        if length != expected_length {
            return Err(EngineError::InvalidArtifact(format!(
                "mixed CUDA segment {} has {length} bytes, expected {expected_length}",
                segment.group_index
            )));
        }
        launches.push(CudaMixedRowSegment {
            descriptor,
            row_start: u32::try_from(row_start)
                .map_err(|_| EngineError::Shape("mixed CUDA row start exceeds u32".into()))?,
            row_count: u32::try_from(row_end - row_start)
                .map_err(|_| EngineError::Shape("mixed CUDA row count exceeds u32".into()))?,
            weight_offset: offset,
        });
        expected_row = row_end;
        expected_offset = expected_offset
            .checked_add(length)
            .ok_or_else(|| EngineError::Shape("mixed CUDA weight size overflows usize".into()))?;
    }
    if expected_row != operation.rows || expected_offset != operation.weights.len() {
        return Err(EngineError::Shape(format!(
            "mixed CUDA segments cover {expected_row}/{} rows and {expected_offset}/{} bytes",
            operation.rows,
            operation.weights.len()
        )));
    }
    Ok(launches)
}

pub(crate) fn validate_recovered_row(
    operation: &RecoveredRow<'_>,
) -> Result<&'static CudaKernelAbi> {
    let descriptor = SM86_MODULE_ABI.descriptor_for(operation.dtype)?;
    if operation.columns == 0 || !operation.columns.is_multiple_of(BLOCK_LEN) {
        return Err(EngineError::Shape(
            "CUDA recovered row columns must be non-zero and divisible by 64".into(),
        ));
    }
    let expected_weights = operation
        .columns
        .checked_div(BLOCK_LEN)
        .and_then(|blocks| blocks.checked_mul(descriptor.block_bytes))
        .ok_or_else(|| EngineError::Shape("CUDA recovered row size overflows usize".into()))?;
    if operation.weights.len() != expected_weights {
        return Err(EngineError::Shape(format!(
            "CUDA recovered row has {} weight bytes, expected {expected_weights}",
            operation.weights.len()
        )));
    }
    if !matches!(operation.s_in, ScaleSlice::F16Le(_)) {
        return Err(EngineError::UnsupportedDType(
            "CUDA recovered-row s_in must remain packed FP16".into(),
        ));
    }
    if operation.s_in.len() != operation.columns {
        return Err(EngineError::Shape(format!(
            "CUDA recovered-row s_in has {} values, expected {}",
            operation.s_in.len(),
            operation.columns
        )));
    }
    if !operation.s_out.is_finite() {
        return Err(EngineError::InvalidArtifact(
            "CUDA recovered-row s_out is non-finite".into(),
        ));
    }
    u32::try_from(operation.columns)
        .map_err(|_| EngineError::Shape("recovered-row columns exceed CUDA u32".into()))?;
    Ok(descriptor)
}

impl CudaModuleAbi {
    /// Every `CUfunction` symbol a conforming module must export.
    pub fn expected_symbols(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.kernels.iter().map(|kernel| kernel.symbol)
    }

    /// Selects the kernel descriptor for a packed dtype, or fails closed for
    /// anything that is not Q2_B64/Q4_B64.
    pub fn descriptor_for(&self, dtype: TensorDType) -> Result<&'static CudaKernelAbi> {
        self.kernels
            .iter()
            .find(|kernel| kernel.dtype == dtype)
            .ok_or(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "fused matvec profile selection",
                reason: format!("no SM86 kernel descriptor for dtype {dtype:?}"),
            })
    }

    /// Fail-closed validation of a candidate module against this ABI.
    ///
    /// `reported_cc` is the compute capability baked into the image and
    /// `exported_symbols` is the `CUfunction` symbol table the driver
    /// resolved. Any mismatch is a hard error; there is no degraded mode.
    pub fn validate_module(
        &self,
        reported_cc: (u32, u32),
        exported_symbols: &[&str],
    ) -> Result<()> {
        if reported_cc != self.compute_capability {
            return Err(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "module ABI validation",
                reason: format!(
                    "module targets compute capability {}.{}, required {}.{} for this profile",
                    reported_cc.0,
                    reported_cc.1,
                    self.compute_capability.0,
                    self.compute_capability.1
                ),
            });
        }
        for kernel in self.kernels {
            if kernel.block_len != BLOCK_LEN {
                return Err(EngineError::InvalidArtifact(format!(
                    "kernel {} block length {} does not match quant layout {BLOCK_LEN}",
                    kernel.symbol, kernel.block_len
                )));
            }
            let expected_block_bytes = match kernel.dtype {
                TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                other => {
                    return Err(EngineError::UnsupportedDType(format!(
                        "kernel {} declares unsupported dtype {other:?}",
                        kernel.symbol
                    )))
                }
            };
            if kernel.block_bytes != expected_block_bytes {
                return Err(EngineError::InvalidArtifact(format!(
                    "kernel {} block size {} does not match quant layout {expected_block_bytes}",
                    kernel.symbol, kernel.block_bytes
                )));
            }
            validate_param_layout(kernel)?;
            if !exported_symbols.contains(&kernel.symbol) {
                return Err(EngineError::UnsupportedOperation {
                    backend: "cuda",
                    operation: "module ABI validation",
                    reason: format!("module does not export required symbol {}", kernel.symbol),
                });
            }
        }
        Ok(())
    }
}

/// Parameter buffers must be tightly packed and self-consistent; a sloppy
/// layout would silently corrupt launches, so it is rejected outright.
fn validate_param_layout(kernel: &CudaKernelAbi) -> Result<()> {
    let mut cursor = 0_u32;
    for param in kernel.params {
        if param.offset_bytes != cursor {
            return Err(EngineError::InvalidArtifact(format!(
                "kernel {} parameter {} has offset {}, expected {cursor}",
                kernel.symbol, param.name, param.offset_bytes
            )));
        }
        cursor += param.size_bytes;
    }
    if cursor != FUSED_MATVEC_PARAM_BYTES {
        return Err(EngineError::InvalidArtifact(format!(
            "kernel {} parameter buffer is {cursor} bytes, expected {FUSED_MATVEC_PARAM_BYTES}",
            kernel.symbol
        )));
    }
    Ok(())
}

/// CUDA contract. Kernel promotion remains fail-closed until vendored kernels,
/// immutable pins, per-op verifiers, and SM86 benchmark evidence land together.
pub struct CudaBackend;

impl CudaBackend {
    /// ABI this backend validates candidate modules against.
    pub fn module_abi(&self) -> CudaModuleAbi {
        SM86_MODULE_ABI
    }
}

impl Backend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
    }

    fn promotion_state(&self) -> PromotionState {
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "cuda-contract"
    }

    fn fused_matvec(&self, _operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "q2/q4 fused matvec",
            reason: "kernel has not passed the SM86 verifier and benchmark gates".into(),
        })
    }

    fn recovered_row(&self, _operation: &RecoveredRow<'_>) -> Result<Vec<f32>> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "Q2/Q4 recovered row gather",
            reason: "embedding gather kernel has not passed the SM86 verifier and benchmark gates"
                .into(),
        })
    }

    fn recovered_row_matvec(&self, _operation: &RecoveredRowMatVec<'_>) -> Result<f32> {
        Err(EngineError::UnsupportedOperation {
            backend: "cuda",
            operation: "recovered restricted LM-head row matvec",
            reason: "the gathered Q2/Q4 SM86 proposal kernel is not promoted".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Activation, Backend, ScaleSlice};
    use crate::quant::{Q2Block64, Q4Block64};
    use half::f16;

    fn packed_weights(dtype: TensorDType, rows: usize, columns: usize) -> Vec<u8> {
        let mut packed = Vec::new();
        for block in 0..rows * columns / BLOCK_LEN {
            let values: [f32; BLOCK_LEN] =
                std::array::from_fn(|index| ((block * BLOCK_LEN + index) as f32 * 0.017).sin());
            match dtype {
                TensorDType::Q2B64 => packed.extend_from_slice(
                    &Q2Block64::quantize(&values)
                        .expect("finite Q2 fixture")
                        .encode(),
                ),
                TensorDType::Q4B64 => packed.extend_from_slice(
                    &Q4Block64::quantize(&values)
                        .expect("finite Q4 fixture")
                        .encode(),
                ),
                _ => unreachable!(),
            }
        }
        packed
    }

    fn f16_bytes(len: usize) -> Vec<u8> {
        (0..len)
            .flat_map(|index| {
                f16::from_f32(0.9 + (index % 7) as f32 * 0.01)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect()
    }

    #[test]
    fn profile_selection_maps_q2_and_q4() {
        let abi = SM86_MODULE_ABI;
        let q2 = abi.descriptor_for(TensorDType::Q2B64).unwrap();
        let q4 = abi.descriptor_for(TensorDType::Q4B64).unwrap();
        assert_eq!(q2.symbol, "ctox_q2_b64_fused_matvec_sm86");
        assert_eq!(q4.symbol, "ctox_q4_b64_fused_matvec_sm86");
        assert_eq!(q2.block_bytes, Q2_BLOCK_BYTES);
        assert_eq!(q4.block_bytes, Q4_BLOCK_BYTES);
        assert_ne!(q2.symbol, q4.symbol);
    }

    #[test]
    fn profile_selection_rejects_non_packed_dtypes() {
        let abi = SM86_MODULE_ABI;
        assert!(abi.descriptor_for(TensorDType::F16).is_err());
        assert!(abi.descriptor_for(TensorDType::F32).is_err());
    }

    #[test]
    fn abi_metadata_targets_sm86() {
        let abi = SM86_MODULE_ABI;
        assert_eq!(abi.compute_capability, (8, 6));
        assert_eq!(SM86_WARP_SIZE, 32);
        assert_eq!(abi.kernels.len(), 2);
        let symbols: Vec<_> = abi.expected_symbols().collect();
        assert_eq!(symbols.len(), 2);
        assert!(!symbols.contains(&A8_QUANTIZE_SYMBOL));
        assert!(!symbols.contains(&SWIGLU_A8_QUANTIZE_SYMBOL));
        assert!(!symbols.contains(&SIGMOID_GATE_A8_QUANTIZE_SYMBOL));
        assert!(!symbols.contains(&Q2_B64_A8_MATVEC_SYMBOL));
        assert!(!symbols.contains(&Q4_B64_A8_MATVEC_SYMBOL));
        assert!(!symbols.contains(&A8_BATCHED_QUANTIZE_SYMBOL));
        assert!(!symbols.contains(&Q2_B64_A8_BATCHED_MATMUL_SYMBOL));
        assert!(!symbols.contains(&Q4_B64_A8_BATCHED_MATMUL_SYMBOL));
        assert!(!symbols.contains(&Q2_B64_A8_BATCHED_MMQ_SYMBOL));
        assert!(!symbols.contains(&Q4_B64_A8_BATCHED_MMQ_SYMBOL));
        assert!(!symbols.contains(&TOPK_TOPP_SAMPLE_F32_SYMBOL));
        assert!(!symbols.contains(&Q2_B64_RECOVERED_ROW_SYMBOL));
        assert!(!symbols.contains(&Q4_B64_RECOVERED_ROW_SYMBOL));
        for kernel in abi.kernels {
            assert_eq!(kernel.block_len, BLOCK_LEN);
            assert_eq!(kernel.max_threads_per_block % SM86_WARP_SIZE, 0);
            let end = kernel.params.last().unwrap();
            assert_eq!(
                end.offset_bytes + end.size_bytes,
                FUSED_MATVEC_PARAM_BYTES,
                "parameter buffer must be tightly packed"
            );
        }
    }

    #[test]
    fn validates_a_complete_module() {
        let symbols = [
            "ctox_q2_b64_fused_matvec_sm86",
            "ctox_q4_b64_fused_matvec_sm86",
        ];
        SM86_MODULE_ABI.validate_module((8, 6), &symbols).unwrap();
    }

    #[test]
    fn gated_delta_candidate_pins_exact_qwen_geometry_and_driver_abi() {
        assert_eq!(GATED_DELTA_HEADS, 48);
        assert_eq!(GATED_DELTA_KEY_DIM, 128);
        assert_eq!(GATED_DELTA_VALUE_DIM, 128);
        assert_eq!(GATED_DELTA_KEY_HEADS, 16);
        assert_eq!(GATED_DELTA_STATE_BYTES, 1_572_864);
        assert_eq!(GATED_DELTA_F16_PARAMS.len(), 11);
        assert_eq!(GATED_DELTA_F16_PARAMS[0].offset_bytes, 0);
        assert_eq!(GATED_DELTA_F16_PARAMS[6].offset_bytes, 48);
        assert_eq!(GATED_DELTA_F16_PARAMS[10].offset_bytes, 68);
        assert_eq!(GATED_DELTA_F16_PARAM_BYTES, 72);
        assert_eq!(GATED_DELTA_PREP_F32_PARAMS.len(), 12);
        assert_eq!(GATED_DELTA_PREP_F32_PARAMS[11].offset_bytes, 80);
        assert_eq!(GATED_DELTA_PREP_F32_PARAM_BYTES, 84);
        assert!(!SM86_MODULE_ABI
            .kernels
            .iter()
            .any(|kernel| kernel.symbol == GATED_DELTA_F16_SYMBOL));
        assert!(!SM86_MODULE_ABI
            .kernels
            .iter()
            .any(|kernel| kernel.symbol == GATED_DELTA_PREP_F32_SYMBOL));
    }

    #[test]
    fn swiglu_a8_candidate_has_tightly_packed_driver_abi() {
        assert_eq!(SWIGLU_A8_QUANTIZE_PARAMS.len(), 6);
        assert_eq!(SWIGLU_A8_QUANTIZE_PARAMS[5].offset_bytes, 40);
        assert_eq!(SWIGLU_A8_QUANTIZE_PARAM_BYTES, 44);
        assert!(!SM86_MODULE_ABI
            .kernels
            .iter()
            .any(|kernel| kernel.symbol == SWIGLU_A8_QUANTIZE_SYMBOL));
    }

    #[test]
    fn sigmoid_gate_a8_candidate_shares_the_fused_binary_layout() {
        assert_eq!(SIGMOID_GATE_A8_QUANTIZE_PARAMS.len(), 6);
        assert_eq!(SIGMOID_GATE_A8_QUANTIZE_PARAM_BYTES, 44);
        assert!(!SM86_MODULE_ABI
            .kernels
            .iter()
            .any(|kernel| kernel.symbol == SIGMOID_GATE_A8_QUANTIZE_SYMBOL));
    }

    #[test]
    fn linear_attention_helpers_pin_exact_qwen_geometry_and_stay_unpromoted() {
        assert_eq!(LINEAR_CONV_CHANNELS, 10_240);
        assert_eq!(LINEAR_CONV_KERNEL_WIDTH, 4);
        assert_eq!(LINEAR_CONV_STATE_BYTES, 81_920);
        assert_eq!(GATED_RMS_NORM_ROWS, 48);
        assert_eq!(GATED_RMS_NORM_COLUMNS, 128);
        assert_eq!(CAUSAL_CONV_F16_PARAMS.len(), 6);
        assert_eq!(CAUSAL_CONV_F16_PARAM_BYTES, 40);
        assert_eq!(GATED_RMS_NORM_F16_PARAMS.len(), 7);
        assert_eq!(GATED_RMS_NORM_F16_PARAM_BYTES, 44);
        assert_eq!(QWEN_RMS_NORM_F16_PARAMS.len(), 6);
        assert_eq!(QWEN_RMS_NORM_F16_PARAM_BYTES, 36);
        assert_eq!(RESIDUAL_RMS_NORM_F16_PARAMS.len(), 8);
        assert_eq!(RESIDUAL_RMS_NORM_F16_PARAMS[7].offset_bytes, 48);
        assert_eq!(RESIDUAL_RMS_NORM_F16_PARAM_BYTES, 52);
        assert_eq!(PARTIAL_ROPE_F32_PARAMS.len(), 6);
        assert_eq!(PARTIAL_ROPE_F32_PARAM_BYTES, 36);
        assert_eq!(QUERY_GATE_NORM_ROPE_F32_PARAMS.len(), 10);
        assert_eq!(QUERY_GATE_NORM_ROPE_F32_PARAM_BYTES, 64);
        assert_eq!(PAGED_GQA_DESCRIPTOR_BYTES, 16);
        assert_eq!(PAGED_GQA_PARAMS_BYTES, 48);
        assert_eq!(PACK_PAGED_KV_Q4_F32_PARAMS.len(), 8);
        assert_eq!(PACK_PAGED_KV_Q4_F32_PARAMS[7].offset_bytes, 40);
        assert_eq!(PACK_PAGED_KV_Q4_F32_PARAM_BYTES, 44);
        assert_eq!(DEMOTE_PAGED_KV_Q4_TO_Q2_PARAMS.len(), 4);
        assert_eq!(DEMOTE_PAGED_KV_Q4_TO_Q2_PARAMS[3].offset_bytes, 20);
        assert_eq!(DEMOTE_PAGED_KV_Q4_TO_Q2_PARAM_BYTES, 24);
        assert_eq!(PAGED_Q2Q4_GQA_F32_PARAMS.len(), 6);
        assert_eq!(PAGED_Q2Q4_GQA_F32_PARAMS[5].offset_bytes, 40);
        assert_eq!(PAGED_Q2Q4_GQA_F32_PARAM_BYTES, 48);
        assert_eq!(PAGED_GQA_SPLIT_SEGMENTS, 16);
        assert_eq!(PAGED_GQA_SPLIT_MAX_QUERY_TOKENS, 5);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_PARAMS.len(), 10);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_PARAMS[9].offset_bytes, 68);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_PARAM_BYTES, 72);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_PARAMS.len(), 7);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_PARAMS[6].offset_bytes, 44);
        assert_eq!(PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_PARAM_BYTES, 48);
        for symbol in [
            CAUSAL_CONV_F16_SYMBOL,
            GATED_RMS_NORM_F16_SYMBOL,
            QWEN_RMS_NORM_F16_SYMBOL,
            RESIDUAL_RMS_NORM_F16_SYMBOL,
            PARTIAL_ROPE_F32_SYMBOL,
            QUERY_GATE_NORM_ROPE_F32_SYMBOL,
            PACK_PAGED_KV_Q4_F32_SYMBOL,
            DEMOTE_PAGED_KV_Q4_TO_Q2_SYMBOL,
            PAGED_Q2Q4_GQA_F32_SYMBOL,
            PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_SYMBOL,
            PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_SYMBOL,
        ] {
            assert!(!SM86_MODULE_ABI
                .kernels
                .iter()
                .any(|kernel| kernel.symbol == symbol));
        }
    }

    #[test]
    fn rejects_missing_symbols() {
        let only_q2 = ["ctox_q2_b64_fused_matvec_sm86"];
        let err = SM86_MODULE_ABI
            .validate_module((8, 6), &only_q2)
            .unwrap_err();
        assert!(err.to_string().contains("ctox_q4_b64_fused_matvec_sm86"));

        let err = SM86_MODULE_ABI.validate_module((8, 6), &[]).unwrap_err();
        assert!(err.to_string().contains("ctox_q2_b64_fused_matvec_sm86"));
    }

    #[test]
    fn rejects_wrong_compute_capability() {
        let symbols = [
            "ctox_q2_b64_fused_matvec_sm86",
            "ctox_q4_b64_fused_matvec_sm86",
        ];
        assert!(SM86_MODULE_ABI.validate_module((8, 0), &symbols).is_err());
        assert!(SM86_MODULE_ABI.validate_module((9, 0), &symbols).is_err());
        assert!(SM86_MODULE_ABI.validate_module((7, 5), &symbols).is_err());
    }

    #[test]
    fn operation_validation_accepts_exact_packed_q2_and_q4() {
        for dtype in [TensorDType::Q2B64, TensorDType::Q4B64] {
            let rows = 5;
            let columns = 128;
            let weights = packed_weights(dtype, rows, columns);
            let input = vec![0.25_f32; columns];
            let s_in = f16_bytes(columns);
            let s_out = f16_bytes(rows);
            let bias = vec![0.0_f32; rows];
            let operation = FusedMatVec {
                dtype,
                weights: &weights,
                segments: &[],
                rows,
                columns,
                input: &input,
                s_in: Some(ScaleSlice::F16Le(&s_in)),
                s_out: Some(ScaleSlice::F16Le(&s_out)),
                bias: Some(&bias),
                activation: Activation::Silu,
            };
            let descriptor = validate_operation(&operation).unwrap();
            assert_eq!(descriptor.dtype, dtype);
        }
    }

    #[test]
    fn mixed_validation_preserves_manifest_offsets_and_row_groups() {
        let rows = 5;
        let columns = 128;
        let q2 = packed_weights(TensorDType::Q2B64, 3, columns);
        let q4 = packed_weights(TensorDType::Q4B64, 2, columns);
        let mut weights = q2.clone();
        weights.extend_from_slice(&q4);
        let segments = [
            crate::format::QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: 3,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2.len() as u64,
            },
            crate::format::QuantSegment {
                group_index: 1,
                row_start: 3,
                row_end: 5,
                dtype: TensorDType::Q4B64,
                offset: q2.len() as u64,
                length: q4.len() as u64,
            },
        ];
        let input = vec![0.25_f32; columns];
        let s_in = f16_bytes(columns);
        let s_out = f16_bytes(rows);
        let bias = vec![0.0_f32; rows];
        let operation = FusedMatVec {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F16Le(&s_in)),
            s_out: Some(ScaleSlice::F16Le(&s_out)),
            bias: Some(&bias),
            activation: Activation::Silu,
        };
        let launches = validate_mixed_operation(&operation).unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].descriptor.dtype, TensorDType::Q2B64);
        assert_eq!(launches[0].row_start, 0);
        assert_eq!(launches[0].row_count, 3);
        assert_eq!(launches[0].weight_offset, 0);
        assert_eq!(launches[1].descriptor.dtype, TensorDType::Q4B64);
        assert_eq!(launches[1].row_start, 3);
        assert_eq!(launches[1].row_count, 2);
        assert_eq!(launches[1].weight_offset, q2.len());
    }

    #[test]
    fn mixed_validation_rejects_a_gap_before_device_allocation() {
        let rows = 2;
        let columns = 64;
        let weights = packed_weights(TensorDType::Q2B64, rows, columns);
        let segments = [crate::format::QuantSegment {
            group_index: 0,
            row_start: 1,
            row_end: 2,
            dtype: TensorDType::Q2B64,
            offset: 0,
            length: Q2_BLOCK_BYTES as u64,
        }];
        let input = vec![0.0_f32; columns];
        let operation = FusedMatVec {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        assert!(matches!(
            validate_mixed_operation(&operation),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn recovered_row_validation_binds_one_packed_row_and_fp16_scales() {
        for dtype in [TensorDType::Q2B64, TensorDType::Q4B64] {
            let columns = 128;
            let weights = packed_weights(dtype, 1, columns);
            let s_in = f16_bytes(columns);
            let operation = RecoveredRow {
                dtype,
                weights: &weights,
                columns,
                s_in: ScaleSlice::F16Le(&s_in),
                s_out: 0.875,
            };
            assert_eq!(validate_recovered_row(&operation).unwrap().dtype, dtype);

            let truncated = &weights[..weights.len() - 1];
            let invalid = RecoveredRow {
                weights: truncated,
                ..operation
            };
            assert!(matches!(
                validate_recovered_row(&invalid),
                Err(EngineError::Shape(_))
            ));
        }
    }

    #[test]
    fn operation_validation_rejects_repacking_and_host_scales() {
        let rows = 2;
        let columns = BLOCK_LEN;
        let weights = packed_weights(TensorDType::Q2B64, rows, columns);
        let input = vec![0.25_f32; columns];
        let host_scales = vec![1.0_f32; columns];
        let mut operation = FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &weights,
            segments: &[],
            rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F32(&host_scales)),
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::UnsupportedDType(_))
        ));
        operation.s_in = None;
        let segments = [crate::format::QuantSegment {
            group_index: 0,
            row_start: 0,
            row_end: rows as u64,
            dtype: TensorDType::Q2B64,
            offset: 0,
            length: weights.len() as u64,
        }];
        operation.segments = &segments;
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn execution_stays_fail_closed() {
        let backend = CudaBackend;
        assert_eq!(backend.promotion_state(), PromotionState::Contract);
        assert_eq!(backend.module_abi(), SM86_MODULE_ABI);
        for dtype in [TensorDType::Q2B64, TensorDType::Q4B64] {
            let weights = vec![0_u8; BLOCK_LEN.div_ceil(8).max(2) * BLOCK_LEN];
            let input = vec![0.0_f32; BLOCK_LEN];
            let operation = FusedMatVec {
                dtype,
                weights: &weights,
                segments: &[],
                rows: 1,
                columns: BLOCK_LEN,
                input: &input,
                s_in: None,
                s_out: None,
                bias: None,
                activation: Activation::Identity,
            };
            let err = backend.fused_matvec(&operation).unwrap_err();
            assert!(matches!(
                err,
                EngineError::UnsupportedOperation {
                    backend: "cuda",
                    ..
                }
            ));
        }
    }
}
