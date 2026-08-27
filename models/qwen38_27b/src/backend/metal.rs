use crate::backend::{
    Backend, BackendKind, FusedMatVec, PromotionState, RecoveredRow, RecoveredRowMatVec, ScaleSlice,
};
use crate::format::TensorDType;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

/// Metal candidate for the Q2/Q4 fused matvec. No MLX/MPSGraph inference
/// fallback is permitted, and there is no scalar fallback path: dispatch fails
/// closed until the same-device verifier and benchmark gates pass.
///
/// The candidate MSL source lives in
/// `kernels/metal/q2q4_fused_matvec.metal` and exposes the
/// [`Q2_KERNEL_NAME`] and [`Q4_KERNEL_NAME`] entry points. The ABI below is
/// the single source of truth shared with that source.
pub struct MetalBackend;

/// Metal CSL entry point for the Q2_B64 candidate kernel.
pub const Q2_KERNEL_NAME: &str = "q2_b64_fused_matvec";
/// Metal CSL entry point for the Q4_B64 candidate kernel.
pub const Q4_KERNEL_NAME: &str = "q4_b64_fused_matvec";
pub const Q2_SWIGLU_KERNEL_NAME: &str = "q2_b64_swiglu_matvec";
pub const Q4_SWIGLU_KERNEL_NAME: &str = "q4_b64_swiglu_matvec";
pub const Q2_SIGMOID_GATE_KERNEL_NAME: &str = "q2_b64_sigmoid_gate_matvec";
pub const Q4_SIGMOID_GATE_KERNEL_NAME: &str = "q4_b64_sigmoid_gate_matvec";
pub const Q2_GATHERED_KERNEL_NAME: &str = "q2_b64_gathered_matvec";
pub const Q4_GATHERED_KERNEL_NAME: &str = "q4_b64_gathered_matvec";
pub const Q2_RECOVERED_ROW_KERNEL_NAME: &str = "q2_b64_recovered_row";
pub const Q4_RECOVERED_ROW_KERNEL_NAME: &str = "q4_b64_recovered_row";
pub const Q2_DYNAMIC_EMBEDDING_KERNEL_NAME: &str = "q2_b64_dynamic_embedding_row";
pub const Q4_DYNAMIC_EMBEDDING_KERNEL_NAME: &str = "q4_b64_dynamic_embedding_row";
pub const CONCAT_F32_KERNEL_NAME: &str = "qwen_concat_f32";
pub const RMS_NORM_1P_KERNEL_NAME: &str = "qwen_rms_norm_1p_f32";
pub const RMS_NORM_1P_HEAD256_INPLACE_KERNEL_NAME: &str = "qwen_rms_norm_1p_head256_inplace_f32";
pub const RESIDUAL_RMS_NORM_1P_KERNEL_NAME: &str = "qwen_residual_rms_norm_1p_f32";
pub const RMS_NORM_GATED_KERNEL_NAME: &str = "qwen_rms_norm_gated_f32";
pub const PARTIAL_ROPE_KERNEL_NAME: &str = "qwen_partial_rope_f32";
pub const QUERY_GATE_NORM_ROPE_KERNEL_NAME: &str = "qwen_query_gate_norm_rope_f32";
pub const KV_Q4_PACK_KERNEL_NAME: &str = "qwen_kv_q4_pack_f32";
pub const KV_Q4_TO_Q2_KERNEL_NAME: &str = "qwen_kv_q4_to_q2";
pub const PAGED_GQA_DECODE_KERNEL_NAME: &str = "qwen_paged_q2q4_gqa_decode_f32";
pub const GATED_DELTA_F16_KERNEL_NAME: &str = "qwen_gated_delta_recurrent_f16";
pub const GATED_DELTA_PREP_F32_KERNEL_NAME: &str = "qwen_gated_delta_prepare_f32";
pub const CAUSAL_CONV_F16_KERNEL_NAME: &str = "qwen_causal_conv_silu_f16";
pub const ARGMAX_F32_PARTIAL_KERNEL_NAME: &str = "qwen_argmax_f32_partial";
pub const ARGMAX_F32_FINAL_KERNEL_NAME: &str = "qwen_argmax_f32_final";
/// Vendored candidate kernel source, relative to the crate root.
pub const KERNEL_SOURCE_PATH: &str = "kernels/metal/q2q4_fused_matvec.metal";

/// fp16 scale bytes at the head of every packed block.
pub const BLOCK_SCALE_BYTES: usize = 2;
/// Q2 code bytes following the scale (64 values at 2 bits each).
pub const Q2_CODE_BYTES: usize = 16;
/// Q4 code bytes following the scale (64 values at 4 bits each).
pub const Q4_CODE_BYTES: usize = 32;

/// Values encoded by the Q2 two-bit codes, in code order.
pub const Q2_CODEBOOK: [f32; 4] = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];

/// Largest candidate threadgroup profile: eight 32-wide simdgroups.
pub const MAX_SIMDGROUPS_PER_THREADGROUP: usize = 8;

/// Metal buffer bindings shared by both candidate kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalBufferAbi;

impl MetalBufferAbi {
    pub const WEIGHTS: u32 = 0;
    pub const INPUT: u32 = 1;
    pub const S_IN: u32 = 2;
    pub const S_OUT: u32 = 3;
    pub const BIAS: u32 = 4;
    pub const OUTPUT: u32 = 5;
    pub const PARAMS: u32 = 6;
    pub const ROW_IDS: u32 = 7;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalConcatBufferAbi;

impl MetalConcatBufferAbi {
    pub const FIRST: u32 = 0;
    pub const SECOND: u32 = 1;
    pub const OUTPUT: u32 = 2;
    pub const PARAMS: u32 = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalSwiGluBufferAbi;

impl MetalSwiGluBufferAbi {
    pub const WEIGHTS: u32 = 0;
    pub const GATE: u32 = 1;
    pub const UP: u32 = 2;
    pub const S_IN: u32 = 3;
    pub const S_OUT: u32 = 4;
    pub const BIAS: u32 = 5;
    pub const OUTPUT: u32 = 6;
    pub const PARAMS: u32 = 7;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalSigmoidGateBufferAbi;

impl MetalSigmoidGateBufferAbi {
    pub const WEIGHTS: u32 = 0;
    pub const ATTENTION: u32 = 1;
    pub const GATE: u32 = 2;
    pub const S_IN: u32 = 3;
    pub const S_OUT: u32 = 4;
    pub const BIAS: u32 = 5;
    pub const OUTPUT: u32 = 6;
    pub const PARAMS: u32 = 7;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalRmsNormBufferAbi;

impl MetalRmsNormBufferAbi {
    pub const INPUT: u32 = 0;
    pub const WEIGHT: u32 = 1;
    pub const OUTPUT: u32 = 2;
    pub const PARAMS: u32 = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalGatedRmsNormBufferAbi;

impl MetalGatedRmsNormBufferAbi {
    pub const INPUT: u32 = 0;
    pub const GATE: u32 = 1;
    pub const WEIGHT: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const PARAMS: u32 = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalResidualRmsNormBufferAbi;

impl MetalResidualRmsNormBufferAbi {
    pub const RESIDUAL: u32 = 0;
    pub const UPDATE: u32 = 1;
    pub const WEIGHT: u32 = 2;
    pub const RESIDUAL_OUTPUT: u32 = 3;
    pub const NORMALIZED_OUTPUT: u32 = 4;
    pub const PARAMS: u32 = 5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalPartialRopeBufferAbi;

impl MetalPartialRopeBufferAbi {
    pub const VALUES: u32 = 0;
    pub const COSINE: u32 = 1;
    pub const SINE: u32 = 2;
    pub const PARAMS: u32 = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalQueryGateBufferAbi;

impl MetalQueryGateBufferAbi {
    pub const QUERY_GATE: u32 = 0;
    pub const Q_NORM_WEIGHT: u32 = 1;
    pub const COSINE: u32 = 2;
    pub const SINE: u32 = 3;
    pub const QUERY: u32 = 4;
    pub const GATE: u32 = 5;
    pub const PARAMS: u32 = 6;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKvQ4PackBufferAbi;

impl MetalKvQ4PackBufferAbi {
    pub const KEY: u32 = 0;
    pub const VALUE: u32 = 1;
    pub const OUTPUT: u32 = 2;
    pub const PARAMS: u32 = 3;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKvQ4ToQ2BufferAbi;

impl MetalKvQ4ToQ2BufferAbi {
    pub const Q4: u32 = 0;
    pub const Q2: u32 = 1;
    pub const PARAMS: u32 = 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalPagedGqaBufferAbi;

impl MetalPagedGqaBufferAbi {
    pub const QUERY: u32 = 0;
    pub const Q2_PAGES: u32 = 1;
    pub const Q4_PAGES: u32 = 2;
    pub const DESCRIPTORS: u32 = 3;
    pub const OUTPUT: u32 = 4;
    pub const PARAMS: u32 = 5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalGatedDeltaBufferAbi;

impl MetalGatedDeltaBufferAbi {
    pub const QUERY: u32 = 0;
    pub const KEY: u32 = 1;
    pub const VALUE: u32 = 2;
    pub const LOG_DECAY: u32 = 3;
    pub const BETA: u32 = 4;
    pub const STATE: u32 = 5;
    pub const OUTPUT: u32 = 6;
    pub const PARAMS: u32 = 7;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalGatedDeltaPrepareBufferAbi;

impl MetalGatedDeltaPrepareBufferAbi {
    pub const CONVOLVED_QKV: u32 = 0;
    pub const RAW_A: u32 = 1;
    pub const RAW_B: u32 = 2;
    pub const A_LOG: u32 = 3;
    pub const DT_BIAS: u32 = 4;
    pub const QUERY: u32 = 5;
    pub const KEY: u32 = 6;
    pub const VALUE: u32 = 7;
    pub const LOG_DECAY: u32 = 8;
    pub const BETA: u32 = 9;
    pub const PARAMS: u32 = 10;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalCausalConvBufferAbi;

impl MetalCausalConvBufferAbi {
    pub const INPUT: u32 = 0;
    pub const WEIGHT: u32 = 1;
    pub const STATE: u32 = 2;
    pub const OUTPUT: u32 = 3;
    pub const PARAMS: u32 = 4;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalArgMaxPartialBufferAbi;

impl MetalArgMaxPartialBufferAbi {
    pub const INPUT: u32 = 0;
    pub const PARTIALS: u32 = 1;
    pub const PARAMS: u32 = 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalArgMaxFinalBufferAbi;

impl MetalArgMaxFinalBufferAbi {
    pub const PARTIALS: u32 = 0;
    pub const RESULT: u32 = 1;
    pub const PARAMS: u32 = 2;
}

/// Activation codes consumed by `apply_activation` in the MSL source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetalActivation {
    Identity = 0,
    Silu = 1,
}

impl MetalActivation {
    pub fn from_backend(activation: crate::backend::Activation) -> Self {
        match activation {
            crate::backend::Activation::Identity => Self::Identity,
            crate::backend::Activation::Silu => Self::Silu,
        }
    }
}

/// `FusedMatVecParams` mirror of the MSL struct: eight little-endian u32
/// words, 32 bytes total, bound at buffer index [`MetalBufferAbi::PARAMS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalFusedMatVecParams {
    pub rows: u32,
    pub columns: u32,
    pub blocks_per_row: u32,
    pub has_s_in: u32,
    pub has_s_out: u32,
    pub has_bias: u32,
    pub activation: u32,
    pub reserved0: u32,
}

/// Segment-local dynamic embedding lookup parameters. The selected token is
/// produced by the resident argmax result buffer, so the host never supplies
/// or reads the row between target and MTP execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalDynamicEmbeddingParams {
    pub columns: u32,
    pub blocks_per_row: u32,
    pub row_start: u32,
    pub row_end: u32,
    pub row_bytes: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalConcatParams {
    pub first_values: u32,
    pub second_values: u32,
    pub reserved0: u32,
    pub reserved1: u32,
}

impl MetalConcatParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.first_values,
            self.second_values,
            self.reserved0,
            self.reserved1,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

impl MetalDynamicEmbeddingParams {
    pub const BYTE_LEN: usize = 32;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.columns,
            self.blocks_per_row,
            self.row_start,
            self.row_end,
            self.row_bytes,
            self.reserved0,
            self.reserved1,
            self.reserved2,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

/// Packed ABI shared with `RmsNormParams` in the MSL candidate source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalRmsNormParams {
    pub rows: u32,
    pub columns: u32,
    pub epsilon: f32,
    pub reserved0: u32,
}

impl MetalRmsNormParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        encoded[0..4].copy_from_slice(&self.rows.to_le_bytes());
        encoded[4..8].copy_from_slice(&self.columns.to_le_bytes());
        encoded[8..12].copy_from_slice(&self.epsilon.to_bits().to_le_bytes());
        encoded[12..16].copy_from_slice(&self.reserved0.to_le_bytes());
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalPartialRopeParams {
    pub heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub position: u32,
    pub theta: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalQueryGateParams {
    pub heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub reserved0: u32,
    pub epsilon: f32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalKvPackParams {
    pub component_values: u32,
    pub blocks: u32,
    pub reserved0: u32,
    pub reserved1: u32,
}

impl MetalKvPackParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.component_values,
            self.blocks,
            self.reserved0,
            self.reserved1,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

impl MetalPartialRopeParams {
    pub const BYTE_LEN: usize = 32;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.heads,
            self.head_dim,
            self.rotary_dim,
            self.position,
            self.theta.to_bits(),
            self.reserved0,
            self.reserved1,
            self.reserved2,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

impl MetalQueryGateParams {
    pub const BYTE_LEN: usize = 32;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.heads,
            self.head_dim,
            self.rotary_dim,
            self.reserved0,
            self.epsilon.to_bits(),
            self.reserved1,
            self.reserved2,
            self.reserved3,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalPagedGqaParams {
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub tokens: u32,
    pub page_tokens: u32,
    pub page_count: u32,
    pub combined_values: u32,
    pub q2_token_bytes: u32,
    pub q4_token_bytes: u32,
    pub q2_page_bytes: u32,
    pub q4_page_bytes: u32,
    pub scale: f32,
}

impl MetalPagedGqaParams {
    pub const BYTE_LEN: usize = 48;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.tokens,
            self.page_tokens,
            self.page_count,
            self.combined_values,
            self.q2_token_bytes,
            self.q4_token_bytes,
            self.q2_page_bytes,
            self.q4_page_bytes,
            self.scale.to_bits(),
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalGatedDeltaParams {
    pub heads: u32,
    pub key_dim: u32,
    pub value_dim: u32,
    pub epsilon: f32,
}

impl MetalGatedDeltaParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.heads,
            self.key_dim,
            self.value_dim,
            self.epsilon.to_bits(),
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalGatedDeltaPrepareParams {
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_dim: u32,
    pub reserved0: u32,
}

impl MetalGatedDeltaPrepareParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.reserved0,
        ]
        .iter()
        .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalCausalConvParams {
    pub channels: u32,
    pub kernel: u32,
    pub reserved0: u32,
    pub reserved1: u32,
}

impl MetalCausalConvParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [self.channels, self.kernel, self.reserved0, self.reserved1]
            .iter()
            .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalArgMaxParams {
    pub values: u32,
    pub threads: u32,
    pub groups: u32,
    pub reserved1: u32,
}

impl MetalArgMaxParams {
    pub const BYTE_LEN: usize = 16;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        for (index, word) in [self.values, self.threads, self.groups, self.reserved1]
            .iter()
            .enumerate()
        {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

/// One contiguous homogeneous row range inside a canonical mixed Q2/Q4
/// tensor. The Metal runtime binds offsets into the original artifact and
/// dispatches the existing Q2 or Q4 kernel without repacking any codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MetalMixedRowSegment {
    pub layout: MetalBlockLayout,
    pub params: MetalFusedMatVecParams,
    pub row_start: usize,
    pub weight_offset: usize,
}

impl MetalFusedMatVecParams {
    pub const BYTE_LEN: usize = 32;

    pub fn encode(self) -> [u8; Self::BYTE_LEN] {
        let mut encoded = [0_u8; Self::BYTE_LEN];
        let words = [
            self.rows,
            self.columns,
            self.blocks_per_row,
            self.has_s_in,
            self.has_s_out,
            self.has_bias,
            self.activation,
            self.reserved0,
        ];
        for (index, word) in words.iter().enumerate() {
            encoded[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        encoded
    }
}

/// Packed-block layout descriptor for one candidate dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalBlockLayout {
    pub dtype: TensorDType,
    pub kernel_name: &'static str,
    pub block_len: usize,
    pub block_bytes: usize,
    pub scale_bytes: usize,
    pub code_bytes: usize,
}

/// Layout table for the supported candidate dtypes. Q3 is intentionally
/// absent: only Q2_B64 and Q4_B64 exist in this format.
pub fn block_layout(dtype: TensorDType) -> Result<MetalBlockLayout> {
    match dtype {
        TensorDType::Q2B64 => Ok(MetalBlockLayout {
            dtype,
            kernel_name: Q2_KERNEL_NAME,
            block_len: BLOCK_LEN,
            block_bytes: Q2_BLOCK_BYTES,
            scale_bytes: BLOCK_SCALE_BYTES,
            code_bytes: Q2_CODE_BYTES,
        }),
        TensorDType::Q4B64 => Ok(MetalBlockLayout {
            dtype,
            kernel_name: Q4_KERNEL_NAME,
            block_len: BLOCK_LEN,
            block_bytes: Q4_BLOCK_BYTES,
            scale_bytes: BLOCK_SCALE_BYTES,
            code_bytes: Q4_CODE_BYTES,
        }),
        other => Err(EngineError::UnsupportedDType(format!(
            "metal candidate supports only Q2_B64/Q4_B64, got {other:?}"
        ))),
    }
}

/// Validates an operation against the candidate ABI and derives the dispatch
/// parameters. This runs before any device work; invalid shapes or buffers
/// fail closed with an error instead of dispatching.
pub fn validate_operation(
    operation: &FusedMatVec<'_>,
) -> Result<(MetalBlockLayout, MetalFusedMatVecParams)> {
    let layout = block_layout(operation.dtype)?;
    if !operation.segments.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "pure Metal Q2/Q4 operation declares mixed row segments".into(),
        ));
    }
    if operation.columns == 0 || operation.rows == 0 || !operation.columns.is_multiple_of(BLOCK_LEN)
    {
        return Err(EngineError::Shape(
            "metal fused matvec dimensions must be non-zero and columns divisible by 64".into(),
        ));
    }
    let blocks_per_row = operation.columns / BLOCK_LEN;
    let expected_weights = operation
        .rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(layout.block_bytes))
        .ok_or_else(|| EngineError::Shape("weight buffer size overflows usize".into()))?;
    if operation.weights.len() != expected_weights {
        return Err(EngineError::Shape(format!(
            "weight buffer has {} bytes, expected {expected_weights}",
            operation.weights.len()
        )));
    }
    if operation.input.len() != operation.columns {
        return Err(EngineError::Shape(format!(
            "input has {} values, expected {}",
            operation.input.len(),
            operation.columns
        )));
    }
    if let Some(scales) = operation.s_in {
        if !matches!(scales, ScaleSlice::F16Le(_)) {
            return Err(EngineError::UnsupportedDType(
                "Metal recovery scales must remain packed FP16".into(),
            ));
        }
        if scales.len() != operation.columns {
            return Err(EngineError::Shape(
                "s_in length differs from columns".into(),
            ));
        }
    }
    if operation
        .s_out
        .is_some_and(|values| values.len() != operation.rows)
    {
        return Err(EngineError::Shape("s_out length differs from rows".into()));
    }
    if operation
        .s_out
        .is_some_and(|values| !matches!(values, ScaleSlice::F16Le(_)))
    {
        return Err(EngineError::UnsupportedDType(
            "Metal recovery scales must remain packed FP16".into(),
        ));
    }
    if operation
        .bias
        .is_some_and(|values| values.len() != operation.rows)
    {
        return Err(EngineError::Shape("bias length differs from rows".into()));
    }
    let rows = u32::try_from(operation.rows)
        .map_err(|_| EngineError::Shape("rows exceed u32 dispatch limit".into()))?;
    let columns = u32::try_from(operation.columns)
        .map_err(|_| EngineError::Shape("columns exceed u32 dispatch limit".into()))?;
    let params = MetalFusedMatVecParams {
        rows,
        columns,
        blocks_per_row: blocks_per_row as u32,
        has_s_in: u32::from(operation.s_in.is_some()),
        has_s_out: u32::from(operation.s_out.is_some()),
        has_bias: u32::from(operation.bias.is_some()),
        activation: MetalActivation::from_backend(operation.activation) as u32,
        reserved0: 0,
    };
    Ok((layout, params))
}

/// Validate one loader-resolved embedding row without widening or copying its
/// packed weights and recovery scales. The returned layout selects only the
/// verifier candidate kernel; production dispatch remains fail-closed until
/// the complete Metal promotion gates pass.
pub(crate) fn validate_recovered_row(operation: &RecoveredRow<'_>) -> Result<MetalBlockLayout> {
    let layout = block_layout(operation.dtype)?;
    if operation.columns == 0 || !operation.columns.is_multiple_of(BLOCK_LEN) {
        return Err(EngineError::Shape(
            "Metal recovered-row columns must be non-zero and divisible by 64".into(),
        ));
    }
    let ScaleSlice::F16Le(s_in) = operation.s_in else {
        return Err(EngineError::UnsupportedDType(
            "Metal recovered-row s_in must remain packed FP16".into(),
        ));
    };
    let expected_s_in = operation
        .columns
        .checked_mul(std::mem::size_of::<half::f16>())
        .ok_or_else(|| EngineError::Shape("Metal recovered-row s_in bytes overflow".into()))?;
    if s_in.len() != expected_s_in {
        return Err(EngineError::Shape(format!(
            "Metal recovered-row s_in has {} bytes, expected {}",
            s_in.len(),
            expected_s_in
        )));
    }
    if !operation.s_out.is_finite() {
        return Err(EngineError::InvalidArtifact(
            "Metal recovered-row s_out is non-finite".into(),
        ));
    }
    let expected_weights = (operation.columns / BLOCK_LEN)
        .checked_mul(layout.block_bytes)
        .ok_or_else(|| EngineError::Shape("Metal recovered-row bytes overflow".into()))?;
    if operation.weights.len() != expected_weights {
        return Err(EngineError::Shape(format!(
            "Metal recovered row has {} packed bytes, expected {expected_weights}",
            operation.weights.len()
        )));
    }
    Ok(layout)
}

/// Validate the exact manifest row groups of a mixed Q2/Q4 operation. Every
/// returned dispatch covers one contiguous row and byte range exactly once.
pub(crate) fn validate_mixed_operation(
    operation: &FusedMatVec<'_>,
) -> Result<Vec<MetalMixedRowSegment>> {
    if operation.dtype != TensorDType::MixedQ2Q4B64 {
        return Err(EngineError::UnsupportedDType(format!(
            "mixed Metal validation requires MixedQ2Q4B64, got {:?}",
            operation.dtype
        )));
    }
    if operation.columns == 0 || operation.rows == 0 || !operation.columns.is_multiple_of(BLOCK_LEN)
    {
        return Err(EngineError::Shape(
            "mixed Metal dimensions must be non-zero and columns divisible by 64".into(),
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
                    "Metal {name} recovery scales must remain packed FP16"
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
    if operation.segments.is_empty() {
        return Err(EngineError::InvalidArtifact(
            "mixed Metal Q2/Q4 operation has no row segments".into(),
        ));
    }
    let columns = u32::try_from(operation.columns)
        .map_err(|_| EngineError::Shape("columns exceed u32 dispatch limit".into()))?;
    let blocks_per_row = operation.columns / BLOCK_LEN;
    let mut expected_row = 0_usize;
    let mut expected_offset = 0_usize;
    let mut dispatches = Vec::with_capacity(operation.segments.len());
    for (expected_group, segment) in operation.segments.iter().enumerate() {
        let group_index = usize::try_from(segment.group_index).map_err(|_| {
            EngineError::InvalidArtifact("mixed Metal group index overflows usize".into())
        })?;
        let row_start = usize::try_from(segment.row_start).map_err(|_| {
            EngineError::InvalidArtifact("mixed Metal row start overflows usize".into())
        })?;
        let row_end = usize::try_from(segment.row_end).map_err(|_| {
            EngineError::InvalidArtifact("mixed Metal row end overflows usize".into())
        })?;
        let weight_offset = usize::try_from(segment.offset).map_err(|_| {
            EngineError::InvalidArtifact("mixed Metal weight offset overflows usize".into())
        })?;
        let length = usize::try_from(segment.length).map_err(|_| {
            EngineError::InvalidArtifact("mixed Metal segment length overflows usize".into())
        })?;
        if group_index != expected_group
            || row_start != expected_row
            || row_end <= row_start
            || row_end > operation.rows
            || weight_offset != expected_offset
        {
            return Err(EngineError::InvalidArtifact(format!(
                "mixed Metal operation has non-contiguous segment {}",
                segment.group_index
            )));
        }
        let layout = block_layout(segment.dtype).map_err(|_| {
            EngineError::InvalidArtifact(format!(
                "mixed Metal segment {} has invalid dtype {:?}",
                segment.group_index, segment.dtype
            ))
        })?;
        let row_count = row_end - row_start;
        let expected_length = row_count
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(layout.block_bytes))
            .ok_or_else(|| EngineError::Shape("mixed Metal segment size overflows".into()))?;
        if length != expected_length {
            return Err(EngineError::InvalidArtifact(format!(
                "mixed Metal segment {} has {length} bytes, expected {expected_length}",
                segment.group_index
            )));
        }
        dispatches.push(MetalMixedRowSegment {
            layout,
            params: MetalFusedMatVecParams {
                rows: u32::try_from(row_count)
                    .map_err(|_| EngineError::Shape("mixed Metal row count exceeds u32".into()))?,
                columns,
                blocks_per_row: u32::try_from(blocks_per_row).map_err(|_| {
                    EngineError::Shape("mixed Metal blocks per row exceed u32".into())
                })?,
                has_s_in: u32::from(operation.s_in.is_some()),
                has_s_out: u32::from(operation.s_out.is_some()),
                has_bias: u32::from(operation.bias.is_some()),
                activation: MetalActivation::from_backend(operation.activation) as u32,
                reserved0: 0,
            },
            row_start,
            weight_offset,
        });
        expected_row = row_end;
        expected_offset = expected_offset
            .checked_add(length)
            .ok_or_else(|| EngineError::Shape("mixed Metal byte coverage overflows".into()))?;
    }
    if expected_row != operation.rows || expected_offset != operation.weights.len() {
        return Err(EngineError::InvalidArtifact(format!(
            "mixed Metal segments cover {expected_row}/{} rows and {expected_offset}/{} bytes",
            operation.rows,
            operation.weights.len()
        )));
    }
    Ok(dispatches)
}

impl Backend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn promotion_state(&self) -> PromotionState {
        // Candidate kernels exist but have not passed the same-device
        // verifier and benchmark gates, so the backend stays fail-closed.
        PromotionState::Contract
    }

    fn profile(&self) -> &'static str {
        "metal-candidate"
    }

    fn fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        // Validate eagerly so shape/buffer errors surface even while dispatch
        // remains fail-closed. The isolated candidate has verifier/benchmark
        // evidence, but not the full-graph and ownership evidence required to
        // expose this production Backend entry point.
        let (_layout, _params) = validate_operation(operation)?;
        Err(EngineError::UnsupportedOperation {
            backend: "metal",
            operation: "q2/q4 fused matvec",
            reason: "MSL candidate passed isolated verification but lacks full-graph ownership, golden, and promotion evidence".into(),
        })
    }

    fn recovered_row(&self, operation: &RecoveredRow<'_>) -> Result<Vec<f32>> {
        validate_recovered_row(operation)?;
        Err(EngineError::UnsupportedOperation {
            backend: "metal",
            operation: "Q2/Q4 recovered row gather",
            reason:
                "embedding gather candidate has not passed same-device verifier and benchmark gates"
                    .into(),
        })
    }

    fn recovered_row_matvec(&self, _operation: &RecoveredRowMatVec<'_>) -> Result<f32> {
        Err(EngineError::UnsupportedOperation {
            backend: "metal",
            operation: "recovered restricted LM-head row matvec",
            reason: "the gathered Q2/Q4 Metal proposal kernel is not promoted".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Activation, ExecutionPolicy};
    use crate::format::QuantSegment;
    use crate::quant::{Q2Block64, Q4Block64};
    use half::f16;

    fn encode_weights(dtype: TensorDType, rows: usize, columns: usize) -> Vec<u8> {
        let blocks = rows * (columns / BLOCK_LEN);
        let mut weights = Vec::new();
        for block in 0..blocks {
            let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
                ((block * BLOCK_LEN + index) as f32 * 0.017).sin() * 0.5
            });
            match dtype {
                TensorDType::Q2B64 => {
                    weights.extend_from_slice(&Q2Block64::quantize(&values).unwrap().encode())
                }
                TensorDType::Q4B64 => {
                    weights.extend_from_slice(&Q4Block64::quantize(&values).unwrap().encode())
                }
                _ => unreachable!(),
            }
        }
        weights
    }

    fn valid_operation<'a>(
        dtype: TensorDType,
        weights: &'a [u8],
        input: &'a [f32],
        rows: usize,
        columns: usize,
    ) -> FusedMatVec<'a> {
        FusedMatVec {
            dtype,
            weights,
            segments: &[],
            rows,
            columns,
            input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        }
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    #[test]
    fn abi_constants_match_quant_layouts() {
        assert_eq!(BLOCK_LEN, 64);
        assert_eq!(Q2_BLOCK_BYTES, BLOCK_SCALE_BYTES + Q2_CODE_BYTES);
        assert_eq!(Q4_BLOCK_BYTES, BLOCK_SCALE_BYTES + Q4_CODE_BYTES);
        assert_eq!(Q2_BLOCK_BYTES, 18);
        assert_eq!(Q4_BLOCK_BYTES, 34);
        assert_eq!(Q2_CODE_BYTES * 4, BLOCK_LEN);
        assert_eq!(Q4_CODE_BYTES * 2, BLOCK_LEN);
        assert_eq!(Q2_CODEBOOK, crate::quant::Q2_CODEBOOK);
        assert_eq!(MetalFusedMatVecParams::BYTE_LEN, 32);
        assert_eq!(MAX_SIMDGROUPS_PER_THREADGROUP * 32, 256);
    }

    #[test]
    fn buffer_bindings_are_distinct() {
        let bindings = [
            MetalBufferAbi::WEIGHTS,
            MetalBufferAbi::INPUT,
            MetalBufferAbi::S_IN,
            MetalBufferAbi::S_OUT,
            MetalBufferAbi::BIAS,
            MetalBufferAbi::OUTPUT,
            MetalBufferAbi::PARAMS,
            MetalBufferAbi::ROW_IDS,
        ];
        for (index, binding) in bindings.iter().enumerate() {
            assert_eq!(*binding, index as u32);
        }
        assert_eq!(
            [
                MetalSwiGluBufferAbi::WEIGHTS,
                MetalSwiGluBufferAbi::GATE,
                MetalSwiGluBufferAbi::UP,
                MetalSwiGluBufferAbi::S_IN,
                MetalSwiGluBufferAbi::S_OUT,
                MetalSwiGluBufferAbi::BIAS,
                MetalSwiGluBufferAbi::OUTPUT,
                MetalSwiGluBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            [
                MetalSigmoidGateBufferAbi::WEIGHTS,
                MetalSigmoidGateBufferAbi::ATTENTION,
                MetalSigmoidGateBufferAbi::GATE,
                MetalSigmoidGateBufferAbi::S_IN,
                MetalSigmoidGateBufferAbi::S_OUT,
                MetalSigmoidGateBufferAbi::BIAS,
                MetalSigmoidGateBufferAbi::OUTPUT,
                MetalSigmoidGateBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            [
                MetalPagedGqaBufferAbi::QUERY,
                MetalPagedGqaBufferAbi::Q2_PAGES,
                MetalPagedGqaBufferAbi::Q4_PAGES,
                MetalPagedGqaBufferAbi::DESCRIPTORS,
                MetalPagedGqaBufferAbi::OUTPUT,
                MetalPagedGqaBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            [
                MetalQueryGateBufferAbi::QUERY_GATE,
                MetalQueryGateBufferAbi::Q_NORM_WEIGHT,
                MetalQueryGateBufferAbi::COSINE,
                MetalQueryGateBufferAbi::SINE,
                MetalQueryGateBufferAbi::QUERY,
                MetalQueryGateBufferAbi::GATE,
                MetalQueryGateBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            [
                MetalGatedDeltaBufferAbi::QUERY,
                MetalGatedDeltaBufferAbi::KEY,
                MetalGatedDeltaBufferAbi::VALUE,
                MetalGatedDeltaBufferAbi::LOG_DECAY,
                MetalGatedDeltaBufferAbi::BETA,
                MetalGatedDeltaBufferAbi::STATE,
                MetalGatedDeltaBufferAbi::OUTPUT,
                MetalGatedDeltaBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            [
                MetalGatedDeltaPrepareBufferAbi::CONVOLVED_QKV,
                MetalGatedDeltaPrepareBufferAbi::RAW_A,
                MetalGatedDeltaPrepareBufferAbi::RAW_B,
                MetalGatedDeltaPrepareBufferAbi::A_LOG,
                MetalGatedDeltaPrepareBufferAbi::DT_BIAS,
                MetalGatedDeltaPrepareBufferAbi::QUERY,
                MetalGatedDeltaPrepareBufferAbi::KEY,
                MetalGatedDeltaPrepareBufferAbi::VALUE,
                MetalGatedDeltaPrepareBufferAbi::LOG_DECAY,
                MetalGatedDeltaPrepareBufferAbi::BETA,
                MetalGatedDeltaPrepareBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
        assert_eq!(
            [
                MetalResidualRmsNormBufferAbi::RESIDUAL,
                MetalResidualRmsNormBufferAbi::UPDATE,
                MetalResidualRmsNormBufferAbi::WEIGHT,
                MetalResidualRmsNormBufferAbi::RESIDUAL_OUTPUT,
                MetalResidualRmsNormBufferAbi::NORMALIZED_OUTPUT,
                MetalResidualRmsNormBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            [
                MetalCausalConvBufferAbi::INPUT,
                MetalCausalConvBufferAbi::WEIGHT,
                MetalCausalConvBufferAbi::STATE,
                MetalCausalConvBufferAbi::OUTPUT,
                MetalCausalConvBufferAbi::PARAMS,
            ],
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            [
                MetalArgMaxPartialBufferAbi::INPUT,
                MetalArgMaxPartialBufferAbi::PARTIALS,
                MetalArgMaxPartialBufferAbi::PARAMS,
            ],
            [0, 1, 2]
        );
        assert_eq!(
            [
                MetalArgMaxFinalBufferAbi::PARTIALS,
                MetalArgMaxFinalBufferAbi::RESULT,
                MetalArgMaxFinalBufferAbi::PARAMS,
            ],
            [0, 1, 2]
        );
    }

    #[test]
    fn dispatch_names_match_dtype() {
        assert_eq!(
            block_layout(TensorDType::Q2B64).unwrap().kernel_name,
            Q2_KERNEL_NAME
        );
        assert_eq!(
            block_layout(TensorDType::Q4B64).unwrap().kernel_name,
            Q4_KERNEL_NAME
        );
        assert_ne!(Q2_KERNEL_NAME, Q4_KERNEL_NAME);
        assert!(Q2_KERNEL_NAME.starts_with("q2_b64"));
        assert!(Q4_KERNEL_NAME.starts_with("q4_b64"));
        assert!(Q2_GATHERED_KERNEL_NAME.starts_with("q2_b64"));
        assert!(Q4_GATHERED_KERNEL_NAME.starts_with("q4_b64"));
        assert!(Q2_RECOVERED_ROW_KERNEL_NAME.starts_with("q2_b64"));
        assert!(Q4_RECOVERED_ROW_KERNEL_NAME.starts_with("q4_b64"));
        assert!(Q2_DYNAMIC_EMBEDDING_KERNEL_NAME.contains("dynamic_embedding"));
        assert!(Q4_DYNAMIC_EMBEDDING_KERNEL_NAME.contains("dynamic_embedding"));
        assert_eq!(CONCAT_F32_KERNEL_NAME, "qwen_concat_f32");
        assert!(RMS_NORM_1P_KERNEL_NAME.starts_with("qwen_rms_norm"));
        assert!(RMS_NORM_1P_HEAD256_INPLACE_KERNEL_NAME.contains("head256_inplace"));
        assert!(PARTIAL_ROPE_KERNEL_NAME.starts_with("qwen_partial_rope"));
        assert!(PAGED_GQA_DECODE_KERNEL_NAME.contains("paged_q2q4_gqa"));
        assert!(GATED_DELTA_F16_KERNEL_NAME.contains("gated_delta_recurrent_f16"));
        assert_eq!(
            GATED_DELTA_PREP_F32_KERNEL_NAME,
            "qwen_gated_delta_prepare_f32"
        );
        assert!(CAUSAL_CONV_F16_KERNEL_NAME.contains("causal_conv_silu_f16"));
        assert_eq!(ARGMAX_F32_PARTIAL_KERNEL_NAME, "qwen_argmax_f32_partial");
        assert_eq!(ARGMAX_F32_FINAL_KERNEL_NAME, "qwen_argmax_f32_final");
    }

    #[test]
    fn rejects_q3_and_unsupported_dtypes() {
        assert!(block_layout(TensorDType::F16).is_err());
        assert!(block_layout(TensorDType::F32).is_err());
        let weights = encode_weights(TensorDType::Q2B64, 1, BLOCK_LEN);
        let input = [0.0_f32; BLOCK_LEN];
        let mut operation = valid_operation(TensorDType::F32, &weights, &input, 1, BLOCK_LEN);
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::UnsupportedDType(_))
        ));
        operation.dtype = TensorDType::F16;
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::UnsupportedDType(_))
        ));
    }

    #[test]
    fn params_encode_matches_msl_struct() {
        let weights = encode_weights(TensorDType::Q4B64, 3, 128);
        let input = [1.0_f32; 128];
        let s_in = f16_bytes(&[1.0_f32; 128]);
        let s_out = f16_bytes(&[1.0_f32; 3]);
        let bias = [0.5_f32; 3];
        let mut operation = valid_operation(TensorDType::Q4B64, &weights, &input, 3, 128);
        operation.s_in = Some(ScaleSlice::F16Le(&s_in));
        operation.s_out = Some(ScaleSlice::F16Le(&s_out));
        operation.bias = Some(&bias);
        operation.activation = Activation::Silu;
        let (layout, params) = validate_operation(&operation).unwrap();
        assert_eq!(layout.kernel_name, Q4_KERNEL_NAME);
        assert_eq!(params.blocks_per_row, 2);
        assert_eq!(params.activation, MetalActivation::Silu as u32);
        let encoded = params.encode();
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![3, 128, 2, 1, 1, 1, 1, 0]);

        let dynamic_embedding = MetalDynamicEmbeddingParams {
            columns: 5_120,
            blocks_per_row: 80,
            row_start: 1_024,
            row_end: 2_048,
            row_bytes: 2_720,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let words: Vec<u32> = dynamic_embedding
            .encode()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![5_120, 80, 1_024, 2_048, 2_720, 0, 0, 0]);

        let concat = MetalConcatParams {
            first_values: 5_120,
            second_values: 5_120,
            reserved0: 0,
            reserved1: 0,
        };
        let words: Vec<u32> = concat
            .encode()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![5_120, 5_120, 0, 0]);

        let paged = MetalPagedGqaParams {
            query_heads: 24,
            key_value_heads: 4,
            head_dim: 256,
            tokens: 131_072,
            page_tokens: 128,
            page_count: 1_024,
            combined_values: 2_048,
            q2_token_bytes: 576,
            q4_token_bytes: 1_088,
            q2_page_bytes: 73_728,
            q4_page_bytes: 139_264,
            scale: 0.0625,
        };
        let encoded = paged.encode();
        assert_eq!(encoded.len(), MetalPagedGqaParams::BYTE_LEN);
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            vec![
                24,
                4,
                256,
                131_072,
                128,
                1_024,
                2_048,
                576,
                1_088,
                73_728,
                139_264,
                0.0625_f32.to_bits(),
            ]
        );

        let gated_delta = MetalGatedDeltaParams {
            heads: 48,
            key_dim: 128,
            value_dim: 128,
            epsilon: 1.0e-6,
        };
        let encoded = gated_delta.encode();
        assert_eq!(encoded.len(), MetalGatedDeltaParams::BYTE_LEN);
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![48, 128, 128, 1.0e-6_f32.to_bits()]);

        let gated_delta_prepare = MetalGatedDeltaPrepareParams {
            key_heads: 16,
            value_heads: 48,
            key_dim: 128,
            reserved0: 0,
        };
        let encoded = gated_delta_prepare.encode();
        assert_eq!(encoded.len(), MetalGatedDeltaPrepareParams::BYTE_LEN);
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![16, 48, 128, 0]);

        let convolution = MetalCausalConvParams {
            channels: 10_240,
            kernel: 4,
            reserved0: 0,
            reserved1: 0,
        };
        let encoded = convolution.encode();
        assert_eq!(encoded.len(), MetalCausalConvParams::BYTE_LEN);
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![10_240, 4, 0, 0]);
    }

    #[test]
    fn argmax_params_encode_matches_msl_struct() {
        let encoded = MetalArgMaxParams {
            values: 248_320,
            threads: 256,
            groups: 256,
            reserved1: 0,
        }
        .encode();
        let words: Vec<u32> = encoded
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(words, vec![248_320, 256, 256, 0]);
    }

    #[test]
    fn rms_norm_params_encode_matches_msl_struct() {
        let params = MetalRmsNormParams {
            rows: 7,
            columns: 5_120,
            epsilon: 1.0e-6,
            reserved0: 0,
        };
        let encoded = params.encode();
        assert_eq!(encoded.len(), MetalRmsNormParams::BYTE_LEN);
        assert_eq!(u32::from_le_bytes(encoded[0..4].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 5_120);
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(encoded[8..12].try_into().unwrap())),
            1.0e-6
        );
        assert_eq!(u32::from_le_bytes(encoded[12..16].try_into().unwrap()), 0);
    }

    #[test]
    fn partial_rope_params_encode_matches_msl_struct() {
        let params = MetalPartialRopeParams {
            heads: 24,
            head_dim: 256,
            rotary_dim: 64,
            position: 131_071,
            theta: 10_000_000.0,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let words: Vec<u32> = params
            .encode()
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(
            words,
            vec![24, 256, 64, 131_071, 10_000_000.0_f32.to_bits(), 0, 0, 0]
        );
    }

    #[test]
    fn query_gate_params_encode_matches_msl_struct() {
        let words = MetalQueryGateParams {
            heads: 24,
            head_dim: 256,
            rotary_dim: 64,
            reserved0: 0,
            epsilon: 1.0e-6,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        }
        .encode()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
        assert_eq!(words, vec![24, 256, 64, 0, 1.0e-6_f32.to_bits(), 0, 0, 0]);
    }

    #[test]
    fn kv_pack_params_encode_matches_msl_struct() {
        let words = MetalKvPackParams {
            component_values: 1_024,
            blocks: 32,
            reserved0: 0,
            reserved1: 0,
        }
        .encode()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
        assert_eq!(words, vec![1_024, 32, 0, 0]);
    }

    #[test]
    fn mixed_segments_preserve_manifest_rows_bytes_and_kernel_choice() {
        let columns = 2 * BLOCK_LEN;
        let q2_rows = 3;
        let q4_rows = 5;
        let q2 = encode_weights(TensorDType::Q2B64, q2_rows, columns);
        let q4 = encode_weights(TensorDType::Q4B64, q4_rows, columns);
        let mut weights = q2.clone();
        weights.extend_from_slice(&q4);
        let input = vec![0.5_f32; columns];
        let s_in = f16_bytes(&vec![1.0; columns]);
        let s_out = f16_bytes(&vec![1.0; q2_rows + q4_rows]);
        let segments = vec![
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: q2_rows as u64,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2.len() as u64,
            },
            QuantSegment {
                group_index: 1,
                row_start: q2_rows as u64,
                row_end: (q2_rows + q4_rows) as u64,
                dtype: TensorDType::Q4B64,
                offset: q2.len() as u64,
                length: q4.len() as u64,
            },
        ];
        let operation = FusedMatVec {
            dtype: TensorDType::MixedQ2Q4B64,
            weights: &weights,
            segments: &segments,
            rows: q2_rows + q4_rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F16Le(&s_in)),
            s_out: Some(ScaleSlice::F16Le(&s_out)),
            bias: None,
            activation: Activation::Silu,
        };
        let dispatches = validate_mixed_operation(&operation).expect("valid mixed layout");
        assert_eq!(dispatches.len(), 2);
        assert_eq!(dispatches[0].layout.dtype, TensorDType::Q2B64);
        assert_eq!(dispatches[0].row_start, 0);
        assert_eq!(dispatches[0].weight_offset, 0);
        assert_eq!(dispatches[0].params.rows, q2_rows as u32);
        assert_eq!(dispatches[1].layout.dtype, TensorDType::Q4B64);
        assert_eq!(dispatches[1].row_start, q2_rows);
        assert_eq!(dispatches[1].weight_offset, q2.len());
        assert_eq!(dispatches[1].params.rows, q4_rows as u32);
        assert_eq!(
            dispatches[1].params.activation,
            MetalActivation::Silu as u32
        );

        let mut gapped = segments.clone();
        gapped[1].row_start -= 1;
        let invalid = FusedMatVec {
            segments: &gapped,
            ..operation
        };
        assert!(matches!(
            validate_mixed_operation(&invalid),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn rejects_zero_and_misaligned_shapes() {
        let weights = encode_weights(TensorDType::Q2B64, 1, BLOCK_LEN);
        let input = [0.0_f32; BLOCK_LEN];
        for (rows, columns) in [(0, BLOCK_LEN), (1, 0), (1, BLOCK_LEN - 1), (1, 96)] {
            let operation = valid_operation(TensorDType::Q2B64, &weights, &input, rows, columns);
            assert!(
                matches!(validate_operation(&operation), Err(EngineError::Shape(_))),
                "shape {rows}x{columns} must fail closed"
            );
        }
    }

    #[test]
    fn rejects_wrong_buffer_lengths() {
        let columns = 128;
        let rows = 2;
        let weights = encode_weights(TensorDType::Q2B64, rows, columns);
        let input = [0.0_f32; 128];
        let mut operation = valid_operation(TensorDType::Q2B64, &weights, &input, rows, columns);

        operation.weights = &weights[..weights.len() - 1];
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::Shape(_))
        ));

        let short_input = [0.0_f32; 64];
        operation = valid_operation(TensorDType::Q2B64, &weights, &short_input, rows, columns);
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::Shape(_))
        ));

        let s_in = f16_bytes(&[1.0_f32; 64]);
        operation = valid_operation(TensorDType::Q2B64, &weights, &input, rows, columns);
        operation.s_in = Some(ScaleSlice::F16Le(&s_in));
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::Shape(_))
        ));

        let s_out = f16_bytes(&[1.0_f32; 3]);
        let bias = [0.0_f32; 1];
        operation.s_in = None;
        operation.s_out = Some(ScaleSlice::F16Le(&s_out));
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::Shape(_))
        ));
        operation.s_out = None;
        operation.bias = Some(&bias);
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::Shape(_))
        ));

        let expanded_scales = [1.0_f32; 128];
        operation = valid_operation(TensorDType::Q2B64, &weights, &input, rows, columns);
        operation.s_in = Some(ScaleSlice::F32(&expanded_scales));
        assert!(matches!(
            validate_operation(&operation),
            Err(EngineError::UnsupportedDType(_))
        ));
    }

    #[test]
    fn backend_stays_fail_closed_with_candidate_status() {
        let backend = MetalBackend;
        assert_eq!(backend.kind(), BackendKind::Metal);
        assert_eq!(backend.promotion_state(), PromotionState::Contract);
        assert_eq!(backend.profile(), "metal-candidate");

        let weights = encode_weights(TensorDType::Q2B64, 1, BLOCK_LEN);
        let input = [1.0_f32; BLOCK_LEN];
        let valid = valid_operation(TensorDType::Q2B64, &weights, &input, 1, BLOCK_LEN);
        assert!(matches!(
            backend.fused_matvec(&valid),
            Err(EngineError::UnsupportedOperation {
                backend: "metal",
                ..
            })
        ));

        // Invalid operations fail validation first and never reach dispatch.
        let invalid = valid_operation(TensorDType::Q2B64, &weights, &input, 1, 96);
        assert!(matches!(
            backend.fused_matvec(&invalid),
            Err(EngineError::Shape(_))
        ));

        // No scalar fallback under production policy.
        let _ = ExecutionPolicy::Production;
    }

    #[test]
    fn kernel_source_declares_direct_msl_entry_points() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = crate_root.join(KERNEL_SOURCE_PATH);
        let text = std::fs::read_to_string(&source).expect("kernel source must be vendored");
        for name in [
            Q2_KERNEL_NAME,
            Q4_KERNEL_NAME,
            Q2_SWIGLU_KERNEL_NAME,
            Q4_SWIGLU_KERNEL_NAME,
            Q2_SIGMOID_GATE_KERNEL_NAME,
            Q4_SIGMOID_GATE_KERNEL_NAME,
            Q2_GATHERED_KERNEL_NAME,
            Q4_GATHERED_KERNEL_NAME,
            Q2_RECOVERED_ROW_KERNEL_NAME,
            Q4_RECOVERED_ROW_KERNEL_NAME,
            RMS_NORM_1P_KERNEL_NAME,
            RESIDUAL_RMS_NORM_1P_KERNEL_NAME,
            PARTIAL_ROPE_KERNEL_NAME,
            QUERY_GATE_NORM_ROPE_KERNEL_NAME,
            PAGED_GQA_DECODE_KERNEL_NAME,
            GATED_DELTA_F16_KERNEL_NAME,
            GATED_DELTA_PREP_F32_KERNEL_NAME,
            CAUSAL_CONV_F16_KERNEL_NAME,
            ARGMAX_F32_PARTIAL_KERNEL_NAME,
            ARGMAX_F32_FINAL_KERNEL_NAME,
        ] {
            assert!(
                text.contains(&format!("kernel void {name}")),
                "MSL source must define {name}"
            );
        }
        assert!(!text.contains("mlx"), "no MLX dependency permitted");
        assert!(
            !text.contains("MPSGraph"),
            "no MPSGraph dependency permitted"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn kernel_source_compiles_with_metal_frontend() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = crate_root.join(KERNEL_SOURCE_PATH);
        let output = std::env::temp_dir().join(format!(
            "qwen38-metal-test-{}-{}.air",
            std::process::id(),
            "q2q4"
        ));
        let status = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                source.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
            ])
            .status();
        let _ = std::fs::remove_file(&output);
        match status {
            Ok(status) => assert!(status.success(), "xcrun metal must compile the candidate"),
            Err(error) => panic!("xcrun metal unavailable in verify environment: {error}"),
        }
    }
}
