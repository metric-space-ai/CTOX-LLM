//! Direct Metal execution for the unpromoted Q2/Q4 fused-matvec candidate.
//!
//! This module deliberately does not implement [`super::Backend`]. It exists
//! to generate same-device verifier and benchmark evidence while the public
//! Metal backend remains fail-closed at `PromotionState::Contract`.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::mem::size_of_val;
use std::rc::Rc;
use std::slice;

use metal_driver::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use sha2::{Digest, Sha256};

use super::metal::{
    validate_mixed_operation, validate_operation, validate_recovered_row,
    MetalArgMaxFinalBufferAbi, MetalArgMaxParams, MetalArgMaxPartialBufferAbi, MetalBufferAbi,
    MetalCausalConvBufferAbi, MetalCausalConvParams, MetalFusedMatVecParams,
    MetalGatedDeltaBufferAbi, MetalGatedDeltaParams, MetalGatedDeltaPrepareBufferAbi,
    MetalGatedDeltaPrepareParams, MetalGatedRmsNormBufferAbi, MetalKvPackParams,
    MetalKvQ4PackBufferAbi, MetalKvQ4ToQ2BufferAbi, MetalPagedGqaBufferAbi, MetalPagedGqaParams,
    MetalPartialRopeBufferAbi, MetalPartialRopeParams, MetalQueryGateBufferAbi,
    MetalQueryGateParams, MetalResidualRmsNormBufferAbi, MetalRmsNormBufferAbi, MetalRmsNormParams,
    MetalSigmoidGateBufferAbi, MetalSwiGluBufferAbi, ARGMAX_F32_FINAL_KERNEL_NAME,
    ARGMAX_F32_PARTIAL_KERNEL_NAME, CAUSAL_CONV_F16_KERNEL_NAME, GATED_DELTA_F16_KERNEL_NAME,
    GATED_DELTA_PREP_F32_KERNEL_NAME, KV_Q4_PACK_KERNEL_NAME, KV_Q4_TO_Q2_KERNEL_NAME,
    MAX_SIMDGROUPS_PER_THREADGROUP, PAGED_GQA_DECODE_KERNEL_NAME, PARTIAL_ROPE_KERNEL_NAME,
    Q2_GATHERED_KERNEL_NAME, Q2_KERNEL_NAME, Q2_RECOVERED_ROW_KERNEL_NAME,
    Q2_SIGMOID_GATE_KERNEL_NAME, Q2_SWIGLU_KERNEL_NAME, Q4_GATHERED_KERNEL_NAME, Q4_KERNEL_NAME,
    Q4_RECOVERED_ROW_KERNEL_NAME, Q4_SIGMOID_GATE_KERNEL_NAME, Q4_SWIGLU_KERNEL_NAME,
    QUERY_GATE_NORM_ROPE_KERNEL_NAME, RESIDUAL_RMS_NORM_1P_KERNEL_NAME, RMS_NORM_1P_KERNEL_NAME,
    RMS_NORM_GATED_KERNEL_NAME,
};
use super::metal_graph::{
    MetalBoundDecodeStep, MetalDecodeBindingPlan, MetalDecodeBufferBinding,
    MetalDecodeExecutionCursor, MetalDecodeWorkspacePlan,
};
use super::metal_schedule::{MetalBufferSlot, MetalDecodeOperation};
use super::{Activation, FusedMatVec, ScaleSlice};
use crate::config::LayerKind;
use crate::format::TensorDType;
use crate::kv_cache::KvPrecision;
#[cfg(test)]
use crate::kv_cache::{PagedKvAppendCheckpoint, PagedKvCache};
use crate::loader::{FloatTensorView, ModelArtifact, RecoveredMatrixView};
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Qwen38Config, Result};

const KERNEL_SOURCE: &str = include_str!("../../kernels/metal/q2q4_fused_matvec.metal");
#[cfg(test)]
const COPY_F32_KERNEL_NAME: &str = "qwen_copy_f32";
const MAX_THREADS_PER_GROUP: usize = MAX_SIMDGROUPS_PER_THREADGROUP * 32;
const DEFAULT_SIMDGROUPS: usize = 2;
const ROWS_PER_SIMDGROUP: usize = 4;
const METAL_PAGED_KV_DESCRIPTOR_BYTES: usize = 16;

/// An explicitly verifier-only Metal runtime owning compiled MSL pipelines.
///
/// Creating this object compiles the in-crate source through the native Metal
/// driver. Callers may prepare and retain shared Metal buffers across
/// dispatches; zero-copy artifact import and full-graph ownership remain
/// separate promotion requirements and are not claimed by this type.
pub struct MetalCandidateRuntime {
    device: Device,
    queue: CommandQueue,
    q2_pipeline: ComputePipelineState,
    q4_pipeline: ComputePipelineState,
    q2_swiglu_pipeline: ComputePipelineState,
    q4_swiglu_pipeline: ComputePipelineState,
    q2_sigmoid_gate_pipeline: ComputePipelineState,
    q4_sigmoid_gate_pipeline: ComputePipelineState,
    q2_gathered_pipeline: ComputePipelineState,
    q4_gathered_pipeline: ComputePipelineState,
    q2_recovered_row_pipeline: ComputePipelineState,
    q4_recovered_row_pipeline: ComputePipelineState,
    rms_norm_1p_pipeline: ComputePipelineState,
    residual_rms_norm_1p_pipeline: ComputePipelineState,
    rms_norm_gated_pipeline: ComputePipelineState,
    partial_rope_pipeline: ComputePipelineState,
    query_gate_norm_rope_pipeline: ComputePipelineState,
    #[cfg(test)]
    copy_f32_pipeline: ComputePipelineState,
    kv_q4_pack_pipeline: ComputePipelineState,
    kv_q4_to_q2_pipeline: ComputePipelineState,
    paged_gqa_decode_pipeline: ComputePipelineState,
    gated_delta_f16_pipeline: ComputePipelineState,
    gated_delta_prepare_f32_pipeline: ComputePipelineState,
    causal_conv_f16_pipeline: ComputePipelineState,
    argmax_f32_partial_pipeline: ComputePipelineState,
    argmax_f32_final_pipeline: ComputePipelineState,
}

/// Device buffers for one prepared projection. Weight and recovery buffers
/// are immutable and stay resident across dispatches; only the small input
/// buffer needs updating between decode tokens.
pub struct PreparedMetalMatVec {
    dtype: TensorDType,
    rows: usize,
    columns: usize,
    thread_width: usize,
    weights_buffer: Buffer,
    input_buffer: Buffer,
    s_in_buffer: Buffer,
    s_out_buffer: Buffer,
    bias_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    resident_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MetalCorrectionIdentity {
    columns: usize,
    s_in_sha256: [u8; 32],
}

/// One input/recovery pair shared by a complete model fan-out.
pub struct PreparedMetalActivation {
    columns: usize,
    correction_identity: MetalCorrectionIdentity,
    input_buffer: Buffer,
    s_in_buffer: Buffer,
    resident_bytes: usize,
}

/// Matrix-local state consuming a separately owned shared activation.
pub struct PreparedMetalProjection {
    dtype: TensorDType,
    rows: usize,
    columns: usize,
    thread_width: usize,
    correction_identity: MetalCorrectionIdentity,
    weights_buffer: Buffer,
    s_out_buffer: Buffer,
    bias_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    resident_bytes: usize,
}

/// One device allocation backing every transient single-token decode slot.
///
/// Slot offsets come from the schedule-derived alias plan; this owner never
/// creates a projection-local activation buffer. Persistent model/state
/// allocations remain separate graph resources.
pub struct PreparedMetalDecodeWorkspace {
    plan: MetalDecodeWorkspacePlan,
    buffer: Buffer,
}

/// One exact read or write view into the shared decode arena.
pub struct PreparedMetalDecodeBufferView<'a> {
    binding: MetalDecodeBufferBinding,
    buffer: &'a Buffer,
    offset: u64,
}

/// The actual shared-buffer views and immutable graph resources for one of the
/// 645 frozen decode steps. Kernels consume these views without allocating
/// operation-local activation buffers.
pub struct PreparedMetalDecodeStepView<'a> {
    step: &'a MetalBoundDecodeStep,
    reads: Vec<PreparedMetalDecodeBufferView<'a>>,
    writes: Vec<PreparedMetalDecodeBufferView<'a>>,
}

/// A complete, validated set of real Metal buffer views for one token program.
pub struct PreparedMetalDecodeProgram<'a> {
    workspace: &'a PreparedMetalDecodeWorkspace,
    plan: &'a MetalDecodeBindingPlan,
    steps: Vec<PreparedMetalDecodeStepView<'a>>,
}

/// RAII owner for one token's exact cursor and speculative target+MTP state.
/// An incomplete or failed attempt restores every state owner on drop; only a
/// successful final command-buffer completion can consume the transaction and
/// return a new committed token position.
pub struct PreparedMetalDecodeAttempt<'a> {
    runtime: &'a MetalCandidateRuntime,
    program: &'a PreparedMetalDecodeProgram<'a>,
    cursor: Option<MetalDecodeExecutionCursor<'a>>,
    transaction: &'a mut PreparedMetalSpeculativeTransaction,
    attentions: &'a mut [PreparedMetalPagedGqa],
    convolutions: &'a mut [PreparedMappedMetalCausalConv],
    recurrences: &'a mut [PreparedMetalGatedDelta],
    finished: bool,
}

/// Bounded device-only snapshot used to restore one f32 arena slot after a
/// rejected speculative branch. The checkpoint has no host mirror.
pub struct PreparedMetalF32Checkpoint {
    values: usize,
    snapshot: Buffer,
    active: bool,
}

/// Atomic speculative-state coordinator for the frozen Qwen target+MTP graph.
/// It owns the target-hidden checkpoint while the 17 paged-attention and 48
/// paired linear-state owners remain in their model-layer resources.
pub struct PreparedMetalSpeculativeTransaction {
    target_hidden: PreparedMetalF32Checkpoint,
    active: bool,
    poisoned: bool,
}

/// One shared Metal view over the complete immutable CTOXQ file mapping.
///
/// `new_buffer_with_bytes_no_copy` does not retain the Rust mmap owner. The
/// owner is therefore the final field of the inner object: Metal releases the
/// buffer before the `ModelArtifact` clone can unmap its address range.
#[derive(Clone)]
pub struct MappedMetalArtifact {
    inner: Rc<MappedMetalArtifactInner>,
}

struct MappedMetalArtifactInner {
    buffer: Buffer,
    artifact: ModelArtifact,
}

/// Reusable projection whose immutable weights and recovery scales are
/// offsets into one shared no-copy CTOXQ mapping. Standalone projections own
/// input/output buffers, external-input projections own only output, and
/// shared-arena graph projections own neither activation endpoint.
pub struct PreparedMappedMetalMatVec {
    dtype: TensorDType,
    rows: usize,
    columns: usize,
    weights_base: u64,
    s_in_offset: u64,
    s_out_base: u64,
    dispatches: Vec<MappedMetalDispatch>,
    mapping: MappedMetalArtifact,
    input_buffer: Option<Buffer>,
    bias_buffer: Buffer,
    output_buffer: Option<Buffer>,
    transient_bytes: usize,
}

struct MappedMetalDispatch {
    dtype: TensorDType,
    rows: usize,
    thread_width: usize,
    weights_offset: u64,
    s_out_offset: u64,
    bias_offset: u64,
    output_offset: u64,
    params_buffer: Buffer,
}

/// Batched arbitrary-row projection used by the restricted MTP LM head.
/// Quant codes and both recovery scales remain in the shared CTOXQ mapping;
/// only canonical row IDs, one input vector, and requested scalar logits are
/// transient Metal buffers.
pub struct PreparedMappedMetalGatheredMatVec {
    columns: usize,
    requested_rows: usize,
    s_in_offset: u64,
    mapping: MappedMetalArtifact,
    input_buffer: Buffer,
    bias_buffer: Buffer,
    output_buffer: Buffer,
    dispatches: Vec<MappedMetalGatherDispatch>,
    transient_bytes: usize,
}

struct MappedMetalGatherDispatch {
    dtype: TensorDType,
    requested_rows: usize,
    thread_width: usize,
    weights_offset: u64,
    s_out_offset: u64,
    output_offset: u64,
    row_ids_buffer: Buffer,
    params_buffer: Buffer,
}

/// One recovered embedding row decoded directly from the shared CTOXQ mmap.
/// The complete packed row and FP16 correction tensors are mapping offsets;
/// standalone dispatch allocates the resulting f32 hidden vector; graph
/// dispatch binds an arena view and retains only the 32-byte parameter block.
pub struct PreparedMappedMetalRecoveredRow {
    dtype: TensorDType,
    columns: usize,
    thread_width: usize,
    mapping: MappedMetalArtifact,
    weights_offset: u64,
    s_in_offset: u64,
    s_out_offset: u64,
    output_buffer: Buffer,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// Complete recovered embedding table retained through one no-copy artifact
/// mapping. Token changes select a row by binding a different byte offset;
/// quantized weights and recovery scales are never repacked or duplicated.
pub struct PreparedMappedMetalEmbedding {
    rows: usize,
    columns: usize,
    mapping: MappedMetalArtifact,
    weights_base: u64,
    s_in_offset: u64,
    s_out_base: u64,
    segments: Vec<MappedMetalEmbeddingSegment>,
    output_buffer: Option<Buffer>,
    params_buffer: Buffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy)]
struct MappedMetalEmbeddingSegment {
    dtype: TensorDType,
    row_start: usize,
    row_end: usize,
    weights_offset: u64,
    row_bytes: usize,
    thread_width: usize,
}

/// Qwen `(1 + weight)` RMSNorm with an mmap-backed FP16 weight vector.
/// Standalone input/output can remain reusable f32 buffers, while graph mode
/// binds both endpoints to shared-arena views. No expanded f32 weight copy is
/// created at load time.
pub struct PreparedMappedMetalRmsNorm {
    rows: usize,
    columns: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// GatedDeltaNet's direct-weight RMSNorm fused with `SiLU(z)`. The FP16 norm
/// weight remains an mmap offset. Standalone verification may own reusable
/// f32 I/O, while decode-graph execution binds shared-arena views.
pub struct PreparedMappedMetalGatedRmsNorm {
    rows: usize,
    columns: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Option<Buffer>,
    gate_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// Reusable in-place Qwen partial-RoPE operation for one flattened set of
/// heads. Position can be updated without reallocating the activation buffer.
pub struct PreparedMetalPartialRope {
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    values_buffer: Option<Buffer>,
    cosine_buffer: Buffer,
    sine_buffer: Buffer,
    params_buffer: Buffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalPagedGqaConfig {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub maximum_tokens: usize,
    pub page_tokens: usize,
    pub sink_tokens: usize,
    pub recent_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetalGatedDeltaConfig {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub epsilon: f32,
}

impl MetalGatedDeltaConfig {
    pub const QWEN38_27B: Self = Self {
        heads: 48,
        key_dim: 128,
        value_dim: 128,
        epsilon: 1.0e-6,
    };
}

/// Persistent FP16 GatedDeltaNet recurrence state. Standalone verification may
/// own reusable f32 inputs/output; graph execution binds shared-arena views.
/// State never has an f32 device duplicate.
pub struct PreparedMetalGatedDelta {
    config: MetalGatedDeltaConfig,
    owner_layer: Option<usize>,
    query_buffer: Option<Buffer>,
    key_buffer: Option<Buffer>,
    value_buffer: Option<Buffer>,
    log_decay_buffer: Option<Buffer>,
    beta_buffer: Option<Buffer>,
    state_buffer: Buffer,
    checkpoint_buffer: Buffer,
    output_buffer: Option<Buffer>,
    params_buffer: Buffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    checkpoint_valid: bool,
    poisoned: bool,
}

/// Immutable mmap-backed A_log/dt_bias resources for the exact Qwen
/// GatedDelta preparation step. All five dynamic outputs live in the shared
/// decode arena and no operation-local activation buffer is retained.
pub struct PreparedMappedMetalGatedDeltaPrepare {
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    mapping: MappedMetalArtifact,
    a_log_offset: u64,
    dt_bias_offset: u64,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// Mmap-backed FP16 convolution weight and persistent FP16 history for one
/// linear-attention layer. Standalone verification owns reusable f32 I/O;
/// graph execution binds the in-place `LinearQkv` arena view instead.
pub struct PreparedMappedMetalCausalConv {
    channels: usize,
    kernel: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Option<Buffer>,
    state_buffer: Buffer,
    checkpoint_buffer: Buffer,
    output_buffer: Option<Buffer>,
    params_buffer: Buffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    checkpoint_valid: bool,
    poisoned: bool,
}

/// The three immutable recovered projections for one exact target
/// full-attention fan-out. All projections share one mmap and one packed
/// recovery input scale; their dynamic I/O is supplied by decode-arena views.
pub struct PreparedMappedMetalFullAttentionFanout {
    layer: usize,
    projections: [PreparedMappedMetalMatVec; 3],
}

/// Canonical full-attention sigmoid gate plus recovered output projection.
/// All activation endpoints remain graph-owned shared-arena views.
pub struct PreparedMappedMetalAttentionOutput {
    layer: usize,
    projection: PreparedMappedMetalMatVec,
}

/// Closed resource owner for one exact target full-attention transformer
/// layer, including its packed KV state and the common residual/FFN tail.
pub struct PreparedMappedMetalFullAttentionLayer {
    layer: usize,
    fanout: PreparedMappedMetalFullAttentionFanout,
    query_gate: PreparedMappedMetalQueryGate,
    key_rope: PreparedMetalPartialRope,
    attention: PreparedMetalPagedGqa,
    attention_output: PreparedMappedMetalAttentionOutput,
    residual_rms_norm: PreparedMappedMetalRmsNorm,
    ffn_gate_up: [PreparedMappedMetalMatVec; 2],
    swiglu_down: PreparedMappedMetalMatVec,
    post_ffn_residual_rms_norm: PreparedMappedMetalRmsNorm,
}

/// One exact target transformer layer in frozen model order. The enum keeps
/// the persistent state and immutable mmap-backed resources coupled to the
/// topology decision, so a caller cannot substitute a same-shape attention
/// implementation at dispatch time.
pub enum PreparedMappedMetalTargetLayer {
    LinearAttention(PreparedMappedMetalLinearAttentionLayer),
    FullAttention(PreparedMappedMetalFullAttentionLayer),
}

/// Atomic owner for all 64 target transformer layers. Construction returns
/// only after every tensor identity and persistent state allocation succeeds;
/// an error drops the partial vector and exposes no incomplete target graph.
pub struct PreparedMappedMetalTargetLayers {
    layers: Vec<PreparedMappedMetalTargetLayer>,
    transaction_active: bool,
    poisoned: bool,
}

/// Closed no-copy resource owner for target embedding, initial norm, all 64
/// transformer layers, and the full target LM head. MTP remains a separate
/// resource package until its native Metal draft/verify graph is complete.
pub struct PreparedMappedMetalTargetCore {
    embedding: PreparedMappedMetalEmbedding,
    initial_norm: PreparedMappedMetalRmsNorm,
    layers: PreparedMappedMetalTargetLayers,
    lm_head: PreparedMappedMetalMatVec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetalTargetCheckpointKind {
    FullAttention,
    LinearConvolution,
    LinearRecurrence,
}

/// Mmap-backed query RMSNorm plus reusable partial-RoPE tables for one exact
/// full-attention layer. QueryGate/Query/AttentionGate remain arena views.
pub struct PreparedMappedMetalQueryGate {
    layer: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    epsilon: f32,
    mapping: MappedMetalArtifact,
    q_norm_weight_offset: u64,
    cosine_buffer: Buffer,
    sine_buffer: Buffer,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// Complete immutable resources and persistent state for one target
/// linear-attention transformer layer. Every projection and parameter remains
/// bound to one admitted CTOXQ mmap; only the convolution and recurrent state
/// own persistent mutable device memory.
pub struct PreparedMappedMetalLinearAttentionLayer {
    layer: usize,
    projections: [PreparedMappedMetalMatVec; 4],
    convolution: PreparedMappedMetalCausalConv,
    gated_delta_prepare: PreparedMappedMetalGatedDeltaPrepare,
    recurrence: PreparedMetalGatedDelta,
    gated_rms_norm: PreparedMappedMetalGatedRmsNorm,
    linear_output_projection: PreparedMappedMetalMatVec,
    residual_rms_norm: PreparedMappedMetalRmsNorm,
    ffn_gate_up: [PreparedMappedMetalMatVec; 2],
    swiglu_down: PreparedMappedMetalMatVec,
    post_ffn_residual_rms_norm: PreparedMappedMetalRmsNorm,
}

/// Reusable bounded state for deterministic device-resident target selection.
/// It deliberately owns no logit input; graph execution binds it directly to
/// the resident LM-head output.
pub struct PreparedMetalArgMaxScratch {
    values: usize,
    groups: usize,
    partials_buffer: Buffer,
    result_buffer: Buffer,
    params_buffer: Buffer,
    transient_bytes: usize,
}

impl PreparedMetalArgMaxScratch {
    pub fn values(&self) -> usize {
        self.values
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

/// Standalone verifier owner for deterministic device-resident target
/// selection. Complete graph assembly instead uses
/// [`PreparedMetalArgMaxScratch`] and binds the same pipelines directly to the
/// mapped LM-head output.
pub struct PreparedMetalArgMax {
    input_buffer: Buffer,
    scratch: PreparedMetalArgMaxScratch,
    resident_bytes: usize,
}

impl PreparedMetalArgMax {
    pub fn values(&self) -> usize {
        self.scratch.values
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn groups(&self) -> usize {
        self.scratch.groups
    }

    pub fn write_input(&mut self, input: &[f32]) -> Result<()> {
        if input.len() != self.scratch.values {
            return Err(EngineError::Shape(format!(
                "Metal argmax input has {} values, expected {}",
                input.len(),
                self.scratch.values
            )));
        }
        write_buffer_range(
            &self.input_buffer,
            0,
            as_bytes(input),
            self.scratch.values * std::mem::size_of::<f32>(),
        )
    }
}

/// Decode-only grouped-query attention retaining K/V pages in their packed
/// Q2/Q4 representation. The Q2 arena has one deterministic slot per logical
/// page; the bounded Q4 arena retains only sink, recent, and one boundary page.
pub struct PreparedMetalPagedGqa {
    owner_layer: Option<usize>,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    maximum_tokens: usize,
    page_tokens: usize,
    q2_token_bytes: usize,
    q4_token_bytes: usize,
    q2_page_bytes: usize,
    q4_page_bytes: usize,
    q4_slots: usize,
    cache: MetalPagedKvMetadata,
    #[cfg(test)]
    verifier_cache: PagedKvCache,
    #[cfg(test)]
    verifier_key_snapshot_buffer: Buffer,
    #[cfg(test)]
    verifier_value_snapshot_buffer: Buffer,
    page_to_q4_slot: Vec<Option<usize>>,
    free_q4_slots: Vec<usize>,
    q2_pages_buffer: Buffer,
    q4_pages_buffer: Buffer,
    descriptors_buffer: Buffer,
    query_buffer: Option<Buffer>,
    output_buffer: Option<Buffer>,
    kv_token_pack_params_buffer: Buffer,
    kv_page_demote_params_buffer: Buffer,
    params_buffer: Buffer,
    packed_device_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
    speculative_checkpoint: Option<MetalPagedGqaCheckpoint>,
}

struct MetalPagedGqaCheckpoint {
    cache: MetalPagedKvMetadataCheckpoint,
    #[cfg(test)]
    verifier_cache: PagedKvAppendCheckpoint,
    page_to_q4_slot: Vec<Option<usize>>,
    free_q4_slots: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPagedKvPageMetadata {
    tokens: usize,
    precision: KvPrecision,
}

#[derive(Debug, Clone)]
struct MetalPagedKvMetadata {
    maximum_tokens: usize,
    page_tokens: usize,
    sink_tokens: usize,
    recent_tokens: usize,
    tokens: usize,
    pages: Vec<MetalPagedKvPageMetadata>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetalPagedKvMetadataCheckpoint {
    tokens: usize,
    pages: usize,
    last_page_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MetalPagedKvUpdate {
    page_index: usize,
    token_in_page: usize,
    demoted_pages: Vec<usize>,
}

struct MetalPagedGqaAppendPlan {
    demotions: Vec<(usize, usize)>,
    q4_slot: usize,
    token_in_page: usize,
    #[cfg(test)]
    verifier_update: MetalPagedKvUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ValidatedMetalFullAttentionLayer {
    layer: usize,
    component_values: usize,
    thread_width: usize,
}

impl MetalPagedKvMetadata {
    fn new(
        maximum_tokens: usize,
        page_tokens: usize,
        sink_tokens: usize,
        recent_tokens: usize,
    ) -> Self {
        Self {
            maximum_tokens,
            page_tokens,
            sink_tokens,
            recent_tokens,
            tokens: 0,
            pages: Vec::new(),
        }
    }

    fn tokens(&self) -> usize {
        self.tokens
    }

    fn push(&mut self, retain_q4: bool) -> Result<MetalPagedKvUpdate> {
        if self.tokens >= self.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "Metal paged KV metadata reached {} tokens",
                self.maximum_tokens
            )));
        }
        if self
            .pages
            .last()
            .is_none_or(|page| page.tokens == self.page_tokens)
        {
            self.pages.push(MetalPagedKvPageMetadata {
                tokens: 0,
                precision: KvPrecision::Q4,
            });
        }
        let page_index = self.pages.len() - 1;
        let page = &mut self.pages[page_index];
        let token_in_page = page.tokens;
        page.tokens += 1;
        self.tokens += 1;
        let mut demoted_pages = Vec::new();
        if !retain_q4 {
            let recent_start = self.tokens.saturating_sub(self.recent_tokens);
            for (index, page) in self.pages.iter_mut().enumerate() {
                let page_start = index * self.page_tokens;
                let page_end = page_start + page.tokens;
                if page.precision == KvPrecision::Q4
                    && page_start >= self.sink_tokens
                    && page_end <= recent_start
                {
                    page.precision = KvPrecision::Q2;
                    demoted_pages.push(index);
                }
            }
        }
        Ok(MetalPagedKvUpdate {
            page_index,
            token_in_page,
            demoted_pages,
        })
    }

    fn checkpoint(&self) -> MetalPagedKvMetadataCheckpoint {
        MetalPagedKvMetadataCheckpoint {
            tokens: self.tokens,
            pages: self.pages.len(),
            last_page_tokens: self.pages.last().map_or(0, |page| page.tokens),
        }
    }

    fn restore(&mut self, checkpoint: MetalPagedKvMetadataCheckpoint) -> Result<()> {
        if checkpoint.tokens > self.tokens
            || checkpoint.pages > self.pages.len()
            || (checkpoint.pages == 0 && checkpoint.last_page_tokens != 0)
            || checkpoint.last_page_tokens > self.page_tokens
        {
            return Err(EngineError::InvalidState(
                "Metal paged KV checkpoint is not a metadata prefix".into(),
            ));
        }
        self.pages.truncate(checkpoint.pages);
        if let Some(last) = self.pages.last_mut() {
            last.tokens = checkpoint.last_page_tokens;
        }
        self.tokens = checkpoint.tokens;
        Ok(())
    }

    fn reset(&mut self) {
        self.tokens = 0;
        self.pages.clear();
    }

    #[cfg(test)]
    fn q2_tokens(&self) -> usize {
        self.pages
            .iter()
            .filter(|page| page.precision == KvPrecision::Q2)
            .map(|page| page.tokens)
            .sum()
    }
}

impl PreparedMetalMatVec {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Exact requested Metal buffer bytes owned by this prepared operation.
    /// Allocator page rounding is intentionally not inferred here and must be
    /// measured separately for full-process residency evidence.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// Replaces the decode input without reallocating weights, corrections,
    /// output, or parameter buffers.
    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.columns)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }
}

impl PreparedMetalActivation {
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.columns)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }
}

impl PreparedMetalDecodeWorkspace {
    pub fn total_bytes(&self) -> usize {
        self.plan.total_bytes()
    }

    pub fn binding(&self, slot: MetalBufferSlot) -> Result<MetalDecodeBufferBinding> {
        self.plan.binding(slot)
    }

    /// Returns the single shared buffer and the validated byte offset for a
    /// slot. Encoders must bind the returned offset instead of allocating a
    /// slot-local buffer.
    pub fn buffer_and_offset(&self, slot: MetalBufferSlot) -> Result<(&Buffer, u64)> {
        let binding = self.binding(slot)?;
        let offset = u64::try_from(binding.offset)
            .map_err(|_| EngineError::MemoryBudget("Metal slot offset exceeds u64".into()))?;
        Ok((&self.buffer, offset))
    }

    /// Resolve every logical read/write in the frozen binding plan to this
    /// workspace's one real Metal allocation. The returned program borrows the
    /// arena, preventing it from being dropped while encoders retain views.
    pub fn bind_decode_program<'a>(
        &'a self,
        plan: &'a MetalDecodeBindingPlan,
    ) -> Result<PreparedMetalDecodeProgram<'a>> {
        if plan.steps().len() != 645 {
            return Err(EngineError::InvalidState(format!(
                "Metal decode program has {} steps, expected 645",
                plan.steps().len()
            )));
        }
        let steps = plan
            .steps()
            .iter()
            .enumerate()
            .map(|(expected_index, step)| {
                if step.schedule_index != expected_index {
                    return Err(EngineError::InvalidState(format!(
                        "Metal decode program step {} has schedule index {}",
                        expected_index, step.schedule_index
                    )));
                }
                Ok(PreparedMetalDecodeStepView {
                    step,
                    reads: self.bind_decode_slots(&step.reads, "read")?,
                    writes: self.bind_decode_slots(&step.writes, "write")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedMetalDecodeProgram {
            workspace: self,
            plan,
            steps,
        })
    }

    fn bind_decode_slots<'a>(
        &'a self,
        slots: &[MetalBufferSlot],
        access: &str,
    ) -> Result<Vec<PreparedMetalDecodeBufferView<'a>>> {
        let mut seen = BTreeSet::new();
        slots
            .iter()
            .map(|&slot| {
                if !seen.insert(slot) {
                    return Err(EngineError::InvalidState(format!(
                        "Metal decode step repeats {access} slot {slot:?}"
                    )));
                }
                let binding = self.binding(slot)?;
                let (buffer, offset) = self.buffer_and_offset(slot)?;
                Ok(PreparedMetalDecodeBufferView {
                    binding,
                    buffer,
                    offset,
                })
            })
            .collect()
    }

    /// Verifier-only host write into one exact slot. Complete graph execution
    /// binds producer kernels directly and does not round-trip activations.
    pub fn write_f32(&mut self, slot: MetalBufferSlot, values: &[f32]) -> Result<()> {
        let binding = self.binding(slot)?;
        if values.len() != binding.values {
            return Err(EngineError::Shape(format!(
                "Metal slot {slot:?} has {} values, write has {}",
                binding.values,
                values.len()
            )));
        }
        write_buffer_range(
            &self.buffer,
            binding.offset,
            as_bytes(values),
            self.total_bytes(),
        )
    }

    /// Verifier-only read of one exact slot from the shared allocation.
    pub fn read_f32(&self, slot: MetalBufferSlot) -> Result<Vec<f32>> {
        let binding = self.binding(slot)?;
        let values = unsafe {
            slice::from_raw_parts(
                self.buffer
                    .contents()
                    .cast::<u8>()
                    .add(binding.offset)
                    .cast::<f32>(),
                binding.values,
            )
        };
        Ok(values.to_vec())
    }
}

impl PreparedMetalDecodeBufferView<'_> {
    pub fn slot(&self) -> MetalBufferSlot {
        self.binding.slot
    }

    pub fn values(&self) -> usize {
        self.binding.values
    }

    pub fn bytes(&self) -> usize {
        self.binding.bytes
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn buffer(&self) -> &Buffer {
        self.buffer
    }
}

impl<'a> PreparedMetalDecodeStepView<'a> {
    pub fn step(&self) -> &'a MetalBoundDecodeStep {
        self.step
    }

    pub fn reads(&self) -> &[PreparedMetalDecodeBufferView<'a>] {
        &self.reads
    }

    pub fn writes(&self) -> &[PreparedMetalDecodeBufferView<'a>] {
        &self.writes
    }
}

impl<'a> PreparedMetalDecodeProgram<'a> {
    pub fn steps(&self) -> &[PreparedMetalDecodeStepView<'a>] {
        &self.steps
    }

    /// Return the exact ten-step slice for one frozen linear-attention layer.
    /// Every Qwen target layer occupies ten schedule positions; this accessor
    /// rejects full-attention layers and any reordered or partially bound
    /// program before a reusable layer encoder can consume its arena views.
    pub fn linear_attention_layer_steps(
        &self,
        layer: usize,
    ) -> Result<&[PreparedMetalDecodeStepView<'a>]> {
        const STEPS_PER_LAYER: usize = 10;
        const OPERATIONS: [MetalDecodeOperation; STEPS_PER_LAYER] = [
            MetalDecodeOperation::LinearFanout,
            MetalDecodeOperation::CausalConvolution,
            MetalDecodeOperation::GatedDeltaPrepare,
            MetalDecodeOperation::GatedDeltaRecurrent,
            MetalDecodeOperation::GatedRmsNorm,
            MetalDecodeOperation::LinearOutputProjection,
            MetalDecodeOperation::ResidualRmsNorm,
            MetalDecodeOperation::FfnGateUpFanout,
            MetalDecodeOperation::SwiGluDownProjection,
            MetalDecodeOperation::ResidualRmsNorm,
        ];
        let start = layer
            .checked_mul(STEPS_PER_LAYER)
            .and_then(|offset| offset.checked_add(2))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal layer schedule index overflows".into())
            })?;
        let end = start.checked_add(STEPS_PER_LAYER).ok_or_else(|| {
            EngineError::MemoryBudget("Metal layer schedule range overflows".into())
        })?;
        let steps = self.steps.get(start..end).ok_or_else(|| {
            EngineError::InvalidState(format!(
                "Metal decode program does not contain complete layer {layer}"
            ))
        })?;
        if steps
            .iter()
            .zip(OPERATIONS)
            .enumerate()
            .any(|(offset, (step, operation))| {
                step.step().schedule_index != start + offset
                    || step.step().layer != Some(layer)
                    || step.step().operation != operation
            })
        {
            return Err(EngineError::InvalidState(format!(
                "Metal decode program layer {layer} is not the frozen linear-attention sequence"
            )));
        }
        Ok(steps)
    }

    /// Return the exact ten-step slice for one frozen full-attention layer.
    /// This is the graph-side contract used while binding query/gate RoPE,
    /// packed KV append, paged GQA, and the gated output projection to the
    /// same shared activation arena as the surrounding transformer layers.
    pub fn full_attention_layer_steps(
        &self,
        layer: usize,
    ) -> Result<&[PreparedMetalDecodeStepView<'a>]> {
        const STEPS_PER_LAYER: usize = 10;
        const OPERATIONS: [MetalDecodeOperation; STEPS_PER_LAYER] = [
            MetalDecodeOperation::FullAttentionFanout,
            MetalDecodeOperation::QueryGateNormRope,
            MetalDecodeOperation::KeyRope,
            MetalDecodeOperation::PagedKvAppend,
            MetalDecodeOperation::PagedGqa,
            MetalDecodeOperation::AttentionGateOutputProjection,
            MetalDecodeOperation::ResidualRmsNorm,
            MetalDecodeOperation::FfnGateUpFanout,
            MetalDecodeOperation::SwiGluDownProjection,
            MetalDecodeOperation::ResidualRmsNorm,
        ];
        let start = layer
            .checked_mul(STEPS_PER_LAYER)
            .and_then(|offset| offset.checked_add(2))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal layer schedule index overflows".into())
            })?;
        let end = start.checked_add(STEPS_PER_LAYER).ok_or_else(|| {
            EngineError::MemoryBudget("Metal layer schedule range overflows".into())
        })?;
        let steps = self.steps.get(start..end).ok_or_else(|| {
            EngineError::InvalidState(format!(
                "Metal decode program does not contain complete layer {layer}"
            ))
        })?;
        if steps
            .iter()
            .zip(OPERATIONS)
            .enumerate()
            .any(|(offset, (step, operation))| {
                step.step().schedule_index != start + offset
                    || step.step().layer != Some(layer)
                    || step.step().operation != operation
            })
        {
            return Err(EngineError::InvalidState(format!(
                "Metal decode program layer {layer} is not the frozen full-attention sequence"
            )));
        }
        Ok(steps)
    }
}

impl PreparedMetalDecodeAttempt<'_> {
    pub fn next_step(&self) -> Option<&PreparedMetalDecodeStepView<'_>> {
        let schedule_index = self.cursor.as_ref()?.next_step()?.schedule_index;
        self.program.steps.get(schedule_index)
    }

    pub fn advance_encoded(
        &mut self,
        schedule_index: usize,
        layer: Option<usize>,
        operation: MetalDecodeOperation,
    ) -> Result<()> {
        self.cursor
            .as_mut()
            .ok_or_else(|| EngineError::InvalidState("Metal decode attempt is finished".into()))?
            .advance(schedule_index, layer, operation)
    }

    /// Explicitly reject an incomplete token and return any restore failure.
    pub fn abort(mut self) -> Result<()> {
        let result = self.runtime.restore_speculative_transaction(
            self.transaction,
            self.program.workspace,
            self.attentions,
            self.convolutions,
            self.recurrences,
        );
        self.finished = true;
        result
    }

    /// Commit only after the sole final Metal command buffer completed.
    pub fn commit_after_completion(mut self, schedule_index: usize) -> Result<usize> {
        let expected = self
            .cursor
            .as_ref()
            .and_then(MetalDecodeExecutionCursor::next_step)
            .ok_or_else(|| EngineError::InvalidState("Metal decode attempt is finished".into()))?;
        if expected.schedule_index != schedule_index
            || expected.layer.is_some()
            || expected.operation != MetalDecodeOperation::TokenCommandBufferCommit
        {
            return Err(EngineError::InvalidState(format!(
                "Metal decode cannot commit at step {schedule_index}; next bound step is ({}, {:?}, {:?})",
                expected.schedule_index, expected.layer, expected.operation
            )));
        }
        self.runtime.commit_speculative_transaction(
            self.transaction,
            self.program.workspace,
            self.attentions,
            self.convolutions,
            self.recurrences,
        )?;
        let mut cursor = self.cursor.take().expect("validated active cursor");
        cursor.commit_after_completion(schedule_index)?;
        let committed = cursor.finish()?;
        self.finished = true;
        Ok(committed)
    }
}

impl Drop for PreparedMetalDecodeAttempt<'_> {
    fn drop(&mut self) {
        if !self.finished && self.transaction.is_active() {
            let _ = self.runtime.restore_speculative_transaction(
                self.transaction,
                self.program.workspace,
                self.attentions,
                self.convolutions,
                self.recurrences,
            );
        }
    }
}

impl PreparedMetalF32Checkpoint {
    pub fn values(&self) -> usize {
        self.values
    }

    pub fn resident_bytes(&self) -> usize {
        self.values * std::mem::size_of::<f32>()
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn commit(&mut self) -> Result<()> {
        if !self.active {
            return Err(EngineError::InvalidState(
                "Metal f32 checkpoint is not active".into(),
            ));
        }
        self.active = false;
        Ok(())
    }

    pub fn clear(&mut self) {
        zero_buffer(&self.snapshot, self.resident_bytes());
        self.active = false;
    }
}

impl PreparedMetalSpeculativeTransaction {
    pub const ATTENTION_STATES: usize = 17;
    pub const LINEAR_STATES: usize = 48;

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn target_hidden_checkpoint_bytes(&self) -> usize {
        self.target_hidden.resident_bytes()
    }
}

impl PreparedMetalProjection {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

impl MappedMetalArtifact {
    /// Logical file bytes shared by every projection. This is one mmap-backed
    /// residency, not an additional copied model allocation.
    pub fn mapped_file_bytes(&self) -> u64 {
        self.inner.artifact.file_bytes()
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    fn byte_offset(&self, bytes: &[u8], label: &str) -> Result<u64> {
        let mapped = self.inner.artifact.mapped_bytes();
        let base = mapped.as_ptr() as usize;
        let start = (bytes.as_ptr() as usize).checked_sub(base).ok_or_else(|| {
            EngineError::InvalidArtifact(format!(
                "Metal {label} does not originate from the admitted CTOXQ mapping"
            ))
        })?;
        let end = start.checked_add(bytes.len()).ok_or_else(|| {
            EngineError::Shape(format!("Metal {label} mapped range overflows usize"))
        })?;
        if bytes.is_empty() || end > mapped.len() {
            return Err(EngineError::InvalidArtifact(format!(
                "Metal {label} lies outside the admitted CTOXQ mapping"
            )));
        }
        u64::try_from(start)
            .map_err(|_| EngineError::Shape(format!("Metal {label} offset exceeds u64")))
    }

    fn float_tensor_binding(&self, name: &str) -> Result<(TensorDType, u64, usize)> {
        let tensor = self.inner.artifact.float_tensor(name)?;
        let (dtype, bytes) = match tensor {
            FloatTensorView::F16Le(bytes) => (TensorDType::F16, bytes),
            FloatTensorView::F32Le(bytes) => (TensorDType::F32, bytes),
        };
        Ok((
            dtype,
            self.byte_offset(bytes, "bound float tensor")?,
            tensor.len(),
        ))
    }
}

impl PreparedMappedMetalMatVec {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn mapped_file_bytes(&self) -> u64 {
        self.mapping.mapped_file_bytes()
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_recovered_tensor(&self, name: &str) -> Result<bool> {
        let recovered = self.mapping.inner.artifact.recovered_matrix(name)?;
        let weights_base = self
            .mapping
            .byte_offset(recovered.matrix.weights, "bound projection weights")?;
        let s_in = match recovered.s_in {
            FloatTensorView::F16Le(bytes) => bytes,
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::InvalidArtifact(format!(
                    "Metal bound projection {name} has non-FP16 s_in"
                )))
            }
        };
        let s_out = match recovered.s_out {
            FloatTensorView::F16Le(bytes) => bytes,
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::InvalidArtifact(format!(
                    "Metal bound projection {name} has non-FP16 s_out"
                )))
            }
        };
        Ok(self.dtype == recovered.matrix.dtype
            && self.rows == recovered.matrix.rows
            && self.columns == recovered.matrix.columns
            && self.weights_base == weights_base
            && self.s_in_offset == self.mapping.byte_offset(s_in, "bound projection s_in")?
            && self.s_out_base == self.mapping.byte_offset(s_out, "bound projection s_out")?)
    }

    fn packed_s_in_bytes(&self) -> Result<&[u8]> {
        let bytes = self
            .columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal s_in byte count overflows".into()))?;
        let start = usize::try_from(self.s_in_offset)
            .map_err(|_| EngineError::InvalidArtifact("Metal s_in offset exceeds usize".into()))?;
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| EngineError::InvalidArtifact("Metal s_in range overflows".into()))?;
        self.mapping
            .inner
            .artifact
            .mapped_bytes()
            .get(start..end)
            .ok_or_else(|| EngineError::InvalidArtifact("Metal s_in exceeds mapping".into()))
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.columns)?;
        let input_buffer = self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal projection consumes an upstream graph buffer and has no host input".into(),
            )
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }

    fn owned_output(&self) -> Result<&Buffer> {
        self.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph projection has no operation-local output buffer".into(),
            )
        })
    }
}

impl PreparedMappedMetalGatheredMatVec {
    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn requested_rows(&self) -> usize {
        self.requested_rows
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.columns)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }
}

impl PreparedMappedMetalRecoveredRow {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedMappedMetalEmbedding {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn owned_output(&self) -> Result<&Buffer> {
        self.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph embedding has no operation-local output buffer".into(),
            )
        })
    }
}

impl PreparedMappedMetalRmsNorm {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_weight_tensor(&self, name: &str) -> Result<bool> {
        let (dtype, offset, values) = self.mapping.float_tensor_binding(name)?;
        Ok(dtype == TensorDType::F16 && offset == self.weight_offset && values == self.columns)
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        let expected = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm input shape overflows".into()))?;
        validate_metal_input(input, expected)?;
        let input_buffer = self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal graph RMSNorm has no host input buffer".into())
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }

    fn owned_input(&self) -> Result<&Buffer> {
        self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal graph RMSNorm has no operation-local input".into())
        })
    }

    fn owned_output(&self) -> Result<&Buffer> {
        self.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal graph RMSNorm has no operation-local output".into())
        })
    }
}

impl PreparedMappedMetalGatedRmsNorm {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_weight_tensor(&self, name: &str) -> Result<bool> {
        let (dtype, offset, values) = self.mapping.float_tensor_binding(name)?;
        Ok(dtype == TensorDType::F16 && offset == self.weight_offset && values == self.columns)
    }

    pub fn has_owned_io(&self) -> bool {
        self.input_buffer.is_some() || self.gate_buffer.is_some() || self.output_buffer.is_some()
    }

    pub fn write_inputs(&self, input: &[f32], gate: &[f32]) -> Result<()> {
        let expected = self.rows.checked_mul(self.columns).ok_or_else(|| {
            EngineError::Shape("Metal gated RMSNorm input shape overflows".into())
        })?;
        validate_metal_input(input, expected)?;
        validate_metal_input(gate, expected)?;
        let input_buffer = self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph gated RMSNorm has no operation-local input".into(),
            )
        })?;
        let gate_buffer = self.gate_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph gated RMSNorm has no operation-local gate".into(),
            )
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents().cast::<f32>(),
                input.len(),
            );
            std::ptr::copy_nonoverlapping(
                gate.as_ptr(),
                gate_buffer.contents().cast::<f32>(),
                gate.len(),
            );
        }
        Ok(())
    }
}

impl PreparedMetalPartialRope {
    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn has_owned_values(&self) -> bool {
        self.values_buffer.is_some()
    }

    pub fn write_values(&self, values: &[f32]) -> Result<()> {
        let expected = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal RoPE value shape overflows".into()))?;
        validate_metal_input(values, expected)?;
        let values_buffer = self.values_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph RoPE has no operation-local activation buffer".into(),
            )
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                values_buffer.contents().cast::<f32>(),
                values.len(),
            );
        }
        Ok(())
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        let params = partial_rope_params(
            self.heads,
            self.head_dim,
            self.rotary_dim,
            position,
            self.theta,
        )?;
        let (cosine, sine) = partial_rope_tables(self.rotary_dim, position, self.theta)?;
        let encoded = params.encode();
        unsafe {
            std::ptr::copy_nonoverlapping(
                cosine.as_ptr(),
                self.cosine_buffer.contents().cast::<f32>(),
                cosine.len(),
            );
            std::ptr::copy_nonoverlapping(
                sine.as_ptr(),
                self.sine_buffer.contents().cast::<f32>(),
                sine.len(),
            );
            std::ptr::copy_nonoverlapping(
                encoded.as_ptr(),
                self.params_buffer.contents().cast::<u8>(),
                encoded.len(),
            );
        }
        Ok(())
    }
}

impl PreparedMetalPagedGqa {
    pub fn tokens(&self) -> usize {
        self.cache.tokens()
    }

    pub fn maximum_tokens(&self) -> usize {
        self.maximum_tokens
    }

    /// Fixed packed arenas plus the small descriptor table. No f32 K/V cache
    /// is allocated on the device.
    pub fn packed_device_bytes(&self) -> usize {
        self.packed_device_bytes
    }

    pub fn q2_arena_bytes(&self) -> usize {
        self.q2_page_bytes * self.page_to_q4_slot.len()
    }

    pub fn q4_arena_bytes(&self) -> usize {
        self.q4_page_bytes * self.q4_slots
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    /// Test-only CPU oracle. Release builds contain only metadata plus the
    /// packed Metal arenas and therefore cannot retain duplicate KV bytes.
    #[cfg(test)]
    pub fn verifier_cpu_packed_bytes(&self) -> usize {
        self.verifier_cache.packed_bytes()
    }

    /// Begin an append-only branch without copying the Q2/Q4 device arenas.
    /// The extra retained Q4 boundary slot guarantees that a four-token MTP
    /// branch cannot overwrite pre-branch packed pages.
    pub fn begin_speculative(&mut self) -> Result<()> {
        if self.poisoned || self.speculative_checkpoint.is_some() {
            return Err(EngineError::InvalidState(
                "Metal paged GQA checkpoint requires healthy state without an active branch".into(),
            ));
        }
        if self.free_q4_slots.is_empty() {
            return Err(EngineError::MemoryBudget(
                "Metal paged GQA has no retained Q4 boundary slot for speculation".into(),
            ));
        }
        self.speculative_checkpoint = Some(MetalPagedGqaCheckpoint {
            cache: self.cache.checkpoint(),
            #[cfg(test)]
            verifier_cache: self.verifier_cache.append_checkpoint(),
            page_to_q4_slot: self.page_to_q4_slot.clone(),
            free_q4_slots: self.free_q4_slots.clone(),
        });
        Ok(())
    }

    pub fn restore_speculative(&mut self) -> Result<()> {
        let checkpoint = self.speculative_checkpoint.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal paged GQA has no speculative checkpoint".into())
        })?;
        self.cache.restore(checkpoint.cache)?;
        #[cfg(test)]
        self.verifier_cache
            .restore_append_checkpoint(checkpoint.verifier_cache)?;
        self.page_to_q4_slot = checkpoint.page_to_q4_slot.clone();
        self.free_q4_slots = checkpoint.free_q4_slots.clone();
        write_metal_paged_gqa_descriptors(self)?;
        write_metal_paged_gqa_params(self)?;
        self.speculative_checkpoint = None;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if self.poisoned {
            return Err(EngineError::InvalidState(
                "Metal paged GQA cannot commit a poisoned speculative branch".into(),
            ));
        }
        if self.speculative_checkpoint.take().is_none() {
            return Err(EngineError::InvalidState(
                "Metal paged GQA has no speculative checkpoint".into(),
            ));
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        #[cfg(test)]
        self.verifier_cache.reset();
        self.page_to_q4_slot.fill(None);
        self.free_q4_slots = (0..self.q4_slots).rev().collect();
        zero_buffer(&self.q2_pages_buffer, self.q2_arena_bytes());
        zero_buffer(&self.q4_pages_buffer, self.q4_arena_bytes());
        zero_buffer(
            &self.descriptors_buffer,
            self.page_to_q4_slot.len() * METAL_PAGED_KV_DESCRIPTOR_BYTES,
        );
        if let Some(query) = self.query_buffer.as_ref() {
            zero_buffer(
                query,
                self.query_heads * self.head_dim * std::mem::size_of::<f32>(),
            );
        }
        if let Some(output) = self.output_buffer.as_ref() {
            zero_buffer(
                output,
                self.query_heads * self.head_dim * std::mem::size_of::<f32>(),
            );
        }
        zero_buffer(&self.params_buffer, MetalPagedGqaParams::BYTE_LEN);
        self.poisoned = false;
        self.speculative_checkpoint = None;
    }
}

impl PreparedMetalGatedDelta {
    pub fn config(&self) -> MetalGatedDeltaConfig {
        self.config
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn begin_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        if self.poisoned || self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal gated-delta checkpoint requires healthy state without an active branch"
                    .into(),
            ));
        }
        runtime.copy_buffer_range_sync(
            &self.state_buffer,
            0,
            &self.checkpoint_buffer,
            0,
            self.resident_state_bytes,
            "gated-delta checkpoint snapshot",
        )?;
        self.checkpoint_valid = true;
        Ok(())
    }

    pub fn restore_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal gated-delta has no speculative checkpoint".into(),
            ));
        }
        runtime.copy_buffer_range_sync(
            &self.checkpoint_buffer,
            0,
            &self.state_buffer,
            0,
            self.resident_state_bytes,
            "gated-delta checkpoint restore",
        )?;
        self.checkpoint_valid = false;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if self.poisoned || !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal gated-delta cannot commit a poisoned or absent speculative branch".into(),
            ));
        }
        self.checkpoint_valid = false;
        Ok(())
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn has_owned_io(&self) -> bool {
        self.query_buffer.is_some()
            && self.key_buffer.is_some()
            && self.value_buffer.is_some()
            && self.log_decay_buffer.is_some()
            && self.beta_buffer.is_some()
            && self.output_buffer.is_some()
    }

    pub fn write_step(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
    ) -> Result<()> {
        let qk_values = self
            .config
            .heads
            .checked_mul(self.config.key_dim)
            .ok_or_else(|| EngineError::Shape("Metal delta Q/K shape overflows".into()))?;
        let value_values = self
            .config
            .heads
            .checked_mul(self.config.value_dim)
            .ok_or_else(|| EngineError::Shape("Metal delta value shape overflows".into()))?;
        validate_metal_input(query, qk_values)?;
        validate_metal_input(key, qk_values)?;
        validate_metal_input(value, value_values)?;
        validate_metal_input(log_decay, self.config.heads)?;
        validate_metal_input(beta, self.config.heads)?;
        for (source, target) in [
            (query, self.query_buffer.as_ref()),
            (key, self.key_buffer.as_ref()),
            (value, self.value_buffer.as_ref()),
            (log_decay, self.log_decay_buffer.as_ref()),
            (beta, self.beta_buffer.as_ref()),
        ] {
            let target = target.ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph recurrence has no operation-local input buffers".into(),
                )
            })?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr(),
                    target.contents().cast::<f32>(),
                    source.len(),
                );
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        zero_buffer(&self.state_buffer, self.resident_state_bytes);
        zero_buffer(&self.checkpoint_buffer, self.resident_state_bytes);
        let output_bytes = self.config.heads * self.config.value_dim * size_of_val(&[0.0_f32]);
        if let Some(output_buffer) = &self.output_buffer {
            zero_buffer(output_buffer, output_bytes);
        }
        self.checkpoint_valid = false;
        self.poisoned = false;
    }

    /// Readback exists only for same-device verifier evidence. Production
    /// graph execution never materializes a second host state.
    pub fn verifier_read_state(&self) -> Vec<half::f16> {
        let values = self.resident_state_bytes / std::mem::size_of::<half::f16>();
        unsafe {
            slice::from_raw_parts(self.state_buffer.contents().cast::<half::f16>(), values).to_vec()
        }
    }
}

impl PreparedMappedMetalGatedDeltaPrepare {
    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_parameter_tensors(&self, a_log: &str, dt_bias: &str) -> Result<bool> {
        let (a_dtype, a_offset, a_values) = self.mapping.float_tensor_binding(a_log)?;
        let (dt_dtype, dt_offset, dt_values) = self.mapping.float_tensor_binding(dt_bias)?;
        Ok(a_dtype == TensorDType::F32
            && dt_dtype == TensorDType::F32
            && a_offset == self.a_log_offset
            && dt_offset == self.dt_bias_offset
            && a_values == self.value_heads
            && dt_values == self.value_heads)
    }
}

impl PreparedMappedMetalCausalConv {
    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn kernel(&self) -> usize {
        self.kernel
    }

    pub fn copied_model_bytes(&self) -> u64 {
        self.mapping.copied_model_bytes()
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn begin_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        if self.poisoned || self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal convolution checkpoint requires healthy state without an active branch"
                    .into(),
            ));
        }
        runtime.copy_buffer_range_sync(
            &self.state_buffer,
            0,
            &self.checkpoint_buffer,
            0,
            self.resident_state_bytes,
            "convolution checkpoint snapshot",
        )?;
        self.checkpoint_valid = true;
        Ok(())
    }

    pub fn restore_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal convolution has no speculative checkpoint".into(),
            ));
        }
        runtime.copy_buffer_range_sync(
            &self.checkpoint_buffer,
            0,
            &self.state_buffer,
            0,
            self.resident_state_bytes,
            "convolution checkpoint restore",
        )?;
        self.checkpoint_valid = false;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if self.poisoned || !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "Metal convolution cannot commit a poisoned or absent speculative branch".into(),
            ));
        }
        self.checkpoint_valid = false;
        Ok(())
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_weight_tensor(&self, name: &str) -> Result<bool> {
        let (dtype, offset, values) = self.mapping.float_tensor_binding(name)?;
        Ok(dtype == TensorDType::F16
            && offset == self.weight_offset
            && values == self.channels * self.kernel)
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.channels)?;
        let input_buffer = self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph convolution has no operation-local input buffer".into(),
            )
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        zero_buffer(&self.state_buffer, self.resident_state_bytes);
        zero_buffer(&self.checkpoint_buffer, self.resident_state_bytes);
        if let Some(output_buffer) = &self.output_buffer {
            zero_buffer(output_buffer, self.channels * std::mem::size_of::<f32>());
        }
        self.checkpoint_valid = false;
        self.poisoned = false;
    }

    fn owned_input(&self) -> Result<&Buffer> {
        self.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph convolution has no operation-local input buffer".into(),
            )
        })
    }

    fn owned_output(&self) -> Result<&Buffer> {
        self.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph convolution has no operation-local output buffer".into(),
            )
        })
    }

    pub fn verifier_read_state(&self) -> Vec<half::f16> {
        let values = self.resident_state_bytes / std::mem::size_of::<half::f16>();
        unsafe {
            slice::from_raw_parts(self.state_buffer.contents().cast::<half::f16>(), values).to_vec()
        }
    }
}

impl PreparedMappedMetalLinearAttentionLayer {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.convolution.resident_state_bytes() + self.recurrence.resident_state_bytes()
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }
}

impl PreparedMappedMetalFullAttentionFanout {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn transient_bytes(&self) -> usize {
        self.projections
            .iter()
            .map(PreparedMappedMetalMatVec::transient_bytes)
            .sum()
    }
}

impl PreparedMappedMetalAttentionOutput {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn transient_bytes(&self) -> usize {
        self.projection.transient_bytes()
    }
}

impl PreparedMappedMetalFullAttentionLayer {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.attention.packed_device_bytes()
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn cached_tokens(&self) -> usize {
        self.attention.tokens()
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        self.query_gate.write_position(position)?;
        self.key_rope.write_position(position)
    }
}

impl PreparedMappedMetalTargetLayer {
    pub fn layer(&self) -> usize {
        match self {
            Self::LinearAttention(layer) => layer.layer(),
            Self::FullAttention(layer) => layer.layer(),
        }
    }

    pub fn kind(&self) -> LayerKind {
        match self {
            Self::LinearAttention(_) => LayerKind::LinearAttention,
            Self::FullAttention(_) => LayerKind::FullAttention,
        }
    }

    pub fn resident_state_bytes(&self) -> usize {
        match self {
            Self::LinearAttention(layer) => layer.resident_state_bytes(),
            Self::FullAttention(layer) => layer.resident_state_bytes(),
        }
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        match self {
            Self::LinearAttention(_) => Ok(()),
            Self::FullAttention(layer) => layer.write_position(position),
        }
    }
}

impl PreparedMappedMetalTargetLayers {
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn layers(&self) -> &[PreparedMappedMetalTargetLayer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [PreparedMappedMetalTargetLayer] {
        &mut self.layers
    }

    pub fn resident_state_bytes(&self) -> Result<usize> {
        self.layers.iter().try_fold(0_usize, |total, layer| {
            total
                .checked_add(layer.resident_state_bytes())
                .ok_or_else(|| {
                    EngineError::MemoryBudget(
                        "Metal target-layer persistent state bytes overflow".into(),
                    )
                })
        })
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        for layer in &self.layers {
            layer.write_position(position)?;
        }
        Ok(())
    }

    pub fn transaction_active(&self) -> bool {
        self.transaction_active
    }

    pub fn begin_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        let config = Qwen38Config::default();
        if self.layers.len() != config.num_hidden_layers
            || self.layers.iter().enumerate().any(|(index, layer)| {
                layer.layer() != index || config.layer_kind(index) != Some(layer.kind())
            })
            || self.transaction_active
            || self.poisoned
            || self.layers.iter().any(|layer| match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    layer.convolution.poisoned
                        || layer.convolution.checkpoint_valid
                        || layer.recurrence.poisoned
                        || layer.recurrence.checkpoint_valid
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    layer.attention.poisoned || layer.attention.speculative_checkpoint.is_some()
                }
            })
        {
            return Err(EngineError::InvalidState(
                "Metal target-layer transaction requires healthy idle state".into(),
            ));
        }
        let mut started = Vec::with_capacity(self.layers.len() * 2);
        let begun = (|| {
            for (index, layer) in self.layers.iter_mut().enumerate() {
                match layer {
                    PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                        layer.convolution.begin_speculative(runtime)?;
                        started.push((index, MetalTargetCheckpointKind::LinearConvolution));
                        layer.recurrence.begin_speculative(runtime)?;
                        started.push((index, MetalTargetCheckpointKind::LinearRecurrence));
                    }
                    PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                        layer.attention.begin_speculative()?;
                        started.push((index, MetalTargetCheckpointKind::FullAttention));
                    }
                }
            }
            Ok(())
        })();
        if let Err(primary) = begun {
            let mut rollback_errors = Vec::new();
            for (index, kind) in started.into_iter().rev() {
                let restored = match (&mut self.layers[index], kind) {
                    (
                        PreparedMappedMetalTargetLayer::LinearAttention(layer),
                        MetalTargetCheckpointKind::LinearConvolution,
                    ) => layer.convolution.restore_speculative(runtime),
                    (
                        PreparedMappedMetalTargetLayer::LinearAttention(layer),
                        MetalTargetCheckpointKind::LinearRecurrence,
                    ) => layer.recurrence.restore_speculative(runtime),
                    (
                        PreparedMappedMetalTargetLayer::FullAttention(layer),
                        MetalTargetCheckpointKind::FullAttention,
                    ) => layer.attention.restore_speculative(),
                    _ => Err(EngineError::InvalidState(
                        "Metal target-layer checkpoint kind diverged from topology".into(),
                    )),
                };
                if let Err(error) = restored {
                    rollback_errors.push(error.to_string());
                }
            }
            if rollback_errors.is_empty() {
                return Err(primary);
            }
            self.poisoned = true;
            return Err(EngineError::InvalidState(format!(
                "Metal target-layer checkpoint failed ({primary}) and rollback failed: {}",
                rollback_errors.join("; ")
            )));
        }
        self.transaction_active = true;
        Ok(())
    }

    pub fn restore_speculative(&mut self, runtime: &MetalCandidateRuntime) -> Result<()> {
        if !self.transaction_active {
            return Err(EngineError::InvalidState(
                "Metal target-layer transaction is not active".into(),
            ));
        }
        let mut errors = Vec::new();
        for layer in self.layers.iter_mut().rev() {
            match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    if let Err(error) = layer.recurrence.restore_speculative(runtime) {
                        errors.push(error.to_string());
                    }
                    if let Err(error) = layer.convolution.restore_speculative(runtime) {
                        errors.push(error.to_string());
                    }
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    if let Err(error) = layer.attention.restore_speculative() {
                        errors.push(error.to_string());
                    }
                }
            }
        }
        self.transaction_active = false;
        if !errors.is_empty() {
            self.poisoned = true;
            return Err(EngineError::InvalidState(format!(
                "Metal target-layer rollback failed: {}",
                errors.join("; ")
            )));
        }
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if !self.transaction_active
            || self.poisoned
            || self.layers.iter().any(|layer| match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    layer.convolution.poisoned
                        || !layer.convolution.checkpoint_valid
                        || layer.recurrence.poisoned
                        || !layer.recurrence.checkpoint_valid
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    layer.attention.poisoned || layer.attention.speculative_checkpoint.is_none()
                }
            })
        {
            return Err(EngineError::InvalidState(
                "Metal target-layer commit requires one complete healthy transaction".into(),
            ));
        }
        for layer in &mut self.layers {
            match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    layer.convolution.commit_speculative()?;
                    layer.recurrence.commit_speculative()?;
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    layer.attention.commit_speculative()?;
                }
            }
        }
        self.transaction_active = false;
        Ok(())
    }
}

impl PreparedMappedMetalTargetCore {
    pub fn copied_model_bytes(&self) -> u64 {
        self.embedding.copied_model_bytes()
            + self.initial_norm.copied_model_bytes()
            + self.layers.copied_model_bytes()
            + self.lm_head.copied_model_bytes()
    }

    pub fn resident_state_bytes(&self) -> Result<usize> {
        self.layers.resident_state_bytes()
    }

    pub fn target_layers(&self) -> &PreparedMappedMetalTargetLayers {
        &self.layers
    }

    pub fn target_layers_mut(&mut self) -> &mut PreparedMappedMetalTargetLayers {
        &mut self.layers
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        self.layers.write_position(position)
    }

    pub fn vocabulary_rows(&self) -> usize {
        debug_assert_eq!(self.embedding.rows, self.lm_head.rows);
        self.lm_head.rows
    }

    pub fn hidden_size(&self) -> usize {
        debug_assert_eq!(self.embedding.columns, self.initial_norm.columns);
        self.initial_norm.columns
    }
}

impl PreparedMappedMetalQueryGate {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn copied_model_bytes(&self) -> u64 {
        0
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    fn matches_weight_tensor(&self, name: &str) -> Result<bool> {
        let (dtype, offset, values) = self.mapping.float_tensor_binding(name)?;
        Ok(dtype == TensorDType::F16
            && offset == self.q_norm_weight_offset
            && values == self.head_dim)
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        partial_rope_params(
            self.heads,
            self.head_dim,
            self.rotary_dim,
            position,
            self.theta,
        )?;
        let (cosine, sine) = partial_rope_tables(self.rotary_dim, position, self.theta)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                cosine.as_ptr(),
                self.cosine_buffer.contents().cast::<f32>(),
                cosine.len(),
            );
            std::ptr::copy_nonoverlapping(
                sine.as_ptr(),
                self.sine_buffer.contents().cast::<f32>(),
                sine.len(),
            );
        }
        Ok(())
    }
}

impl MetalCandidateRuntime {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| EngineError::UnsupportedOperation {
            backend: "metal",
            operation: "create candidate runtime",
            reason: "no system Metal device is available".into(),
        })?;
        let options = CompileOptions::new();
        // Numerical evidence must compare the declared equations rather than
        // compiler-specific relaxed transcendental substitutions.
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(|message| {
                EngineError::InvalidState(format!("Metal source compile failed: {message}"))
            })?;
        let q2_function = library
            .get_function(Q2_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!("Metal Q2 function lookup failed: {message}"))
            })?;
        let q4_function = library
            .get_function(Q4_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!("Metal Q4 function lookup failed: {message}"))
            })?;
        let q2_swiglu_function =
            library
                .get_function(Q2_SWIGLU_KERNEL_NAME, None)
                .map_err(|message| {
                    EngineError::InvalidState(format!(
                        "Metal Q2 SwiGLU function lookup failed: {message}"
                    ))
                })?;
        let q4_swiglu_function =
            library
                .get_function(Q4_SWIGLU_KERNEL_NAME, None)
                .map_err(|message| {
                    EngineError::InvalidState(format!(
                        "Metal Q4 SwiGLU function lookup failed: {message}"
                    ))
                })?;
        let q2_sigmoid_gate_function = library
            .get_function(Q2_SIGMOID_GATE_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 sigmoid-gate function lookup failed: {message}"
                ))
            })?;
        let q4_sigmoid_gate_function = library
            .get_function(Q4_SIGMOID_GATE_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 sigmoid-gate function lookup failed: {message}"
                ))
            })?;
        let q2_gathered_function = library
            .get_function(Q2_GATHERED_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 gathered function lookup failed: {message}"
                ))
            })?;
        let q4_gathered_function = library
            .get_function(Q4_GATHERED_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 gathered function lookup failed: {message}"
                ))
            })?;
        let q2_recovered_row_function = library
            .get_function(Q2_RECOVERED_ROW_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 recovered-row function lookup failed: {message}"
                ))
            })?;
        let q4_recovered_row_function = library
            .get_function(Q4_RECOVERED_ROW_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 recovered-row function lookup failed: {message}"
                ))
            })?;
        let rms_norm_1p_function = library
            .get_function(RMS_NORM_1P_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen RMSNorm function lookup failed: {message}"
                ))
            })?;
        let residual_rms_norm_1p_function = library
            .get_function(RESIDUAL_RMS_NORM_1P_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen residual RMSNorm function lookup failed: {message}"
                ))
            })?;
        let rms_norm_gated_function = library
            .get_function(RMS_NORM_GATED_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen gated RMSNorm function lookup failed: {message}"
                ))
            })?;
        let partial_rope_function = library
            .get_function(PARTIAL_ROPE_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen partial-RoPE function lookup failed: {message}"
                ))
            })?;
        let query_gate_norm_rope_function = library
            .get_function(QUERY_GATE_NORM_ROPE_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal query/gate norm+RoPE function lookup failed: {message}"
                ))
            })?;
        #[cfg(test)]
        let copy_f32_function =
            library
                .get_function(COPY_F32_KERNEL_NAME, None)
                .map_err(|message| {
                    EngineError::InvalidState(format!(
                        "Metal verifier copy function lookup failed: {message}"
                    ))
                })?;
        let kv_q4_pack_function =
            library
                .get_function(KV_Q4_PACK_KERNEL_NAME, None)
                .map_err(|message| {
                    EngineError::InvalidState(format!(
                        "Metal Q4 KV pack function lookup failed: {message}"
                    ))
                })?;
        let kv_q4_to_q2_function = library
            .get_function(KV_Q4_TO_Q2_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4-to-Q2 KV function lookup failed: {message}"
                ))
            })?;
        let paged_gqa_decode_function = library
            .get_function(PAGED_GQA_DECODE_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal paged GQA function lookup failed: {message}"
                ))
            })?;
        let gated_delta_f16_function = library
            .get_function(GATED_DELTA_F16_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal gated-delta function lookup failed: {message}"
                ))
            })?;
        let gated_delta_prepare_f32_function = library
            .get_function(GATED_DELTA_PREP_F32_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal gated-delta preparation function lookup failed: {message}"
                ))
            })?;
        let causal_conv_f16_function = library
            .get_function(CAUSAL_CONV_F16_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal causal-convolution function lookup failed: {message}"
                ))
            })?;
        let argmax_f32_partial_function = library
            .get_function(ARGMAX_F32_PARTIAL_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal partial argmax function lookup failed: {message}"
                ))
            })?;
        let argmax_f32_final_function = library
            .get_function(ARGMAX_F32_FINAL_KERNEL_NAME, None)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal final argmax function lookup failed: {message}"
                ))
            })?;
        let q2_pipeline = device
            .new_compute_pipeline_state_with_function(&q2_function)
            .map_err(|message| {
                EngineError::InvalidState(format!("Metal Q2 pipeline creation failed: {message}"))
            })?;
        let q4_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_function)
            .map_err(|message| {
                EngineError::InvalidState(format!("Metal Q4 pipeline creation failed: {message}"))
            })?;
        let q2_swiglu_pipeline = device
            .new_compute_pipeline_state_with_function(&q2_swiglu_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 SwiGLU pipeline creation failed: {message}"
                ))
            })?;
        let q4_swiglu_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_swiglu_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 SwiGLU pipeline creation failed: {message}"
                ))
            })?;
        if q2_swiglu_pipeline.thread_execution_width() != 32
            || q4_swiglu_pipeline.thread_execution_width() != 32
        {
            return Err(EngineError::InvalidState(format!(
                "Metal SwiGLU Q2/Q4 kernels require 32-wide simdgroups, device reports {}/{}",
                q2_swiglu_pipeline.thread_execution_width(),
                q4_swiglu_pipeline.thread_execution_width()
            )));
        }
        let q2_sigmoid_gate_pipeline = device
            .new_compute_pipeline_state_with_function(&q2_sigmoid_gate_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 sigmoid-gate pipeline creation failed: {message}"
                ))
            })?;
        let q4_sigmoid_gate_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_sigmoid_gate_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 sigmoid-gate pipeline creation failed: {message}"
                ))
            })?;
        if q2_sigmoid_gate_pipeline.thread_execution_width() != 32
            || q4_sigmoid_gate_pipeline.thread_execution_width() != 32
        {
            return Err(EngineError::InvalidState(format!(
                "Metal sigmoid-gate Q2/Q4 kernels require 32-wide simdgroups, device reports {}/{}",
                q2_sigmoid_gate_pipeline.thread_execution_width(),
                q4_sigmoid_gate_pipeline.thread_execution_width()
            )));
        }
        let q2_gathered_pipeline = device
            .new_compute_pipeline_state_with_function(&q2_gathered_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 gathered pipeline creation failed: {message}"
                ))
            })?;
        let q4_gathered_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_gathered_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 gathered pipeline creation failed: {message}"
                ))
            })?;
        let q2_recovered_row_pipeline = device
            .new_compute_pipeline_state_with_function(&q2_recovered_row_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q2 recovered-row pipeline creation failed: {message}"
                ))
            })?;
        let q4_recovered_row_pipeline = device
            .new_compute_pipeline_state_with_function(&q4_recovered_row_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 recovered-row pipeline creation failed: {message}"
                ))
            })?;
        let rms_norm_1p_pipeline = device
            .new_compute_pipeline_state_with_function(&rms_norm_1p_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen RMSNorm pipeline creation failed: {message}"
                ))
            })?;
        if rms_norm_1p_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal Qwen RMSNorm requires a 32-wide simdgroup, device reports {}",
                rms_norm_1p_pipeline.thread_execution_width()
            )));
        }
        let residual_rms_norm_1p_pipeline = device
            .new_compute_pipeline_state_with_function(&residual_rms_norm_1p_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen residual RMSNorm pipeline creation failed: {message}"
                ))
            })?;
        if residual_rms_norm_1p_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal Qwen residual RMSNorm requires a 32-wide simdgroup, device reports {}",
                residual_rms_norm_1p_pipeline.thread_execution_width()
            )));
        }
        let rms_norm_gated_pipeline = device
            .new_compute_pipeline_state_with_function(&rms_norm_gated_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen gated RMSNorm pipeline creation failed: {message}"
                ))
            })?;
        if rms_norm_gated_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal Qwen gated RMSNorm requires a 32-wide simdgroup, device reports {}",
                rms_norm_gated_pipeline.thread_execution_width()
            )));
        }
        let partial_rope_pipeline = device
            .new_compute_pipeline_state_with_function(&partial_rope_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Qwen partial-RoPE pipeline creation failed: {message}"
                ))
            })?;
        let query_gate_norm_rope_pipeline = device
            .new_compute_pipeline_state_with_function(&query_gate_norm_rope_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal query/gate norm+RoPE pipeline creation failed: {message}"
                ))
            })?;
        if query_gate_norm_rope_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal query/gate norm+RoPE requires a 32-wide simdgroup, device reports {}",
                query_gate_norm_rope_pipeline.thread_execution_width()
            )));
        }
        #[cfg(test)]
        let copy_f32_pipeline = device
            .new_compute_pipeline_state_with_function(&copy_f32_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal verifier copy pipeline creation failed: {message}"
                ))
            })?;
        let kv_q4_pack_pipeline = device
            .new_compute_pipeline_state_with_function(&kv_q4_pack_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4 KV pack pipeline creation failed: {message}"
                ))
            })?;
        let kv_q4_to_q2_pipeline = device
            .new_compute_pipeline_state_with_function(&kv_q4_to_q2_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal Q4-to-Q2 KV pipeline creation failed: {message}"
                ))
            })?;
        if kv_q4_pack_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal Q4 KV pack requires a 32-wide simdgroup, device reports {}",
                kv_q4_pack_pipeline.thread_execution_width()
            )));
        }
        let paged_gqa_decode_pipeline = device
            .new_compute_pipeline_state_with_function(&paged_gqa_decode_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal paged GQA pipeline creation failed: {message}"
                ))
            })?;
        if paged_gqa_decode_pipeline.thread_execution_width() != 32 {
            return Err(EngineError::InvalidState(format!(
                "Metal paged GQA requires a 32-wide simdgroup, device reports {}",
                paged_gqa_decode_pipeline.thread_execution_width()
            )));
        }
        let gated_delta_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&gated_delta_f16_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal gated-delta pipeline creation failed: {message}"
                ))
            })?;
        let gated_delta_prepare_f32_pipeline = device
            .new_compute_pipeline_state_with_function(&gated_delta_prepare_f32_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal gated-delta preparation pipeline creation failed: {message}"
                ))
            })?;
        let causal_conv_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&causal_conv_f16_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal causal-convolution pipeline creation failed: {message}"
                ))
            })?;
        let argmax_f32_partial_pipeline = device
            .new_compute_pipeline_state_with_function(&argmax_f32_partial_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal partial argmax pipeline creation failed: {message}"
                ))
            })?;
        let argmax_f32_final_pipeline = device
            .new_compute_pipeline_state_with_function(&argmax_f32_final_function)
            .map_err(|message| {
                EngineError::InvalidState(format!(
                    "Metal final argmax pipeline creation failed: {message}"
                ))
            })?;
        if argmax_f32_partial_pipeline.max_total_threads_per_threadgroup() < 256
            || argmax_f32_final_pipeline.max_total_threads_per_threadgroup() < 256
        {
            return Err(EngineError::InvalidState(format!(
                "Metal argmax requires 256 threads, device reports partial/final maxima of {}/{}",
                argmax_f32_partial_pipeline.max_total_threads_per_threadgroup(),
                argmax_f32_final_pipeline.max_total_threads_per_threadgroup()
            )));
        }
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            q2_pipeline,
            q4_pipeline,
            q2_swiglu_pipeline,
            q4_swiglu_pipeline,
            q2_sigmoid_gate_pipeline,
            q4_sigmoid_gate_pipeline,
            q2_gathered_pipeline,
            q4_gathered_pipeline,
            q2_recovered_row_pipeline,
            q4_recovered_row_pipeline,
            rms_norm_1p_pipeline,
            residual_rms_norm_1p_pipeline,
            rms_norm_gated_pipeline,
            partial_rope_pipeline,
            query_gate_norm_rope_pipeline,
            #[cfg(test)]
            copy_f32_pipeline,
            kv_q4_pack_pipeline,
            kv_q4_to_q2_pipeline,
            paged_gqa_decode_pipeline,
            gated_delta_f16_pipeline,
            gated_delta_prepare_f32_pipeline,
            causal_conv_f16_pipeline,
            argmax_f32_partial_pipeline,
            argmax_f32_final_pipeline,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Materializes the schedule-derived decode arena as exactly one shared
    /// Metal allocation. The plan is immutable after construction, so every
    /// later encoder observes the same slot offsets and alias decisions.
    pub fn prepare_decode_workspace(
        &self,
        plan: &MetalDecodeWorkspacePlan,
    ) -> Result<PreparedMetalDecodeWorkspace> {
        Ok(PreparedMetalDecodeWorkspace {
            plan: plan.clone(),
            buffer: new_zeroed_buffer(&self.device, plan.total_bytes())?,
        })
    }

    pub fn prepare_f32_checkpoint(&self, values: usize) -> Result<PreparedMetalF32Checkpoint> {
        let bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal checkpoint bytes overflow".into()))?;
        if bytes == 0 {
            return Err(EngineError::Shape(
                "Metal checkpoint must contain at least one value".into(),
            ));
        }
        Ok(PreparedMetalF32Checkpoint {
            values,
            snapshot: new_zeroed_buffer(&self.device, bytes)?,
            active: false,
        })
    }

    pub fn prepare_speculative_transaction(
        &self,
        config: &crate::Qwen38Config,
    ) -> Result<PreparedMetalSpeculativeTransaction> {
        if config != &crate::Qwen38Config::default() {
            return Err(EngineError::Shape(
                "Metal speculative transaction requires the frozen Qwen3.8-27B topology".into(),
            ));
        }
        Ok(PreparedMetalSpeculativeTransaction {
            target_hidden: self.prepare_f32_checkpoint(config.hidden_size)?,
            active: false,
            poisoned: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin_decode_attempt<'a>(
        &'a self,
        program: &'a PreparedMetalDecodeProgram<'a>,
        transaction: &'a mut PreparedMetalSpeculativeTransaction,
        attentions: &'a mut [PreparedMetalPagedGqa],
        convolutions: &'a mut [PreparedMappedMetalCausalConv],
        recurrences: &'a mut [PreparedMetalGatedDelta],
        token_position: usize,
        committed_tokens: usize,
        admitted_context: usize,
    ) -> Result<PreparedMetalDecodeAttempt<'a>> {
        let cursor =
            program
                .plan
                .execution_cursor(token_position, committed_tokens, admitted_context)?;
        self.begin_speculative_transaction(
            transaction,
            program.workspace,
            attentions,
            convolutions,
            recurrences,
        )?;
        Ok(PreparedMetalDecodeAttempt {
            runtime: self,
            program,
            cursor: Some(cursor),
            transaction,
            attentions,
            convolutions,
            recurrences,
            finished: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn begin_speculative_transaction(
        &self,
        transaction: &mut PreparedMetalSpeculativeTransaction,
        workspace: &PreparedMetalDecodeWorkspace,
        attentions: &mut [PreparedMetalPagedGqa],
        convolutions: &mut [PreparedMappedMetalCausalConv],
        recurrences: &mut [PreparedMetalGatedDelta],
    ) -> Result<()> {
        validate_metal_speculative_shape(
            transaction,
            workspace,
            attentions,
            convolutions,
            recurrences,
        )?;
        if transaction.active
            || transaction.poisoned
            || transaction.target_hidden.is_active()
            || attentions.iter().any(|state| {
                state.poisoned
                    || state.speculative_checkpoint.is_some()
                    || state.free_q4_slots.is_empty()
            })
            || convolutions
                .iter()
                .any(|state| state.poisoned || state.checkpoint_valid)
            || recurrences
                .iter()
                .any(|state| state.poisoned || state.checkpoint_valid)
        {
            return Err(EngineError::InvalidState(
                "Metal speculative transaction requires every state owner to be healthy and idle"
                    .into(),
            ));
        }

        let mut attention_started = 0_usize;
        let mut convolution_started = 0_usize;
        let mut recurrence_started = 0_usize;
        let begun = (|| {
            self.snapshot_workspace_f32(
                &mut transaction.target_hidden,
                workspace,
                MetalBufferSlot::Normalized,
            )?;
            for attention in attentions.iter_mut() {
                attention.begin_speculative()?;
                attention_started += 1;
            }
            for convolution in convolutions.iter_mut() {
                convolution.begin_speculative(self)?;
                convolution_started += 1;
            }
            for recurrence in recurrences.iter_mut() {
                recurrence.begin_speculative(self)?;
                recurrence_started += 1;
            }
            Ok(())
        })();
        if let Err(primary) = begun {
            let mut rollback_errors = Vec::new();
            for recurrence in recurrences[..recurrence_started].iter_mut().rev() {
                if let Err(error) = recurrence.restore_speculative(self) {
                    rollback_errors.push(error.to_string());
                }
            }
            for convolution in convolutions[..convolution_started].iter_mut().rev() {
                if let Err(error) = convolution.restore_speculative(self) {
                    rollback_errors.push(error.to_string());
                }
            }
            for attention in attentions[..attention_started].iter_mut().rev() {
                if let Err(error) = attention.restore_speculative() {
                    rollback_errors.push(error.to_string());
                }
            }
            if transaction.target_hidden.is_active() {
                if let Err(error) = self.restore_workspace_f32(
                    &mut transaction.target_hidden,
                    workspace,
                    MetalBufferSlot::Normalized,
                ) {
                    rollback_errors.push(error.to_string());
                }
            }
            if rollback_errors.is_empty() {
                return Err(primary);
            }
            transaction.poisoned = true;
            return Err(EngineError::InvalidState(format!(
                "Metal speculative begin failed ({primary}) and rollback failed: {}",
                rollback_errors.join("; ")
            )));
        }
        transaction.active = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restore_speculative_transaction(
        &self,
        transaction: &mut PreparedMetalSpeculativeTransaction,
        workspace: &PreparedMetalDecodeWorkspace,
        attentions: &mut [PreparedMetalPagedGqa],
        convolutions: &mut [PreparedMappedMetalCausalConv],
        recurrences: &mut [PreparedMetalGatedDelta],
    ) -> Result<()> {
        validate_metal_speculative_shape(
            transaction,
            workspace,
            attentions,
            convolutions,
            recurrences,
        )?;
        if !transaction.active
            || transaction.poisoned
            || !transaction.target_hidden.is_active()
            || attentions
                .iter()
                .any(|state| state.speculative_checkpoint.is_none())
            || convolutions.iter().any(|state| !state.checkpoint_valid)
            || recurrences.iter().any(|state| !state.checkpoint_valid)
        {
            return Err(EngineError::InvalidState(
                "Metal speculative restore requires one complete active transaction".into(),
            ));
        }
        let mut errors = Vec::new();
        for recurrence in recurrences.iter_mut().rev() {
            if let Err(error) = recurrence.restore_speculative(self) {
                errors.push(error.to_string());
            }
        }
        for convolution in convolutions.iter_mut().rev() {
            if let Err(error) = convolution.restore_speculative(self) {
                errors.push(error.to_string());
            }
        }
        for attention in attentions.iter_mut().rev() {
            if let Err(error) = attention.restore_speculative() {
                errors.push(error.to_string());
            }
        }
        if let Err(error) = self.restore_workspace_f32(
            &mut transaction.target_hidden,
            workspace,
            MetalBufferSlot::Normalized,
        ) {
            errors.push(error.to_string());
        }
        transaction.active = false;
        if !errors.is_empty() {
            transaction.poisoned = true;
            return Err(EngineError::InvalidState(format!(
                "Metal speculative transaction restore failed: {}",
                errors.join("; ")
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn commit_speculative_transaction(
        &self,
        transaction: &mut PreparedMetalSpeculativeTransaction,
        workspace: &PreparedMetalDecodeWorkspace,
        attentions: &mut [PreparedMetalPagedGqa],
        convolutions: &mut [PreparedMappedMetalCausalConv],
        recurrences: &mut [PreparedMetalGatedDelta],
    ) -> Result<()> {
        validate_metal_speculative_shape(
            transaction,
            workspace,
            attentions,
            convolutions,
            recurrences,
        )?;
        if !transaction.active
            || transaction.poisoned
            || !transaction.target_hidden.is_active()
            || attentions
                .iter()
                .any(|state| state.poisoned || state.speculative_checkpoint.is_none())
            || convolutions
                .iter()
                .any(|state| state.poisoned || !state.checkpoint_valid)
            || recurrences
                .iter()
                .any(|state| state.poisoned || !state.checkpoint_valid)
        {
            return Err(EngineError::InvalidState(
                "Metal speculative commit requires one complete healthy transaction".into(),
            ));
        }
        let committed = (|| {
            for recurrence in recurrences.iter_mut() {
                recurrence.commit_speculative()?;
            }
            for convolution in convolutions.iter_mut() {
                convolution.commit_speculative()?;
            }
            for attention in attentions.iter_mut() {
                attention.commit_speculative()?;
            }
            transaction.target_hidden.commit()
        })();
        transaction.active = false;
        if let Err(error) = committed {
            transaction.poisoned = true;
            return Err(EngineError::InvalidState(format!(
                "Metal speculative transaction commit failed after validation: {error}"
            )));
        }
        Ok(())
    }

    /// Snapshot one exact decode-arena slot through a device-to-device blit.
    /// No f32 values are materialized on the host.
    pub fn snapshot_workspace_f32(
        &self,
        checkpoint: &mut PreparedMetalF32Checkpoint,
        workspace: &PreparedMetalDecodeWorkspace,
        slot: MetalBufferSlot,
    ) -> Result<()> {
        let binding = workspace.binding(slot)?;
        if checkpoint.active || checkpoint.values != binding.values {
            return Err(EngineError::InvalidState(format!(
                "Metal checkpoint is active or has {} values for {slot:?} with {} values",
                checkpoint.values, binding.values
            )));
        }
        let (source, source_offset) = workspace.buffer_and_offset(slot)?;
        self.copy_buffer_range_sync(
            source,
            source_offset,
            &checkpoint.snapshot,
            0,
            binding.bytes,
            "decode checkpoint snapshot",
        )?;
        checkpoint.active = true;
        Ok(())
    }

    /// Restore one exact decode-arena slot through a device-to-device blit.
    /// Successful restore consumes the checkpoint.
    pub fn restore_workspace_f32(
        &self,
        checkpoint: &mut PreparedMetalF32Checkpoint,
        workspace: &PreparedMetalDecodeWorkspace,
        slot: MetalBufferSlot,
    ) -> Result<()> {
        let binding = workspace.binding(slot)?;
        if !checkpoint.active || checkpoint.values != binding.values {
            return Err(EngineError::InvalidState(format!(
                "Metal checkpoint is absent or has {} values for {slot:?} with {} values",
                checkpoint.values, binding.values
            )));
        }
        let (destination, destination_offset) = workspace.buffer_and_offset(slot)?;
        self.copy_buffer_range_sync(
            &checkpoint.snapshot,
            0,
            destination,
            destination_offset,
            binding.bytes,
            "decode checkpoint restore",
        )?;
        checkpoint.active = false;
        Ok(())
    }

    fn copy_buffer_range_sync(
        &self,
        source: &Buffer,
        source_offset: u64,
        destination: &Buffer,
        destination_offset: u64,
        bytes: usize,
        label: &'static str,
    ) -> Result<()> {
        let bytes = u64::try_from(bytes)
            .map_err(|_| EngineError::MemoryBudget("Metal blit length exceeds u64".into()))?;
        let source_end = source_offset
            .checked_add(bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal blit source range overflows".into()))?;
        let destination_end = destination_offset.checked_add(bytes).ok_or_else(|| {
            EngineError::MemoryBudget("Metal blit destination range overflows".into())
        })?;
        if bytes == 0 || source_end > source.length() || destination_end > destination.length() {
            return Err(EngineError::MemoryBudget(format!(
                "Metal {label} range exceeds its source or destination buffer"
            )));
        }
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label(label);
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(
            source,
            source_offset,
            destination,
            destination_offset,
            bytes,
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal {label} failed with status {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    pub fn prepare_argmax_f32(&self, input: &[f32]) -> Result<PreparedMetalArgMax> {
        self.prepare_argmax_f32_with_groups(input, 32)
    }

    pub fn prepare_argmax_f32_with_groups(
        &self,
        input: &[f32],
        groups: usize,
    ) -> Result<PreparedMetalArgMax> {
        let scratch = self.prepare_argmax_f32_scratch_with_groups(input.len(), groups)?;
        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal argmax input bytes overflow".into()))?;
        let resident_bytes = input_bytes
            .checked_add(scratch.transient_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal argmax bytes overflow".into()))?;
        Ok(PreparedMetalArgMax {
            input_buffer: buffer_with_data(&self.device, as_bytes(input)),
            scratch,
            resident_bytes,
        })
    }

    pub fn prepare_argmax_f32_scratch(&self, values: usize) -> Result<PreparedMetalArgMaxScratch> {
        self.prepare_argmax_f32_scratch_with_groups(values, 32)
    }

    pub fn prepare_argmax_f32_scratch_with_groups(
        &self,
        values: usize,
        groups: usize,
    ) -> Result<PreparedMetalArgMaxScratch> {
        if values == 0 {
            return Err(EngineError::Shape(
                "Metal argmax input must be non-empty".into(),
            ));
        }
        let encoded_values = u32::try_from(values)
            .map_err(|_| EngineError::Shape("Metal argmax input exceeds u32".into()))?;
        if groups == 0 || groups > 256 || !groups.is_power_of_two() {
            return Err(EngineError::Shape(format!(
                "Metal argmax group count must be a power of two from 1 through 256, got {groups}"
            )));
        }
        let threads = 256_u32;
        let params = MetalArgMaxParams {
            values: encoded_values,
            threads,
            groups: groups as u32,
            reserved1: 0,
        };
        let result_bytes = 2 * std::mem::size_of::<u32>();
        let partials_bytes = groups * 4 * std::mem::size_of::<u32>();
        let partials_buffer = new_zeroed_buffer(&self.device, partials_bytes)?;
        let result_buffer = new_zeroed_buffer(&self.device, result_bytes)?;
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = result_bytes
            .checked_add(partials_bytes)
            .and_then(|bytes| bytes.checked_add(MetalArgMaxParams::BYTE_LEN))
            .ok_or_else(|| EngineError::MemoryBudget("Metal argmax bytes overflow".into()))?;
        Ok(PreparedMetalArgMaxScratch {
            values,
            groups,
            partials_buffer,
            result_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    pub fn dispatch_argmax_f32(&self, prepared: &PreparedMetalArgMax) -> Result<u32> {
        self.dispatch_argmax_f32_repeated(prepared, 1)
    }

    /// Encode repeated resident selections into one command buffer. This is a
    /// graph-overhead verifier: production uses one selection after a much
    /// larger decoder graph, so a standalone commit/wait per argmax is not a
    /// representative kernel-bandwidth measurement.
    pub fn dispatch_argmax_f32_repeated(
        &self,
        prepared: &PreparedMetalArgMax,
        dispatches: usize,
    ) -> Result<u32> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "Metal argmax dispatch count must be positive".into(),
            ));
        }
        zero_buffer(
            &prepared.scratch.result_buffer,
            2 * std::mem::size_of::<u32>(),
        );
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-argmax-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        for _ in 0..dispatches {
            self.encode_argmax_f32(encoder, &prepared.input_buffer, &prepared.scratch);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal argmax command ended with {:?}",
                command_buffer.status()
            )));
        }
        self.read_argmax_result(&prepared.scratch)
    }

    fn encode_argmax_f32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        scratch: &PreparedMetalArgMaxScratch,
    ) {
        encoder.set_compute_pipeline_state(&self.argmax_f32_partial_pipeline);
        encoder.set_buffer(MetalArgMaxPartialBufferAbi::INPUT as u64, Some(input), 0);
        encoder.set_buffer(
            MetalArgMaxPartialBufferAbi::PARTIALS as u64,
            Some(&scratch.partials_buffer),
            0,
        );
        encoder.set_buffer(
            MetalArgMaxPartialBufferAbi::PARAMS as u64,
            Some(&scratch.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: scratch.groups as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&self.argmax_f32_final_pipeline);
        encoder.set_buffer(
            MetalArgMaxFinalBufferAbi::PARTIALS as u64,
            Some(&scratch.partials_buffer),
            0,
        );
        encoder.set_buffer(
            MetalArgMaxFinalBufferAbi::RESULT as u64,
            Some(&scratch.result_buffer),
            0,
        );
        encoder.set_buffer(
            MetalArgMaxFinalBufferAbi::PARAMS as u64,
            Some(&scratch.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
    }

    fn read_argmax_result(&self, scratch: &PreparedMetalArgMaxScratch) -> Result<u32> {
        let result =
            unsafe { slice::from_raw_parts(scratch.result_buffer.contents().cast::<u32>(), 2) };
        if result[1] != 0 {
            return Err(EngineError::InvalidArtifact(format!(
                "Metal argmax rejected {} non-finite logits",
                result[1]
            )));
        }
        if result[0] as usize >= scratch.values {
            return Err(EngineError::InvalidState(format!(
                "Metal argmax selected {}, input has {} values",
                result[0], scratch.values
            )));
        }
        Ok(result[0])
    }

    /// Import the complete immutable CTOXQ mmap once through Metal shared
    /// virtual memory. The returned owner must be reused by every prepared
    /// model projection so loading never creates a second full weight copy.
    pub fn map_artifact_no_copy(&self, artifact: &ModelArtifact) -> Result<MappedMetalArtifact> {
        let mapped = artifact.mapped_bytes();
        if mapped.is_empty() {
            return Err(EngineError::InvalidArtifact(
                "cannot map an empty CTOXQ artifact into Metal".into(),
            ));
        }
        let length = u64::try_from(mapped.len())
            .map_err(|_| EngineError::Shape("Metal artifact length exceeds u64".into()))?;
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            mapped.as_ptr().cast::<c_void>(),
            length,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(MappedMetalArtifact {
            inner: Rc::new(MappedMetalArtifactInner {
                buffer,
                artifact: artifact.clone(),
            }),
        })
    }

    /// Prepare a projection without copying its quantized payload or packed
    /// FP16 recovery scales. The operation must have been resolved from the
    /// exact artifact represented by `mapping`; arbitrary same-valued slices
    /// are rejected by address-range validation.
    pub fn prepare_mapped_fused_matvec(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(
            mapping,
            operation,
            DEFAULT_SIMDGROUPS,
            true,
            true,
        )
    }

    /// Prepare a projection whose input will be supplied by an upstream Metal
    /// graph buffer. No duplicate activation buffer is allocated.
    pub fn prepare_mapped_fused_matvec_external_input(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(
            mapping,
            operation,
            DEFAULT_SIMDGROUPS,
            false,
            true,
        )
    }

    /// Prepare immutable projection resources for shared-arena graph I/O.
    /// Neither input nor output activation storage is allocated here.
    pub fn prepare_mapped_fused_matvec_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(
            mapping,
            operation,
            DEFAULT_SIMDGROUPS,
            false,
            false,
        )
    }

    fn prepare_named_mapped_projection_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        name: &str,
    ) -> Result<PreparedMappedMetalMatVec> {
        let matrix = mapping.inner.artifact.recovered_matrix(name)?;
        let validation_input = vec![0.0_f32; matrix.matrix.columns];
        let operation = matrix.operation(&validation_input, Activation::Identity)?;
        let prepared = self.prepare_mapped_fused_matvec_graph_io(mapping, &operation)?;
        if !prepared.matches_recovered_tensor(name)? {
            return Err(EngineError::InvalidState(format!(
                "Metal prepared projection does not retain canonical tensor identity {name}"
            )));
        }
        Ok(prepared)
    }

    /// Prepare the canonical Q/K/V recovered projections for one frozen
    /// full-attention layer. The three matrices must share the exact packed
    /// `s_in` tensor so the fan-out has one logical corrected activation.
    pub fn prepare_mapped_full_attention_fanout(
        &self,
        mapping: &MappedMetalArtifact,
        layer: usize,
    ) -> Result<PreparedMappedMetalFullAttentionFanout> {
        let config = Qwen38Config::default();
        if config.layer_kind(layer) != Some(LayerKind::FullAttention) {
            return Err(EngineError::InvalidState(format!(
                "Metal layer {layer} is not a frozen Qwen full-attention layer"
            )));
        }
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        let projections = [
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{prefix}.q_proj.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{prefix}.k_proj.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{prefix}.v_proj.weight"),
            )?,
        ];
        let query_values = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal query width overflows".into()))?;
        let key_value_values = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV width overflows".into()))?;
        let query_gate_values = query_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("Metal query/gate width overflows".into()))?;
        if projections[0].columns != config.hidden_size
            || projections[1].columns != config.hidden_size
            || projections[2].columns != config.hidden_size
            || projections[0].rows != query_gate_values
            || projections[1].rows != key_value_values
            || projections[2].rows != key_value_values
            || projections[1].packed_s_in_bytes()? != projections[0].packed_s_in_bytes()?
            || projections[2].packed_s_in_bytes()? != projections[0].packed_s_in_bytes()?
        {
            return Err(EngineError::InvalidArtifact(format!(
                "Metal full-attention layer {layer} fan-out has incompatible shape or recovery input"
            )));
        }
        Ok(PreparedMappedMetalFullAttentionFanout { layer, projections })
    }

    /// Prepare the exact per-layer query RMSNorm and partial-RoPE tables for
    /// graph-owned QueryGate/Query/AttentionGate arena views.
    pub fn prepare_mapped_query_gate_norm_rope(
        &self,
        mapping: &MappedMetalArtifact,
        layer: usize,
        position: u64,
    ) -> Result<PreparedMappedMetalQueryGate> {
        let config = Qwen38Config::default();
        if config.layer_kind(layer) != Some(LayerKind::FullAttention) {
            return Err(EngineError::InvalidState(format!(
                "Metal layer {layer} is not a frozen Qwen full-attention layer"
            )));
        }
        partial_rope_params(
            config.num_attention_heads,
            config.head_dim,
            config.rotary_dim,
            position,
            config.rope_theta,
        )?;
        let name = format!("model.language_model.layers.{layer}.self_attn.q_norm.weight");
        let weight = mapping.inner.artifact.float_tensor(&name)?;
        let expected_weight_bytes = config
            .head_dim
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q norm weight overflows".into()))?;
        let weight_bytes = match weight {
            FloatTensorView::F16Le(bytes) if bytes.len() == expected_weight_bytes => bytes,
            FloatTensorView::F16Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal Q norm weight has {} bytes, expected {expected_weight_bytes}",
                    bytes.len()
                )))
            }
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal Q norm weight must remain packed FP16".into(),
                ))
            }
        };
        let q_norm_weight_offset = mapping.byte_offset(weight_bytes, "query RMSNorm weight")?;
        let (cosine, sine) = partial_rope_tables(config.rotary_dim, position, config.rope_theta)?;
        let params = MetalQueryGateParams {
            heads: u32::try_from(config.num_attention_heads)
                .map_err(|_| EngineError::Shape("Metal query heads exceed u32".into()))?,
            head_dim: u32::try_from(config.head_dim)
                .map_err(|_| EngineError::Shape("Metal query head dim exceeds u32".into()))?,
            rotary_dim: u32::try_from(config.rotary_dim)
                .map_err(|_| EngineError::Shape("Metal query rotary dim exceeds u32".into()))?,
            reserved0: 0,
            epsilon: config.rms_norm_epsilon,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };
        let transient_bytes = size_of_val(cosine.as_slice())
            .checked_add(size_of_val(sine.as_slice()))
            .and_then(|bytes| bytes.checked_add(MetalQueryGateParams::BYTE_LEN))
            .ok_or_else(|| EngineError::MemoryBudget("Metal query/gate bytes overflow".into()))?;
        Ok(PreparedMappedMetalQueryGate {
            layer,
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            theta: config.rope_theta,
            epsilon: config.rms_norm_epsilon,
            mapping: mapping.clone(),
            q_norm_weight_offset,
            cosine_buffer: buffer_with_data(&self.device, as_bytes(&cosine)),
            sine_buffer: buffer_with_data(&self.device, as_bytes(&sine)),
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            transient_bytes,
        })
    }

    /// Prepare the canonical recovered output projection that consumes the
    /// full-attention sigmoid gate directly from graph-owned arena views.
    pub fn prepare_mapped_attention_gate_output_projection(
        &self,
        mapping: &MappedMetalArtifact,
        layer: usize,
    ) -> Result<PreparedMappedMetalAttentionOutput> {
        let config = Qwen38Config::default();
        if config.layer_kind(layer) != Some(LayerKind::FullAttention) {
            return Err(EngineError::InvalidState(format!(
                "Metal layer {layer} is not a frozen Qwen full-attention layer"
            )));
        }
        let name = format!("model.language_model.layers.{layer}.self_attn.o_proj.weight");
        let projection = self.prepare_named_mapped_projection_graph_io(mapping, &name)?;
        let attention_values = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal attention width overflows".into()))?;
        if projection.columns != attention_values || projection.rows != config.hidden_size {
            return Err(EngineError::InvalidArtifact(format!(
                "Metal full-attention layer {layer} output projection has {}x{}, expected {}x{}",
                projection.rows, projection.columns, config.hidden_size, attention_values
            )));
        }
        Ok(PreparedMappedMetalAttentionOutput { layer, projection })
    }

    /// Prepare one closed full-attention layer owner. All immutable tensors
    /// must come from the same admitted mapping; only the packed Q2/Q4 KV
    /// cache owns persistent mutable device memory.
    pub fn prepare_mapped_full_attention_layer(
        &self,
        mapping: &MappedMetalArtifact,
        layer: usize,
        position: u64,
        cache: MetalPagedGqaConfig,
    ) -> Result<PreparedMappedMetalFullAttentionLayer> {
        let config = Qwen38Config::default();
        if config.layer_kind(layer) != Some(LayerKind::FullAttention)
            || cache.query_heads != config.num_attention_heads
            || cache.key_value_heads != config.num_key_value_heads
            || cache.head_dim != config.head_dim
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} has incompatible frozen topology or cache geometry"
            )));
        }
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let hidden_validation = vec![0.0_f32; config.hidden_size];
        let residual_rms_norm = self.prepare_mapped_rms_norm_1p_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            &hidden_validation,
            1,
            config.hidden_size,
            config.rms_norm_epsilon,
        )?;
        let ffn_gate_up = [
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{mlp_prefix}.gate_proj.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{mlp_prefix}.up_proj.weight"),
            )?,
        ];
        let swiglu_down = self.prepare_named_mapped_projection_graph_io(
            mapping,
            &format!("{mlp_prefix}.down_proj.weight"),
        )?;
        let next_norm_name = if layer + 1 == config.num_hidden_layers {
            "model.language_model.norm.weight".to_owned()
        } else {
            format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            )
        };
        let post_ffn_residual_rms_norm = self.prepare_mapped_rms_norm_1p_graph_io(
            mapping,
            mapping.inner.artifact.float_tensor(&next_norm_name)?,
            &hidden_validation,
            1,
            config.hidden_size,
            config.rms_norm_epsilon,
        )?;
        Ok(PreparedMappedMetalFullAttentionLayer {
            layer,
            fanout: self.prepare_mapped_full_attention_fanout(mapping, layer)?,
            query_gate: self.prepare_mapped_query_gate_norm_rope(mapping, layer, position)?,
            key_rope: self.prepare_partial_rope_graph(
                config.num_key_value_heads,
                config.head_dim,
                config.rotary_dim,
                position,
                config.rope_theta,
            )?,
            attention: self.prepare_paged_gqa_decode_graph(layer, cache)?,
            attention_output: self
                .prepare_mapped_attention_gate_output_projection(mapping, layer)?,
            residual_rms_norm,
            ffn_gate_up,
            swiglu_down,
            post_ffn_residual_rms_norm,
        })
    }

    /// Prepare all 16 target full-attention layers in canonical model order.
    /// A failure drops every already prepared cache and immutable resource;
    /// callers can never observe a partially admitted attention graph.
    pub fn prepare_all_mapped_full_attention_layers(
        &self,
        mapping: &MappedMetalArtifact,
        position: u64,
        cache: MetalPagedGqaConfig,
    ) -> Result<Vec<PreparedMappedMetalFullAttentionLayer>> {
        let config = Qwen38Config::default();
        let mut layers = Vec::with_capacity(config.full_attention_layers());
        for layer in 0..config.num_hidden_layers {
            if config.layer_kind(layer) == Some(LayerKind::FullAttention) {
                layers.push(
                    self.prepare_mapped_full_attention_layer(mapping, layer, position, cache)?,
                );
            }
        }
        if layers.len() != config.full_attention_layers() {
            return Err(EngineError::InvalidState(format!(
                "Metal prepared {} full-attention layers, expected {}",
                layers.len(),
                config.full_attention_layers()
            )));
        }
        Ok(layers)
    }

    /// Prepare every immutable tensor and persistent state owner for one exact
    /// target linear-attention layer. This is the model-specific resource
    /// loader used by the reusable ten-step layer encoder.
    pub fn prepare_mapped_linear_attention_layer(
        &self,
        mapping: &MappedMetalArtifact,
        layer: usize,
    ) -> Result<PreparedMappedMetalLinearAttentionLayer> {
        let config = Qwen38Config::default();
        if config.layer_kind(layer) != Some(LayerKind::LinearAttention) {
            return Err(EngineError::InvalidState(format!(
                "Metal layer {layer} is not a frozen Qwen linear-attention layer"
            )));
        }
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let linear_prefix = format!("{layer_prefix}.linear_attn");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        let projections = [
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{linear_prefix}.in_proj_qkv.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{linear_prefix}.in_proj_z.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{linear_prefix}.in_proj_a.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{linear_prefix}.in_proj_b.weight"),
            )?,
        ];
        let convolution_channels = projections[0].rows;
        let convolution_validation = vec![0.0_f32; convolution_channels];
        let convolution = self.prepare_mapped_causal_conv_f16_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{linear_prefix}.conv1d.weight"))?,
            &convolution_validation,
            convolution_channels,
            config.linear_conv_kernel_dim,
        )?;
        let gated_delta_prepare = self.prepare_mapped_gated_delta_prepare_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{linear_prefix}.A_log"))?,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{linear_prefix}.dt_bias"))?,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
        )?;
        let recurrence =
            self.prepare_gated_delta_f16_graph_io(MetalGatedDeltaConfig::QWEN38_27B, layer)?;
        let gated_values = MetalGatedDeltaConfig::QWEN38_27B
            .heads
            .checked_mul(MetalGatedDeltaConfig::QWEN38_27B.value_dim)
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal gated RMSNorm values overflow".into())
            })?;
        let gated_validation = vec![0.0_f32; gated_values];
        let gated_rms_norm = self.prepare_mapped_rms_norm_gated_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{linear_prefix}.norm.weight"))?,
            &gated_validation,
            &gated_validation,
            MetalGatedDeltaConfig::QWEN38_27B.heads,
            MetalGatedDeltaConfig::QWEN38_27B.value_dim,
            MetalGatedDeltaConfig::QWEN38_27B.epsilon,
        )?;
        let linear_output_projection = self.prepare_named_mapped_projection_graph_io(
            mapping,
            &format!("{linear_prefix}.out_proj.weight"),
        )?;
        let hidden_validation = vec![0.0_f32; config.hidden_size];
        let residual_rms_norm = self.prepare_mapped_rms_norm_1p_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
            &hidden_validation,
            1,
            config.hidden_size,
            config.rms_norm_epsilon,
        )?;
        let ffn_gate_up = [
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{mlp_prefix}.gate_proj.weight"),
            )?,
            self.prepare_named_mapped_projection_graph_io(
                mapping,
                &format!("{mlp_prefix}.up_proj.weight"),
            )?,
        ];
        let swiglu_down = self.prepare_named_mapped_projection_graph_io(
            mapping,
            &format!("{mlp_prefix}.down_proj.weight"),
        )?;
        let post_ffn_residual_rms_norm = self.prepare_mapped_rms_norm_1p_graph_io(
            mapping,
            mapping.inner.artifact.float_tensor(&format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            ))?,
            &hidden_validation,
            1,
            config.hidden_size,
            config.rms_norm_epsilon,
        )?;
        Ok(PreparedMappedMetalLinearAttentionLayer {
            layer,
            projections,
            convolution,
            gated_delta_prepare,
            recurrence,
            gated_rms_norm,
            linear_output_projection,
            residual_rms_norm,
            ffn_gate_up,
            swiglu_down,
            post_ffn_residual_rms_norm,
        })
    }

    /// Prepare all 48 target linear-attention layers in canonical layer order.
    /// Any absent or mismatched tensor aborts the complete load; already
    /// prepared partial state is dropped instead of admitting a partial graph.
    pub fn prepare_all_mapped_linear_attention_layers(
        &self,
        mapping: &MappedMetalArtifact,
    ) -> Result<Vec<PreparedMappedMetalLinearAttentionLayer>> {
        let config = Qwen38Config::default();
        let layers = (0..config.num_hidden_layers)
            .filter(|layer| config.layer_kind(*layer) == Some(LayerKind::LinearAttention))
            .map(|layer| self.prepare_mapped_linear_attention_layer(mapping, layer))
            .collect::<Result<Vec<_>>>()?;
        if layers.len() != config.linear_attention_layers() {
            return Err(EngineError::InvalidState(format!(
                "Metal prepared {} linear layers, expected {}",
                layers.len(),
                config.linear_attention_layers()
            )));
        }
        Ok(layers)
    }

    /// Prepare all 64 target layers as one topology-ordered, all-or-nothing
    /// resource set. This is the model-level ownership boundary consumed by
    /// the forthcoming complete token encoder; separate 48/16 vectors are
    /// retained only as focused verifier entry points.
    pub fn prepare_all_mapped_target_layers(
        &self,
        mapping: &MappedMetalArtifact,
        position: u64,
        cache: MetalPagedGqaConfig,
    ) -> Result<PreparedMappedMetalTargetLayers> {
        let config = Qwen38Config::default();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let expected_kind = config.layer_kind(layer).ok_or_else(|| {
                EngineError::InvalidState(format!("Metal target topology omits layer {layer}"))
            })?;
            let prepared = match expected_kind {
                LayerKind::LinearAttention => PreparedMappedMetalTargetLayer::LinearAttention(
                    self.prepare_mapped_linear_attention_layer(mapping, layer)?,
                ),
                LayerKind::FullAttention => PreparedMappedMetalTargetLayer::FullAttention(
                    self.prepare_mapped_full_attention_layer(mapping, layer, position, cache)?,
                ),
            };
            if prepared.layer() != layer || prepared.kind() != expected_kind {
                return Err(EngineError::InvalidState(format!(
                    "Metal target layer {layer} was prepared with the wrong identity"
                )));
            }
            layers.push(prepared);
        }
        if layers.len() != config.num_hidden_layers
            || layers
                .iter()
                .filter(|layer| layer.kind() == LayerKind::LinearAttention)
                .count()
                != config.linear_attention_layers()
            || layers
                .iter()
                .filter(|layer| layer.kind() == LayerKind::FullAttention)
                .count()
                != config.full_attention_layers()
        {
            return Err(EngineError::InvalidState(
                "Metal target-layer resource set does not match the frozen 48/16 topology".into(),
            ));
        }
        Ok(PreparedMappedMetalTargetLayers {
            layers,
            transaction_active: false,
            poisoned: false,
        })
    }

    /// Load the complete target-side model core from one admitted CTOXQ mmap.
    /// A missing or mismatched frontend, layer, or LM-head tensor drops every
    /// resource already created and returns no partial core.
    pub fn prepare_mapped_target_core(
        &self,
        mapping: &MappedMetalArtifact,
        position: u64,
        cache: MetalPagedGqaConfig,
    ) -> Result<PreparedMappedMetalTargetCore> {
        let config = Qwen38Config::default();
        let embedding = self.prepare_mapped_embedding_graph_output(
            mapping,
            mapping
                .inner
                .artifact
                .recovered_matrix("model.language_model.embed_tokens.weight")?,
        )?;
        let validation_hidden = vec![0.0_f32; config.hidden_size];
        let initial_norm = self.prepare_mapped_rms_norm_1p_graph_io(
            mapping,
            mapping
                .inner
                .artifact
                .float_tensor("model.language_model.layers.0.input_layernorm.weight")?,
            &validation_hidden,
            1,
            config.hidden_size,
            config.rms_norm_epsilon,
        )?;
        let layers = self.prepare_all_mapped_target_layers(mapping, position, cache)?;
        let lm_head = self.prepare_named_mapped_projection_graph_io(mapping, "lm_head.weight")?;
        let canonical = &mapping.inner;
        if embedding.rows != config.vocab_size
            || embedding.columns != config.hidden_size
            || embedding.output_buffer.is_some()
            || initial_norm.rows != 1
            || initial_norm.columns != config.hidden_size
            || initial_norm.input_buffer.is_some()
            || initial_norm.output_buffer.is_some()
            || lm_head.rows != config.vocab_size
            || lm_head.columns != config.hidden_size
            || lm_head.input_buffer.is_some()
            || lm_head.output_buffer.is_some()
            || !Rc::ptr_eq(&embedding.mapping.inner, canonical)
            || !Rc::ptr_eq(&initial_norm.mapping.inner, canonical)
            || !Rc::ptr_eq(&lm_head.mapping.inner, canonical)
            || layers.layers.iter().any(|layer| match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    !Rc::ptr_eq(&layer.projections[0].mapping.inner, canonical)
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    !Rc::ptr_eq(&layer.fanout.projections[0].mapping.inner, canonical)
                }
            })
            || !lm_head.matches_recovered_tensor("lm_head.weight")?
            || !initial_norm
                .matches_weight_tensor("model.language_model.layers.0.input_layernorm.weight")?
        {
            return Err(EngineError::InvalidState(
                "Metal target core resources are not one canonical frozen model".into(),
            ));
        }
        Ok(PreparedMappedMetalTargetCore {
            embedding,
            initial_norm,
            layers,
            lm_head,
        })
    }

    pub fn prepare_mapped_fused_matvec_with_simdgroups(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(mapping, operation, simdgroups, true, true)
    }

    fn prepare_mapped_fused_matvec_internal(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
        own_input: bool,
        own_output: bool,
    ) -> Result<PreparedMappedMetalMatVec> {
        let segment_contracts: Vec<(TensorDType, usize, usize, MetalFusedMatVecParams)> =
            if operation.dtype == TensorDType::MixedQ2Q4B64 {
                validate_mixed_operation(operation)?
                    .into_iter()
                    .map(|segment| {
                        (
                            segment.layout.dtype,
                            segment.row_start,
                            segment.weight_offset,
                            segment.params,
                        )
                    })
                    .collect()
            } else {
                let (layout, params) = validate_operation(operation)?;
                vec![(layout.dtype, 0, 0, params)]
            };
        let s_in = match operation.s_in {
            Some(ScaleSlice::F16Le(bytes)) => bytes,
            _ => {
                return Err(EngineError::InvalidArtifact(
                    "mapped Metal projection requires artifact-backed FP16 s_in".into(),
                ))
            }
        };
        let s_out = match operation.s_out {
            Some(ScaleSlice::F16Le(bytes)) => bytes,
            _ => {
                return Err(EngineError::InvalidArtifact(
                    "mapped Metal projection requires artifact-backed FP16 s_out".into(),
                ))
            }
        };
        let weights_base = mapping.byte_offset(operation.weights, "weights")?;
        let s_in_offset = mapping.byte_offset(s_in, "s_in")?;
        let s_out_base = mapping.byte_offset(s_out, "s_out")?;
        let input_buffer =
            own_input.then(|| buffer_with_data(&self.device, as_bytes(operation.input)));
        let dummy_bias = [0.0_f32];
        let bias = operation.bias.unwrap_or(&dummy_bias);
        let bias_buffer = buffer_with_data(&self.device, as_bytes(bias));
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal output byte size overflows".into()))?;
        let output_buffer = own_output.then(|| {
            self.device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared)
        });
        let mut dispatches = Vec::with_capacity(segment_contracts.len());
        for (dtype, row_start, weight_offset, params) in segment_contracts {
            let pipeline = match dtype {
                TensorDType::Q2B64 => &self.q2_pipeline,
                TensorDType::Q4B64 => &self.q4_pipeline,
                _ => unreachable!("mapped Metal segment is Q2/Q4"),
            };
            let thread_width = dispatch_width(pipeline, simdgroups)?;
            let weight_offset = weights_base
                .checked_add(u64::try_from(weight_offset).map_err(|_| {
                    EngineError::Shape("Metal segment weight offset exceeds u64".into())
                })?)
                .ok_or_else(|| EngineError::Shape("Metal weight binding overflows u64".into()))?;
            let s_out_offset = s_out_base
                .checked_add(
                    u64::try_from(row_start)
                        .map_err(|_| EngineError::Shape("Metal row start exceeds u64".into()))?
                        .checked_mul(2)
                        .ok_or_else(|| {
                            EngineError::Shape("Metal s_out binding overflows u64".into())
                        })?,
                )
                .ok_or_else(|| EngineError::Shape("Metal s_out binding overflows u64".into()))?;
            let bias_offset = if operation.bias.is_some() {
                u64::try_from(row_start)
                    .map_err(|_| EngineError::Shape("Metal row start exceeds u64".into()))?
                    .checked_mul(std::mem::size_of::<f32>() as u64)
                    .ok_or_else(|| EngineError::Shape("Metal bias binding overflows u64".into()))?
            } else {
                0
            };
            let output_offset = u64::try_from(row_start)
                .map_err(|_| EngineError::Shape("Metal row start exceeds u64".into()))?
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| EngineError::Shape("Metal output binding overflows u64".into()))?;
            dispatches.push(MappedMetalDispatch {
                dtype,
                rows: params.rows as usize,
                thread_width,
                weights_offset: weight_offset,
                s_out_offset,
                bias_offset,
                output_offset,
                params_buffer: buffer_with_data(&self.device, &params.encode()),
            });
        }
        let parameter_bytes = MetalFusedMatVecParams::BYTE_LEN
            .checked_mul(dispatches.len())
            .ok_or_else(|| EngineError::Shape("Metal parameter bytes overflow usize".into()))?;
        let input_bytes = if own_input {
            size_of_val(operation.input)
        } else {
            0
        };
        let transient_bytes = input_bytes
            .checked_add(size_of_val(bias))
            .and_then(|total| total.checked_add(if own_output { output_bytes } else { 0 }))
            .and_then(|total| total.checked_add(parameter_bytes))
            .ok_or_else(|| EngineError::Shape("Metal transient byte count overflows".into()))?;
        Ok(PreparedMappedMetalMatVec {
            dtype: operation.dtype,
            rows: operation.rows,
            columns: operation.columns,
            weights_base,
            s_in_offset,
            s_out_base,
            dispatches,
            mapping: mapping.clone(),
            input_buffer,
            bias_buffer,
            output_buffer,
            transient_bytes,
        })
    }

    /// Prepare a single batched restricted-LM-head projection. Canonical row
    /// IDs must be sorted and unique, matching the signed draft-vocabulary
    /// contract. Mixed Q2/Q4 row groups become separate dispatches in one
    /// command encoder while output order remains canonical ID order.
    pub fn prepare_mapped_gathered_matvec(
        &self,
        mapping: &MappedMetalArtifact,
        matrix: RecoveredMatrixView<'_>,
        input: &[f32],
        row_ids: &[u32],
    ) -> Result<PreparedMappedMetalGatheredMatVec> {
        if row_ids.is_empty()
            || row_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || row_ids
                .iter()
                .any(|row| usize::try_from(*row).map_or(true, |row| row >= matrix.matrix.rows))
        {
            return Err(EngineError::InvalidArtifact(
                "Metal gathered row IDs must be non-empty, canonical, and in range".into(),
            ));
        }
        let operation = matrix.operation(input, Activation::Identity)?;
        let contracts: Vec<(TensorDType, usize, usize, usize, MetalFusedMatVecParams)> =
            if operation.dtype == TensorDType::MixedQ2Q4B64 {
                validate_mixed_operation(&operation)?
                    .into_iter()
                    .map(|segment| {
                        let row_count = segment.params.rows as usize;
                        (
                            segment.layout.dtype,
                            segment.row_start,
                            segment.row_start + row_count,
                            segment.weight_offset,
                            segment.params,
                        )
                    })
                    .collect()
            } else {
                let (layout, params) = validate_operation(&operation)?;
                vec![(layout.dtype, 0, operation.rows, 0, params)]
            };
        let s_in = match operation.s_in {
            Some(ScaleSlice::F16Le(bytes)) => bytes,
            _ => {
                return Err(EngineError::InvalidArtifact(
                    "Metal gathered projection requires artifact-backed FP16 s_in".into(),
                ))
            }
        };
        let s_out = match operation.s_out {
            Some(ScaleSlice::F16Le(bytes)) => bytes,
            _ => {
                return Err(EngineError::InvalidArtifact(
                    "Metal gathered projection requires artifact-backed FP16 s_out".into(),
                ))
            }
        };
        let weights_base = mapping.byte_offset(operation.weights, "gathered weights")?;
        let s_in_offset = mapping.byte_offset(s_in, "gathered s_in")?;
        let s_out_base = mapping.byte_offset(s_out, "gathered s_out")?;
        let input_buffer = buffer_with_data(&self.device, as_bytes(input));
        let bias_buffer = buffer_with_data(&self.device, as_bytes(&[0.0_f32]));
        let output_bytes = row_ids
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal gathered output bytes overflow".into()))?;
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let mut dispatches = Vec::new();
        let mut selected_rows = 0_usize;
        for (dtype, row_start, row_end, weight_offset, mut params) in contracts {
            let request_start = row_ids.partition_point(|row| (*row as usize) < row_start);
            let request_end = row_ids.partition_point(|row| (*row as usize) < row_end);
            if request_start == request_end {
                continue;
            }
            let local_ids: Vec<u32> = row_ids[request_start..request_end]
                .iter()
                .map(|row| {
                    row.checked_sub(u32::try_from(row_start).map_err(|_| {
                        EngineError::Shape("Metal gathered segment row exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        EngineError::InvalidArtifact(
                            "Metal gathered row precedes its segment".into(),
                        )
                    })
                })
                .collect::<Result<_>>()?;
            params.rows = u32::try_from(local_ids.len()).map_err(|_| {
                EngineError::Shape("Metal gathered request count exceeds u32".into())
            })?;
            params.has_bias = 0;
            params.activation = 0;
            let pipeline = match dtype {
                TensorDType::Q2B64 => &self.q2_gathered_pipeline,
                TensorDType::Q4B64 => &self.q4_gathered_pipeline,
                _ => unreachable!("Metal gathered segment is Q2/Q4"),
            };
            let weights_offset = weights_base
                .checked_add(u64::try_from(weight_offset).map_err(|_| {
                    EngineError::Shape("Metal gathered weight offset exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    EngineError::Shape("Metal gathered weight binding overflows u64".into())
                })?;
            let s_out_offset = s_out_base
                .checked_add(
                    u64::try_from(row_start)
                        .map_err(|_| {
                            EngineError::Shape("Metal gathered row start exceeds u64".into())
                        })?
                        .checked_mul(2)
                        .ok_or_else(|| {
                            EngineError::Shape("Metal gathered s_out offset overflows".into())
                        })?,
                )
                .ok_or_else(|| {
                    EngineError::Shape("Metal gathered s_out binding overflows".into())
                })?;
            let output_offset = u64::try_from(request_start)
                .map_err(|_| EngineError::Shape("Metal gathered output index exceeds u64".into()))?
                .checked_mul(std::mem::size_of::<f32>() as u64)
                .ok_or_else(|| {
                    EngineError::Shape("Metal gathered output offset overflows".into())
                })?;
            selected_rows = selected_rows
                .checked_add(local_ids.len())
                .ok_or_else(|| EngineError::Shape("Metal gathered row count overflows".into()))?;
            dispatches.push(MappedMetalGatherDispatch {
                dtype,
                requested_rows: local_ids.len(),
                thread_width: dispatch_width(pipeline, DEFAULT_SIMDGROUPS)?,
                weights_offset,
                s_out_offset,
                output_offset,
                row_ids_buffer: buffer_with_data(&self.device, as_bytes(&local_ids)),
                params_buffer: buffer_with_data(&self.device, &params.encode()),
            });
        }
        if selected_rows != row_ids.len() || dispatches.is_empty() {
            return Err(EngineError::InvalidArtifact(
                "Metal gathered row groups do not cover every requested ID".into(),
            ));
        }
        let parameter_bytes = dispatches
            .len()
            .checked_mul(MetalFusedMatVecParams::BYTE_LEN)
            .ok_or_else(|| EngineError::Shape("Metal gathered parameter bytes overflow".into()))?;
        let row_id_bytes = row_ids
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| EngineError::Shape("Metal gathered ID bytes overflow".into()))?;
        let transient_bytes = size_of_val(input)
            .checked_add(std::mem::size_of::<f32>())
            .and_then(|total| total.checked_add(output_bytes))
            .and_then(|total| total.checked_add(row_id_bytes))
            .and_then(|total| total.checked_add(parameter_bytes))
            .ok_or_else(|| EngineError::Shape("Metal gathered transient bytes overflow".into()))?;
        Ok(PreparedMappedMetalGatheredMatVec {
            columns: operation.columns,
            requested_rows: row_ids.len(),
            s_in_offset,
            mapping: mapping.clone(),
            input_buffer,
            bias_buffer,
            output_buffer,
            dispatches,
            transient_bytes,
        })
    }

    /// Prepare one embedding lookup from an exact recovered matrix row. The
    /// selected pure or mixed Q2/Q4 row is resolved by the loader, then bound
    /// as an offset into the same complete CTOXQ mapping used by projections.
    pub fn prepare_mapped_recovered_row(
        &self,
        mapping: &MappedMetalArtifact,
        matrix: RecoveredMatrixView<'_>,
        row: usize,
    ) -> Result<PreparedMappedMetalRecoveredRow> {
        let operation = matrix.row_operation(row)?;
        let layout = validate_recovered_row(&operation)?;
        let ScaleSlice::F16Le(s_in) = operation.s_in else {
            unreachable!("validated Metal recovered-row s_in is FP16")
        };
        let ScaleSlice::F16Le(s_out) = matrix.s_out.as_recovery_scales()? else {
            unreachable!("loader recovery scales are FP16")
        };
        let weights_offset = mapping.byte_offset(operation.weights, "recovered-row weights")?;
        let s_in_offset = mapping.byte_offset(s_in, "recovered-row s_in")?;
        let s_out_base = mapping.byte_offset(s_out, "recovered-row s_out")?;
        let s_out_offset = s_out_base
            .checked_add(
                u64::try_from(row)
                    .map_err(|_| EngineError::Shape("Metal embedding row exceeds u64".into()))?
                    .checked_mul(std::mem::size_of::<half::f16>() as u64)
                    .ok_or_else(|| {
                        EngineError::Shape("Metal embedding s_out index overflows".into())
                    })?,
            )
            .ok_or_else(|| EngineError::Shape("Metal embedding s_out offset overflows".into()))?;
        let pipeline = match layout.dtype {
            TensorDType::Q2B64 => &self.q2_recovered_row_pipeline,
            TensorDType::Q4B64 => &self.q4_recovered_row_pipeline,
            _ => unreachable!("Metal recovered row is Q2 or Q4"),
        };
        let params = MetalFusedMatVecParams {
            rows: 1,
            columns: u32::try_from(operation.columns)
                .map_err(|_| EngineError::Shape("Metal embedding width exceeds u32".into()))?,
            blocks_per_row: u32::try_from(operation.columns / crate::quant::BLOCK_LEN).map_err(
                |_| EngineError::Shape("Metal embedding block count exceeds u32".into()),
            )?,
            has_s_in: 1,
            has_s_out: 1,
            has_bias: 0,
            activation: 0,
            reserved0: 0,
        };
        let output_bytes = operation
            .columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal embedding output bytes overflow".into()))?;
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = output_bytes
            .checked_add(MetalFusedMatVecParams::BYTE_LEN)
            .ok_or_else(|| EngineError::Shape("Metal embedding transient bytes overflow".into()))?;
        Ok(PreparedMappedMetalRecoveredRow {
            dtype: layout.dtype,
            columns: operation.columns,
            thread_width: dispatch_width(pipeline, DEFAULT_SIMDGROUPS)?,
            mapping: mapping.clone(),
            weights_offset,
            s_in_offset,
            s_out_offset,
            output_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    /// Prepare the complete embedding table once. Unlike
    /// [`Self::prepare_mapped_recovered_row`], this owner can select any token
    /// row at dispatch time without constructing another Metal object.
    pub fn prepare_mapped_embedding(
        &self,
        mapping: &MappedMetalArtifact,
        matrix: RecoveredMatrixView<'_>,
    ) -> Result<PreparedMappedMetalEmbedding> {
        self.prepare_mapped_embedding_internal(mapping, matrix, true)
    }

    /// Prepare the resident embedding table for a graph-owned output view.
    pub fn prepare_mapped_embedding_graph_output(
        &self,
        mapping: &MappedMetalArtifact,
        matrix: RecoveredMatrixView<'_>,
    ) -> Result<PreparedMappedMetalEmbedding> {
        self.prepare_mapped_embedding_internal(mapping, matrix, false)
    }

    fn prepare_mapped_embedding_internal(
        &self,
        mapping: &MappedMetalArtifact,
        matrix: RecoveredMatrixView<'_>,
        own_output: bool,
    ) -> Result<PreparedMappedMetalEmbedding> {
        let validation_input = vec![0.0_f32; matrix.matrix.columns];
        let operation = matrix.operation(&validation_input, Activation::Identity)?;
        let contracts = if operation.dtype == TensorDType::MixedQ2Q4B64 {
            validate_mixed_operation(&operation)?
                .into_iter()
                .map(|segment| {
                    let row_end = segment
                        .row_start
                        .checked_add(segment.params.rows as usize)
                        .ok_or_else(|| {
                            EngineError::Shape("Metal embedding segment rows overflow".into())
                        })?;
                    Ok((
                        segment.layout.dtype,
                        segment.row_start,
                        row_end,
                        segment.weight_offset,
                        segment.layout.block_bytes,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let (layout, _) = validate_operation(&operation)?;
            vec![(layout.dtype, 0, operation.rows, 0, layout.block_bytes)]
        };
        let ScaleSlice::F16Le(s_in) = operation
            .s_in
            .ok_or_else(|| EngineError::InvalidArtifact("Metal embedding has no s_in".into()))?
        else {
            unreachable!("validated Metal embedding s_in is FP16")
        };
        let ScaleSlice::F16Le(s_out) = operation
            .s_out
            .ok_or_else(|| EngineError::InvalidArtifact("Metal embedding has no s_out".into()))?
        else {
            unreachable!("validated Metal embedding s_out is FP16")
        };
        let weights_base = mapping.byte_offset(operation.weights, "embedding weights")?;
        let s_in_offset = mapping.byte_offset(s_in, "embedding s_in")?;
        let s_out_base = mapping.byte_offset(s_out, "embedding s_out")?;
        let blocks_per_row = operation.columns / BLOCK_LEN;
        let mut segments = Vec::with_capacity(contracts.len());
        for (dtype, row_start, row_end, weight_offset, block_bytes) in contracts {
            let pipeline = match dtype {
                TensorDType::Q2B64 => &self.q2_recovered_row_pipeline,
                TensorDType::Q4B64 => &self.q4_recovered_row_pipeline,
                _ => unreachable!("validated Metal embedding segment is Q2/Q4"),
            };
            let row_bytes = blocks_per_row.checked_mul(block_bytes).ok_or_else(|| {
                EngineError::Shape("Metal embedding row byte size overflows".into())
            })?;
            segments.push(MappedMetalEmbeddingSegment {
                dtype,
                row_start,
                row_end,
                weights_offset: u64::try_from(weight_offset).map_err(|_| {
                    EngineError::Shape("Metal embedding weight offset exceeds u64".into())
                })?,
                row_bytes,
                thread_width: dispatch_width(pipeline, DEFAULT_SIMDGROUPS)?,
            });
        }
        let output_bytes = operation
            .columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal embedding output bytes overflow".into()))?;
        let output_buffer = own_output.then(|| {
            self.device
                .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared)
        });
        let params = MetalFusedMatVecParams {
            rows: 1,
            columns: u32::try_from(operation.columns)
                .map_err(|_| EngineError::Shape("Metal embedding width exceeds u32".into()))?,
            blocks_per_row: u32::try_from(blocks_per_row).map_err(|_| {
                EngineError::Shape("Metal embedding block count exceeds u32".into())
            })?,
            has_s_in: 1,
            has_s_out: 1,
            has_bias: 0,
            activation: 0,
            reserved0: 0,
        };
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = (if own_output { output_bytes } else { 0 })
            .checked_add(MetalFusedMatVecParams::BYTE_LEN)
            .ok_or_else(|| EngineError::Shape("Metal embedding transient bytes overflow".into()))?;
        Ok(PreparedMappedMetalEmbedding {
            rows: operation.rows,
            columns: operation.columns,
            mapping: mapping.clone(),
            weights_base,
            s_in_offset,
            s_out_base,
            segments,
            output_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    /// Prepare Qwen's `(1 + weight)` RMSNorm with an exact mmap-backed FP16
    /// weight vector. The candidate supports one decode row and multi-row
    /// prefill without changing the weight representation.
    pub fn prepare_mapped_rms_norm_1p(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<PreparedMappedMetalRmsNorm> {
        self.prepare_mapped_rms_norm_1p_internal(
            mapping, weight, input, rows, columns, epsilon, true,
        )
    }

    /// Prepare immutable RMSNorm resources for graph-owned input/output views.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mapped_rms_norm_1p_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        validation_input: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<PreparedMappedMetalRmsNorm> {
        self.prepare_mapped_rms_norm_1p_internal(
            mapping,
            weight,
            validation_input,
            rows,
            columns,
            epsilon,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_mapped_rms_norm_1p_internal(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
        own_io: bool,
    ) -> Result<PreparedMappedMetalRmsNorm> {
        let value_count = rows
            .checked_mul(columns)
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm shape overflows".into()))?;
        if rows == 0 || columns == 0 || input.len() != value_count {
            return Err(EngineError::Shape(format!(
                "Metal RMSNorm input has {} values, expected {rows}x{columns}",
                input.len()
            )));
        }
        validate_metal_input(input, value_count)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(EngineError::Shape(
                "Metal RMSNorm epsilon must be finite and positive".into(),
            ));
        }
        let expected_weight_bytes = columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm weight bytes overflow".into()))?;
        let weight_bytes = match weight {
            FloatTensorView::F16Le(bytes) if bytes.len() == expected_weight_bytes => bytes,
            FloatTensorView::F16Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal RMSNorm weight has {} bytes, expected {}",
                    bytes.len(),
                    expected_weight_bytes
                )))
            }
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal RMSNorm weight must remain packed FP16".into(),
                ))
            }
        };
        let weight_offset = mapping.byte_offset(weight_bytes, "RMSNorm weight")?;
        let params = MetalRmsNormParams {
            rows: u32::try_from(rows)
                .map_err(|_| EngineError::Shape("Metal RMSNorm rows exceed u32".into()))?,
            columns: u32::try_from(columns)
                .map_err(|_| EngineError::Shape("Metal RMSNorm columns exceed u32".into()))?,
            epsilon,
            reserved0: 0,
        };
        let value_bytes = value_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm value bytes overflow".into()))?;
        let input_buffer = own_io.then(|| buffer_with_data(&self.device, as_bytes(input)));
        let output_buffer = own_io.then(|| {
            self.device
                .new_buffer(value_bytes as u64, MTLResourceOptions::StorageModeShared)
        });
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = value_bytes
            .checked_mul(if own_io { 2 } else { 0 })
            .and_then(|bytes| bytes.checked_add(MetalRmsNormParams::BYTE_LEN))
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm transient bytes overflow".into()))?;
        Ok(PreparedMappedMetalRmsNorm {
            rows,
            columns,
            mapping: mapping.clone(),
            weight_offset,
            input_buffer,
            output_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    /// Prepare GatedDeltaNet's direct-weight RMSNorm fused with `SiLU(gate)`.
    /// The learned FP16 weight remains inside the shared CTOXQ mapping.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mapped_rms_norm_gated(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        gate: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<PreparedMappedMetalGatedRmsNorm> {
        self.prepare_mapped_rms_norm_gated_internal(
            mapping, weight, input, gate, rows, columns, epsilon, true,
        )
    }

    /// Prepare the exact Qwen GatedDelta output normalization for graph-owned
    /// shared-arena I/O. Only its packed weight view and 16-byte geometry block
    /// remain operation-local.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_mapped_rms_norm_gated_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        validation_input: &[f32],
        validation_gate: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<PreparedMappedMetalGatedRmsNorm> {
        if rows != MetalGatedDeltaConfig::QWEN38_27B.heads
            || columns != MetalGatedDeltaConfig::QWEN38_27B.value_dim
            || epsilon != MetalGatedDeltaConfig::QWEN38_27B.epsilon
        {
            return Err(EngineError::Shape(
                "Metal graph gated RMSNorm requires the exact Qwen3.8-27B geometry".into(),
            ));
        }
        self.prepare_mapped_rms_norm_gated_internal(
            mapping,
            weight,
            validation_input,
            validation_gate,
            rows,
            columns,
            epsilon,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_mapped_rms_norm_gated_internal(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        gate: &[f32],
        rows: usize,
        columns: usize,
        epsilon: f32,
        own_io: bool,
    ) -> Result<PreparedMappedMetalGatedRmsNorm> {
        let value_count = rows
            .checked_mul(columns)
            .ok_or_else(|| EngineError::Shape("Metal gated RMSNorm shape overflows".into()))?;
        if rows == 0 || columns == 0 {
            return Err(EngineError::Shape(
                "Metal gated RMSNorm rows and columns must be positive".into(),
            ));
        }
        validate_metal_input(input, value_count)?;
        validate_metal_input(gate, value_count)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(EngineError::Shape(
                "Metal gated RMSNorm epsilon must be finite and positive".into(),
            ));
        }
        let expected_weight_bytes = columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| {
                EngineError::Shape("Metal gated RMSNorm weight bytes overflow".into())
            })?;
        let weight_bytes = match weight {
            FloatTensorView::F16Le(bytes) if bytes.len() == expected_weight_bytes => bytes,
            FloatTensorView::F16Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal gated RMSNorm weight has {} bytes, expected {}",
                    bytes.len(),
                    expected_weight_bytes
                )))
            }
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal gated RMSNorm weight must remain packed FP16".into(),
                ))
            }
        };
        let weight_offset = mapping.byte_offset(weight_bytes, "gated RMSNorm weight")?;
        let params = MetalRmsNormParams {
            rows: usize_to_u32(rows, "Metal gated RMSNorm rows")?,
            columns: usize_to_u32(columns, "Metal gated RMSNorm columns")?,
            epsilon,
            reserved0: 0,
        };
        let value_bytes = value_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal gated RMSNorm values overflow".into()))?;
        let transient_bytes = value_bytes
            .checked_mul(if own_io { 3 } else { 0 })
            .and_then(|bytes| bytes.checked_add(MetalRmsNormParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::Shape("Metal gated RMSNorm transient bytes overflow".into())
            })?;
        Ok(PreparedMappedMetalGatedRmsNorm {
            rows,
            columns,
            mapping: mapping.clone(),
            weight_offset,
            input_buffer: own_io.then(|| buffer_with_data(&self.device, as_bytes(input))),
            gate_buffer: own_io.then(|| buffer_with_data(&self.device, as_bytes(gate))),
            output_buffer: if own_io {
                Some(new_zeroed_buffer(&self.device, value_bytes)?)
            } else {
                None
            },
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            transient_bytes,
        })
    }

    /// Prepare the immutable parameters for the exact Qwen GatedDelta input
    /// transformation. Dynamic inputs and all five outputs are graph-owned
    /// shared-arena views; only the 16-byte geometry block is allocated.
    pub fn prepare_mapped_gated_delta_prepare_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        a_log: FloatTensorView<'_>,
        dt_bias: FloatTensorView<'_>,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
    ) -> Result<PreparedMappedMetalGatedDeltaPrepare> {
        if (key_heads, value_heads, key_dim) != (16, 48, 128) {
            return Err(EngineError::Shape(format!(
                "Metal GatedDelta preparation requires exact Qwen geometry 16/48/128, got {key_heads}/{value_heads}/{key_dim}"
            )));
        }
        let expected_values = value_heads;
        let expected_bytes = expected_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal GatedDelta parameters overflow".into())
            })?;
        let a_log_bytes = match a_log {
            FloatTensorView::F32Le(bytes) if bytes.len() == expected_bytes => bytes,
            FloatTensorView::F32Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal A_log has {} bytes, expected {expected_bytes}",
                    bytes.len()
                )))
            }
            FloatTensorView::F16Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal A_log must remain packed FP32".into(),
                ))
            }
        };
        let dt_bias_bytes = match dt_bias {
            FloatTensorView::F32Le(bytes) if bytes.len() == expected_bytes => bytes,
            FloatTensorView::F32Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal dt_bias has {} bytes, expected {expected_bytes}",
                    bytes.len()
                )))
            }
            FloatTensorView::F16Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal dt_bias must remain packed FP32".into(),
                ))
            }
        };
        for index in 0..expected_values {
            a_log.value(index)?;
            dt_bias.value(index)?;
        }
        let params = MetalGatedDeltaPrepareParams {
            key_heads: usize_to_u32(key_heads, "Metal GatedDelta key heads")?,
            value_heads: usize_to_u32(value_heads, "Metal GatedDelta value heads")?,
            key_dim: usize_to_u32(key_dim, "Metal GatedDelta key dimension")?,
            reserved0: 0,
        };
        Ok(PreparedMappedMetalGatedDeltaPrepare {
            key_heads,
            value_heads,
            key_dim,
            mapping: mapping.clone(),
            a_log_offset: mapping.byte_offset(a_log_bytes, "GatedDelta A_log")?,
            dt_bias_offset: mapping.byte_offset(dt_bias_bytes, "GatedDelta dt_bias")?,
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            transient_bytes: MetalGatedDeltaPrepareParams::BYTE_LEN,
        })
    }

    /// Prepare Qwen's depthwise causal convolution with an mmap-backed FP16
    /// weight and an exclusively FP16 persistent history buffer.
    pub fn prepare_mapped_causal_conv_f16(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        channels: usize,
        kernel: usize,
    ) -> Result<PreparedMappedMetalCausalConv> {
        self.prepare_mapped_causal_conv_f16_internal(mapping, weight, input, channels, kernel, true)
    }

    /// Prepare convolution weights/state for graph-owned in-place arena I/O.
    pub fn prepare_mapped_causal_conv_f16_graph_io(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        validation_input: &[f32],
        channels: usize,
        kernel: usize,
    ) -> Result<PreparedMappedMetalCausalConv> {
        self.prepare_mapped_causal_conv_f16_internal(
            mapping,
            weight,
            validation_input,
            channels,
            kernel,
            false,
        )
    }

    fn prepare_mapped_causal_conv_f16_internal(
        &self,
        mapping: &MappedMetalArtifact,
        weight: FloatTensorView<'_>,
        input: &[f32],
        channels: usize,
        kernel: usize,
        own_io: bool,
    ) -> Result<PreparedMappedMetalCausalConv> {
        if channels == 0 || kernel == 0 || kernel > 32 {
            return Err(EngineError::Shape(
                "invalid Metal causal-convolution geometry".into(),
            ));
        }
        validate_metal_input(input, channels)?;
        let state_values = channels
            .checked_mul(kernel)
            .ok_or_else(|| EngineError::MemoryBudget("Metal convolution state overflows".into()))?;
        let weight_bytes_expected = state_values
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal convolution weight overflows".into())
            })?;
        let weight_bytes = match weight {
            FloatTensorView::F16Le(bytes) if bytes.len() == weight_bytes_expected => bytes,
            FloatTensorView::F16Le(bytes) => {
                return Err(EngineError::Shape(format!(
                    "Metal convolution weight has {} bytes, expected {weight_bytes_expected}",
                    bytes.len()
                )))
            }
            FloatTensorView::F32Le(_) => {
                return Err(EngineError::UnsupportedDType(
                    "Metal convolution weight must remain packed FP16".into(),
                ))
            }
        };
        let weight_offset = mapping.byte_offset(weight_bytes, "causal-convolution weight")?;
        let value_bytes = channels
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal convolution values overflow".into()))?;
        let params = MetalCausalConvParams {
            channels: usize_to_u32(channels, "Metal convolution channels")?,
            kernel: usize_to_u32(kernel, "Metal convolution kernel")?,
            reserved0: 0,
            reserved1: 0,
        };
        let transient_bytes = value_bytes
            .checked_mul(if own_io { 2 } else { 0 })
            .and_then(|bytes| bytes.checked_add(MetalCausalConvParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal convolution transient bytes overflow".into())
            })?;
        Ok(PreparedMappedMetalCausalConv {
            channels,
            kernel,
            mapping: mapping.clone(),
            weight_offset,
            input_buffer: own_io.then(|| buffer_with_data(&self.device, as_bytes(input))),
            state_buffer: new_zeroed_buffer(&self.device, weight_bytes_expected)?,
            checkpoint_buffer: new_zeroed_buffer(&self.device, weight_bytes_expected)?,
            output_buffer: if own_io {
                Some(new_zeroed_buffer(&self.device, value_bytes)?)
            } else {
                None
            },
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            resident_state_bytes: weight_bytes_expected,
            transient_bytes,
            checkpoint_valid: false,
            poisoned: false,
        })
    }

    pub fn prepare_partial_rope(
        &self,
        values: &[f32],
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: u64,
        theta: f32,
    ) -> Result<PreparedMetalPartialRope> {
        let value_count = heads
            .checked_mul(head_dim)
            .ok_or_else(|| EngineError::Shape("Metal RoPE value shape overflows".into()))?;
        validate_metal_input(values, value_count)?;
        let params = partial_rope_params(heads, head_dim, rotary_dim, position, theta)?;
        let (cosine, sine) = partial_rope_tables(rotary_dim, position, theta)?;
        let values_buffer = buffer_with_data(&self.device, as_bytes(values));
        let cosine_buffer = buffer_with_data(&self.device, as_bytes(&cosine));
        let sine_buffer = buffer_with_data(&self.device, as_bytes(&sine));
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = size_of_val(values)
            .checked_add(size_of_val(cosine.as_slice()))
            .and_then(|bytes| bytes.checked_add(size_of_val(sine.as_slice())))
            .and_then(|bytes| bytes.checked_add(MetalPartialRopeParams::BYTE_LEN))
            .ok_or_else(|| EngineError::Shape("Metal RoPE transient bytes overflow".into()))?;
        Ok(PreparedMetalPartialRope {
            heads,
            head_dim,
            rotary_dim,
            theta,
            values_buffer: Some(values_buffer),
            cosine_buffer,
            sine_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    /// Prepare only immutable/tiny RoPE tables and parameters. The activation
    /// itself is supplied as a view into the schedule-derived decode arena.
    pub fn prepare_partial_rope_graph(
        &self,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        position: u64,
        theta: f32,
    ) -> Result<PreparedMetalPartialRope> {
        heads
            .checked_mul(head_dim)
            .ok_or_else(|| EngineError::Shape("Metal graph RoPE value shape overflows".into()))?;
        let params = partial_rope_params(heads, head_dim, rotary_dim, position, theta)?;
        let (cosine, sine) = partial_rope_tables(rotary_dim, position, theta)?;
        let transient_bytes = size_of_val(cosine.as_slice())
            .checked_add(size_of_val(sine.as_slice()))
            .and_then(|bytes| bytes.checked_add(MetalPartialRopeParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::Shape("Metal graph RoPE transient bytes overflow".into())
            })?;
        Ok(PreparedMetalPartialRope {
            heads,
            head_dim,
            rotary_dim,
            theta,
            values_buffer: None,
            cosine_buffer: buffer_with_data(&self.device, as_bytes(&cosine)),
            sine_buffer: buffer_with_data(&self.device, as_bytes(&sine)),
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            transient_bytes,
        })
    }

    /// Allocate a standalone verifier cache with reusable local Q/output I/O.
    pub fn prepare_paged_gqa_decode(
        &self,
        config: MetalPagedGqaConfig,
    ) -> Result<PreparedMetalPagedGqa> {
        self.prepare_paged_gqa_decode_internal(None, config, true)
    }

    /// Allocate one layer-owned packed cache whose activation I/O is supplied
    /// exclusively by shared decode-arena views.
    pub fn prepare_paged_gqa_decode_graph(
        &self,
        layer: usize,
        config: MetalPagedGqaConfig,
    ) -> Result<PreparedMetalPagedGqa> {
        if Qwen38Config::default().layer_kind(layer) != Some(LayerKind::FullAttention) {
            return Err(EngineError::Shape(format!(
                "Metal graph paged GQA layer {layer} is not full attention"
            )));
        }
        self.prepare_paged_gqa_decode_internal(Some(layer), config, false)
    }

    fn prepare_paged_gqa_decode_internal(
        &self,
        owner_layer: Option<usize>,
        config: MetalPagedGqaConfig,
        own_io: bool,
    ) -> Result<PreparedMetalPagedGqa> {
        let MetalPagedGqaConfig {
            query_heads,
            key_value_heads,
            head_dim,
            maximum_tokens,
            page_tokens,
            sink_tokens,
            recent_tokens,
        } = config;
        if query_heads == 0
            || key_value_heads == 0
            || !query_heads.is_multiple_of(key_value_heads)
            || head_dim == 0
            || head_dim > 256
            || !head_dim.is_multiple_of(32)
        {
            return Err(EngineError::Shape(
                "invalid Metal paged GQA head geometry".into(),
            ));
        }
        let component_values = key_value_heads
            .checked_mul(head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV width overflows".into()))?;
        let combined_values = component_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("Metal combined KV width overflows".into()))?;
        if !combined_values.is_multiple_of(BLOCK_LEN) {
            return Err(EngineError::Shape(format!(
                "Metal combined KV width must be a multiple of {BLOCK_LEN}"
            )));
        }
        let cache =
            MetalPagedKvMetadata::new(maximum_tokens, page_tokens, sink_tokens, recent_tokens);
        #[cfg(test)]
        let verifier_cache = PagedKvCache::new(
            maximum_tokens,
            component_values,
            page_tokens,
            sink_tokens,
            recent_tokens,
        )?;
        let maximum_pages = maximum_tokens.div_ceil(page_tokens);
        let q2_token_bytes = combined_values
            .checked_div(BLOCK_LEN)
            .and_then(|blocks| blocks.checked_mul(Q2_BLOCK_BYTES))
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q2 KV token bytes overflow".into()))?;
        let q4_token_bytes = combined_values
            .checked_div(BLOCK_LEN)
            .and_then(|blocks| blocks.checked_mul(Q4_BLOCK_BYTES))
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 KV token bytes overflow".into()))?;
        let q2_page_bytes = page_tokens
            .checked_mul(q2_token_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q2 KV page bytes overflow".into()))?;
        let q4_page_bytes = page_tokens
            .checked_mul(q4_token_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 KV page bytes overflow".into()))?;
        let q4_slots = sink_tokens
            .checked_div(page_tokens)
            .and_then(|slots| slots.checked_add(recent_tokens.div_ceil(page_tokens)))
            .and_then(|slots| slots.checked_add(1))
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 KV slot count overflows".into()))?
            .clamp(1, maximum_pages);
        let q2_arena_bytes = maximum_pages
            .checked_mul(q2_page_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q2 KV arena overflows".into()))?;
        let q4_arena_bytes = q4_slots
            .checked_mul(q4_page_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 KV arena overflows".into()))?;
        let descriptor_bytes = maximum_pages
            .checked_mul(METAL_PAGED_KV_DESCRIPTOR_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV descriptors overflow".into()))?;
        let value_bytes = query_heads
            .checked_mul(head_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| EngineError::MemoryBudget("Metal GQA values overflow".into()))?;
        let packed_device_bytes = q2_arena_bytes
            .checked_add(q4_arena_bytes)
            .and_then(|bytes| bytes.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal packed KV residency overflows".into())
            })?;
        let transient_bytes = value_bytes
            .checked_mul(if own_io { 2 } else { 0 })
            .and_then(|bytes| bytes.checked_add(2 * MetalKvPackParams::BYTE_LEN))
            .and_then(|bytes| bytes.checked_add(MetalPagedGqaParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal GQA transient bytes overflow".into())
            })?;

        let token_blocks = combined_values / BLOCK_LEN;
        let page_blocks = page_tokens
            .checked_mul(token_blocks)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV page blocks overflow".into()))?;
        let token_pack_params = MetalKvPackParams {
            component_values: u32::try_from(component_values)
                .map_err(|_| EngineError::Shape("Metal KV component width exceeds u32".into()))?,
            blocks: u32::try_from(token_blocks)
                .map_err(|_| EngineError::Shape("Metal KV token blocks exceed u32".into()))?,
            reserved0: 0,
            reserved1: 0,
        };
        let page_demote_params = MetalKvPackParams {
            component_values: token_pack_params.component_values,
            blocks: u32::try_from(page_blocks)
                .map_err(|_| EngineError::Shape("Metal KV page blocks exceed u32".into()))?,
            reserved0: 0,
            reserved1: 0,
        };
        Ok(PreparedMetalPagedGqa {
            owner_layer,
            query_heads,
            key_value_heads,
            head_dim,
            maximum_tokens,
            page_tokens,
            q2_token_bytes,
            q4_token_bytes,
            q2_page_bytes,
            q4_page_bytes,
            q4_slots,
            cache,
            #[cfg(test)]
            verifier_cache,
            #[cfg(test)]
            verifier_key_snapshot_buffer: new_zeroed_buffer(
                &self.device,
                component_values * std::mem::size_of::<f32>(),
            )?,
            #[cfg(test)]
            verifier_value_snapshot_buffer: new_zeroed_buffer(
                &self.device,
                component_values * std::mem::size_of::<f32>(),
            )?,
            page_to_q4_slot: vec![None; maximum_pages],
            free_q4_slots: (0..q4_slots).rev().collect(),
            q2_pages_buffer: new_zeroed_buffer(&self.device, q2_arena_bytes)?,
            q4_pages_buffer: new_zeroed_buffer(&self.device, q4_arena_bytes)?,
            descriptors_buffer: new_zeroed_buffer(&self.device, descriptor_bytes)?,
            query_buffer: if own_io {
                Some(new_zeroed_buffer(&self.device, value_bytes)?)
            } else {
                None
            },
            output_buffer: if own_io {
                Some(new_zeroed_buffer(&self.device, value_bytes)?)
            } else {
                None
            },
            kv_token_pack_params_buffer: buffer_with_data(
                &self.device,
                &token_pack_params.encode(),
            ),
            kv_page_demote_params_buffer: buffer_with_data(
                &self.device,
                &page_demote_params.encode(),
            ),
            params_buffer: new_zeroed_buffer(&self.device, MetalPagedGqaParams::BYTE_LEN)?,
            packed_device_bytes,
            transient_bytes,
            poisoned: false,
            speculative_checkpoint: None,
        })
    }

    /// Allocate one persistent FP16 GatedDelta recurrence state. Inputs and
    /// output are reusable f32 buffers; no f32 state shadow is retained.
    pub fn prepare_gated_delta_f16(
        &self,
        config: MetalGatedDeltaConfig,
    ) -> Result<PreparedMetalGatedDelta> {
        self.prepare_gated_delta_f16_internal(config, true, None)
    }

    /// Allocate only persistent FP16 recurrence/checkpoint state for graph
    /// execution. All five inputs and the output remain shared-arena views.
    pub fn prepare_gated_delta_f16_graph_io(
        &self,
        config: MetalGatedDeltaConfig,
        layer: usize,
    ) -> Result<PreparedMetalGatedDelta> {
        if config != MetalGatedDeltaConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "Metal graph recurrence requires exact Qwen3.8-27B geometry".into(),
            ));
        }
        self.prepare_gated_delta_f16_internal(config, false, Some(layer))
    }

    fn prepare_gated_delta_f16_internal(
        &self,
        config: MetalGatedDeltaConfig,
        own_io: bool,
        owner_layer: Option<usize>,
    ) -> Result<PreparedMetalGatedDelta> {
        if config.heads == 0
            || config.key_dim == 0
            || config.value_dim == 0
            || config.value_dim > MAX_THREADS_PER_GROUP
            || !config.value_dim.is_multiple_of(32)
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
        {
            return Err(EngineError::Shape(
                "invalid Metal gated-delta geometry or epsilon".into(),
            ));
        }
        if config.value_dim as u64
            > self
                .gated_delta_f16_pipeline
                .max_total_threads_per_threadgroup()
        {
            return Err(EngineError::InvalidState(
                "Metal gated-delta value dimension exceeds pipeline threadgroup capacity".into(),
            ));
        }
        let qk_values = config
            .heads
            .checked_mul(config.key_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta Q/K values overflow".into()))?;
        let value_values = config
            .heads
            .checked_mul(config.value_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta values overflow".into()))?;
        let state_values = qk_values
            .checked_mul(config.value_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta state values overflow".into()))?;
        let qk_bytes = qk_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta Q/K bytes overflow".into()))?;
        let value_bytes = value_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta value bytes overflow".into()))?;
        let head_bytes = config
            .heads
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta head bytes overflow".into()))?;
        let resident_state_bytes = state_values
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal delta state bytes overflow".into()))?;
        let io_bytes = qk_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(value_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(head_bytes.checked_mul(2)?))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal delta graph I/O bytes overflow".into())
            })?;
        let transient_bytes = io_bytes
            .checked_mul(usize::from(own_io))
            .and_then(|bytes| bytes.checked_add(MetalGatedDeltaParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal delta transient bytes overflow".into())
            })?;
        let params = MetalGatedDeltaParams {
            heads: usize_to_u32(config.heads, "Metal delta heads")?,
            key_dim: usize_to_u32(config.key_dim, "Metal delta key dimension")?,
            value_dim: usize_to_u32(config.value_dim, "Metal delta value dimension")?,
            epsilon: config.epsilon,
        };
        Ok(PreparedMetalGatedDelta {
            config,
            owner_layer,
            query_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, qk_bytes))
                .transpose()?,
            key_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, qk_bytes))
                .transpose()?,
            value_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, value_bytes))
                .transpose()?,
            log_decay_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, head_bytes))
                .transpose()?,
            beta_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, head_bytes))
                .transpose()?,
            state_buffer: new_zeroed_buffer(&self.device, resident_state_bytes)?,
            checkpoint_buffer: new_zeroed_buffer(&self.device, resident_state_bytes)?,
            output_buffer: own_io
                .then(|| new_zeroed_buffer(&self.device, value_bytes))
                .transpose()?,
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            resident_state_bytes,
            transient_bytes,
            checkpoint_valid: false,
            poisoned: false,
        })
    }

    /// Executes the candidate kernel with its exact CTOXQ FP16 recovery-scale
    /// ABI. There is no CPU or alternate-kernel fallback.
    pub fn dispatch_fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        let prepared = self.prepare_fused_matvec(operation)?;
        self.dispatch_prepared(&prepared)
    }

    /// Copies one operation into reusable shared Metal buffers. This is the
    /// verifier precursor to the final mmap/no-copy graph loader: repeated
    /// dispatches no longer duplicate immutable tensor data.
    pub fn prepare_fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<PreparedMetalMatVec> {
        self.prepare_fused_matvec_with_simdgroups(operation, DEFAULT_SIMDGROUPS)
    }

    /// Autotuning entry point. Release profiles pin the winning simdgroup
    /// count per operation shape instead of relying on a global default.
    pub fn prepare_fused_matvec_with_simdgroups(
        &self,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
    ) -> Result<PreparedMetalMatVec> {
        let (layout, params) = validate_operation(operation)?;
        let pipeline = match layout.dtype {
            TensorDType::Q2B64 => &self.q2_pipeline,
            TensorDType::Q4B64 => &self.q4_pipeline,
            _ => unreachable!("Metal validation accepts only Q2/Q4"),
        };
        let thread_width = dispatch_width(pipeline, simdgroups)?;
        let dummy_half = [0_u8; 2];
        let dummy_float = [0.0_f32];
        let s_in = fp16_bytes_or_dummy(operation.s_in, &dummy_half);
        let s_out = fp16_bytes_or_dummy(operation.s_out, &dummy_half);
        let bias = operation.bias.unwrap_or(&dummy_float);
        let params = params.encode();

        let weights_buffer = buffer_with_data(&self.device, operation.weights);
        let input_buffer = buffer_with_data(&self.device, as_bytes(operation.input));
        let s_in_buffer = buffer_with_data(&self.device, s_in);
        let s_out_buffer = buffer_with_data(&self.device, s_out);
        let bias_buffer = buffer_with_data(&self.device, as_bytes(bias));
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal output byte size overflows".into()))?;
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params_buffer = buffer_with_data(&self.device, &params);

        let resident_bytes = operation
            .weights
            .len()
            .checked_add(size_of_val(operation.input))
            .and_then(|total| total.checked_add(s_in.len()))
            .and_then(|total| total.checked_add(s_out.len()))
            .and_then(|total| total.checked_add(size_of_val(bias)))
            .and_then(|total| total.checked_add(output_bytes))
            .and_then(|total| total.checked_add(params.len()))
            .ok_or_else(|| EngineError::Shape("Metal resident byte count overflows".into()))?;
        Ok(PreparedMetalMatVec {
            dtype: layout.dtype,
            rows: operation.rows,
            columns: operation.columns,
            thread_width,
            weights_buffer,
            input_buffer,
            s_in_buffer,
            s_out_buffer,
            bias_buffer,
            output_buffer,
            params_buffer,
            resident_bytes,
        })
    }

    /// Prepares one input and exact packed FP16 recovery correction without
    /// retaining any projection-local buffers.
    pub fn prepare_shared_activation(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMetalActivation> {
        validate_operation(operation)?;
        let dummy_half = [0_u8; 2];
        let s_in = fp16_bytes_or_dummy(operation.s_in, &dummy_half);
        let correction_identity = metal_correction_identity(operation.columns, operation.s_in)?;
        let input_buffer = buffer_with_data(&self.device, as_bytes(operation.input));
        let s_in_buffer = buffer_with_data(&self.device, s_in);
        let resident_bytes = size_of_val(operation.input)
            .checked_add(s_in.len())
            .ok_or_else(|| EngineError::Shape("Metal activation byte count overflows".into()))?;
        Ok(PreparedMetalActivation {
            columns: operation.columns,
            correction_identity,
            input_buffer,
            s_in_buffer,
            resident_bytes,
        })
    }

    pub fn prepare_shared_projection(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMetalProjection> {
        self.prepare_shared_projection_with_simdgroups(operation, DEFAULT_SIMDGROUPS)
    }

    pub fn prepare_shared_projection_with_simdgroups(
        &self,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
    ) -> Result<PreparedMetalProjection> {
        let (layout, params) = validate_operation(operation)?;
        let pipeline = match layout.dtype {
            TensorDType::Q2B64 => &self.q2_pipeline,
            TensorDType::Q4B64 => &self.q4_pipeline,
            _ => unreachable!("Metal validation accepts only Q2/Q4"),
        };
        let thread_width = dispatch_width(pipeline, simdgroups)?;
        let dummy_half = [0_u8; 2];
        let dummy_float = [0.0_f32];
        let s_out = fp16_bytes_or_dummy(operation.s_out, &dummy_half);
        let bias = operation.bias.unwrap_or(&dummy_float);
        let params = params.encode();
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("Metal output byte size overflows".into()))?;
        let weights_buffer = buffer_with_data(&self.device, operation.weights);
        let s_out_buffer = buffer_with_data(&self.device, s_out);
        let bias_buffer = buffer_with_data(&self.device, as_bytes(bias));
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params_buffer = buffer_with_data(&self.device, &params);
        let resident_bytes = operation
            .weights
            .len()
            .checked_add(s_out.len())
            .and_then(|total| total.checked_add(size_of_val(bias)))
            .and_then(|total| total.checked_add(output_bytes))
            .and_then(|total| total.checked_add(params.len()))
            .ok_or_else(|| EngineError::Shape("Metal projection byte count overflows".into()))?;
        Ok(PreparedMetalProjection {
            dtype: layout.dtype,
            rows: operation.rows,
            columns: operation.columns,
            thread_width,
            correction_identity: metal_correction_identity(operation.columns, operation.s_in)?,
            weights_buffer,
            s_out_buffer,
            bias_buffer,
            output_buffer,
            params_buffer,
            resident_bytes,
        })
    }

    /// Encodes all projections into one command buffer and synchronizes once.
    /// Output order is identical to caller order.
    pub fn dispatch_shared_fanout(
        &self,
        activation: &PreparedMetalActivation,
        projections: &[&PreparedMetalProjection],
    ) -> Result<Vec<Vec<f32>>> {
        if projections.is_empty() {
            return Err(EngineError::Shape(
                "Metal fan-out requires at least one projection".into(),
            ));
        }
        for projection in projections {
            if projection.columns != activation.columns
                || projection.correction_identity != activation.correction_identity
            {
                return Err(EngineError::InvalidArtifact(
                    "Metal fan-out projection s_in identity differs".into(),
                ));
            }
        }

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-fanout-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        for projection in projections {
            let pipeline = match projection.dtype {
                TensorDType::Q2B64 => &self.q2_pipeline,
                TensorDType::Q4B64 => &self.q4_pipeline,
                _ => unreachable!("prepared Metal projection is Q2/Q4"),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalBufferAbi::WEIGHTS as u64,
                Some(&projection.weights_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::INPUT as u64,
                Some(&activation.input_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::S_IN as u64,
                Some(&activation.s_in_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::S_OUT as u64,
                Some(&projection.s_out_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::BIAS as u64,
                Some(&projection.bias_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::OUTPUT as u64,
                Some(&projection.output_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::PARAMS as u64,
                Some(&projection.params_buffer),
                0,
            );
            let grid = MTLSize {
                width: projection
                    .rows
                    .div_ceil((projection.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: projection.thread_width as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal fan-out command ended with {:?}",
                command_buffer.status()
            )));
        }
        projections
            .iter()
            .map(|projection| {
                let output = unsafe {
                    slice::from_raw_parts(
                        projection.output_buffer.contents().cast::<f32>(),
                        projection.rows,
                    )
                    .to_vec()
                };
                if output.iter().any(|value| !value.is_finite()) {
                    return Err(EngineError::InvalidState(
                        "Metal fan-out produced a non-finite output".into(),
                    ));
                }
                Ok(output)
            })
            .collect()
    }

    /// Dispatches an already resident projection. Command encoding and
    /// completion remain synchronous so verifier and benchmark callers obtain
    /// an unambiguous interval and completed output.
    pub fn dispatch_prepared(&self, prepared: &PreparedMetalMatVec) -> Result<Vec<f32>> {
        self.dispatch_prepared_repeated(prepared, 1)
    }

    pub fn dispatch_mapped(&self, prepared: &PreparedMappedMetalMatVec) -> Result<Vec<f32>> {
        let input_buffer = prepared.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal external-input projection requires an upstream graph dispatch".into(),
            )
        })?;
        let output_buffer = prepared.owned_output()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-q2q4-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_buffer(MetalBufferAbi::INPUT as u64, Some(input_buffer), 0);
        encoder.set_buffer(
            MetalBufferAbi::S_IN as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.s_in_offset,
        );
        for dispatch in &prepared.dispatches {
            let pipeline = match dispatch.dtype {
                TensorDType::Q2B64 => &self.q2_pipeline,
                TensorDType::Q4B64 => &self.q4_pipeline,
                _ => unreachable!("mapped Metal dispatch is Q2/Q4"),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalBufferAbi::WEIGHTS as u64,
                Some(&prepared.mapping.inner.buffer),
                dispatch.weights_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::S_OUT as u64,
                Some(&prepared.mapping.inner.buffer),
                dispatch.s_out_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::BIAS as u64,
                Some(&prepared.bias_buffer),
                dispatch.bias_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::OUTPUT as u64,
                Some(output_buffer),
                dispatch.output_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::PARAMS as u64,
                Some(&dispatch.params_buffer),
                0,
            );
            let grid = MTLSize {
                width: dispatch
                    .rows
                    .div_ceil((dispatch.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: dispatch.thread_width as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal mmap command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), prepared.rows).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal mmap projection produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_mapped_gathered(
        &self,
        prepared: &PreparedMappedMetalGatheredMatVec,
    ) -> Result<Vec<f32>> {
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-gathered-lm-head-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_buffer(
            MetalBufferAbi::INPUT as u64,
            Some(&prepared.input_buffer),
            0,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_IN as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.s_in_offset,
        );
        encoder.set_buffer(MetalBufferAbi::BIAS as u64, Some(&prepared.bias_buffer), 0);
        for dispatch in &prepared.dispatches {
            let pipeline = match dispatch.dtype {
                TensorDType::Q2B64 => &self.q2_gathered_pipeline,
                TensorDType::Q4B64 => &self.q4_gathered_pipeline,
                _ => unreachable!("Metal gathered dispatch is Q2/Q4"),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalBufferAbi::WEIGHTS as u64,
                Some(&prepared.mapping.inner.buffer),
                dispatch.weights_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::S_OUT as u64,
                Some(&prepared.mapping.inner.buffer),
                dispatch.s_out_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::OUTPUT as u64,
                Some(&prepared.output_buffer),
                dispatch.output_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::PARAMS as u64,
                Some(&dispatch.params_buffer),
                0,
            );
            encoder.set_buffer(
                MetalBufferAbi::ROW_IDS as u64,
                Some(&dispatch.row_ids_buffer),
                0,
            );
            let grid = MTLSize {
                width: dispatch
                    .requested_rows
                    .div_ceil((dispatch.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: dispatch.thread_width as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal gathered command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                prepared.requested_rows,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal gathered projection produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Decode one prepared embedding row directly from the no-copy artifact
    /// mapping. No host implementation or alternate Metal kernel is used.
    pub fn dispatch_mapped_recovered_row(
        &self,
        prepared: &PreparedMappedMetalRecoveredRow,
    ) -> Result<Vec<f32>> {
        self.dispatch_mapped_recovered_row_repeated(prepared, 1)
    }

    /// Record repeated resident embedding decodes in one command buffer so
    /// verifier benchmarks can separate kernel latency from synchronization.
    pub fn dispatch_mapped_recovered_row_repeated(
        &self,
        prepared: &PreparedMappedMetalRecoveredRow,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "Metal recovered-row dispatch count must be positive".into(),
            ));
        }
        let (pipeline, values_per_thread) = match prepared.dtype {
            TensorDType::Q2B64 => (&self.q2_recovered_row_pipeline, 4_usize),
            TensorDType::Q4B64 => (&self.q4_recovered_row_pipeline, 2_usize),
            _ => unreachable!("prepared Metal recovered row is Q2 or Q4"),
        };
        let work_items = prepared.columns / values_per_thread;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-recovered-embedding-row-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            MetalBufferAbi::WEIGHTS as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weights_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_IN as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.s_in_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_OUT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.s_out_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::OUTPUT as u64,
            Some(&prepared.output_buffer),
            0,
        );
        encoder.set_buffer(
            MetalBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        let grid = MTLSize {
            width: work_items.div_ceil(prepared.thread_width) as u64,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: prepared.thread_width as u64,
            height: 1,
            depth: 1,
        };
        for _ in 0..dispatches {
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal recovered-row command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                prepared.columns,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal recovered embedding row produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    fn encode_mapped_embedding(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalEmbedding,
        token: usize,
    ) -> Result<()> {
        self.encode_mapped_embedding_to(encoder, prepared, token, prepared.owned_output()?, 0)
    }

    fn encode_mapped_embedding_to(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalEmbedding,
        token: usize,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        if token >= prepared.rows {
            return Err(EngineError::Shape(format!(
                "Metal embedding token {token} exceeds {} rows",
                prepared.rows
            )));
        }
        let segment = prepared
            .segments
            .iter()
            .find(|segment| token >= segment.row_start && token < segment.row_end)
            .ok_or_else(|| {
                EngineError::InvalidArtifact(format!(
                    "Metal embedding token {token} has no quantized segment"
                ))
            })?;
        let local_row = token - segment.row_start;
        let local_weight_offset = local_row
            .checked_mul(segment.row_bytes)
            .ok_or_else(|| EngineError::Shape("Metal embedding row offset overflows".into()))?;
        let weights_offset = prepared
            .weights_base
            .checked_add(segment.weights_offset)
            .and_then(|offset| offset.checked_add(u64::try_from(local_weight_offset).ok()?))
            .ok_or_else(|| EngineError::Shape("Metal embedding weight binding overflows".into()))?;
        let s_out_offset = prepared
            .s_out_base
            .checked_add(
                u64::try_from(token)
                    .map_err(|_| EngineError::Shape("Metal embedding token exceeds u64".into()))?
                    .checked_mul(std::mem::size_of::<half::f16>() as u64)
                    .ok_or_else(|| {
                        EngineError::Shape("Metal embedding s_out index overflows".into())
                    })?,
            )
            .ok_or_else(|| EngineError::Shape("Metal embedding s_out binding overflows".into()))?;
        let (pipeline, values_per_thread) = match segment.dtype {
            TensorDType::Q2B64 => (&self.q2_recovered_row_pipeline, 4_usize),
            TensorDType::Q4B64 => (&self.q4_recovered_row_pipeline, 2_usize),
            _ => unreachable!("prepared Metal embedding segment is Q2/Q4"),
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            MetalBufferAbi::WEIGHTS as u64,
            Some(&prepared.mapping.inner.buffer),
            weights_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_IN as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.s_in_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_OUT as u64,
            Some(&prepared.mapping.inner.buffer),
            s_out_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            output_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        let work_items = prepared.columns / values_per_thread;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: work_items.div_ceil(segment.thread_width) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: segment.thread_width as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Select and decode one token from a complete resident embedding table.
    /// Only the row offsets change between calls; the table remains one shared
    /// mmap-backed Metal allocation.
    pub fn dispatch_mapped_embedding(
        &self,
        prepared: &PreparedMappedMetalEmbedding,
        token: usize,
    ) -> Result<Vec<f32>> {
        if token >= prepared.rows {
            return Err(EngineError::Shape(format!(
                "Metal embedding token {token} exceeds {} rows",
                prepared.rows
            )));
        }
        let output_buffer = prepared.owned_output()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-resident-embedding-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_embedding(encoder, prepared, token)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal resident embedding command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), prepared.columns).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal resident embedding produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_mapped_rms_norm_1p(
        &self,
        prepared: &PreparedMappedMetalRmsNorm,
    ) -> Result<Vec<f32>> {
        self.dispatch_mapped_rms_norm_1p_repeated(prepared, 1)
    }

    /// Record repeated resident RMSNorm operations in one command buffer.
    /// This remains an isolated verifier primitive, not production graph
    /// promotion or evidence of a complete decoder.
    pub fn dispatch_mapped_rms_norm_1p_repeated(
        &self,
        prepared: &PreparedMappedMetalRmsNorm,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "Metal RMSNorm dispatch count must be positive".into(),
            ));
        }
        let input_buffer = prepared.owned_input()?;
        let output_buffer = prepared.owned_output()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-rms-norm-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_1p_pipeline);
        encoder.set_buffer(MetalRmsNormBufferAbi::INPUT as u64, Some(input_buffer), 0);
        encoder.set_buffer(
            MetalRmsNormBufferAbi::WEIGHT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weight_offset,
        );
        encoder.set_buffer(MetalRmsNormBufferAbi::OUTPUT as u64, Some(output_buffer), 0);
        encoder.set_buffer(
            MetalRmsNormBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        let grid = MTLSize {
            width: prepared.rows as u64,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        for _ in 0..dispatches {
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal RMSNorm command ended with {:?}",
                command_buffer.status()
            )));
        }
        let value_count = prepared
            .rows
            .checked_mul(prepared.columns)
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm output shape overflows".into()))?;
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), value_count).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_mapped_rms_norm_gated(
        &self,
        prepared: &PreparedMappedMetalGatedRmsNorm,
    ) -> Result<Vec<f32>> {
        let input_buffer = prepared.input_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph gated RMSNorm has no operation-local input".into(),
            )
        })?;
        let gate_buffer = prepared.gate_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph gated RMSNorm has no operation-local gate".into(),
            )
        })?;
        let output_buffer = prepared.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "Metal graph gated RMSNorm has no operation-local output".into(),
            )
        })?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-gated-rms-norm-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_gated_pipeline);
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::INPUT as u64,
            Some(input_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::GATE as u64,
            Some(gate_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::WEIGHT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weight_offset,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal gated RMSNorm command ended with {:?}",
                command_buffer.status()
            )));
        }
        let value_count = prepared.rows.checked_mul(prepared.columns).ok_or_else(|| {
            EngineError::Shape("Metal gated RMSNorm output shape overflows".into())
        })?;
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), value_count).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal gated RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_partial_rope(&self, prepared: &PreparedMetalPartialRope) -> Result<Vec<f32>> {
        self.dispatch_partial_rope_many(&[prepared])?
            .pop()
            .ok_or_else(|| EngineError::InvalidState("Metal RoPE output is missing".into()))
    }

    /// Apply the independent query/key RoPE transforms in one command encoder
    /// and synchronize once. Output buffers are updated in place.
    pub fn dispatch_partial_rope_pair(
        &self,
        query: &PreparedMetalPartialRope,
        key: &PreparedMetalPartialRope,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut outputs = self.dispatch_partial_rope_many(&[query, key])?;
        let key = outputs
            .pop()
            .ok_or_else(|| EngineError::InvalidState("Metal key RoPE output is missing".into()))?;
        let query = outputs.pop().ok_or_else(|| {
            EngineError::InvalidState("Metal query RoPE output is missing".into())
        })?;
        Ok((query, key))
    }

    fn dispatch_partial_rope_many(
        &self,
        prepared: &[&PreparedMetalPartialRope],
    ) -> Result<Vec<Vec<f32>>> {
        if prepared.is_empty() {
            return Err(EngineError::Shape(
                "Metal RoPE dispatch requires at least one tensor".into(),
            ));
        }
        let thread_width = dispatch_width(&self.partial_rope_pipeline, DEFAULT_SIMDGROUPS)?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-partial-rope-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        for operation in prepared {
            let values_buffer = operation.values_buffer.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph RoPE requires an explicit shared-arena dispatch".into(),
                )
            })?;
            self.encode_partial_rope_between(encoder, operation, values_buffer, 0, thread_width)?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal partial-RoPE command ended with {:?}",
                command_buffer.status()
            )));
        }
        prepared
            .iter()
            .map(|operation| {
                let value_count =
                    operation
                        .heads
                        .checked_mul(operation.head_dim)
                        .ok_or_else(|| {
                            EngineError::Shape("Metal RoPE output shape overflows".into())
                        })?;
                let output = unsafe {
                    slice::from_raw_parts(
                        operation
                            .values_buffer
                            .as_ref()
                            .expect("owned RoPE buffer checked before dispatch")
                            .contents()
                            .cast::<f32>(),
                        value_count,
                    )
                    .to_vec()
                };
                if output.iter().any(|value| !value.is_finite()) {
                    return Err(EngineError::InvalidState(
                        "Metal partial RoPE produced a non-finite output".into(),
                    ));
                }
                Ok(output)
            })
            .collect()
    }

    /// Apply in-place partial RoPE directly to one exact shared-arena view.
    /// The view is never copied to an operation-local activation buffer.
    pub fn dispatch_partial_rope_view(
        &self,
        prepared: &PreparedMetalPartialRope,
        view: &PreparedMetalDecodeBufferView<'_>,
    ) -> Result<()> {
        if prepared.has_owned_values() {
            return Err(EngineError::InvalidState(
                "Metal graph RoPE unexpectedly owns an activation buffer".into(),
            ));
        }
        let expected = prepared
            .heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal graph RoPE shape overflows".into()))?;
        if view.values() < expected
            || !matches!(view.slot(), MetalBufferSlot::Query | MetalBufferSlot::Key)
        {
            return Err(EngineError::Shape(format!(
                "Metal graph RoPE view {:?} has {} values, expected {expected}",
                view.slot(),
                view.values()
            )));
        }
        let thread_width = dispatch_width(&self.partial_rope_pipeline, DEFAULT_SIMDGROUPS)?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-partial-rope-graph-view");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_partial_rope_between(
            encoder,
            prepared,
            view.buffer(),
            view.offset(),
            thread_width,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal graph partial-RoPE command ended with {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    fn encode_partial_rope_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMetalPartialRope,
        values_buffer: &Buffer,
        values_offset: u64,
        thread_width: usize,
    ) -> Result<()> {
        encoder.set_compute_pipeline_state(&self.partial_rope_pipeline);
        encoder.set_buffer(
            MetalPartialRopeBufferAbi::VALUES as u64,
            Some(values_buffer),
            values_offset,
        );
        encoder.set_buffer(
            MetalPartialRopeBufferAbi::COSINE as u64,
            Some(&prepared.cosine_buffer),
            0,
        );
        encoder.set_buffer(
            MetalPartialRopeBufferAbi::SINE as u64,
            Some(&prepared.sine_buffer),
            0,
        );
        encoder.set_buffer(
            MetalPartialRopeBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        let pair_count = prepared
            .heads
            .checked_mul(prepared.rotary_dim / 2)
            .ok_or_else(|| EngineError::Shape("Metal RoPE pair count overflows".into()))?;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: pair_count.div_ceil(thread_width) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: thread_width as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Verifier for the device-only KV transition used by the production
    /// paged cache: pack one resident K/V token to Q4, then demote that packed
    /// representation to Q2 in the same command encoder. Returned bytes are
    /// verifier evidence only; the graph path binds page-arena offsets instead.
    pub fn dispatch_kv_q4_pack_and_demote(
        &self,
        key: &[f32],
        value: &[f32],
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        if key.len() != value.len() || key.is_empty() || !(key.len() * 2).is_multiple_of(BLOCK_LEN)
        {
            return Err(EngineError::Shape(
                "Metal KV pack requires equal non-empty K/V widths divisible by 32".into(),
            ));
        }
        if key.iter().chain(value).any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "Metal KV pack input contains a non-finite value".into(),
            ));
        }
        let combined_values = key
            .len()
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV width overflows".into()))?;
        let blocks = combined_values / BLOCK_LEN;
        let q4_bytes = blocks
            .checked_mul(Q4_BLOCK_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 KV bytes overflow".into()))?;
        let q2_bytes = blocks
            .checked_mul(Q2_BLOCK_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q2 KV bytes overflow".into()))?;
        let params = MetalKvPackParams {
            component_values: u32::try_from(key.len())
                .map_err(|_| EngineError::Shape("Metal KV component width exceeds u32".into()))?,
            blocks: u32::try_from(blocks)
                .map_err(|_| EngineError::Shape("Metal KV block count exceeds u32".into()))?,
            reserved0: 0,
            reserved1: 0,
        };
        let key_buffer = buffer_with_data(&self.device, as_bytes(key));
        let value_buffer = buffer_with_data(&self.device, as_bytes(value));
        let q4_buffer = new_zeroed_buffer(&self.device, q4_bytes)?;
        let q2_buffer = new_zeroed_buffer(&self.device, q2_bytes)?;
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-kv-q4-pack-demote-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.kv_q4_pack_pipeline);
        encoder.set_buffer(MetalKvQ4PackBufferAbi::KEY as u64, Some(&key_buffer), 0);
        encoder.set_buffer(MetalKvQ4PackBufferAbi::VALUE as u64, Some(&value_buffer), 0);
        encoder.set_buffer(MetalKvQ4PackBufferAbi::OUTPUT as u64, Some(&q4_buffer), 0);
        encoder.set_buffer(
            MetalKvQ4PackBufferAbi::PARAMS as u64,
            Some(&params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: blocks as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.set_compute_pipeline_state(&self.kv_q4_to_q2_pipeline);
        encoder.set_buffer(MetalKvQ4ToQ2BufferAbi::Q4 as u64, Some(&q4_buffer), 0);
        encoder.set_buffer(MetalKvQ4ToQ2BufferAbi::Q2 as u64, Some(&q2_buffer), 0);
        encoder.set_buffer(
            MetalKvQ4ToQ2BufferAbi::PARAMS as u64,
            Some(&params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: blocks as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal KV pack/demotion command ended with {:?}",
                command_buffer.status()
            )));
        }
        let q4 = unsafe { slice::from_raw_parts(q4_buffer.contents().cast::<u8>(), q4_bytes) };
        let q2 = unsafe { slice::from_raw_parts(q2_buffer.contents().cast::<u8>(), q2_bytes) };
        Ok((q4.to_vec(), q2.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_kv_q4_pack_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        key_buffer: &Buffer,
        key_offset: u64,
        value_buffer: &Buffer,
        value_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
        params_buffer: &Buffer,
        blocks: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_q4_pack_pipeline);
        encoder.set_buffer(
            MetalKvQ4PackBufferAbi::KEY as u64,
            Some(key_buffer),
            key_offset,
        );
        encoder.set_buffer(
            MetalKvQ4PackBufferAbi::VALUE as u64,
            Some(value_buffer),
            value_offset,
        );
        encoder.set_buffer(
            MetalKvQ4PackBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            output_offset,
        );
        encoder.set_buffer(
            MetalKvQ4PackBufferAbi::PARAMS as u64,
            Some(params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: blocks as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_kv_q4_to_q2_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q4_buffer: &Buffer,
        q4_offset: u64,
        q2_buffer: &Buffer,
        q2_offset: u64,
        params_buffer: &Buffer,
        blocks: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_q4_to_q2_pipeline);
        encoder.set_buffer(
            MetalKvQ4ToQ2BufferAbi::Q4 as u64,
            Some(q4_buffer),
            q4_offset,
        );
        encoder.set_buffer(
            MetalKvQ4ToQ2BufferAbi::Q2 as u64,
            Some(q2_buffer),
            q2_offset,
        );
        encoder.set_buffer(
            MetalKvQ4ToQ2BufferAbi::PARAMS as u64,
            Some(params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: blocks as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    fn plan_paged_gqa_append(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
    ) -> Result<MetalPagedGqaAppendPlan> {
        let retain_q4 = prepared.speculative_checkpoint.is_some();
        let update = prepared.cache.push(retain_q4)?;

        let mut demotions = Vec::with_capacity(update.demoted_pages.len());
        for &page_index in &update.demoted_pages {
            let page = prepared.cache.pages.get(page_index).ok_or_else(|| {
                EngineError::InvalidState("Metal demoted KV page is missing".into())
            })?;
            if page.precision != KvPrecision::Q2 || page.tokens != prepared.page_tokens {
                return Err(EngineError::InvalidState(
                    "Metal demoted KV page is not one complete Q2 page".into(),
                ));
            }
            let slot = prepared.page_to_q4_slot[page_index].take().ok_or_else(|| {
                EngineError::InvalidState("Metal demoted page has no Q4 arena slot".into())
            })?;
            demotions.push((page_index, slot));
            prepared.free_q4_slots.push(slot);
        }

        let current =
            prepared.cache.pages.get(update.page_index).ok_or_else(|| {
                EngineError::InvalidState("Metal current KV page is missing".into())
            })?;
        if current.precision != KvPrecision::Q4 {
            return Err(EngineError::InvalidState(
                "Metal current KV page is not Q4".into(),
            ));
        }
        let q4_slot = match prepared.page_to_q4_slot[update.page_index] {
            Some(slot) => slot,
            None => {
                let slot = prepared.free_q4_slots.pop().ok_or_else(|| {
                    EngineError::MemoryBudget(
                        "Metal bounded Q4 KV arena has no free retention slot".into(),
                    )
                })?;
                prepared.page_to_q4_slot[update.page_index] = Some(slot);
                slot
            }
        };
        Ok(MetalPagedGqaAppendPlan {
            demotions,
            q4_slot,
            token_in_page: update.token_in_page,
            #[cfg(test)]
            verifier_update: update,
        })
    }

    #[cfg(test)]
    fn commit_paged_gqa_verifier_append(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
        plan: &MetalPagedGqaAppendPlan,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let verifier_update = if prepared.speculative_checkpoint.is_some() {
            prepared.verifier_cache.push_retaining_q4(key, value)?
        } else {
            prepared.verifier_cache.push(key, value)?
        };
        if verifier_update.page_index != plan.verifier_update.page_index
            || verifier_update.token_in_page != plan.verifier_update.token_in_page
            || verifier_update.demoted_pages != plan.verifier_update.demoted_pages
        {
            return Err(EngineError::InvalidState(
                "Metal paged KV metadata diverged from verifier policy".into(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_paged_gqa_append_and_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMetalPagedGqa,
        plan: &MetalPagedGqaAppendPlan,
        query_buffer: &Buffer,
        query_offset: u64,
        key_buffer: &Buffer,
        key_offset: u64,
        value_buffer: &Buffer,
        value_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        let token_blocks = (prepared.key_value_heads * prepared.head_dim * 2) / BLOCK_LEN;
        let page_blocks = prepared
            .page_tokens
            .checked_mul(token_blocks)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV page blocks overflow".into()))?;
        for &(page_index, slot) in &plan.demotions {
            let q4_offset = slot
                .checked_mul(prepared.q4_page_bytes)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or_else(|| {
                    EngineError::MemoryBudget("Metal Q4 page offset overflows".into())
                })?;
            let q2_offset = page_index
                .checked_mul(prepared.q2_page_bytes)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or_else(|| {
                    EngineError::MemoryBudget("Metal Q2 page offset overflows".into())
                })?;
            self.encode_kv_q4_to_q2_between(
                encoder,
                &prepared.q4_pages_buffer,
                q4_offset,
                &prepared.q2_pages_buffer,
                q2_offset,
                &prepared.kv_page_demote_params_buffer,
                page_blocks,
            );
        }
        let q4_token_offset = plan
            .q4_slot
            .checked_mul(prepared.q4_page_bytes)
            .and_then(|offset| {
                plan.token_in_page
                    .checked_mul(prepared.q4_token_bytes)
                    .and_then(|token_offset| offset.checked_add(token_offset))
            })
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 token offset overflows".into()))?;
        self.encode_kv_q4_pack_between(
            encoder,
            key_buffer,
            key_offset,
            value_buffer,
            value_offset,
            &prepared.q4_pages_buffer,
            q4_token_offset,
            &prepared.kv_token_pack_params_buffer,
            token_blocks,
        );
        encoder.set_compute_pipeline_state(&self.paged_gqa_decode_pipeline);
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::QUERY as u64,
            Some(query_buffer),
            query_offset,
        );
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::Q2_PAGES as u64,
            Some(&prepared.q2_pages_buffer),
            0,
        );
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::Q4_PAGES as u64,
            Some(&prepared.q4_pages_buffer),
            0,
        );
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::DESCRIPTORS as u64,
            Some(&prepared.descriptors_buffer),
            0,
        );
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            output_offset,
        );
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.query_heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// Append one K/V token to the persistent packed cache and execute one
    /// decode-only GQA step. There is no CPU attention fallback and no f32 K/V
    /// allocation. Any failure after cache mutation poisons the prepared state
    /// until the caller explicitly resets it.
    pub fn append_and_dispatch_paged_gqa(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
        if prepared.poisoned || prepared.owner_layer.is_some() {
            return Err(EngineError::InvalidState(
                "Metal standalone paged GQA is poisoned or graph-owned".into(),
            ));
        }
        let query_values = prepared
            .query_heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal GQA query shape overflows".into()))?;
        let component_values = prepared
            .key_value_heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal GQA KV shape overflows".into()))?;
        validate_metal_input(query, query_values)?;
        validate_metal_input(key, component_values)?;
        validate_metal_input(value, component_values)?;
        if prepared.cache.tokens() >= prepared.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "Metal paged GQA reached {} tokens",
                prepared.maximum_tokens
            )));
        }

        prepared.poisoned = true;
        let result = self.append_and_dispatch_paged_gqa_inner(prepared, query, key, value);
        if result.is_ok() {
            prepared.poisoned = false;
        }
        result
    }

    /// Append one graph-produced K/V token and execute decode-only GQA using
    /// only the frozen shared-arena views. The packed KV arenas remain the
    /// sole persistent token store; release builds perform no activation
    /// upload or readback on this edge.
    pub fn append_and_dispatch_paged_gqa_views(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
        append_step: &PreparedMetalDecodeStepView<'_>,
        gqa_step: &PreparedMetalDecodeStepView<'_>,
    ) -> Result<()> {
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "Metal graph paged GQA is poisoned; reset is required".into(),
            ));
        }
        let layer = prepared.owner_layer.ok_or_else(|| {
            EngineError::InvalidState("Metal graph paged GQA has no layer owner".into())
        })?;
        if prepared.query_buffer.is_some() || prepared.output_buffer.is_some() {
            return Err(EngineError::InvalidState(
                "Metal graph paged GQA unexpectedly owns activation buffers".into(),
            ));
        }
        if append_step.step().layer != Some(layer)
            || append_step.step().operation != MetalDecodeOperation::PagedKvAppend
            || append_step.reads().len() != 2
            || !append_step.writes().is_empty()
            || append_step.reads()[0].slot() != MetalBufferSlot::Key
            || append_step.reads()[1].slot() != MetalBufferSlot::Value
            || gqa_step.step().layer != Some(layer)
            || gqa_step.step().operation != MetalDecodeOperation::PagedGqa
            || gqa_step.reads().len() != 1
            || gqa_step.writes().len() != 1
            || gqa_step.reads()[0].slot() != MetalBufferSlot::Query
            || gqa_step.writes()[0].slot() != MetalBufferSlot::AttentionOutput
        {
            return Err(EngineError::InvalidState(format!(
                "Metal graph paged GQA layer {layer} does not match its append/GQA schedule views"
            )));
        }
        let component_values = prepared
            .key_value_heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal graph GQA KV shape overflows".into()))?;
        let query_values = prepared
            .query_heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal graph GQA query shape overflows".into()))?;
        let key_view = &append_step.reads()[0];
        let value_view = &append_step.reads()[1];
        let query_view = &gqa_step.reads()[0];
        let output_view = &gqa_step.writes()[0];
        let arena = key_view.buffer();
        if [value_view, query_view, output_view]
            .iter()
            .any(|view| !std::ptr::eq(arena, view.buffer()))
            || key_view.values() < component_values
            || value_view.values() < component_values
            || query_view.values() < query_values
            || output_view.values() < query_values
        {
            return Err(EngineError::InvalidState(format!(
                "Metal graph paged GQA layer {layer} has incompatible arena identity or shape"
            )));
        }
        if prepared.cache.tokens() >= prepared.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "Metal paged GQA reached {} tokens",
                prepared.maximum_tokens
            )));
        }

        #[cfg(test)]
        let verifier_key = unsafe {
            slice::from_raw_parts(
                arena
                    .contents()
                    .cast::<u8>()
                    .add(usize::try_from(key_view.offset()).map_err(|_| {
                        EngineError::MemoryBudget("Metal graph key offset exceeds usize".into())
                    })?)
                    .cast::<f32>(),
                component_values,
            )
            .to_vec()
        };
        #[cfg(test)]
        let verifier_value = unsafe {
            slice::from_raw_parts(
                arena
                    .contents()
                    .cast::<u8>()
                    .add(usize::try_from(value_view.offset()).map_err(|_| {
                        EngineError::MemoryBudget("Metal graph value offset exceeds usize".into())
                    })?)
                    .cast::<f32>(),
                component_values,
            )
            .to_vec()
        };
        prepared.poisoned = true;
        let result = (|| {
            let plan = self.plan_paged_gqa_append(prepared)?;
            write_metal_paged_gqa_descriptors(prepared)?;
            write_metal_paged_gqa_params(prepared)?;
            let command_buffer = self.queue.new_command_buffer();
            command_buffer.set_label("ctox-qwen38-shared-arena-paged-q2q4-gqa");
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_paged_gqa_append_and_attention(
                encoder,
                prepared,
                &plan,
                query_view.buffer(),
                query_view.offset(),
                key_view.buffer(),
                key_view.offset(),
                value_view.buffer(),
                value_view.offset(),
                output_view.buffer(),
                output_view.offset(),
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(EngineError::InvalidState(format!(
                    "Metal graph paged GQA layer {layer} ended with {:?}",
                    command_buffer.status()
                )));
            }
            #[cfg(test)]
            self.commit_paged_gqa_verifier_append(prepared, &plan, &verifier_key, &verifier_value)?;
            Ok(())
        })();
        if result.is_ok() {
            prepared.poisoned = false;
        }
        result
    }

    fn append_and_dispatch_paged_gqa_inner(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
        let plan = self.plan_paged_gqa_append(prepared)?;
        write_metal_paged_gqa_descriptors(prepared)?;
        let query_buffer = prepared.query_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal standalone GQA query buffer is missing".into())
        })?;
        let output_buffer = prepared.output_buffer.as_ref().ok_or_else(|| {
            EngineError::InvalidState("Metal standalone GQA output buffer is missing".into())
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                query_buffer.contents().cast::<f32>(),
                query.len(),
            );
        }
        write_metal_paged_gqa_params(prepared)?;

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-paged-q2q4-gqa-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        let key_buffer = buffer_with_data(&self.device, as_bytes(key));
        let value_buffer = buffer_with_data(&self.device, as_bytes(value));
        self.encode_paged_gqa_append_and_attention(
            encoder,
            prepared,
            &plan,
            query_buffer,
            0,
            &key_buffer,
            0,
            &value_buffer,
            0,
            output_buffer,
            0,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal paged GQA command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output_values = prepared
            .query_heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal GQA output shape overflows".into()))?;
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), output_values).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal paged GQA produced a non-finite output".into(),
            ));
        }
        #[cfg(test)]
        self.commit_paged_gqa_verifier_append(prepared, &plan, key, value)?;
        Ok(output)
    }

    /// Execute one recurrent GatedDeltaNet step against the persistent FP16
    /// state. No CPU recurrence or f32 state fallback exists.
    pub fn dispatch_gated_delta_f16(
        &self,
        prepared: &mut PreparedMetalGatedDelta,
    ) -> Result<Vec<f32>> {
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "Metal gated-delta state is poisoned; reset is required".into(),
            ));
        }
        if !prepared.has_owned_io() {
            return Err(EngineError::InvalidState(
                "Metal graph recurrence requires an explicit shared-arena dispatch".into(),
            ));
        }
        let query_buffer = prepared.query_buffer.as_ref().expect("owned I/O checked");
        let key_buffer = prepared.key_buffer.as_ref().expect("owned I/O checked");
        let value_buffer = prepared.value_buffer.as_ref().expect("owned I/O checked");
        let log_decay_buffer = prepared
            .log_decay_buffer
            .as_ref()
            .expect("owned I/O checked");
        let beta_buffer = prepared.beta_buffer.as_ref().expect("owned I/O checked");
        let output_buffer = prepared.output_buffer.as_ref().expect("owned I/O checked");
        prepared.poisoned = true;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-gated-delta-f16-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gated_delta_f16_pipeline);
        for (binding, buffer) in [
            (MetalGatedDeltaBufferAbi::QUERY, query_buffer),
            (MetalGatedDeltaBufferAbi::KEY, key_buffer),
            (MetalGatedDeltaBufferAbi::VALUE, value_buffer),
            (MetalGatedDeltaBufferAbi::LOG_DECAY, log_decay_buffer),
            (MetalGatedDeltaBufferAbi::BETA, beta_buffer),
            (MetalGatedDeltaBufferAbi::STATE, &prepared.state_buffer),
            (MetalGatedDeltaBufferAbi::OUTPUT, output_buffer),
            (MetalGatedDeltaBufferAbi::PARAMS, &prepared.params_buffer),
        ] {
            encoder.set_buffer(binding as u64, Some(buffer), 0);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.config.heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: prepared.config.value_dim as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal gated-delta command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output_values = prepared
            .config
            .heads
            .checked_mul(prepared.config.value_dim)
            .ok_or_else(|| EngineError::Shape("Metal delta output shape overflows".into()))?;
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), output_values).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal gated-delta produced a non-finite output".into(),
            ));
        }
        prepared.poisoned = false;
        Ok(output)
    }

    /// Execute one depthwise causal-convolution step against mmap-backed FP16
    /// weights and persistent FP16 history. No CPU or f32-state fallback.
    pub fn dispatch_mapped_causal_conv_f16(
        &self,
        prepared: &mut PreparedMappedMetalCausalConv,
    ) -> Result<Vec<f32>> {
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "Metal causal-convolution state is poisoned; reset is required".into(),
            ));
        }
        if prepared.input_buffer.is_none() || prepared.output_buffer.is_none() {
            return Err(EngineError::InvalidState(
                "Metal graph convolution requires an explicit shared-arena dispatch".into(),
            ));
        }
        prepared.poisoned = true;
        let input_buffer = prepared.owned_input()?;
        let output_buffer = prepared.owned_output()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-causal-conv-f16-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_causal_conv_between(
            encoder,
            prepared,
            input_buffer,
            0,
            output_buffer,
            0,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal causal-convolution command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), prepared.channels)
                .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal causal convolution produced a non-finite output".into(),
            ));
        }
        prepared.poisoned = false;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_causal_conv_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalCausalConv,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        let thread_width = dispatch_width(&self.causal_conv_f16_pipeline, DEFAULT_SIMDGROUPS)?;
        encoder.set_compute_pipeline_state(&self.causal_conv_f16_pipeline);
        encoder.set_buffer(
            MetalCausalConvBufferAbi::INPUT as u64,
            Some(input_buffer),
            input_offset,
        );
        encoder.set_buffer(
            MetalCausalConvBufferAbi::WEIGHT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weight_offset,
        );
        encoder.set_buffer(
            MetalCausalConvBufferAbi::STATE as u64,
            Some(&prepared.state_buffer),
            0,
        );
        encoder.set_buffer(
            MetalCausalConvBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            output_offset,
        );
        encoder.set_buffer(
            MetalCausalConvBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.channels.div_ceil(thread_width) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: thread_width as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_gated_delta_prepare_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalGatedDeltaPrepare,
        arena: &Buffer,
        convolved_qkv_offset: u64,
        raw_a_offset: u64,
        raw_b_offset: u64,
        query_offset: u64,
        key_offset: u64,
        value_offset: u64,
        log_decay_offset: u64,
        beta_offset: u64,
    ) -> Result<()> {
        let thread_width =
            dispatch_width(&self.gated_delta_prepare_f32_pipeline, DEFAULT_SIMDGROUPS)?;
        let output_values = prepared
            .value_heads
            .checked_mul(prepared.key_dim)
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal GatedDelta output shape overflows".into())
            })?;
        encoder.set_compute_pipeline_state(&self.gated_delta_prepare_f32_pipeline);
        for (binding, buffer, offset) in [
            (
                MetalGatedDeltaPrepareBufferAbi::CONVOLVED_QKV,
                arena,
                convolved_qkv_offset,
            ),
            (MetalGatedDeltaPrepareBufferAbi::RAW_A, arena, raw_a_offset),
            (MetalGatedDeltaPrepareBufferAbi::RAW_B, arena, raw_b_offset),
            (
                MetalGatedDeltaPrepareBufferAbi::A_LOG,
                &prepared.mapping.inner.buffer,
                prepared.a_log_offset,
            ),
            (
                MetalGatedDeltaPrepareBufferAbi::DT_BIAS,
                &prepared.mapping.inner.buffer,
                prepared.dt_bias_offset,
            ),
            (MetalGatedDeltaPrepareBufferAbi::QUERY, arena, query_offset),
            (MetalGatedDeltaPrepareBufferAbi::KEY, arena, key_offset),
            (MetalGatedDeltaPrepareBufferAbi::VALUE, arena, value_offset),
            (
                MetalGatedDeltaPrepareBufferAbi::LOG_DECAY,
                arena,
                log_decay_offset,
            ),
            (MetalGatedDeltaPrepareBufferAbi::BETA, arena, beta_offset),
        ] {
            encoder.set_buffer(binding as u64, Some(buffer), offset);
        }
        encoder.set_buffer(
            MetalGatedDeltaPrepareBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: output_values.div_ceil(thread_width) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: thread_width as u64,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_gated_rms_norm_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalGatedRmsNorm,
        input_buffer: &Buffer,
        input_offset: u64,
        gate_buffer: &Buffer,
        gate_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.rms_norm_gated_pipeline);
        for (binding, buffer, offset) in [
            (
                MetalGatedRmsNormBufferAbi::INPUT,
                input_buffer,
                input_offset,
            ),
            (MetalGatedRmsNormBufferAbi::GATE, gate_buffer, gate_offset),
            (
                MetalGatedRmsNormBufferAbi::WEIGHT,
                &prepared.mapping.inner.buffer,
                prepared.weight_offset,
            ),
            (
                MetalGatedRmsNormBufferAbi::OUTPUT,
                output_buffer,
                output_offset,
            ),
        ] {
            encoder.set_buffer(binding as u64, Some(buffer), offset);
        }
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_gated_delta_f16_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMetalGatedDelta,
        arena: &Buffer,
        query_offset: u64,
        key_offset: u64,
        value_offset: u64,
        log_decay_offset: u64,
        beta_offset: u64,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.gated_delta_f16_pipeline);
        for (binding, offset) in [
            (MetalGatedDeltaBufferAbi::QUERY, query_offset),
            (MetalGatedDeltaBufferAbi::KEY, key_offset),
            (MetalGatedDeltaBufferAbi::VALUE, value_offset),
            (MetalGatedDeltaBufferAbi::LOG_DECAY, log_decay_offset),
            (MetalGatedDeltaBufferAbi::BETA, beta_offset),
            (MetalGatedDeltaBufferAbi::OUTPUT, output_offset),
        ] {
            encoder.set_buffer(binding as u64, Some(arena), offset);
        }
        encoder.set_buffer(
            MetalGatedDeltaBufferAbi::STATE as u64,
            Some(&prepared.state_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedDeltaBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.config.heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: prepared.config.value_dim as u64,
                height: 1,
                depth: 1,
            },
        );
    }

    fn validate_mapped_norm_projection(
        &self,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
    ) -> Result<()> {
        if norm.rows != 1 || norm.columns != projection.columns {
            return Err(EngineError::Shape(format!(
                "Metal norm/projection chain has norm {}x{} and projection width {}",
                norm.rows, norm.columns, projection.columns
            )));
        }
        if projection.input_buffer.is_some() {
            return Err(EngineError::InvalidState(
                "Metal norm/projection chain requires an external-input projection".into(),
            ));
        }
        if !Rc::ptr_eq(&norm.mapping.inner, &projection.mapping.inner) {
            return Err(EngineError::InvalidState(
                "Metal norm and projection do not share one artifact mapping".into(),
            ));
        }
        Ok(())
    }

    fn encode_mapped_norm_projection(
        &self,
        encoder: &ComputeCommandEncoderRef,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
    ) -> Result<()> {
        let input = norm.owned_input()?;
        self.encode_mapped_norm_projection_with_input(encoder, norm, projection, input)
    }

    fn encode_mapped_norm_projection_with_input(
        &self,
        encoder: &ComputeCommandEncoderRef,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
        input_buffer: &Buffer,
    ) -> Result<()> {
        let norm_output = norm.owned_output()?;
        let projection_output = projection.owned_output()?;
        self.encode_mapped_norm_between(encoder, norm, input_buffer, 0, norm_output, 0);
        self.encode_mapped_projection_between(
            encoder,
            projection,
            norm_output,
            0,
            projection_output,
            0,
        )
    }

    fn encode_mapped_norm_with_input(
        &self,
        encoder: &ComputeCommandEncoderRef,
        norm: &PreparedMappedMetalRmsNorm,
        input_buffer: &Buffer,
    ) -> Result<()> {
        let output_buffer = norm.owned_output()?;
        self.encode_mapped_norm_between(encoder, norm, input_buffer, 0, output_buffer, 0);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_norm_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        norm: &PreparedMappedMetalRmsNorm,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.rms_norm_1p_pipeline);
        encoder.set_buffer(
            MetalRmsNormBufferAbi::INPUT as u64,
            Some(input_buffer),
            input_offset,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::WEIGHT as u64,
            Some(&norm.mapping.inner.buffer),
            norm.weight_offset,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::OUTPUT as u64,
            Some(output_buffer),
            output_offset,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::PARAMS as u64,
            Some(&norm.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_residual_rms_norm_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        norm: &PreparedMappedMetalRmsNorm,
        residual_buffer: &Buffer,
        residual_offset: u64,
        update_buffer: &Buffer,
        update_offset: u64,
        residual_output_buffer: &Buffer,
        residual_output_offset: u64,
        normalized_output_buffer: &Buffer,
        normalized_output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.residual_rms_norm_1p_pipeline);
        for (binding, buffer, offset) in [
            (
                MetalResidualRmsNormBufferAbi::RESIDUAL,
                residual_buffer,
                residual_offset,
            ),
            (
                MetalResidualRmsNormBufferAbi::UPDATE,
                update_buffer,
                update_offset,
            ),
            (
                MetalResidualRmsNormBufferAbi::WEIGHT,
                &norm.mapping.inner.buffer,
                norm.weight_offset,
            ),
            (
                MetalResidualRmsNormBufferAbi::RESIDUAL_OUTPUT,
                residual_output_buffer,
                residual_output_offset,
            ),
            (
                MetalResidualRmsNormBufferAbi::NORMALIZED_OUTPUT,
                normalized_output_buffer,
                normalized_output_offset,
            ),
        ] {
            encoder.set_buffer(binding as u64, Some(buffer), offset);
        }
        encoder.set_buffer(
            MetalResidualRmsNormBufferAbi::PARAMS as u64,
            Some(&norm.params_buffer),
            0,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: norm.rows as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    fn encode_mapped_projection_with_input(
        &self,
        encoder: &ComputeCommandEncoderRef,
        projection: &PreparedMappedMetalMatVec,
        input_buffer: &Buffer,
    ) -> Result<()> {
        let output_buffer = projection.owned_output()?;
        self.encode_mapped_projection_between(
            encoder,
            projection,
            input_buffer,
            0,
            output_buffer,
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_projection_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        projection: &PreparedMappedMetalMatVec,
        input_buffer: &Buffer,
        input_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        encoder.set_buffer(
            MetalBufferAbi::INPUT as u64,
            Some(input_buffer),
            input_offset,
        );
        encoder.set_buffer(
            MetalBufferAbi::S_IN as u64,
            Some(&projection.mapping.inner.buffer),
            projection.s_in_offset,
        );
        for dispatch in &projection.dispatches {
            let pipeline = match dispatch.dtype {
                TensorDType::Q2B64 => &self.q2_pipeline,
                TensorDType::Q4B64 => &self.q4_pipeline,
                _ => unreachable!("mapped Metal dispatch is Q2/Q4"),
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalBufferAbi::WEIGHTS as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.weights_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::S_OUT as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.s_out_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::BIAS as u64,
                Some(&projection.bias_buffer),
                dispatch.bias_offset,
            );
            encoder.set_buffer(
                MetalBufferAbi::OUTPUT as u64,
                Some(output_buffer),
                output_offset
                    .checked_add(dispatch.output_offset)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget(
                            "Metal projection output view offset overflows".into(),
                        )
                    })?,
            );
            encoder.set_buffer(
                MetalBufferAbi::PARAMS as u64,
                Some(&dispatch.params_buffer),
                0,
            );
            let grid = MTLSize {
                width: dispatch
                    .rows
                    .div_ceil((dispatch.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: dispatch.thread_width as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(grid, threads);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_swiglu_projection_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        projection: &PreparedMappedMetalMatVec,
        gate_buffer: &Buffer,
        gate_offset: u64,
        up_buffer: &Buffer,
        up_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        encoder.set_buffer(
            MetalSwiGluBufferAbi::GATE as u64,
            Some(gate_buffer),
            gate_offset,
        );
        encoder.set_buffer(MetalSwiGluBufferAbi::UP as u64, Some(up_buffer), up_offset);
        encoder.set_buffer(
            MetalSwiGluBufferAbi::S_IN as u64,
            Some(&projection.mapping.inner.buffer),
            projection.s_in_offset,
        );
        for dispatch in &projection.dispatches {
            let pipeline = match dispatch.dtype {
                TensorDType::Q2B64 => &self.q2_swiglu_pipeline,
                TensorDType::Q4B64 => &self.q4_swiglu_pipeline,
                _ => unreachable!("mapped Metal SwiGLU projection is Q2/Q4"),
            };
            if dispatch.thread_width > pipeline.max_total_threads_per_threadgroup() as usize {
                return Err(EngineError::InvalidState(format!(
                    "Metal SwiGLU projection requires {} threads but pipeline admits {}",
                    dispatch.thread_width,
                    pipeline.max_total_threads_per_threadgroup()
                )));
            }
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalSwiGluBufferAbi::WEIGHTS as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.weights_offset,
            );
            encoder.set_buffer(
                MetalSwiGluBufferAbi::S_OUT as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.s_out_offset,
            );
            encoder.set_buffer(
                MetalSwiGluBufferAbi::BIAS as u64,
                Some(&projection.bias_buffer),
                dispatch.bias_offset,
            );
            encoder.set_buffer(
                MetalSwiGluBufferAbi::OUTPUT as u64,
                Some(output_buffer),
                output_offset
                    .checked_add(dispatch.output_offset)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget(
                            "Metal SwiGLU output view offset overflows".into(),
                        )
                    })?,
            );
            encoder.set_buffer(
                MetalSwiGluBufferAbi::PARAMS as u64,
                Some(&dispatch.params_buffer),
                0,
            );
            let grid = MTLSize {
                width: dispatch
                    .rows
                    .div_ceil((dispatch.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(
                grid,
                MTLSize {
                    width: dispatch.thread_width as u64,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_sigmoid_gate_projection_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        projection: &PreparedMappedMetalMatVec,
        attention_buffer: &Buffer,
        attention_offset: u64,
        gate_buffer: &Buffer,
        gate_offset: u64,
        output_buffer: &Buffer,
        output_offset: u64,
    ) -> Result<()> {
        encoder.set_buffer(
            MetalSigmoidGateBufferAbi::ATTENTION as u64,
            Some(attention_buffer),
            attention_offset,
        );
        encoder.set_buffer(
            MetalSigmoidGateBufferAbi::GATE as u64,
            Some(gate_buffer),
            gate_offset,
        );
        encoder.set_buffer(
            MetalSigmoidGateBufferAbi::S_IN as u64,
            Some(&projection.mapping.inner.buffer),
            projection.s_in_offset,
        );
        for dispatch in &projection.dispatches {
            let pipeline = match dispatch.dtype {
                TensorDType::Q2B64 => &self.q2_sigmoid_gate_pipeline,
                TensorDType::Q4B64 => &self.q4_sigmoid_gate_pipeline,
                _ => unreachable!("mapped Metal sigmoid-gate projection is Q2/Q4"),
            };
            if dispatch.thread_width > pipeline.max_total_threads_per_threadgroup() as usize {
                return Err(EngineError::InvalidState(format!(
                    "Metal sigmoid-gate projection requires {} threads but pipeline admits {}",
                    dispatch.thread_width,
                    pipeline.max_total_threads_per_threadgroup()
                )));
            }
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(
                MetalSigmoidGateBufferAbi::WEIGHTS as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.weights_offset,
            );
            encoder.set_buffer(
                MetalSigmoidGateBufferAbi::S_OUT as u64,
                Some(&projection.mapping.inner.buffer),
                dispatch.s_out_offset,
            );
            encoder.set_buffer(
                MetalSigmoidGateBufferAbi::BIAS as u64,
                Some(&projection.bias_buffer),
                dispatch.bias_offset,
            );
            encoder.set_buffer(
                MetalSigmoidGateBufferAbi::OUTPUT as u64,
                Some(output_buffer),
                output_offset
                    .checked_add(dispatch.output_offset)
                    .ok_or_else(|| {
                        EngineError::MemoryBudget(
                            "Metal sigmoid-gate output view offset overflows".into(),
                        )
                    })?,
            );
            encoder.set_buffer(
                MetalSigmoidGateBufferAbi::PARAMS as u64,
                Some(&dispatch.params_buffer),
                0,
            );
            let grid = MTLSize {
                width: dispatch
                    .rows
                    .div_ceil((dispatch.thread_width / 32) * ROWS_PER_SIMDGROUP)
                    as u64,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(
                grid,
                MTLSize {
                    width: dispatch.thread_width as u64,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    /// Verifier-only wrapper for the fused SwiGLU projection entry points.
    /// Production graph execution binds arena views through the schedule-bound
    /// path instead of allocating these temporary input/output buffers.
    pub fn dispatch_mapped_swiglu_projection(
        &self,
        projection: &PreparedMappedMetalMatVec,
        gate: &[f32],
        up: &[f32],
    ) -> Result<Vec<f32>> {
        if projection.input_buffer.is_some()
            || projection.output_buffer.is_some()
            || gate.len() != projection.columns
            || up.len() != projection.columns
            || gate.iter().chain(up).any(|value| !value.is_finite())
        {
            return Err(EngineError::Shape(
                "Metal fused SwiGLU verifier requires graph-I/O projection and finite exact-width gate/up inputs"
                    .into(),
            ));
        }
        let gate_buffer = buffer_with_data(&self.device, as_bytes(gate));
        let up_buffer = buffer_with_data(&self.device, as_bytes(up));
        let output_bytes = projection
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal SwiGLU output bytes overflow".into())
            })?;
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-swiglu-projection-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_swiglu_projection_between(
            encoder,
            projection,
            &gate_buffer,
            0,
            &up_buffer,
            0,
            &output_buffer,
            0,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal fused SwiGLU verifier ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), projection.rows).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal fused SwiGLU verifier produced non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Encode one decode-row RMSNorm followed by a recovered Q2/Q4
    /// projection in a single command encoder. The projection consumes the
    /// RMSNorm output buffer directly and therefore owns no second activation
    /// allocation and performs no host readback between operations.
    pub fn dispatch_mapped_embedding_rms_norm_projection(
        &self,
        embedding: &PreparedMappedMetalEmbedding,
        token: usize,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
    ) -> Result<Vec<f32>> {
        self.validate_mapped_norm_projection(norm, projection)?;
        if token >= embedding.rows
            || embedding.columns != norm.columns
            || norm.rows != 1
            || !Rc::ptr_eq(&embedding.mapping.inner, &norm.mapping.inner)
        {
            return Err(EngineError::InvalidState(
                "Metal embedding/norm/projection chain has incompatible token, shape, or mapping"
                    .into(),
            ));
        }

        let embedding_output = embedding.owned_output()?;
        let projection_output = projection.owned_output()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-embedding-norm-projection-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_embedding(encoder, embedding, token)?;
        self.encode_mapped_norm_projection_with_input(encoder, norm, projection, embedding_output)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal embedding/norm/projection command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(projection_output.contents().cast::<f32>(), projection.rows)
                .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal embedding/norm/projection chain produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Encode one resident embedding lookup, one RMSNorm, and a complete
    /// recovery-bound projection fan-out in a single command encoder. All
    /// projections consume the exact same normalized buffer and packed
    /// `s_in`; only matrix-local outputs remain distinct.
    pub fn dispatch_mapped_embedding_rms_norm_fanout(
        &self,
        embedding: &PreparedMappedMetalEmbedding,
        token: usize,
        norm: &PreparedMappedMetalRmsNorm,
        projections: &[&PreparedMappedMetalMatVec],
    ) -> Result<Vec<Vec<f32>>> {
        let first = projections.first().ok_or_else(|| {
            EngineError::Shape("Metal embedding fan-out requires at least one projection".into())
        })?;
        if token >= embedding.rows
            || embedding.columns != norm.columns
            || norm.rows != 1
            || !Rc::ptr_eq(&embedding.mapping.inner, &norm.mapping.inner)
        {
            return Err(EngineError::InvalidState(
                "Metal embedding/norm/fan-out has incompatible token, shape, or mapping".into(),
            ));
        }
        for projection in projections {
            self.validate_mapped_norm_projection(norm, projection)?;
            if projection.s_in_offset != first.s_in_offset {
                return Err(EngineError::InvalidArtifact(
                    "Metal embedding fan-out projections do not share exact packed s_in".into(),
                ));
            }
        }

        let embedding_output = embedding.owned_output()?;
        let norm_output = norm.owned_output()?;
        let projection_outputs = projections
            .iter()
            .map(|projection| projection.owned_output())
            .collect::<Result<Vec<_>>>()?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-embedding-norm-fanout-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_embedding(encoder, embedding, token)?;
        self.encode_mapped_norm_with_input(encoder, norm, embedding_output)?;
        for projection in projections {
            self.encode_mapped_projection_with_input(encoder, projection, norm_output)?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal embedding/norm/fan-out command ended with {:?}",
                command_buffer.status()
            )));
        }
        projections
            .iter()
            .zip(projection_outputs)
            .map(|(projection, output_buffer)| {
                let output = unsafe {
                    slice::from_raw_parts(output_buffer.contents().cast::<f32>(), projection.rows)
                        .to_vec()
                };
                if output.iter().any(|value| !value.is_finite()) {
                    return Err(EngineError::InvalidState(
                        "Metal embedding/norm/fan-out produced a non-finite output".into(),
                    ));
                }
                Ok(output)
            })
            .collect()
    }

    /// Dispatch up to the exact first twelve operations of the frozen decode graph:
    /// embedding, layer-0 RMSNorm, four-way linear-attention fan-out, in-place
    /// convolution, five-output GatedDelta preparation, recurrent update, and
    /// direct-weight gated RMSNorm, the recovered linear output projection, and
    /// fused residual-add plus Qwen RMSNorm into the next hidden/normalized views,
    /// the mixed-Q2/Q4 two-way FFN gate/up fan-out, and fused SwiGLU down
    /// projection without a materialized SwiGLU activation, followed by the
    /// fused post-FFN residual-add and next-layer Qwen RMSNorm.
    /// All activations are typed views into one schedule-derived arena. The
    /// prepared graph resources own only immutable parameters and tiny command
    /// metadata; no operation-local input or output activation is allocated.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_mapped_embedding_rms_norm_linear_fanout_views(
        &self,
        embedding: &PreparedMappedMetalEmbedding,
        token: usize,
        norm: &PreparedMappedMetalRmsNorm,
        projections: [&PreparedMappedMetalMatVec; 4],
        mut convolution: Option<&mut PreparedMappedMetalCausalConv>,
        gated_delta_prepare: Option<(
            &PreparedMappedMetalGatedDeltaPrepare,
            [&PreparedMetalDecodeBufferView<'_>; 5],
        )>,
        mut recurrence: Option<(
            &mut PreparedMetalGatedDelta,
            &PreparedMetalDecodeStepView<'_>,
        )>,
        gated_rms_norm: Option<(
            &PreparedMappedMetalGatedRmsNorm,
            &PreparedMetalDecodeStepView<'_>,
        )>,
        linear_output_projection: Option<(
            &PreparedMappedMetalMatVec,
            &PreparedMetalDecodeStepView<'_>,
        )>,
        residual_rms_norm: Option<(
            &PreparedMappedMetalRmsNorm,
            &PreparedMetalDecodeStepView<'_>,
        )>,
        ffn_gate_up: Option<(
            [&PreparedMappedMetalMatVec; 2],
            &PreparedMetalDecodeStepView<'_>,
        )>,
        swiglu_down: Option<(&PreparedMappedMetalMatVec, &PreparedMetalDecodeStepView<'_>)>,
        post_ffn_residual_rms_norm: Option<(
            &PreparedMappedMetalRmsNorm,
            &PreparedMetalDecodeStepView<'_>,
        )>,
        embedding_output: &PreparedMetalDecodeBufferView<'_>,
        normalized_output: &PreparedMetalDecodeBufferView<'_>,
        projection_outputs: [&PreparedMetalDecodeBufferView<'_>; 4],
    ) -> Result<Vec<Vec<f32>>> {
        const OUTPUT_SLOTS: [MetalBufferSlot; 4] = [
            MetalBufferSlot::LinearQkv,
            MetalBufferSlot::LinearZ,
            MetalBufferSlot::LinearA,
            MetalBufferSlot::LinearB,
        ];
        const DELTA_OUTPUT_SLOTS: [MetalBufferSlot; 5] = [
            MetalBufferSlot::Query,
            MetalBufferSlot::Key,
            MetalBufferSlot::Value,
            MetalBufferSlot::LogDecay,
            MetalBufferSlot::Beta,
        ];
        if token >= embedding.rows
            || embedding.columns != norm.columns
            || norm.rows != 1
            || embedding.output_buffer.is_some()
            || norm.input_buffer.is_some()
            || norm.output_buffer.is_some()
            || !Rc::ptr_eq(&embedding.mapping.inner, &norm.mapping.inner)
        {
            return Err(EngineError::InvalidState(
                "Metal graph embedding/norm resources or shapes are incompatible".into(),
            ));
        }
        if embedding_output.slot() != MetalBufferSlot::HiddenA
            || embedding_output.values() != embedding.columns
            || normalized_output.slot() != MetalBufferSlot::Normalized
            || normalized_output.values() != norm.columns
            || embedding_output.bytes()
                != embedding
                    .columns
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("Metal embedding arena view overflows".into())
                    })?
            || normalized_output.bytes()
                != norm
                    .columns
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::MemoryBudget("Metal normalized arena view overflows".into())
                    })?
        {
            return Err(EngineError::Shape(
                "Metal graph embedding/norm arena views do not match the frozen slots".into(),
            ));
        }
        let first = projections[0];
        let correction_bytes = first
            .columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal s_in byte count overflows".into()))?;
        let first_s_in_start = usize::try_from(first.s_in_offset)
            .map_err(|_| EngineError::InvalidArtifact("Metal s_in offset exceeds usize".into()))?;
        let first_s_in_end = first_s_in_start
            .checked_add(correction_bytes)
            .ok_or_else(|| EngineError::InvalidArtifact("Metal s_in range overflows".into()))?;
        let first_s_in = first
            .mapping
            .inner
            .artifact
            .mapped_bytes()
            .get(first_s_in_start..first_s_in_end)
            .ok_or_else(|| {
                EngineError::InvalidArtifact("Metal s_in range exceeds mapping".into())
            })?;
        for ((projection, output), expected_slot) in projections
            .iter()
            .zip(projection_outputs.iter())
            .zip(OUTPUT_SLOTS)
        {
            self.validate_mapped_norm_projection(norm, projection)?;
            if projection.output_buffer.is_some()
                || output.slot() != expected_slot
                || output.values() != projection.rows
                || output.bytes()
                    != projection
                        .rows
                        .checked_mul(std::mem::size_of::<f32>())
                        .ok_or_else(|| {
                            EngineError::MemoryBudget(
                                "Metal projection arena view overflows".into(),
                            )
                        })?
            {
                return Err(EngineError::InvalidArtifact(format!(
                    "Metal graph projection does not match arena slot {expected_slot:?}"
                )));
            }
            let s_in_start = usize::try_from(projection.s_in_offset).map_err(|_| {
                EngineError::InvalidArtifact("Metal fan-out s_in offset exceeds usize".into())
            })?;
            let s_in_end = s_in_start.checked_add(correction_bytes).ok_or_else(|| {
                EngineError::InvalidArtifact("Metal fan-out s_in range overflows".into())
            })?;
            if projection
                .mapping
                .inner
                .artifact
                .mapped_bytes()
                .get(s_in_start..s_in_end)
                != Some(first_s_in)
            {
                return Err(EngineError::InvalidArtifact(
                    "Metal graph fan-out projections do not share byte-identical s_in".into(),
                ));
            }
        }
        let arena = embedding_output.buffer();
        if !std::ptr::eq(arena, normalized_output.buffer())
            || projection_outputs
                .iter()
                .any(|output| !std::ptr::eq(arena, output.buffer()))
        {
            return Err(EngineError::InvalidState(
                "Metal graph chain requires one shared activation arena".into(),
            ));
        }
        if let Some(convolution) = convolution.as_deref() {
            if convolution.poisoned
                || convolution.channels != projections[0].rows
                || convolution.input_buffer.is_some()
                || convolution.output_buffer.is_some()
                || !Rc::ptr_eq(&convolution.mapping.inner, &embedding.mapping.inner)
            {
                return Err(EngineError::InvalidState(
                    "Metal graph convolution does not match the LinearQkv arena output".into(),
                ));
            }
        }
        if let Some((prepared, outputs)) = gated_delta_prepare.as_ref() {
            if convolution.is_none()
                || prepared.key_heads != 16
                || prepared.value_heads != 48
                || prepared.key_dim != 128
                || !Rc::ptr_eq(&prepared.mapping.inner, &embedding.mapping.inner)
                || projections[0].rows != 10_240
                || projections[2].rows != prepared.value_heads
                || projections[3].rows != prepared.value_heads
            {
                return Err(EngineError::InvalidState(
                    "Metal GatedDelta preparation does not match the exact layer-0 fan-out".into(),
                ));
            }
            let qk_values = prepared
                .value_heads
                .checked_mul(prepared.key_dim)
                .ok_or_else(|| {
                    EngineError::MemoryBudget("Metal GatedDelta view shape overflows".into())
                })?;
            let expected_values = [
                qk_values,
                qk_values,
                qk_values,
                prepared.value_heads,
                prepared.value_heads,
            ];
            for ((output, slot), values) in
                outputs.iter().zip(DELTA_OUTPUT_SLOTS).zip(expected_values)
            {
                if output.slot() != slot
                    || output.values() != values
                    || !std::ptr::eq(arena, output.buffer())
                {
                    return Err(EngineError::InvalidArtifact(format!(
                        "Metal GatedDelta preparation does not match arena slot {slot:?}"
                    )));
                }
            }
        }
        if let Some((recurrence, step)) = recurrence.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, prepared_outputs) = gated_delta_prepare.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph recurrence requires GatedDelta preparation".into(),
                )
            })?;
            if recurrence.poisoned
                || recurrence.has_owned_io()
                || !recurrence.checkpoint_valid
                || !convolution
                    .as_deref()
                    .is_some_and(|convolution| convolution.checkpoint_valid)
                || recurrence.config != MetalGatedDeltaConfig::QWEN38_27B
                || step.step().schedule_index != 5
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::GatedDeltaRecurrent
                || reads.len() != 5
                || writes.len() != 1
            {
                return Err(EngineError::InvalidState(
                    "Metal graph recurrence does not match frozen schedule step 5".into(),
                ));
            }
            for ((read, prepared_output), expected_slot) in reads
                .iter()
                .zip(prepared_outputs.iter())
                .zip(DELTA_OUTPUT_SLOTS)
            {
                if read.slot() != expected_slot
                    || read.offset() != prepared_output.offset()
                    || read.values() != prepared_output.values()
                    || !std::ptr::eq(arena, read.buffer())
                {
                    return Err(EngineError::InvalidState(
                        "Metal graph recurrence reads do not match preparation outputs".into(),
                    ));
                }
            }
            let output = &writes[0];
            if output.slot() != MetalBufferSlot::AttentionOutput
                || output.values() != recurrence.config.heads * recurrence.config.value_dim
                || !std::ptr::eq(arena, output.buffer())
            {
                return Err(EngineError::InvalidState(
                    "Metal graph recurrence output does not match AttentionOutput".into(),
                ));
            }
        }
        if let Some((gated_norm, step)) = gated_rms_norm.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, recurrence_step) = recurrence.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph gated RMSNorm requires the recurrent update".into(),
                )
            })?;
            let recurrence_output = &recurrence_step.writes()[0];
            if gated_norm.has_owned_io()
                || gated_norm.rows != MetalGatedDeltaConfig::QWEN38_27B.heads
                || gated_norm.columns != MetalGatedDeltaConfig::QWEN38_27B.value_dim
                || !Rc::ptr_eq(&gated_norm.mapping.inner, &embedding.mapping.inner)
                || step.step().schedule_index != 6
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::GatedRmsNorm
                || reads.len() != 2
                || writes.len() != 1
            {
                return Err(EngineError::InvalidState(
                    "Metal graph gated RMSNorm does not match frozen schedule step 6".into(),
                ));
            }
            let input = &reads[0];
            let gate = &reads[1];
            let output = &writes[0];
            if input.slot() != MetalBufferSlot::AttentionOutput
                || input.offset() != recurrence_output.offset()
                || gate.slot() != MetalBufferSlot::LinearZ
                || gate.offset() != projection_outputs[1].offset()
                || output.slot() != MetalBufferSlot::AttentionOutput
                || output.offset() != input.offset()
                || input.values() != gated_norm.rows * gated_norm.columns
                || gate.values() != input.values()
                || output.values() != input.values()
                || !std::ptr::eq(arena, input.buffer())
                || !std::ptr::eq(arena, gate.buffer())
                || !std::ptr::eq(arena, output.buffer())
            {
                return Err(EngineError::InvalidState(
                    "Metal graph gated RMSNorm views do not match recurrence and LinearZ outputs"
                        .into(),
                ));
            }
        }
        if let Some((projection, step)) = linear_output_projection.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, gated_norm_step) = gated_rms_norm.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph linear output projection requires gated RMSNorm".into(),
                )
            })?;
            let gated_norm_output = &gated_norm_step.writes()[0];
            if projection.input_buffer.is_some()
                || projection.output_buffer.is_some()
                || !Rc::ptr_eq(&projection.mapping.inner, &embedding.mapping.inner)
                || projection.columns != gated_norm_output.values()
                || projection.rows != embedding.columns
                || step.step().schedule_index != 7
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::LinearOutputProjection
                || reads.len() != 1
                || writes.len() != 1
            {
                return Err(EngineError::InvalidState(
                    "Metal graph linear output projection does not match frozen schedule step 7"
                        .into(),
                ));
            }
            let input = &reads[0];
            let output = &writes[0];
            if input.slot() != MetalBufferSlot::AttentionOutput
                || input.offset() != gated_norm_output.offset()
                || input.values() != projection.columns
                || output.slot() != MetalBufferSlot::MixerOutput
                || output.values() != projection.rows
                || !std::ptr::eq(arena, input.buffer())
                || !std::ptr::eq(arena, output.buffer())
            {
                return Err(EngineError::InvalidState(
                    "Metal graph linear output projection views do not match gated RMSNorm and MixerOutput"
                        .into(),
                ));
            }
        }
        if let Some((residual_norm, step)) = residual_rms_norm.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, output_projection_step) =
                linear_output_projection.as_ref().ok_or_else(|| {
                    EngineError::InvalidState(
                        "Metal graph residual RMSNorm requires linear output projection".into(),
                    )
                })?;
            let mixer_output = &output_projection_step.writes()[0];
            if residual_norm.input_buffer.is_some()
                || residual_norm.output_buffer.is_some()
                || !Rc::ptr_eq(&residual_norm.mapping.inner, &embedding.mapping.inner)
                || residual_norm.rows != 1
                || residual_norm.columns != embedding.columns
                || step.step().schedule_index != 8
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::ResidualRmsNorm
                || reads.len() != 2
                || writes.len() != 2
            {
                return Err(EngineError::InvalidState(
                    "Metal graph residual RMSNorm does not match frozen schedule step 8".into(),
                ));
            }
            let residual = &reads[0];
            let update = &reads[1];
            let residual_output = &writes[0];
            let normalized = &writes[1];
            if residual.slot() != MetalBufferSlot::HiddenA
                || residual.offset() != embedding_output.offset()
                || update.slot() != MetalBufferSlot::MixerOutput
                || update.offset() != mixer_output.offset()
                || residual_output.slot() != MetalBufferSlot::HiddenB
                || normalized.slot() != MetalBufferSlot::Normalized
                || [residual, update, residual_output, normalized]
                    .iter()
                    .any(|view| {
                        view.values() != residual_norm.columns
                            || !std::ptr::eq(arena, view.buffer())
                    })
            {
                return Err(EngineError::InvalidState(
                    "Metal graph residual RMSNorm views do not match HiddenA/MixerOutput to HiddenB/Normalized"
                        .into(),
                ));
            }
        }
        if let Some((projections, step)) = ffn_gate_up.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, residual_norm_step) = residual_rms_norm.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph FFN gate/up fan-out requires residual RMSNorm".into(),
                )
            })?;
            let normalized = &residual_norm_step.writes()[1];
            if step.step().schedule_index != 9
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::FfnGateUpFanout
                || reads.len() != 1
                || writes.len() != 2
                || reads[0].slot() != MetalBufferSlot::Normalized
                || reads[0].offset() != normalized.offset()
                || reads[0].values() != embedding.columns
                || writes[0].slot() != MetalBufferSlot::FfnGate
                || writes[1].slot() != MetalBufferSlot::FfnUp
                || !std::ptr::eq(arena, reads[0].buffer())
            {
                return Err(EngineError::InvalidState(
                    "Metal graph FFN gate/up fan-out does not match frozen schedule step 9".into(),
                ));
            }
            for (projection, output) in projections.iter().zip(writes) {
                if projection.input_buffer.is_some()
                    || projection.output_buffer.is_some()
                    || !Rc::ptr_eq(&projection.mapping.inner, &embedding.mapping.inner)
                    || projection.columns != embedding.columns
                    || projection.rows != output.values()
                    || !std::ptr::eq(arena, output.buffer())
                {
                    return Err(EngineError::InvalidState(
                        "Metal graph FFN gate/up projection resource or output view is incompatible"
                            .into(),
                    ));
                }
            }
        }
        if let Some((projection, step)) = swiglu_down.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, ffn_step) = ffn_gate_up.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph SwiGLU down projection requires FFN gate/up fan-out".into(),
                )
            })?;
            let ffn_writes = ffn_step.writes();
            if step.step().schedule_index != 10
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::SwiGluDownProjection
                || reads.len() != 2
                || writes.len() != 1
                || reads[0].slot() != MetalBufferSlot::FfnGate
                || reads[0].offset() != ffn_writes[0].offset()
                || reads[1].slot() != MetalBufferSlot::FfnUp
                || reads[1].offset() != ffn_writes[1].offset()
                || writes[0].slot() != MetalBufferSlot::FfnDown
                || projection.input_buffer.is_some()
                || projection.output_buffer.is_some()
                || !Rc::ptr_eq(&projection.mapping.inner, &embedding.mapping.inner)
                || projection.columns != reads[0].values()
                || reads[1].values() != projection.columns
                || projection.rows != embedding.columns
                || writes[0].values() != projection.rows
                || [&reads[0], &reads[1], &writes[0]]
                    .iter()
                    .any(|view| !std::ptr::eq(arena, view.buffer()))
            {
                return Err(EngineError::InvalidState(
                    "Metal graph fused SwiGLU down projection does not match frozen schedule step 10"
                        .into(),
                ));
            }
        }
        if let Some((residual_norm, step)) = post_ffn_residual_rms_norm.as_ref() {
            let reads = step.reads();
            let writes = step.writes();
            let (_, attention_residual_step) = residual_rms_norm.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph post-FFN residual RMSNorm requires attention residual RMSNorm"
                        .into(),
                )
            })?;
            let (_, swiglu_step) = swiglu_down.as_ref().ok_or_else(|| {
                EngineError::InvalidState(
                    "Metal graph post-FFN residual RMSNorm requires fused SwiGLU down projection"
                        .into(),
                )
            })?;
            if step.step().schedule_index != 11
                || step.step().layer != Some(0)
                || step.step().operation != MetalDecodeOperation::ResidualRmsNorm
                || reads.len() != 2
                || writes.len() != 2
                || reads[0].slot() != MetalBufferSlot::HiddenB
                || reads[0].offset() != attention_residual_step.writes()[0].offset()
                || reads[1].slot() != MetalBufferSlot::FfnDown
                || reads[1].offset() != swiglu_step.writes()[0].offset()
                || writes[0].slot() != MetalBufferSlot::HiddenA
                || writes[1].slot() != MetalBufferSlot::Normalized
                || residual_norm.input_buffer.is_some()
                || residual_norm.output_buffer.is_some()
                || !Rc::ptr_eq(&residual_norm.mapping.inner, &embedding.mapping.inner)
                || residual_norm.rows != 1
                || residual_norm.columns != embedding.columns
                || [&reads[0], &reads[1], &writes[0], &writes[1]]
                    .iter()
                    .any(|view| {
                        view.values() != residual_norm.columns
                            || !std::ptr::eq(arena, view.buffer())
                    })
            {
                return Err(EngineError::InvalidState(
                    "Metal graph post-FFN residual RMSNorm does not match frozen schedule step 11"
                        .into(),
                ));
            }
        }

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-decode-prefix");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_embedding_to(
            encoder,
            embedding,
            token,
            arena,
            embedding_output.offset(),
        )?;
        self.encode_mapped_norm_between(
            encoder,
            norm,
            arena,
            embedding_output.offset(),
            arena,
            normalized_output.offset(),
        );
        for (projection, output) in projections.iter().zip(projection_outputs.iter()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                normalized_output.offset(),
                arena,
                output.offset(),
            )?;
        }
        if let Some(convolution) = convolution.as_deref_mut() {
            self.encode_mapped_causal_conv_between(
                encoder,
                convolution,
                arena,
                projection_outputs[0].offset(),
                arena,
                projection_outputs[0].offset(),
            )?;
            convolution.poisoned = true;
        }
        if let Some((prepared, outputs)) = gated_delta_prepare.as_ref() {
            self.encode_mapped_gated_delta_prepare_between(
                encoder,
                prepared,
                arena,
                projection_outputs[0].offset(),
                projection_outputs[2].offset(),
                projection_outputs[3].offset(),
                outputs[0].offset(),
                outputs[1].offset(),
                outputs[2].offset(),
                outputs[3].offset(),
                outputs[4].offset(),
            )?;
        }
        if let Some((recurrence, step)) = recurrence.as_mut() {
            let reads = step.reads();
            let output = &step.writes()[0];
            self.encode_gated_delta_f16_between(
                encoder,
                recurrence,
                arena,
                reads[0].offset(),
                reads[1].offset(),
                reads[2].offset(),
                reads[3].offset(),
                reads[4].offset(),
                output.offset(),
            );
            recurrence.poisoned = true;
        }
        if let Some((gated_norm, step)) = gated_rms_norm.as_ref() {
            self.encode_mapped_gated_rms_norm_between(
                encoder,
                gated_norm,
                arena,
                step.reads()[0].offset(),
                arena,
                step.reads()[1].offset(),
                arena,
                step.writes()[0].offset(),
            );
        }
        if let Some((projection, step)) = linear_output_projection.as_ref() {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                step.reads()[0].offset(),
                arena,
                step.writes()[0].offset(),
            )?;
        }
        if let Some((residual_norm, step)) = residual_rms_norm.as_ref() {
            self.encode_mapped_residual_rms_norm_between(
                encoder,
                residual_norm,
                arena,
                step.reads()[0].offset(),
                arena,
                step.reads()[1].offset(),
                arena,
                step.writes()[0].offset(),
                arena,
                step.writes()[1].offset(),
            );
        }
        if let Some((projections, step)) = ffn_gate_up.as_ref() {
            for (projection, output) in projections.iter().zip(step.writes()) {
                self.encode_mapped_projection_between(
                    encoder,
                    projection,
                    arena,
                    step.reads()[0].offset(),
                    arena,
                    output.offset(),
                )?;
            }
        }
        if let Some((projection, step)) = swiglu_down.as_ref() {
            self.encode_mapped_swiglu_projection_between(
                encoder,
                projection,
                arena,
                step.reads()[0].offset(),
                arena,
                step.reads()[1].offset(),
                arena,
                step.writes()[0].offset(),
            )?;
        }
        if let Some((residual_norm, step)) = post_ffn_residual_rms_norm.as_ref() {
            self.encode_mapped_residual_rms_norm_between(
                encoder,
                residual_norm,
                arena,
                step.reads()[0].offset(),
                arena,
                step.reads()[1].offset(),
                arena,
                step.writes()[0].offset(),
                arena,
                step.writes()[1].offset(),
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal shared-arena graph chain ended with {:?}",
                command_buffer.status()
            )));
        }

        let outputs = if recurrence.is_none() {
            projection_outputs
                .iter()
                .map(|output| {
                    let offset = usize::try_from(output.offset()).map_err(|_| {
                        EngineError::MemoryBudget("Metal output view offset exceeds usize".into())
                    })?;
                    let values = unsafe {
                        slice::from_raw_parts(
                            arena.contents().cast::<u8>().add(offset).cast::<f32>(),
                            output.values(),
                        )
                        .to_vec()
                    };
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(EngineError::InvalidState(
                            "Metal shared-arena graph chain produced non-finite output".into(),
                        ));
                    }
                    Ok(values)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        if recurrence.is_none() {
            if let Some((_, delta_outputs)) = gated_delta_prepare {
                for output in delta_outputs {
                    let offset = usize::try_from(output.offset()).map_err(|_| {
                        EngineError::MemoryBudget(
                            "Metal GatedDelta output offset exceeds usize".into(),
                        )
                    })?;
                    let values = unsafe {
                        slice::from_raw_parts(
                            arena.contents().cast::<u8>().add(offset).cast::<f32>(),
                            output.values(),
                        )
                    };
                    if values.iter().any(|value| !value.is_finite()) {
                        return Err(EngineError::InvalidState(
                            "Metal GatedDelta preparation produced non-finite output".into(),
                        ));
                    }
                }
            }
        }
        if gated_rms_norm.is_none() {
            if let Some((_, step)) = recurrence.as_ref() {
                let output = &step.writes()[0];
                let offset = usize::try_from(output.offset()).map_err(|_| {
                    EngineError::MemoryBudget("Metal recurrence output offset exceeds usize".into())
                })?;
                let values = unsafe {
                    slice::from_raw_parts(
                        arena.contents().cast::<u8>().add(offset).cast::<f32>(),
                        output.values(),
                    )
                };
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(EngineError::InvalidState(
                        "Metal graph recurrence produced non-finite output".into(),
                    ));
                }
            }
        }
        if let Some(convolution) = convolution {
            convolution.poisoned = false;
        }
        if let Some((recurrence, _)) = recurrence {
            recurrence.poisoned = false;
        }
        Ok(outputs)
    }

    /// Execute one exact full-attention Q/K/V fan-out from the already
    /// normalized shared-arena input. All three recovered projections consume
    /// the same arena view and write their canonical QueryGate/Key/Value views
    /// in one command encoder without graph-local activation buffers.
    pub fn dispatch_mapped_full_attention_fanout_views(
        &self,
        step: &PreparedMetalDecodeStepView<'_>,
        prepared: &PreparedMappedMetalFullAttentionFanout,
    ) -> Result<()> {
        let layer = prepared.layer;
        if step.step().layer != Some(layer)
            || step.step().operation != MetalDecodeOperation::FullAttentionFanout
            || step.reads().len() != 1
            || step.writes().len() != 3
            || step.reads()[0].slot() != MetalBufferSlot::Normalized
            || step.writes()[0].slot() != MetalBufferSlot::QueryGate
            || step.writes()[1].slot() != MetalBufferSlot::Key
            || step.writes()[2].slot() != MetalBufferSlot::Value
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} fan-out does not match its frozen schedule view"
            )));
        }
        let input = &step.reads()[0];
        let arena = input.buffer();
        let mapping = &prepared.projections[0].mapping.inner;
        if step
            .writes()
            .iter()
            .any(|view| !std::ptr::eq(arena, view.buffer()))
            || prepared.projections.iter().any(|projection| {
                projection.input_buffer.is_some()
                    || projection.output_buffer.is_some()
                    || !Rc::ptr_eq(&projection.mapping.inner, mapping)
                    || projection.columns != input.values()
            })
            || prepared
                .projections
                .iter()
                .zip(step.writes())
                .any(|(projection, output)| projection.rows > output.values())
            || prepared.projections[1].packed_s_in_bytes()?
                != prepared.projections[0].packed_s_in_bytes()?
            || prepared.projections[2].packed_s_in_bytes()?
                != prepared.projections[0].packed_s_in_bytes()?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} fan-out has incompatible arena, mapping, shape, or recovery input"
            )));
        }
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        for (projection, name) in prepared.projections.iter().zip([
            format!("{prefix}.q_proj.weight"),
            format!("{prefix}.k_proj.weight"),
            format!("{prefix}.v_proj.weight"),
        ]) {
            if !projection.matches_recovered_tensor(&name)? {
                return Err(EngineError::InvalidState(format!(
                    "Metal full-attention layer {layer} projection does not match {name}"
                )));
            }
        }

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-full-attention-fanout");
        let encoder = command_buffer.new_compute_command_encoder();
        for (projection, output) in prepared.projections.iter().zip(step.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                input.offset(),
                arena,
                output.offset(),
            )?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} fan-out ended with {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    /// Deinterleave, Qwen-normalize, and partially rotate the query branch of
    /// one full-attention layer directly between shared-arena views. The gate
    /// branch is copied to its canonical arena slot in the same kernel.
    pub fn dispatch_mapped_query_gate_norm_rope_view(
        &self,
        step: &PreparedMetalDecodeStepView<'_>,
        prepared: &PreparedMappedMetalQueryGate,
    ) -> Result<()> {
        let layer = prepared.layer;
        if step.step().layer != Some(layer)
            || step.step().operation != MetalDecodeOperation::QueryGateNormRope
            || step.reads().len() != 1
            || step.writes().len() != 2
            || step.reads()[0].slot() != MetalBufferSlot::QueryGate
            || step.writes()[0].slot() != MetalBufferSlot::Query
            || step.writes()[1].slot() != MetalBufferSlot::AttentionGate
            || prepared.heads != 24
            || prepared.head_dim != 256
            || prepared.rotary_dim != 64
            || !prepared.epsilon.is_finite()
            || prepared.epsilon <= 0.0
        {
            return Err(EngineError::InvalidState(format!(
                "Metal query/gate layer {layer} does not match the frozen Qwen schedule or geometry"
            )));
        }
        let query_values = prepared
            .heads
            .checked_mul(prepared.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal query width overflows".into()))?;
        let input_values = query_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("Metal query/gate width overflows".into()))?;
        let input = &step.reads()[0];
        let query = &step.writes()[0];
        let gate = &step.writes()[1];
        let arena = input.buffer();
        if input.values() < input_values
            || query.values() < query_values
            || gate.values() < query_values
            || !std::ptr::eq(arena, query.buffer())
            || !std::ptr::eq(arena, gate.buffer())
            || !prepared.matches_weight_tensor(&format!(
                "model.language_model.layers.{layer}.self_attn.q_norm.weight"
            ))?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal query/gate layer {layer} has incompatible arena views or Q-norm identity"
            )));
        }
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-query-gate-norm-rope");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_query_gate_norm_rope_between(
            encoder,
            prepared,
            arena,
            input.offset(),
            arena,
            query.offset(),
            arena,
            gate.offset(),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal query/gate layer {layer} ended with {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_query_gate_norm_rope_between(
        &self,
        encoder: &ComputeCommandEncoderRef,
        prepared: &PreparedMappedMetalQueryGate,
        input_buffer: &Buffer,
        input_offset: u64,
        query_buffer: &Buffer,
        query_offset: u64,
        gate_buffer: &Buffer,
        gate_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.query_gate_norm_rope_pipeline);
        for (binding, buffer, offset) in [
            (
                MetalQueryGateBufferAbi::QUERY_GATE,
                input_buffer,
                input_offset,
            ),
            (
                MetalQueryGateBufferAbi::Q_NORM_WEIGHT,
                &prepared.mapping.inner.buffer,
                prepared.q_norm_weight_offset,
            ),
            (MetalQueryGateBufferAbi::COSINE, &prepared.cosine_buffer, 0),
            (MetalQueryGateBufferAbi::SINE, &prepared.sine_buffer, 0),
            (MetalQueryGateBufferAbi::QUERY, query_buffer, query_offset),
            (MetalQueryGateBufferAbi::GATE, gate_buffer, gate_offset),
            (MetalQueryGateBufferAbi::PARAMS, &prepared.params_buffer, 0),
        ] {
            encoder.set_buffer(binding as u64, Some(buffer), offset);
        }
        encoder.dispatch_thread_groups(
            MTLSize {
                width: prepared.heads as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    }

    /// Apply the per-channel sigmoid attention gate, recovery `s_in`, and the
    /// mixed Q2/Q4 output projection in one kernel family. The gated 6,144-wide
    /// tensor is never materialized and every activation remains in the shared
    /// decode arena.
    pub fn dispatch_mapped_attention_gate_output_projection_view(
        &self,
        step: &PreparedMetalDecodeStepView<'_>,
        prepared: &PreparedMappedMetalAttentionOutput,
    ) -> Result<()> {
        let layer = prepared.layer;
        let projection = &prepared.projection;
        if step.step().layer != Some(layer)
            || step.step().operation != MetalDecodeOperation::AttentionGateOutputProjection
            || step.reads().len() != 2
            || step.writes().len() != 1
            || step.reads()[0].slot() != MetalBufferSlot::AttentionOutput
            || step.reads()[1].slot() != MetalBufferSlot::AttentionGate
            || step.writes()[0].slot() != MetalBufferSlot::MixerOutput
            || projection.input_buffer.is_some()
            || projection.output_buffer.is_some()
        {
            return Err(EngineError::InvalidState(format!(
                "Metal attention output layer {layer} does not match its frozen schedule view"
            )));
        }
        let attention = &step.reads()[0];
        let gate = &step.reads()[1];
        let output = &step.writes()[0];
        let arena = attention.buffer();
        let name = format!("model.language_model.layers.{layer}.self_attn.o_proj.weight");
        if !std::ptr::eq(arena, gate.buffer())
            || !std::ptr::eq(arena, output.buffer())
            || attention.values() < projection.columns
            || gate.values() < projection.columns
            || output.values() < projection.rows
            || !projection.matches_recovered_tensor(&name)?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal attention output layer {layer} has incompatible arena, shape, or projection identity"
            )));
        }

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-attention-gate-output");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_sigmoid_gate_projection_between(
            encoder,
            projection,
            arena,
            attention.offset(),
            arena,
            gate.offset(),
            arena,
            output.offset(),
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal attention output layer {layer} ended with {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    pub fn dispatch_mapped_transformer_tail_views(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        layer: usize,
        residual_rms_norm: &PreparedMappedMetalRmsNorm,
        ffn_gate_up: [&PreparedMappedMetalMatVec; 2],
        swiglu_down: &PreparedMappedMetalMatVec,
        post_ffn_residual_rms_norm: &PreparedMappedMetalRmsNorm,
    ) -> Result<()> {
        const OPERATIONS: [MetalDecodeOperation; 4] = [
            MetalDecodeOperation::ResidualRmsNorm,
            MetalDecodeOperation::FfnGateUpFanout,
            MetalDecodeOperation::SwiGluDownProjection,
            MetalDecodeOperation::ResidualRmsNorm,
        ];
        if steps.len() != OPERATIONS.len()
            || steps.iter().zip(OPERATIONS).any(|(step, operation)| {
                step.step().layer != Some(layer) || step.step().operation != operation
            })
        {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} does not match its four-step schedule"
            )));
        }
        let residual = &steps[0];
        let ffn = &steps[1];
        let down = &steps[2];
        let post_ffn = &steps[3];
        if residual.reads().len() != 2
            || residual.writes().len() != 2
            || residual.reads()[0].slot() != MetalBufferSlot::HiddenA
            || residual.reads()[1].slot() != MetalBufferSlot::MixerOutput
            || residual.writes()[0].slot() != MetalBufferSlot::HiddenB
            || residual.writes()[1].slot() != MetalBufferSlot::Normalized
            || ffn.reads().len() != 1
            || ffn.writes().len() != 2
            || ffn.reads()[0].slot() != MetalBufferSlot::Normalized
            || ffn.writes()[0].slot() != MetalBufferSlot::FfnGate
            || ffn.writes()[1].slot() != MetalBufferSlot::FfnUp
            || down.reads().len() != 2
            || down.writes().len() != 1
            || down.reads()[0].slot() != MetalBufferSlot::FfnGate
            || down.reads()[1].slot() != MetalBufferSlot::FfnUp
            || down.writes()[0].slot() != MetalBufferSlot::FfnDown
            || post_ffn.reads().len() != 2
            || post_ffn.writes().len() != 2
            || post_ffn.reads()[0].slot() != MetalBufferSlot::HiddenB
            || post_ffn.reads()[1].slot() != MetalBufferSlot::FfnDown
            || post_ffn.writes()[0].slot() != MetalBufferSlot::HiddenA
            || post_ffn.writes()[1].slot() != MetalBufferSlot::Normalized
        {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} has incompatible arena slots"
            )));
        }
        let arena = residual.reads()[0].buffer();
        if steps
            .iter()
            .flat_map(|step| step.reads().iter().chain(step.writes()))
            .any(|view| !std::ptr::eq(arena, view.buffer()))
        {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} does not use one activation arena"
            )));
        }
        let mapping = &ffn_gate_up[0].mapping.inner;
        let hidden = Qwen38Config::default().hidden_size;
        let intermediate = Qwen38Config::default().intermediate_size;
        if residual_rms_norm.input_buffer.is_some()
            || residual_rms_norm.output_buffer.is_some()
            || post_ffn_residual_rms_norm.input_buffer.is_some()
            || post_ffn_residual_rms_norm.output_buffer.is_some()
            || ffn_gate_up.iter().any(|projection| {
                projection.input_buffer.is_some()
                    || projection.output_buffer.is_some()
                    || !Rc::ptr_eq(&projection.mapping.inner, mapping)
                    || projection.columns != hidden
                    || projection.rows != intermediate
            })
            || swiglu_down.input_buffer.is_some()
            || swiglu_down.output_buffer.is_some()
            || !Rc::ptr_eq(&swiglu_down.mapping.inner, mapping)
            || !Rc::ptr_eq(&residual_rms_norm.mapping.inner, mapping)
            || !Rc::ptr_eq(&post_ffn_residual_rms_norm.mapping.inner, mapping)
            || swiglu_down.columns != intermediate
            || swiglu_down.rows != hidden
            || residual_rms_norm.rows != 1
            || residual_rms_norm.columns != hidden
            || post_ffn_residual_rms_norm.rows != 1
            || post_ffn_residual_rms_norm.columns != hidden
            || ffn_gate_up[1].packed_s_in_bytes()? != ffn_gate_up[0].packed_s_in_bytes()?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} resources are not graph-owned or shape-compatible"
            )));
        }
        let layer_prefix = format!("model.language_model.layers.{layer}");
        let mlp_prefix = format!("{layer_prefix}.mlp");
        for (projection, name) in [
            (ffn_gate_up[0], format!("{mlp_prefix}.gate_proj.weight")),
            (ffn_gate_up[1], format!("{mlp_prefix}.up_proj.weight")),
            (swiglu_down, format!("{mlp_prefix}.down_proj.weight")),
        ] {
            if !projection.matches_recovered_tensor(&name)? {
                return Err(EngineError::InvalidState(format!(
                    "Metal transformer tail layer {layer} projection does not match {name}"
                )));
            }
        }
        let config = Qwen38Config::default();
        let next_norm_name = if layer + 1 == config.num_hidden_layers {
            "model.language_model.norm.weight".to_owned()
        } else {
            format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            )
        };
        if !residual_rms_norm
            .matches_weight_tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?
            || !post_ffn_residual_rms_norm.matches_weight_tensor(&next_norm_name)?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} norm identity is incompatible"
            )));
        }

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-transformer-tail");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            residual_rms_norm,
            arena,
            residual.reads()[0].offset(),
            arena,
            residual.reads()[1].offset(),
            arena,
            residual.writes()[0].offset(),
            arena,
            residual.writes()[1].offset(),
        );
        for (projection, output) in ffn_gate_up.iter().zip(ffn.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                ffn.reads()[0].offset(),
                arena,
                output.offset(),
            )?;
        }
        self.encode_mapped_swiglu_projection_between(
            encoder,
            swiglu_down,
            arena,
            down.reads()[0].offset(),
            arena,
            down.reads()[1].offset(),
            arena,
            down.writes()[0].offset(),
        )?;
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            post_ffn_residual_rms_norm,
            arena,
            post_ffn.reads()[0].offset(),
            arena,
            post_ffn.reads()[1].offset(),
            arena,
            post_ffn.writes()[0].offset(),
            arena,
            post_ffn.writes()[1].offset(),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal transformer tail layer {layer} ended with {:?}",
                command_buffer.status()
            )));
        }
        Ok(())
    }

    /// Validate one complete target full-attention layer before cache metadata
    /// is advanced or any command is encoded.
    fn validate_prepared_mapped_full_attention_layer(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        prepared: &PreparedMappedMetalFullAttentionLayer,
    ) -> Result<ValidatedMetalFullAttentionLayer> {
        const OPERATIONS: [MetalDecodeOperation; 10] = [
            MetalDecodeOperation::FullAttentionFanout,
            MetalDecodeOperation::QueryGateNormRope,
            MetalDecodeOperation::KeyRope,
            MetalDecodeOperation::PagedKvAppend,
            MetalDecodeOperation::PagedGqa,
            MetalDecodeOperation::AttentionGateOutputProjection,
            MetalDecodeOperation::ResidualRmsNorm,
            MetalDecodeOperation::FfnGateUpFanout,
            MetalDecodeOperation::SwiGluDownProjection,
            MetalDecodeOperation::ResidualRmsNorm,
        ];
        let layer = prepared.layer;
        let expected_start = layer
            .checked_mul(OPERATIONS.len())
            .and_then(|offset| offset.checked_add(2))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal full-attention schedule index overflows".into())
            })?;
        if steps.len() != OPERATIONS.len()
            || steps
                .iter()
                .zip(OPERATIONS)
                .enumerate()
                .any(|(offset, (step, operation))| {
                    step.step().schedule_index != expected_start + offset
                        || step.step().layer != Some(layer)
                        || step.step().operation != operation
                })
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} does not match its frozen ten-step schedule"
            )));
        }
        let mapping = &prepared.fanout.projections[0].mapping.inner;
        if prepared.fanout.layer != layer
            || prepared.query_gate.layer != layer
            || prepared.attention.owner_layer != Some(layer)
            || prepared.attention_output.layer != layer
            || !Rc::ptr_eq(&prepared.query_gate.mapping.inner, mapping)
            || !Rc::ptr_eq(&prepared.attention_output.projection.mapping.inner, mapping)
            || prepared
                .fanout
                .projections
                .iter()
                .any(|projection| !Rc::ptr_eq(&projection.mapping.inner, mapping))
            || prepared
                .ffn_gate_up
                .iter()
                .any(|projection| !Rc::ptr_eq(&projection.mapping.inner, mapping))
            || !Rc::ptr_eq(&prepared.swiglu_down.mapping.inner, mapping)
            || !Rc::ptr_eq(&prepared.residual_rms_norm.mapping.inner, mapping)
            || !Rc::ptr_eq(&prepared.post_ffn_residual_rms_norm.mapping.inner, mapping)
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} resources do not share one canonical mapping"
            )));
        }
        let arena = steps[0].reads()[0].buffer();
        if steps
            .iter()
            .flat_map(|step| step.reads().iter().chain(step.writes()))
            .any(|view| !std::ptr::eq(arena, view.buffer()))
            || prepared.attention.poisoned
            || prepared.attention.query_buffer.is_some()
            || prepared.attention.output_buffer.is_some()
            || prepared.key_rope.has_owned_values()
            || prepared.attention.cache.tokens() >= prepared.attention.maximum_tokens
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} arena or persistent state is not ready"
            )));
        }
        let fanout = &steps[0];
        let query_gate = &steps[1];
        let key_rope = &steps[2];
        let append = &steps[3];
        let gqa = &steps[4];
        let attention_output = &steps[5];
        let residual = &steps[6];
        let ffn = &steps[7];
        let down = &steps[8];
        let post_ffn = &steps[9];
        let query_values = prepared
            .attention
            .query_heads
            .checked_mul(prepared.attention.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal query width overflows".into()))?;
        let component_values = prepared
            .attention
            .key_value_heads
            .checked_mul(prepared.attention.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("Metal KV width overflows".into()))?;
        if fanout.reads().len() != 1
            || fanout.writes().len() != 3
            || query_gate.reads().len() != 1
            || query_gate.writes().len() != 2
            || key_rope.reads().len() != 1
            || key_rope.writes().len() != 1
            || append.reads().len() != 2
            || !append.writes().is_empty()
            || gqa.reads().len() != 1
            || gqa.writes().len() != 1
            || attention_output.reads().len() != 2
            || attention_output.writes().len() != 1
            || fanout.reads()[0].slot() != MetalBufferSlot::Normalized
            || fanout.writes()[0].slot() != MetalBufferSlot::QueryGate
            || fanout.writes()[1].slot() != MetalBufferSlot::Key
            || fanout.writes()[2].slot() != MetalBufferSlot::Value
            || query_gate.reads()[0].slot() != MetalBufferSlot::QueryGate
            || query_gate.writes()[0].slot() != MetalBufferSlot::Query
            || query_gate.writes()[1].slot() != MetalBufferSlot::AttentionGate
            || key_rope.reads()[0].slot() != MetalBufferSlot::Key
            || key_rope.writes()[0].slot() != MetalBufferSlot::Key
            || append.reads()[0].slot() != MetalBufferSlot::Key
            || append.reads()[1].slot() != MetalBufferSlot::Value
            || gqa.reads()[0].slot() != MetalBufferSlot::Query
            || gqa.writes()[0].slot() != MetalBufferSlot::AttentionOutput
            || attention_output.reads()[0].slot() != MetalBufferSlot::AttentionOutput
            || attention_output.reads()[1].slot() != MetalBufferSlot::AttentionGate
            || attention_output.writes()[0].slot() != MetalBufferSlot::MixerOutput
            || residual.reads().len() != 2
            || residual.writes().len() != 2
            || residual.reads()[0].slot() != MetalBufferSlot::HiddenA
            || residual.reads()[1].slot() != MetalBufferSlot::MixerOutput
            || residual.writes()[0].slot() != MetalBufferSlot::HiddenB
            || residual.writes()[1].slot() != MetalBufferSlot::Normalized
            || ffn.reads().len() != 1
            || ffn.writes().len() != 2
            || ffn.reads()[0].slot() != MetalBufferSlot::Normalized
            || ffn.writes()[0].slot() != MetalBufferSlot::FfnGate
            || ffn.writes()[1].slot() != MetalBufferSlot::FfnUp
            || down.reads().len() != 2
            || down.writes().len() != 1
            || down.reads()[0].slot() != MetalBufferSlot::FfnGate
            || down.reads()[1].slot() != MetalBufferSlot::FfnUp
            || down.writes()[0].slot() != MetalBufferSlot::FfnDown
            || post_ffn.reads().len() != 2
            || post_ffn.writes().len() != 2
            || post_ffn.reads()[0].slot() != MetalBufferSlot::HiddenB
            || post_ffn.reads()[1].slot() != MetalBufferSlot::FfnDown
            || post_ffn.writes()[0].slot() != MetalBufferSlot::HiddenA
            || post_ffn.writes()[1].slot() != MetalBufferSlot::Normalized
            || gqa.reads()[0].values() < query_values
            || gqa.writes()[0].values() < query_values
            || append.reads()[0].values() < component_values
            || append.reads()[1].values() < component_values
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} views are not the frozen Qwen graph"
            )));
        }
        let config = Qwen38Config::default();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        if prepared.fanout.projections[0].columns != hidden
            || prepared.fanout.projections[0].rows != query_values * 2
            || prepared.fanout.projections[1].columns != hidden
            || prepared.fanout.projections[1].rows != component_values
            || prepared.fanout.projections[2].columns != hidden
            || prepared.fanout.projections[2].rows != component_values
            || prepared.attention_output.projection.columns != query_values
            || prepared.attention_output.projection.rows != hidden
            || prepared.residual_rms_norm.columns != hidden
            || prepared.post_ffn_residual_rms_norm.columns != hidden
            || prepared.ffn_gate_up.iter().any(|projection| {
                projection.columns != hidden
                    || projection.rows != intermediate
                    || projection.input_buffer.is_some()
                    || projection.output_buffer.is_some()
            })
            || prepared.swiglu_down.columns != intermediate
            || prepared.swiglu_down.rows != hidden
            || prepared.swiglu_down.input_buffer.is_some()
            || prepared.swiglu_down.output_buffer.is_some()
            || prepared.residual_rms_norm.input_buffer.is_some()
            || prepared.residual_rms_norm.output_buffer.is_some()
            || prepared.post_ffn_residual_rms_norm.input_buffer.is_some()
            || prepared.post_ffn_residual_rms_norm.output_buffer.is_some()
            || prepared.ffn_gate_up[1].packed_s_in_bytes()?
                != prepared.ffn_gate_up[0].packed_s_in_bytes()?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal full-attention layer {layer} resources are not graph-owned or shape-compatible"
            )));
        }
        let thread_width = dispatch_width(&self.partial_rope_pipeline, DEFAULT_SIMDGROUPS)?;
        Ok(ValidatedMetalFullAttentionLayer {
            layer,
            component_values,
            thread_width,
        })
    }

    /// Execute one admitted target full-attention layer through its exact ten
    /// schedule views in one command encoder and one wait. CPU cache metadata
    /// is planned before encoding; K/V projection output is consumed only by
    /// later kernels in the same command buffer.
    pub fn dispatch_prepared_mapped_full_attention_layer(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        prepared: &mut PreparedMappedMetalFullAttentionLayer,
    ) -> Result<()> {
        let validated = self.validate_prepared_mapped_full_attention_layer(steps, prepared)?;
        prepared.attention.poisoned = true;
        let result = (|| {
            let plan = self.plan_paged_gqa_append(&mut prepared.attention)?;
            write_metal_paged_gqa_descriptors(&prepared.attention)?;
            write_metal_paged_gqa_params(&prepared.attention)?;
            let command_buffer = self.queue.new_command_buffer();
            command_buffer.set_label("ctox-qwen38-shared-arena-full-attention-layer");
            let encoder = command_buffer.new_compute_command_encoder();
            self.encode_prepared_mapped_full_attention_layer(
                encoder,
                steps,
                prepared,
                &plan,
                validated.thread_width,
                validated.component_values,
            )?;
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(EngineError::InvalidState(format!(
                    "Metal full-attention layer {} ended with {:?}",
                    validated.layer,
                    command_buffer.status()
                )));
            }
            #[cfg(test)]
            {
                let key = unsafe {
                    slice::from_raw_parts(
                        prepared
                            .attention
                            .verifier_key_snapshot_buffer
                            .contents()
                            .cast::<f32>(),
                        validated.component_values,
                    )
                    .to_vec()
                };
                let value = unsafe {
                    slice::from_raw_parts(
                        prepared
                            .attention
                            .verifier_value_snapshot_buffer
                            .contents()
                            .cast::<f32>(),
                        validated.component_values,
                    )
                    .to_vec()
                };
                self.commit_paged_gqa_verifier_append(
                    &mut prepared.attention,
                    &plan,
                    &key,
                    &value,
                )?;
            }
            Ok(())
        })();
        if result.is_ok() {
            prepared.attention.poisoned = false;
        }
        result
    }

    /// Validate the entire topology-ordered target layer set before a graph
    /// encoder advances any cache metadata or touches persistent device state.
    /// Every layer reuses the exact standalone admission contract, and all 64
    /// owners must point at one canonical artifact mapping.
    pub fn validate_prepared_mapped_target_layers(
        &self,
        program: &PreparedMetalDecodeProgram<'_>,
        prepared: &PreparedMappedMetalTargetLayers,
    ) -> Result<()> {
        let config = Qwen38Config::default();
        if prepared.layers.len() != config.num_hidden_layers
            || prepared.copied_model_bytes() != 0
            || prepared.resident_state_bytes()? == 0
        {
            return Err(EngineError::InvalidState(
                "Metal target graph does not own one complete 64-layer state set".into(),
            ));
        }
        let canonical_mapping = match &prepared.layers[0] {
            PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                &layer.projections[0].mapping.inner
            }
            PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                &layer.fanout.projections[0].mapping.inner
            }
        };
        let mut linear_layers = 0_usize;
        let mut full_layers = 0_usize;
        for (expected_layer, layer) in prepared.layers.iter().enumerate() {
            if layer.layer() != expected_layer
                || Some(layer.kind()) != config.layer_kind(expected_layer)
            {
                return Err(EngineError::InvalidState(format!(
                    "Metal target graph layer {expected_layer} has the wrong topology identity"
                )));
            }
            match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    if !Rc::ptr_eq(&layer.projections[0].mapping.inner, canonical_mapping) {
                        return Err(EngineError::InvalidState(format!(
                            "Metal target graph linear layer {expected_layer} uses a different artifact mapping"
                        )));
                    }
                    let steps = program.linear_attention_layer_steps(expected_layer)?;
                    let validated = self.validate_mapped_linear_attention_layer_views(
                        steps,
                        [
                            &layer.projections[0],
                            &layer.projections[1],
                            &layer.projections[2],
                            &layer.projections[3],
                        ],
                        &layer.convolution,
                        &layer.gated_delta_prepare,
                        &layer.recurrence,
                        &layer.gated_rms_norm,
                        &layer.linear_output_projection,
                        &layer.residual_rms_norm,
                        [&layer.ffn_gate_up[0], &layer.ffn_gate_up[1]],
                        &layer.swiglu_down,
                        &layer.post_ffn_residual_rms_norm,
                    )?;
                    if validated != expected_layer {
                        return Err(EngineError::InvalidState(format!(
                            "Metal target graph linear layer {expected_layer} validated as {validated}"
                        )));
                    }
                    linear_layers += 1;
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    if !Rc::ptr_eq(
                        &layer.fanout.projections[0].mapping.inner,
                        canonical_mapping,
                    ) {
                        return Err(EngineError::InvalidState(format!(
                            "Metal target graph full-attention layer {expected_layer} uses a different artifact mapping"
                        )));
                    }
                    let steps = program.full_attention_layer_steps(expected_layer)?;
                    let validated =
                        self.validate_prepared_mapped_full_attention_layer(steps, layer)?;
                    if validated.layer != expected_layer {
                        return Err(EngineError::InvalidState(format!(
                            "Metal target graph full-attention layer {expected_layer} validated as {}",
                            validated.layer
                        )));
                    }
                    full_layers += 1;
                }
            }
        }
        if linear_layers != config.linear_attention_layers()
            || full_layers != config.full_attention_layers()
        {
            return Err(EngineError::InvalidState(format!(
                "Metal target graph has {linear_layers} linear and {full_layers} full-attention layers"
            )));
        }
        Ok(())
    }

    fn encode_prepared_mapped_target_layers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        program: &PreparedMetalDecodeProgram<'_>,
        prepared: &mut PreparedMappedMetalTargetLayers,
    ) -> Result<Vec<(usize, MetalPagedGqaAppendPlan)>> {
        if !prepared.transaction_active {
            return Err(EngineError::InvalidState(
                "Metal target graph encoding requires an active state transaction".into(),
            ));
        }
        let mut full_plans = Vec::with_capacity(Qwen38Config::default().full_attention_layers());
        for (layer_index, layer) in prepared.layers.iter_mut().enumerate() {
            match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    let steps = program.linear_attention_layer_steps(layer_index)?;
                    self.encode_mapped_linear_attention_layer_views(
                        encoder,
                        steps,
                        [
                            &layer.projections[0],
                            &layer.projections[1],
                            &layer.projections[2],
                            &layer.projections[3],
                        ],
                        &mut layer.convolution,
                        &layer.gated_delta_prepare,
                        &mut layer.recurrence,
                        &layer.gated_rms_norm,
                        &layer.linear_output_projection,
                        &layer.residual_rms_norm,
                        [&layer.ffn_gate_up[0], &layer.ffn_gate_up[1]],
                        &layer.swiglu_down,
                        &layer.post_ffn_residual_rms_norm,
                    )?;
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    let steps = program.full_attention_layer_steps(layer_index)?;
                    let validated =
                        self.validate_prepared_mapped_full_attention_layer(steps, layer)?;
                    layer.attention.poisoned = true;
                    let plan = self.plan_paged_gqa_append(&mut layer.attention)?;
                    write_metal_paged_gqa_descriptors(&layer.attention)?;
                    write_metal_paged_gqa_params(&layer.attention)?;
                    self.encode_prepared_mapped_full_attention_layer(
                        encoder,
                        steps,
                        layer,
                        &plan,
                        validated.thread_width,
                        validated.component_values,
                    )?;
                    full_plans.push((layer_index, plan));
                }
            }
        }
        Ok(full_plans)
    }

    /// Execute all 64 target transformer layers in topology order through one
    /// shared activation arena, one compute encoder, one command buffer, and
    /// one completion wait. Persistent KV/convolution/recurrent state is
    /// checkpointed as one target-layer transaction and restored on every
    /// encoding, GPU, or verifier failure.
    pub fn dispatch_prepared_mapped_target_layers(
        &self,
        program: &PreparedMetalDecodeProgram<'_>,
        prepared: &mut PreparedMappedMetalTargetLayers,
    ) -> Result<()> {
        self.validate_prepared_mapped_target_layers(program, prepared)?;
        prepared.begin_speculative(self)?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-complete-target-layers");
        let encoder = command_buffer.new_compute_command_encoder();
        let full_plans = match self.encode_prepared_mapped_target_layers(encoder, program, prepared)
        {
            Ok(plans) => plans,
            Err(primary) => {
                encoder.end_encoding();
                return match prepared.restore_speculative(self) {
                    Ok(()) => Err(primary),
                    Err(rollback) => Err(EngineError::InvalidState(format!(
                    "Metal target graph encoding failed ({primary}) and rollback failed: {rollback}"
                ))),
                };
            }
        };
        #[cfg(not(test))]
        let _ = &full_plans;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            let primary = EngineError::InvalidState(format!(
                "Metal target graph ended with {:?}",
                command_buffer.status()
            ));
            return match prepared.restore_speculative(self) {
                Ok(()) => Err(primary),
                Err(rollback) => Err(EngineError::InvalidState(format!(
                    "Metal target graph failed ({primary}) and rollback failed: {rollback}"
                ))),
            };
        }
        for layer in &mut prepared.layers {
            match layer {
                PreparedMappedMetalTargetLayer::LinearAttention(layer) => {
                    layer.convolution.poisoned = false;
                    layer.recurrence.poisoned = false;
                }
                PreparedMappedMetalTargetLayer::FullAttention(layer) => {
                    layer.attention.poisoned = false;
                }
            }
        }
        #[cfg(test)]
        for (layer_index, plan) in &full_plans {
            let PreparedMappedMetalTargetLayer::FullAttention(layer) =
                &mut prepared.layers[*layer_index]
            else {
                let primary = EngineError::InvalidState(
                    "Metal target graph verifier plan diverged from topology".into(),
                );
                return match prepared.restore_speculative(self) {
                    Ok(()) => Err(primary),
                    Err(rollback) => Err(EngineError::InvalidState(format!(
                        "Metal target graph verification failed ({primary}) and rollback failed: {rollback}"
                    ))),
                };
            };
            let component_values = layer.attention.key_value_heads * layer.attention.head_dim;
            let key = unsafe {
                slice::from_raw_parts(
                    layer
                        .attention
                        .verifier_key_snapshot_buffer
                        .contents()
                        .cast::<f32>(),
                    component_values,
                )
                .to_vec()
            };
            let value = unsafe {
                slice::from_raw_parts(
                    layer
                        .attention
                        .verifier_value_snapshot_buffer
                        .contents()
                        .cast::<f32>(),
                    component_values,
                )
                .to_vec()
            };
            if let Err(primary) =
                self.commit_paged_gqa_verifier_append(&mut layer.attention, plan, &key, &value)
            {
                return match prepared.restore_speculative(self) {
                    Ok(()) => Err(primary),
                    Err(rollback) => Err(EngineError::InvalidState(format!(
                        "Metal target graph verification failed ({primary}) and rollback failed: {rollback}"
                    ))),
                };
            }
        }
        prepared.commit_speculative()
    }

    fn encode_prepared_mapped_full_attention_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        steps: &[PreparedMetalDecodeStepView<'_>],
        prepared: &PreparedMappedMetalFullAttentionLayer,
        plan: &MetalPagedGqaAppendPlan,
        thread_width: usize,
        component_values: usize,
    ) -> Result<()> {
        #[cfg(not(test))]
        let _ = component_values;
        let arena = steps[0].reads()[0].buffer();
        let fanout = &steps[0];
        let query_gate = &steps[1];
        let key_rope = &steps[2];
        let append = &steps[3];
        let gqa = &steps[4];
        let attention_output = &steps[5];
        let residual = &steps[6];
        let ffn = &steps[7];
        let down = &steps[8];
        let post_ffn = &steps[9];
        for (projection, output) in prepared.fanout.projections.iter().zip(fanout.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                fanout.reads()[0].offset(),
                arena,
                output.offset(),
            )?;
        }
        self.encode_mapped_query_gate_norm_rope_between(
            encoder,
            &prepared.query_gate,
            arena,
            query_gate.reads()[0].offset(),
            arena,
            query_gate.writes()[0].offset(),
            arena,
            query_gate.writes()[1].offset(),
        );
        self.encode_partial_rope_between(
            encoder,
            &prepared.key_rope,
            arena,
            key_rope.writes()[0].offset(),
            thread_width,
        )?;
        #[cfg(test)]
        {
            encoder.set_compute_pipeline_state(&self.copy_f32_pipeline);
            encoder.set_buffer(0, Some(arena), append.reads()[0].offset());
            encoder.set_buffer(1, Some(&prepared.attention.verifier_key_snapshot_buffer), 0);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: component_values as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.set_buffer(0, Some(arena), append.reads()[1].offset());
            encoder.set_buffer(
                1,
                Some(&prepared.attention.verifier_value_snapshot_buffer),
                0,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: component_values as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
        }
        self.encode_paged_gqa_append_and_attention(
            encoder,
            &prepared.attention,
            plan,
            arena,
            gqa.reads()[0].offset(),
            arena,
            append.reads()[0].offset(),
            arena,
            append.reads()[1].offset(),
            arena,
            gqa.writes()[0].offset(),
        )?;
        self.encode_mapped_sigmoid_gate_projection_between(
            encoder,
            &prepared.attention_output.projection,
            arena,
            attention_output.reads()[0].offset(),
            arena,
            attention_output.reads()[1].offset(),
            arena,
            attention_output.writes()[0].offset(),
        )?;
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            &prepared.residual_rms_norm,
            arena,
            residual.reads()[0].offset(),
            arena,
            residual.reads()[1].offset(),
            arena,
            residual.writes()[0].offset(),
            arena,
            residual.writes()[1].offset(),
        );
        for (projection, output) in prepared.ffn_gate_up.iter().zip(ffn.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                ffn.reads()[0].offset(),
                arena,
                output.offset(),
            )?;
        }
        self.encode_mapped_swiglu_projection_between(
            encoder,
            &prepared.swiglu_down,
            arena,
            down.reads()[0].offset(),
            arena,
            down.reads()[1].offset(),
            arena,
            down.writes()[0].offset(),
        )?;
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            &prepared.post_ffn_residual_rms_norm,
            arena,
            post_ffn.reads()[0].offset(),
            arena,
            post_ffn.reads()[1].offset(),
            arena,
            post_ffn.writes()[0].offset(),
            arena,
            post_ffn.writes()[1].offset(),
        );
        Ok(())
    }

    /// Validate one complete frozen linear-attention transformer layer before
    /// any command is encoded or persistent state is mutated.
    #[allow(clippy::too_many_arguments)]
    fn validate_mapped_linear_attention_layer_views(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        projections: [&PreparedMappedMetalMatVec; 4],
        convolution: &PreparedMappedMetalCausalConv,
        gated_delta_prepare: &PreparedMappedMetalGatedDeltaPrepare,
        recurrence: &PreparedMetalGatedDelta,
        gated_rms_norm: &PreparedMappedMetalGatedRmsNorm,
        linear_output_projection: &PreparedMappedMetalMatVec,
        residual_rms_norm: &PreparedMappedMetalRmsNorm,
        ffn_gate_up: [&PreparedMappedMetalMatVec; 2],
        swiglu_down: &PreparedMappedMetalMatVec,
        post_ffn_residual_rms_norm: &PreparedMappedMetalRmsNorm,
    ) -> Result<usize> {
        const OPERATIONS: [MetalDecodeOperation; 10] = [
            MetalDecodeOperation::LinearFanout,
            MetalDecodeOperation::CausalConvolution,
            MetalDecodeOperation::GatedDeltaPrepare,
            MetalDecodeOperation::GatedDeltaRecurrent,
            MetalDecodeOperation::GatedRmsNorm,
            MetalDecodeOperation::LinearOutputProjection,
            MetalDecodeOperation::ResidualRmsNorm,
            MetalDecodeOperation::FfnGateUpFanout,
            MetalDecodeOperation::SwiGluDownProjection,
            MetalDecodeOperation::ResidualRmsNorm,
        ];
        if steps.len() != OPERATIONS.len() {
            return Err(EngineError::InvalidState(format!(
                "Metal linear layer has {} bound steps, expected {}",
                steps.len(),
                OPERATIONS.len()
            )));
        }
        let layer = steps[0].step().layer.ok_or_else(|| {
            EngineError::InvalidState("Metal linear layer has no layer identity".into())
        })?;
        let expected_start = layer
            .checked_mul(OPERATIONS.len())
            .and_then(|offset| offset.checked_add(2))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal linear layer schedule index overflows".into())
            })?;
        if steps
            .iter()
            .zip(OPERATIONS)
            .enumerate()
            .any(|(offset, (step, operation))| {
                step.step().schedule_index != expected_start + offset
                    || step.step().layer != Some(layer)
                    || step.step().operation != operation
            })
        {
            return Err(EngineError::InvalidState(format!(
                "Metal linear layer {layer} does not match its frozen ten-step schedule"
            )));
        }

        let input = steps[0].reads().first().ok_or_else(|| {
            EngineError::InvalidState("Metal linear layer has no normalized input view".into())
        })?;
        let arena = input.buffer();
        if steps
            .iter()
            .flat_map(|step| step.reads().iter().chain(step.writes()))
            .any(|view| !std::ptr::eq(arena, view.buffer()))
        {
            return Err(EngineError::InvalidState(
                "Metal linear layer does not use one shared activation arena".into(),
            ));
        }
        let hidden = input.values();
        let mapping = &projections[0].mapping.inner;
        let all_projections = [
            projections[0],
            projections[1],
            projections[2],
            projections[3],
            linear_output_projection,
            ffn_gate_up[0],
            ffn_gate_up[1],
            swiglu_down,
        ];
        if all_projections.iter().any(|projection| {
            projection.input_buffer.is_some()
                || projection.output_buffer.is_some()
                || !Rc::ptr_eq(&projection.mapping.inner, mapping)
        }) || !Rc::ptr_eq(&convolution.mapping.inner, mapping)
            || !Rc::ptr_eq(&gated_delta_prepare.mapping.inner, mapping)
            || !Rc::ptr_eq(&gated_rms_norm.mapping.inner, mapping)
            || !Rc::ptr_eq(&residual_rms_norm.mapping.inner, mapping)
            || !Rc::ptr_eq(&post_ffn_residual_rms_norm.mapping.inner, mapping)
        {
            return Err(EngineError::InvalidState(
                "Metal linear layer resources do not share one artifact mapping".into(),
            ));
        }
        let linear_prefix = format!("model.language_model.layers.{layer}.linear_attn");
        for (projection, name) in projections.iter().zip([
            format!("{linear_prefix}.in_proj_qkv.weight"),
            format!("{linear_prefix}.in_proj_z.weight"),
            format!("{linear_prefix}.in_proj_a.weight"),
            format!("{linear_prefix}.in_proj_b.weight"),
        ]) {
            if !projection.matches_recovered_tensor(&name)? {
                return Err(EngineError::InvalidState(format!(
                    "Metal linear layer {layer} projection does not match {name}"
                )));
            }
        }
        let mlp_prefix = format!("model.language_model.layers.{layer}.mlp");
        for (projection, name) in [
            (
                linear_output_projection,
                format!("{linear_prefix}.out_proj.weight"),
            ),
            (ffn_gate_up[0], format!("{mlp_prefix}.gate_proj.weight")),
            (ffn_gate_up[1], format!("{mlp_prefix}.up_proj.weight")),
            (swiglu_down, format!("{mlp_prefix}.down_proj.weight")),
        ] {
            if !projection.matches_recovered_tensor(&name)? {
                return Err(EngineError::InvalidState(format!(
                    "Metal linear layer {layer} projection does not match {name}"
                )));
            }
        }
        let layer_prefix = format!("model.language_model.layers.{layer}");
        if !convolution.matches_weight_tensor(&format!("{linear_prefix}.conv1d.weight"))?
            || !gated_delta_prepare.matches_parameter_tensors(
                &format!("{linear_prefix}.A_log"),
                &format!("{linear_prefix}.dt_bias"),
            )?
            || recurrence.owner_layer != Some(layer)
            || !gated_rms_norm.matches_weight_tensor(&format!("{linear_prefix}.norm.weight"))?
            || !residual_rms_norm
                .matches_weight_tensor(&format!("{layer_prefix}.post_attention_layernorm.weight"))?
            || !post_ffn_residual_rms_norm.matches_weight_tensor(&format!(
                "model.language_model.layers.{}.input_layernorm.weight",
                layer + 1
            ))?
        {
            return Err(EngineError::InvalidState(format!(
                "Metal linear layer {layer} state or norm identity is incompatible"
            )));
        }
        if convolution.poisoned
            || recurrence.poisoned
            || convolution.input_buffer.is_some()
            || convolution.output_buffer.is_some()
            || recurrence.has_owned_io()
            || recurrence.config != MetalGatedDeltaConfig::QWEN38_27B
            || gated_rms_norm.has_owned_io()
            || residual_rms_norm.input_buffer.is_some()
            || residual_rms_norm.output_buffer.is_some()
            || post_ffn_residual_rms_norm.input_buffer.is_some()
            || post_ffn_residual_rms_norm.output_buffer.is_some()
        {
            return Err(EngineError::InvalidState(
                "Metal linear layer state or graph-owned I/O is not ready".into(),
            ));
        }

        let linear = &steps[0];
        if linear.reads().len() != 1
            || linear.writes().len() != 4
            || linear.reads()[0].slot() != MetalBufferSlot::Normalized
            || projections
                .iter()
                .zip(linear.writes())
                .any(|(projection, output)| {
                    projection.columns != hidden || projection.rows != output.values()
                })
            || convolution.channels != linear.writes()[0].values()
        {
            return Err(EngineError::InvalidState(
                "Metal linear fan-out or convolution shape is incompatible".into(),
            ));
        }
        let prepare = &steps[2];
        let recurrent = &steps[3];
        let gated = &steps[4];
        if prepare.reads().len() != 3
            || prepare.writes().len() != 5
            || recurrent.reads().len() != 5
            || recurrent.writes().len() != 1
            || gated.reads().len() != 2
            || gated.writes().len() != 1
            || prepare
                .writes()
                .iter()
                .zip(recurrent.reads())
                .any(|(output, input)| {
                    output.slot() != input.slot()
                        || output.offset() != input.offset()
                        || output.values() != input.values()
                })
            || recurrent.writes()[0].values()
                != recurrence.config.heads * recurrence.config.value_dim
            || gated_rms_norm.rows != recurrence.config.heads
            || gated_rms_norm.columns != recurrence.config.value_dim
            || gated.reads()[0].offset() != recurrent.writes()[0].offset()
            || gated.writes()[0].offset() != gated.reads()[0].offset()
        {
            return Err(EngineError::InvalidState(
                "Metal GatedDelta layer views or resources are incompatible".into(),
            ));
        }
        let output_projection = &steps[5];
        let residual = &steps[6];
        let ffn = &steps[7];
        let down = &steps[8];
        let post_ffn = &steps[9];
        if output_projection.reads().len() != 1
            || output_projection.writes().len() != 1
            || linear_output_projection.columns != output_projection.reads()[0].values()
            || linear_output_projection.rows != hidden
            || residual.reads().len() != 2
            || residual.writes().len() != 2
            || residual_rms_norm.rows != 1
            || residual_rms_norm.columns != hidden
            || ffn.reads().len() != 1
            || ffn.writes().len() != 2
            || ffn_gate_up
                .iter()
                .zip(ffn.writes())
                .any(|(projection, output)| {
                    projection.columns != hidden || projection.rows != output.values()
                })
            || down.reads().len() != 2
            || down.writes().len() != 1
            || swiglu_down.columns != down.reads()[0].values()
            || down.reads()[1].values() != swiglu_down.columns
            || swiglu_down.rows != hidden
            || post_ffn.reads().len() != 2
            || post_ffn.writes().len() != 2
            || post_ffn_residual_rms_norm.rows != 1
            || post_ffn_residual_rms_norm.columns != hidden
        {
            return Err(EngineError::InvalidState(
                "Metal linear output, FFN, or residual resources are incompatible".into(),
            ));
        }

        Ok(layer)
    }

    /// Execute one admitted linear-attention layer in one command encoder and
    /// one wait. The separate validator is reusable by the complete graph
    /// encoder before it mutates any of the 64 layer states.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_mapped_linear_attention_layer_views(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        projections: [&PreparedMappedMetalMatVec; 4],
        convolution: &mut PreparedMappedMetalCausalConv,
        gated_delta_prepare: &PreparedMappedMetalGatedDeltaPrepare,
        recurrence: &mut PreparedMetalGatedDelta,
        gated_rms_norm: &PreparedMappedMetalGatedRmsNorm,
        linear_output_projection: &PreparedMappedMetalMatVec,
        residual_rms_norm: &PreparedMappedMetalRmsNorm,
        ffn_gate_up: [&PreparedMappedMetalMatVec; 2],
        swiglu_down: &PreparedMappedMetalMatVec,
        post_ffn_residual_rms_norm: &PreparedMappedMetalRmsNorm,
    ) -> Result<()> {
        let layer = self.validate_mapped_linear_attention_layer_views(
            steps,
            projections,
            convolution,
            gated_delta_prepare,
            recurrence,
            gated_rms_norm,
            linear_output_projection,
            residual_rms_norm,
            ffn_gate_up,
            swiglu_down,
            post_ffn_residual_rms_norm,
        )?;
        if !convolution.checkpoint_valid || !recurrence.checkpoint_valid {
            return Err(EngineError::InvalidState(format!(
                "Metal linear layer {layer} requires active device checkpoints"
            )));
        }
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-shared-arena-linear-layer");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_linear_attention_layer_views(
            encoder,
            steps,
            projections,
            convolution,
            gated_delta_prepare,
            recurrence,
            gated_rms_norm,
            linear_output_projection,
            residual_rms_norm,
            ffn_gate_up,
            swiglu_down,
            post_ffn_residual_rms_norm,
        )?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal linear layer {layer} ended with {:?}",
                command_buffer.status()
            )));
        }
        convolution.poisoned = false;
        recurrence.poisoned = false;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mapped_linear_attention_layer_views(
        &self,
        encoder: &ComputeCommandEncoderRef,
        steps: &[PreparedMetalDecodeStepView<'_>],
        projections: [&PreparedMappedMetalMatVec; 4],
        convolution: &mut PreparedMappedMetalCausalConv,
        gated_delta_prepare: &PreparedMappedMetalGatedDeltaPrepare,
        recurrence: &mut PreparedMetalGatedDelta,
        gated_rms_norm: &PreparedMappedMetalGatedRmsNorm,
        linear_output_projection: &PreparedMappedMetalMatVec,
        residual_rms_norm: &PreparedMappedMetalRmsNorm,
        ffn_gate_up: [&PreparedMappedMetalMatVec; 2],
        swiglu_down: &PreparedMappedMetalMatVec,
        post_ffn_residual_rms_norm: &PreparedMappedMetalRmsNorm,
    ) -> Result<()> {
        let arena = steps[0].reads()[0].buffer();
        let linear = &steps[0];
        let prepare = &steps[2];
        let recurrent = &steps[3];
        let gated = &steps[4];
        let output_projection = &steps[5];
        let residual = &steps[6];
        let ffn = &steps[7];
        let down = &steps[8];
        let post_ffn = &steps[9];
        for (projection, output) in projections.iter().zip(linear.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                linear.reads()[0].offset(),
                arena,
                output.offset(),
            )?;
        }
        self.encode_mapped_causal_conv_between(
            encoder,
            convolution,
            arena,
            steps[1].reads()[0].offset(),
            arena,
            steps[1].writes()[0].offset(),
        )?;
        convolution.poisoned = true;
        self.encode_mapped_gated_delta_prepare_between(
            encoder,
            gated_delta_prepare,
            arena,
            prepare.reads()[0].offset(),
            prepare.reads()[1].offset(),
            prepare.reads()[2].offset(),
            prepare.writes()[0].offset(),
            prepare.writes()[1].offset(),
            prepare.writes()[2].offset(),
            prepare.writes()[3].offset(),
            prepare.writes()[4].offset(),
        )?;
        self.encode_gated_delta_f16_between(
            encoder,
            recurrence,
            arena,
            recurrent.reads()[0].offset(),
            recurrent.reads()[1].offset(),
            recurrent.reads()[2].offset(),
            recurrent.reads()[3].offset(),
            recurrent.reads()[4].offset(),
            recurrent.writes()[0].offset(),
        );
        recurrence.poisoned = true;
        self.encode_mapped_gated_rms_norm_between(
            encoder,
            gated_rms_norm,
            arena,
            gated.reads()[0].offset(),
            arena,
            gated.reads()[1].offset(),
            arena,
            gated.writes()[0].offset(),
        );
        self.encode_mapped_projection_between(
            encoder,
            linear_output_projection,
            arena,
            output_projection.reads()[0].offset(),
            arena,
            output_projection.writes()[0].offset(),
        )?;
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            residual_rms_norm,
            arena,
            residual.reads()[0].offset(),
            arena,
            residual.reads()[1].offset(),
            arena,
            residual.writes()[0].offset(),
            arena,
            residual.writes()[1].offset(),
        );
        for (projection, output) in ffn_gate_up.iter().zip(ffn.writes()) {
            self.encode_mapped_projection_between(
                encoder,
                projection,
                arena,
                ffn.reads()[0].offset(),
                arena,
                output.offset(),
            )?;
        }
        self.encode_mapped_swiglu_projection_between(
            encoder,
            swiglu_down,
            arena,
            down.reads()[0].offset(),
            arena,
            down.reads()[1].offset(),
            arena,
            down.writes()[0].offset(),
        )?;
        self.encode_mapped_residual_rms_norm_between(
            encoder,
            post_ffn_residual_rms_norm,
            arena,
            post_ffn.reads()[0].offset(),
            arena,
            post_ffn.reads()[1].offset(),
            arena,
            post_ffn.writes()[0].offset(),
            arena,
            post_ffn.writes()[1].offset(),
        );
        Ok(())
    }

    /// Dispatch one fully prepared layer bundle through its exact bound
    /// schedule slice. The bundle keeps layer ownership and canonical tensor
    /// identity together so callers cannot accidentally assemble a valid-shape
    /// but semantically mixed layer at the call site.
    pub fn dispatch_prepared_mapped_linear_attention_layer(
        &self,
        steps: &[PreparedMetalDecodeStepView<'_>],
        prepared: &mut PreparedMappedMetalLinearAttentionLayer,
    ) -> Result<()> {
        let PreparedMappedMetalLinearAttentionLayer {
            projections,
            convolution,
            gated_delta_prepare,
            recurrence,
            gated_rms_norm,
            linear_output_projection,
            residual_rms_norm,
            ffn_gate_up,
            swiglu_down,
            post_ffn_residual_rms_norm,
            ..
        } = prepared;
        self.dispatch_mapped_linear_attention_layer_views(
            steps,
            [
                &projections[0],
                &projections[1],
                &projections[2],
                &projections[3],
            ],
            convolution,
            gated_delta_prepare,
            recurrence,
            gated_rms_norm,
            linear_output_projection,
            residual_rms_norm,
            [&ffn_gate_up[0], &ffn_gate_up[1]],
            swiglu_down,
            post_ffn_residual_rms_norm,
        )
    }

    /// Encode one decode-row RMSNorm followed by a recovered Q2/Q4
    /// projection in a single command encoder. The projection consumes the
    /// RMSNorm output buffer directly and therefore owns no second activation
    /// allocation and performs no host readback between operations.
    pub fn dispatch_mapped_rms_norm_then_projection(
        &self,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
    ) -> Result<Vec<f32>> {
        self.validate_mapped_norm_projection(norm, projection)?;
        let projection_output = projection.owned_output()?;

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-rmsnorm-projection-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_norm_projection(encoder, norm, projection)?;
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal norm/projection command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(projection_output.contents().cast::<f32>(), projection.rows)
                .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal norm/projection chain produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Execute final RMSNorm, the complete recovered LM-head projection, and
    /// deterministic target selection in one command encoder. The selector
    /// may cover fewer values than the physical matrix has rows so padded
    /// vocabulary rows remain permanently unselectable. No logits are read by
    /// the host; only `{token_id, invalid_count}` is observed after completion.
    pub fn dispatch_mapped_rms_norm_projection_argmax(
        &self,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
        selector: &PreparedMetalArgMaxScratch,
    ) -> Result<u32> {
        self.validate_mapped_norm_projection(norm, projection)?;
        let projection_output = projection.owned_output()?;
        if selector.values > projection.rows {
            return Err(EngineError::Shape(format!(
                "Metal selector has {} valid values, projection has {} rows",
                selector.values, projection.rows
            )));
        }
        zero_buffer(&selector.result_buffer, 2 * std::mem::size_of::<u32>());
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-lm-head-selection-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_mapped_norm_projection(encoder, norm, projection)?;
        self.encode_argmax_f32(encoder, projection_output, selector);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal LM-head selection command ended with {:?}",
                command_buffer.status()
            )));
        }
        self.read_argmax_result(selector)
    }

    /// Records the same resident operation repeatedly into one command
    /// encoder and pays commit/wait only once. This is a benchmark/graph
    /// construction primitive; a production decoder will record distinct
    /// prepared operations with explicit buffer dependencies in the same way.
    pub fn dispatch_prepared_repeated(
        &self,
        prepared: &PreparedMetalMatVec,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "Metal repeated dispatch count must be positive".into(),
            ));
        }
        let pipeline = match prepared.dtype {
            TensorDType::Q2B64 => &self.q2_pipeline,
            TensorDType::Q4B64 => &self.q4_pipeline,
            _ => unreachable!("prepared Metal operation is Q2/Q4"),
        };

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-q2q4-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(
            MetalBufferAbi::WEIGHTS as u64,
            Some(&prepared.weights_buffer),
            0,
        );
        encoder.set_buffer(
            MetalBufferAbi::INPUT as u64,
            Some(&prepared.input_buffer),
            0,
        );
        encoder.set_buffer(MetalBufferAbi::S_IN as u64, Some(&prepared.s_in_buffer), 0);
        encoder.set_buffer(
            MetalBufferAbi::S_OUT as u64,
            Some(&prepared.s_out_buffer),
            0,
        );
        encoder.set_buffer(MetalBufferAbi::BIAS as u64, Some(&prepared.bias_buffer), 0);
        encoder.set_buffer(
            MetalBufferAbi::OUTPUT as u64,
            Some(&prepared.output_buffer),
            0,
        );
        encoder.set_buffer(
            MetalBufferAbi::PARAMS as u64,
            Some(&prepared.params_buffer),
            0,
        );
        let grid = MTLSize {
            width: prepared
                .rows
                .div_ceil((prepared.thread_width / 32) * ROWS_PER_SIMDGROUP)
                as u64,
            height: 1,
            depth: 1,
        };
        let threads = MTLSize {
            width: prepared.thread_width as u64,
            height: 1,
            depth: 1,
        };
        for _ in 0..dispatches {
            encoder.dispatch_thread_groups(grid, threads);
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal candidate command ended with {:?}",
                command_buffer.status()
            )));
        }

        // StorageModeShared makes the completed result coherently visible to
        // the CPU on Apple Silicon. Copy before the Metal buffer is dropped.
        let output = unsafe {
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                prepared.rows,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }
}

fn partial_rope_params(
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    position: u64,
    theta: f32,
) -> Result<MetalPartialRopeParams> {
    if heads == 0
        || head_dim == 0
        || rotary_dim == 0
        || rotary_dim > head_dim
        || !rotary_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return Err(EngineError::Shape(
            "invalid Metal partial-RoPE contract".into(),
        ));
    }
    Ok(MetalPartialRopeParams {
        heads: u32::try_from(heads)
            .map_err(|_| EngineError::Shape("Metal RoPE heads exceed u32".into()))?,
        head_dim: u32::try_from(head_dim)
            .map_err(|_| EngineError::Shape("Metal RoPE head dimension exceeds u32".into()))?,
        rotary_dim: u32::try_from(rotary_dim)
            .map_err(|_| EngineError::Shape("Metal RoPE dimension exceeds u32".into()))?,
        position: u32::try_from(position)
            .map_err(|_| EngineError::Shape("Metal RoPE position exceeds u32".into()))?,
        theta,
        reserved0: 0,
        reserved1: 0,
        reserved2: 0,
    })
}

fn partial_rope_tables(
    rotary_dim: usize,
    position: u64,
    theta: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || !theta.is_finite() || theta <= 0.0 {
        return Err(EngineError::Shape(
            "invalid Metal partial-RoPE table contract".into(),
        ));
    }
    let half_dim = rotary_dim / 2;
    let mut cosine = Vec::with_capacity(half_dim);
    let mut sine = Vec::with_capacity(half_dim);
    for index in 0..half_dim {
        let inverse_frequency = theta.powf(-((2 * index) as f32) / rotary_dim as f32);
        let angle = position as f32 * inverse_frequency;
        cosine.push(angle.cos());
        sine.push(angle.sin());
    }
    Ok((cosine, sine))
}

fn validate_metal_speculative_shape(
    transaction: &PreparedMetalSpeculativeTransaction,
    workspace: &PreparedMetalDecodeWorkspace,
    attentions: &[PreparedMetalPagedGqa],
    convolutions: &[PreparedMappedMetalCausalConv],
    recurrences: &[PreparedMetalGatedDelta],
) -> Result<()> {
    if attentions.len() != PreparedMetalSpeculativeTransaction::ATTENTION_STATES
        || convolutions.len() != PreparedMetalSpeculativeTransaction::LINEAR_STATES
        || recurrences.len() != PreparedMetalSpeculativeTransaction::LINEAR_STATES
    {
        return Err(EngineError::Shape(format!(
            "Metal speculative transaction requires exactly {} attention and {} paired linear states, got {}/{}/{}",
            PreparedMetalSpeculativeTransaction::ATTENTION_STATES,
            PreparedMetalSpeculativeTransaction::LINEAR_STATES,
            attentions.len(),
            convolutions.len(),
            recurrences.len()
        )));
    }
    let normalized = workspace.binding(MetalBufferSlot::Normalized)?;
    if normalized.values != transaction.target_hidden.values() {
        return Err(EngineError::Shape(format!(
            "Metal speculative target-hidden checkpoint has {} values, normalized workspace slot has {}",
            transaction.target_hidden.values(),
            normalized.values
        )));
    }
    Ok(())
}

fn validate_metal_input(input: &[f32], columns: usize) -> Result<()> {
    if input.len() != columns {
        return Err(EngineError::Shape(format!(
            "Metal prepared input has {} values, expected {columns}",
            input.len()
        )));
    }
    if input.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::InvalidArtifact(
            "Metal prepared input contains a non-finite value".into(),
        ));
    }
    Ok(())
}

fn metal_correction_identity(
    columns: usize,
    s_in: Option<ScaleSlice<'_>>,
) -> Result<MetalCorrectionIdentity> {
    let mut digest = Sha256::new();
    digest.update(b"ctox.metal.correction-identity.v1\0");
    digest.update(
        u64::try_from(columns)
            .map_err(|_| EngineError::Shape("Metal columns exceed u64".into()))?
            .to_le_bytes(),
    );
    match s_in {
        None => digest.update([0]),
        Some(ScaleSlice::F16Le(bytes)) => {
            let expected = columns
                .checked_mul(2)
                .ok_or_else(|| EngineError::Shape("Metal s_in size overflows".into()))?;
            if bytes.len() != expected {
                return Err(EngineError::Shape(format!(
                    "Metal s_in has {} bytes, expected {expected}",
                    bytes.len()
                )));
            }
            digest.update([1]);
            digest.update(bytes);
        }
        Some(ScaleSlice::F32(_)) => {
            return Err(EngineError::UnsupportedDType(
                "Metal correction identity requires packed FP16 s_in".into(),
            ));
        }
    }
    Ok(MetalCorrectionIdentity {
        columns,
        s_in_sha256: digest.finalize().into(),
    })
}

fn dispatch_width(pipeline: &ComputePipelineState, simdgroups: usize) -> Result<usize> {
    let execution_width = pipeline.thread_execution_width() as usize;
    let maximum = pipeline.max_total_threads_per_threadgroup() as usize;
    if execution_width == 0 || maximum < execution_width {
        return Err(EngineError::InvalidState(
            "Metal pipeline reports an invalid execution width".into(),
        ));
    }
    if !matches!(simdgroups, 1 | 2 | 4 | 8) {
        return Err(EngineError::Shape(format!(
            "Metal simdgroup count must be one of 1,2,4,8, got {simdgroups}"
        )));
    }
    let width = execution_width
        .checked_mul(simdgroups)
        .ok_or_else(|| EngineError::Shape("Metal thread width overflows".into()))?;
    if width > maximum || width > MAX_THREADS_PER_GROUP {
        return Err(EngineError::InvalidState(format!(
            "Metal pipeline cannot dispatch {simdgroups} simdgroups"
        )));
    }
    if width == 0 || width.div_ceil(32) > MAX_SIMDGROUPS_PER_THREADGROUP {
        return Err(EngineError::InvalidState(
            "Metal pipeline exceeds the supported simdgroup count".into(),
        ));
    }
    Ok(width)
}

fn fp16_bytes_or_dummy<'a>(scales: Option<ScaleSlice<'a>>, dummy: &'a [u8; 2]) -> &'a [u8] {
    match scales {
        Some(ScaleSlice::F16Le(bytes)) => bytes,
        Some(ScaleSlice::F32(_)) => unreachable!("validated Metal scales are FP16"),
        None => dummy,
    }
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
}

fn buffer_with_data(device: &Device, bytes: &[u8]) -> metal_driver::Buffer {
    debug_assert!(!bytes.is_empty());
    device.new_buffer_with_data(
        bytes.as_ptr().cast::<c_void>(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn write_metal_paged_gqa_descriptors(prepared: &PreparedMetalPagedGqa) -> Result<()> {
    let capacity = prepared
        .page_to_q4_slot
        .len()
        .checked_mul(METAL_PAGED_KV_DESCRIPTOR_BYTES)
        .ok_or_else(|| EngineError::MemoryBudget("Metal KV descriptors overflow".into()))?;
    let mut descriptor_words =
        Vec::with_capacity(prepared.cache.pages.len() * (METAL_PAGED_KV_DESCRIPTOR_BYTES / 4));
    for (page_index, page) in prepared.cache.pages.iter().enumerate() {
        let (precision, slot) = match page.precision {
            KvPrecision::Q2 => (0_u32, page_index),
            KvPrecision::Q4 => (
                1_u32,
                prepared.page_to_q4_slot[page_index].ok_or_else(|| {
                    EngineError::InvalidState("Metal Q4 page has no arena slot".into())
                })?,
            ),
        };
        descriptor_words.extend_from_slice(&[
            precision,
            u32::try_from(slot)
                .map_err(|_| EngineError::Shape("Metal KV slot exceeds u32".into()))?,
            u32::try_from(page.tokens)
                .map_err(|_| EngineError::Shape("Metal KV page tokens exceed u32".into()))?,
            u32::try_from(page_index * prepared.page_tokens)
                .map_err(|_| EngineError::Shape("Metal KV token index exceeds u32".into()))?,
        ]);
    }
    zero_buffer(&prepared.descriptors_buffer, capacity);
    write_buffer_range(
        &prepared.descriptors_buffer,
        0,
        as_bytes(&descriptor_words),
        capacity,
    )
}

fn write_metal_paged_gqa_params(prepared: &PreparedMetalPagedGqa) -> Result<()> {
    let combined_values = prepared
        .key_value_heads
        .checked_mul(prepared.head_dim)
        .and_then(|values| values.checked_mul(2))
        .ok_or_else(|| EngineError::Shape("Metal combined KV width overflows".into()))?;
    let params = MetalPagedGqaParams {
        query_heads: usize_to_u32(prepared.query_heads, "Metal GQA query heads")?,
        key_value_heads: usize_to_u32(prepared.key_value_heads, "Metal GQA KV heads")?,
        head_dim: usize_to_u32(prepared.head_dim, "Metal GQA head dimension")?,
        tokens: usize_to_u32(prepared.cache.tokens(), "Metal GQA token count")?,
        page_tokens: usize_to_u32(prepared.page_tokens, "Metal GQA page tokens")?,
        page_count: usize_to_u32(prepared.cache.pages.len(), "Metal GQA page count")?,
        combined_values: usize_to_u32(combined_values, "Metal combined KV width")?,
        q2_token_bytes: usize_to_u32(prepared.q2_token_bytes, "Metal Q2 token bytes")?,
        q4_token_bytes: usize_to_u32(prepared.q4_token_bytes, "Metal Q4 token bytes")?,
        q2_page_bytes: usize_to_u32(prepared.q2_page_bytes, "Metal Q2 page bytes")?,
        q4_page_bytes: usize_to_u32(prepared.q4_page_bytes, "Metal Q4 page bytes")?,
        scale: 1.0 / (prepared.head_dim as f32).sqrt(),
    };
    write_buffer_range(
        &prepared.params_buffer,
        0,
        &params.encode(),
        MetalPagedGqaParams::BYTE_LEN,
    )
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| EngineError::Shape(format!("{label} exceeds u32")))
}

fn new_zeroed_buffer(device: &Device, bytes: usize) -> Result<Buffer> {
    if bytes == 0 {
        return Err(EngineError::Shape(
            "Metal cannot allocate a zero-length verifier buffer".into(),
        ));
    }
    let length = u64::try_from(bytes)
        .map_err(|_| EngineError::MemoryBudget("Metal buffer length exceeds u64".into()))?;
    let buffer = device.new_buffer(length, MTLResourceOptions::StorageModeShared);
    zero_buffer(&buffer, bytes);
    Ok(buffer)
}

fn write_buffer_range(buffer: &Buffer, offset: usize, bytes: &[u8], capacity: usize) -> Result<()> {
    let end = offset
        .checked_add(bytes.len())
        .ok_or_else(|| EngineError::MemoryBudget("Metal buffer write range overflows".into()))?;
    if end > capacity {
        return Err(EngineError::MemoryBudget(format!(
            "Metal buffer write ends at {end}, capacity is {capacity}"
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().cast::<u8>().add(offset),
            bytes.len(),
        );
    }
    Ok(())
}

fn zero_buffer(buffer: &Buffer, bytes: usize) {
    unsafe {
        std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::metal_graph::MetalProjectionPlan;
    use crate::backend::metal_schedule::{MetalDecodeOperation, MetalDecodeSchedule};
    use crate::backend::{Activation, Backend, RecoveredRowMatVec};
    use crate::format::{
        ArtifactBuilder, FileHeader, ModelManifest, PackedTensor, QuantSegment, TensorEntry,
        DEFAULT_ALIGNMENT, HEADER_BYTES,
    };
    use crate::loader::{ChecksumPolicy, ModelArtifact};
    use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
    use crate::Qwen38Config;
    use half::f16;
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::tempdir;

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn packed_weights(dtype: TensorDType, rows: usize, columns: usize) -> Vec<u8> {
        let mut packed = Vec::new();
        for block in 0..rows * columns / BLOCK_LEN {
            let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
                let position = block * BLOCK_LEN + index;
                (position as f32 * 0.031).sin() * 0.7 + (position as f32 * 0.013).cos() * 0.2
            });
            match dtype {
                TensorDType::Q2B64 => packed.extend_from_slice(
                    &Q2Block64::quantize(&values)
                        .expect("finite Q2 block")
                        .encode(),
                ),
                TensorDType::Q4B64 => packed.extend_from_slice(
                    &Q4Block64::quantize(&values)
                        .expect("finite Q4 block")
                        .encode(),
                ),
                _ => unreachable!(),
            }
        }
        packed
    }

    fn repeated_packed_weights(dtype: TensorDType, rows: usize, columns: usize) -> Vec<u8> {
        let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
            (index as f32 * 0.031).sin() * 0.7 + (index as f32 * 0.013).cos() * 0.2
        });
        let block = match dtype {
            TensorDType::Q2B64 => Q2Block64::quantize(&values)
                .expect("finite repeated Q2 block")
                .encode()
                .to_vec(),
            TensorDType::Q4B64 => Q4Block64::quantize(&values)
                .expect("finite repeated Q4 block")
                .encode()
                .to_vec(),
            _ => unreachable!(),
        };
        block.repeat(rows * columns / BLOCK_LEN)
    }

    fn repeated_recovered_tensors(
        name: &str,
        dtype: TensorDType,
        rows: usize,
        columns: usize,
        s_in: &[u8],
        s_out: f32,
    ) -> Vec<PackedTensor> {
        vec![
            PackedTensor {
                name: name.into(),
                dtype,
                shape: vec![rows as u64, columns as u64],
                bytes: repeated_packed_weights(dtype, rows, columns),
            },
            PackedTensor {
                name: format!("{name}.s_in"),
                dtype: TensorDType::F16,
                shape: vec![columns as u64],
                bytes: s_in.to_vec(),
            },
            PackedTensor {
                name: format!("{name}.s_out"),
                dtype: TensorDType::F16,
                shape: vec![rows as u64],
                bytes: f16_bytes(&vec![s_out; rows]),
            },
        ]
    }

    fn aligned(value: usize) -> usize {
        value.div_ceil(DEFAULT_ALIGNMENT as usize) * DEFAULT_ALIGNMENT as usize
    }

    fn write_mixed_fixture(
        path: &std::path::Path,
        rows_q2: usize,
        rows_q4: usize,
        columns: usize,
    ) -> Vec<QuantSegment> {
        write_named_mixed_fixture(path, "matrix.weight", rows_q2, rows_q4, columns)
    }

    fn write_named_mixed_fixture(
        path: &std::path::Path,
        name: &str,
        rows_q2: usize,
        rows_q4: usize,
        columns: usize,
    ) -> Vec<QuantSegment> {
        let q2 = packed_weights(TensorDType::Q2B64, rows_q2, columns);
        let q4 = packed_weights(TensorDType::Q4B64, rows_q4, columns);
        let mut weights = q2.clone();
        weights.extend_from_slice(&q4);
        let s_in = f16_bytes(&vec![1.125; columns]);
        let s_out = f16_bytes(&vec![0.875; rows_q2 + rows_q4]);
        let s_in_offset = aligned(weights.len());
        let s_out_offset = aligned(s_in_offset + s_in.len());
        let segments = vec![
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: rows_q2 as u64,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2.len() as u64,
            },
            QuantSegment {
                group_index: 1,
                row_start: rows_q2 as u64,
                row_end: (rows_q2 + rows_q4) as u64,
                dtype: TensorDType::Q4B64,
                offset: q2.len() as u64,
                length: q4.len() as u64,
            },
        ];
        let tensors = vec![
            TensorEntry {
                name: name.into(),
                dtype: TensorDType::MixedQ2Q4B64,
                shape: vec![(rows_q2 + rows_q4) as u64, columns as u64],
                offset: 0,
                length: weights.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&weights)),
                segments: segments.clone(),
            },
            TensorEntry {
                name: format!("{name}.s_in"),
                dtype: TensorDType::F16,
                shape: vec![columns as u64],
                offset: s_in_offset as u64,
                length: s_in.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&s_in)),
                segments: Vec::new(),
            },
            TensorEntry {
                name: format!("{name}.s_out"),
                dtype: TensorDType::F16,
                shape: vec![(rows_q2 + rows_q4) as u64],
                offset: s_out_offset as u64,
                length: s_out.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&s_out)),
                segments: Vec::new(),
            },
        ];
        let manifest = ModelManifest {
            format: "ctox.q2q4.v2".into(),
            model: "test/qwen38".into(),
            revision: "0123456789abcdef".into(),
            alignment: DEFAULT_ALIGNMENT,
            target: "canonical-b64".into(),
            recovery: None,
            tensors,
        };
        let data_bytes = s_out_offset + s_out.len();
        manifest
            .validate(data_bytes as u64)
            .expect("mixed manifest");
        let manifest_bytes = serde_json::to_vec(&manifest).expect("serialize mixed manifest");
        let file_data_offset = aligned(HEADER_BYTES + manifest_bytes.len());
        let header = FileHeader {
            version: 2,
            manifest_len: manifest_bytes.len() as u64,
            data_offset: file_data_offset as u64,
            tensor_count: manifest.tensors.len() as u32,
            alignment: DEFAULT_ALIGNMENT,
        };
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create mixed artifact");
        file.write_all(&header.encode()).expect("write header");
        file.write_all(&manifest_bytes).expect("write manifest");
        file.write_all(&vec![
            0;
            file_data_offset - HEADER_BYTES - manifest_bytes.len()
        ])
        .expect("align data");
        file.write_all(&weights).expect("write mixed weights");
        file.write_all(&vec![0; s_in_offset - weights.len()])
            .expect("align s_in");
        file.write_all(&s_in).expect("write s_in");
        file.write_all(&vec![0; s_out_offset - s_in_offset - s_in.len()])
            .expect("align s_out");
        file.write_all(&s_out).expect("write s_out");
        file.sync_all().expect("sync mixed artifact");
        segments
    }

    #[test]
    fn decode_workspace_materializes_one_shared_buffer_with_exact_views() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let mut workspace = runtime
            .prepare_decode_workspace(&plan)
            .expect("allocate one decode arena");
        assert_eq!(workspace.total_bytes(), 1_173_760);

        let hidden = workspace
            .binding(MetalBufferSlot::HiddenA)
            .expect("hidden binding");
        let logits = workspace
            .binding(MetalBufferSlot::TargetLogits)
            .expect("target-logit binding");
        let (hidden_buffer, hidden_offset) = workspace
            .buffer_and_offset(MetalBufferSlot::HiddenA)
            .expect("hidden view");
        let (logit_buffer, logit_offset) = workspace
            .buffer_and_offset(MetalBufferSlot::TargetLogits)
            .expect("target-logit view");
        assert!(std::ptr::eq(hidden_buffer, logit_buffer));
        assert_eq!(hidden_offset, hidden.offset as u64);
        assert_eq!(logit_offset, logits.offset as u64);

        let values: Vec<f32> = (0..hidden.values)
            .map(|index| index as f32 * 0.125 - 3.0)
            .collect();
        workspace
            .write_f32(MetalBufferSlot::HiddenA, &values)
            .expect("write exact hidden view");
        assert_eq!(
            workspace
                .read_f32(MetalBufferSlot::HiddenA)
                .expect("read exact hidden view"),
            values
        );
        assert!(workspace
            .write_f32(MetalBufferSlot::HiddenA, &[0.0; 3])
            .is_err());

        drop(workspace);
        assert_eq!(
            runtime
                .prepare_decode_workspace(&plan)
                .expect("reallocate arena after drop")
                .total_bytes(),
            plan.total_bytes()
        );
    }

    #[test]
    fn decode_program_binds_all_645_steps_to_one_real_metal_buffer() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projections = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate one decode arena");
        let program = workspace
            .bind_decode_program(&bindings)
            .expect("bind real buffer views for every decode step");

        assert_eq!(program.steps().len(), 645);
        assert_eq!(program.steps().len(), bindings.steps().len());
        for layer in [0, 1, 2, 4, 62] {
            let layer_steps = program
                .linear_attention_layer_steps(layer)
                .expect("bind reusable linear-attention layer slice");
            assert_eq!(layer_steps.len(), 10);
            assert_eq!(layer_steps[0].step().schedule_index, 2 + layer * 10);
            assert_eq!(layer_steps[9].step().schedule_index, 11 + layer * 10);
        }
        assert!(program.linear_attention_layer_steps(3).is_err());
        assert!(program.linear_attention_layer_steps(64).is_err());
        for layer in [3, 7, 11, 15, 63] {
            let layer_steps = program
                .full_attention_layer_steps(layer)
                .expect("bind reusable full-attention layer slice");
            assert_eq!(layer_steps.len(), 10);
            assert_eq!(layer_steps[0].step().schedule_index, 2 + layer * 10);
            assert_eq!(layer_steps[9].step().schedule_index, 11 + layer * 10);
        }
        assert!(program.full_attention_layer_steps(0).is_err());
        assert!(program.full_attention_layer_steps(64).is_err());
        let expected_views = schedule
            .steps
            .iter()
            .map(|step| step.reads.len() + step.writes.len())
            .sum::<usize>();
        assert_eq!(
            program
                .steps()
                .iter()
                .map(|step| step.reads().len() + step.writes().len())
                .sum::<usize>(),
            expected_views
        );

        let (shared_buffer, _) = workspace
            .buffer_and_offset(MetalBufferSlot::HiddenA)
            .expect("shared arena buffer");
        for (index, prepared) in program.steps().iter().enumerate() {
            let bound = &bindings.steps()[index];
            assert_eq!(prepared.step(), bound);
            assert_eq!(bound.schedule_index, index);
            assert_eq!(
                prepared
                    .reads()
                    .iter()
                    .map(PreparedMetalDecodeBufferView::slot)
                    .collect::<Vec<_>>(),
                bound.reads
            );
            assert_eq!(
                prepared
                    .writes()
                    .iter()
                    .map(PreparedMetalDecodeBufferView::slot)
                    .collect::<Vec<_>>(),
                bound.writes
            );
            for view in prepared.reads().iter().chain(prepared.writes()) {
                let expected = workspace
                    .binding(view.slot())
                    .expect("scheduled workspace slot");
                assert!(std::ptr::eq(view.buffer(), shared_buffer));
                assert_eq!(view.values(), expected.values);
                assert_eq!(view.bytes(), expected.bytes);
                assert_eq!(view.offset(), expected.offset as u64);
            }
        }
        let final_step = program.steps().last().expect("final token barrier");
        assert_eq!(
            final_step.step().operation,
            MetalDecodeOperation::TokenCommandBufferCommit
        );
        assert_eq!(
            final_step
                .reads()
                .iter()
                .map(PreparedMetalDecodeBufferView::slot)
                .collect::<Vec<_>>(),
            vec![MetalBufferSlot::TargetLogits, MetalBufferSlot::MtpDraft]
        );
        assert!(final_step.writes().is_empty());
    }

    #[test]
    fn decode_workspace_checkpoint_restores_on_device_and_is_single_use() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let mut workspace = runtime
            .prepare_decode_workspace(&plan)
            .expect("allocate one decode arena");
        let hidden = workspace
            .binding(MetalBufferSlot::HiddenA)
            .expect("hidden binding");
        let original: Vec<f32> = (0..hidden.values)
            .map(|index| index as f32 * 0.0625 - 11.0)
            .collect();
        let changed = vec![7.25_f32; hidden.values];
        workspace
            .write_f32(MetalBufferSlot::HiddenA, &original)
            .expect("write committed hidden state");

        let mut checkpoint = runtime
            .prepare_f32_checkpoint(hidden.values)
            .expect("allocate bounded f32 checkpoint");
        assert_eq!(checkpoint.values(), hidden.values);
        assert_eq!(checkpoint.resident_bytes(), hidden.bytes);
        assert!(!checkpoint.is_active());
        runtime
            .snapshot_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::HiddenA)
            .expect("device snapshot");
        assert!(checkpoint.is_active());
        assert!(runtime
            .snapshot_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::HiddenA)
            .is_err());
        workspace
            .write_f32(MetalBufferSlot::HiddenA, &changed)
            .expect("mutate speculative hidden state");
        runtime
            .restore_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::HiddenA)
            .expect("device restore");
        assert!(!checkpoint.is_active());
        assert_eq!(
            workspace
                .read_f32(MetalBufferSlot::HiddenA)
                .expect("read restored hidden state"),
            original
        );
        assert!(runtime
            .restore_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::HiddenA)
            .is_err());

        runtime
            .snapshot_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::HiddenA)
            .expect("second device snapshot");
        workspace
            .write_f32(MetalBufferSlot::HiddenA, &changed)
            .expect("mutate committed branch");
        checkpoint.commit().expect("commit speculative branch");
        assert_eq!(
            workspace
                .read_f32(MetalBufferSlot::HiddenA)
                .expect("read committed hidden state"),
            changed
        );
        assert!(checkpoint.commit().is_err());
        assert!(runtime
            .snapshot_workspace_f32(&mut checkpoint, &workspace, MetalBufferSlot::TargetLogits)
            .is_err());
        checkpoint.clear();
        assert!(!checkpoint.is_active());
    }

    #[test]
    fn complete_speculative_transaction_restores_or_commits_all_state_classes() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let mut workspace = runtime
            .prepare_decode_workspace(&plan)
            .expect("allocate shared decode arena");
        let normalized = workspace
            .binding(MetalBufferSlot::Normalized)
            .expect("normalized target-hidden binding");
        let original_hidden: Vec<f32> = (0..normalized.values)
            .map(|index| index as f32 * 0.000_125 - 0.25)
            .collect();
        let speculative_hidden = vec![0.625_f32; normalized.values];
        workspace
            .write_f32(MetalBufferSlot::Normalized, &original_hidden)
            .expect("write committed target hidden");

        let attention_config = MetalPagedGqaConfig {
            query_heads: 4,
            key_value_heads: 2,
            head_dim: 64,
            maximum_tokens: 16,
            page_tokens: 4,
            sink_tokens: 4,
            recent_tokens: 4,
        };
        let mut attentions = (0..PreparedMetalSpeculativeTransaction::ATTENTION_STATES)
            .map(|_| {
                runtime
                    .prepare_paged_gqa_decode(attention_config)
                    .expect("prepare graph attention state")
            })
            .collect::<Vec<_>>();
        let recurrence_config = MetalGatedDeltaConfig {
            heads: 1,
            key_dim: 64,
            value_dim: 64,
            epsilon: 1.0e-6,
        };
        let mut recurrences = (0..PreparedMetalSpeculativeTransaction::LINEAR_STATES)
            .map(|_| {
                runtime
                    .prepare_gated_delta_f16(recurrence_config)
                    .expect("prepare graph recurrent state")
            })
            .collect::<Vec<_>>();

        let channels = 64;
        let kernel = 4;
        let directory = tempdir().expect("temporary graph-state artifact directory");
        let path = directory.path().join("graph-state-convolution.ctoxq");
        write_mixed_fixture(&path, 3, 5, channels * kernel);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open graph-state convolution fixture");
        let weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve graph-state convolution weight");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import graph-state convolution mapping");
        let mut convolutions = (0..PreparedMetalSpeculativeTransaction::LINEAR_STATES)
            .map(|_| {
                runtime
                    .prepare_mapped_causal_conv_f16(&mapping, weight, &[0.0; 64], channels, kernel)
                    .expect("prepare graph convolution state")
            })
            .collect::<Vec<_>>();
        drop(mapping);
        drop(artifact);

        let original_conv = convolutions[0].verifier_read_state();
        let original_recurrence = recurrences[0].verifier_read_state();
        let mut transaction = runtime
            .prepare_speculative_transaction(&config)
            .expect("prepare frozen graph transaction");
        assert!(!transaction.is_active());
        assert!(!transaction.is_poisoned());
        assert_eq!(
            transaction.target_hidden_checkpoint_bytes(),
            normalized.bytes
        );
        assert!(runtime
            .begin_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions[..16],
                &mut convolutions,
                &mut recurrences,
            )
            .is_err());
        assert!(!transaction.is_active());

        runtime
            .begin_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .expect("begin complete speculative graph transaction");
        assert!(transaction.is_active());
        assert!(runtime
            .begin_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .is_err());
        workspace
            .write_f32(MetalBufferSlot::Normalized, &speculative_hidden)
            .expect("write speculative target hidden");
        let query = vec![0.13_f32; attention_config.query_heads * attention_config.head_dim];
        let key = vec![0.19_f32; attention_config.key_value_heads * attention_config.head_dim];
        let value = vec![-0.17_f32; attention_config.key_value_heads * attention_config.head_dim];
        runtime
            .append_and_dispatch_paged_gqa(&mut attentions[0], &query, &key, &value)
            .expect("advance speculative attention state");
        let conv_input = vec![0.23_f32; channels];
        convolutions[0]
            .write_input(&conv_input)
            .expect("write speculative convolution input");
        runtime
            .dispatch_mapped_causal_conv_f16(&mut convolutions[0])
            .expect("advance speculative convolution state");
        let recurrent_qk = vec![0.11_f32; recurrence_config.key_dim];
        let recurrent_value = vec![-0.09_f32; recurrence_config.value_dim];
        recurrences[0]
            .write_step(
                &recurrent_qk,
                &recurrent_qk,
                &recurrent_value,
                &[-0.02],
                &[0.55],
            )
            .expect("write speculative recurrent inputs");
        runtime
            .dispatch_gated_delta_f16(&mut recurrences[0])
            .expect("advance speculative recurrent state");
        assert_ne!(convolutions[0].verifier_read_state(), original_conv);
        assert_ne!(recurrences[0].verifier_read_state(), original_recurrence);
        assert_eq!(attentions[0].tokens(), 1);

        runtime
            .restore_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .expect("restore complete speculative graph transaction");
        assert!(!transaction.is_active());
        assert!(!transaction.is_poisoned());
        assert_eq!(
            workspace
                .read_f32(MetalBufferSlot::Normalized)
                .expect("read restored target hidden"),
            original_hidden
        );
        assert_eq!(attentions[0].tokens(), 0);
        assert_eq!(convolutions[0].verifier_read_state(), original_conv);
        assert_eq!(recurrences[0].verifier_read_state(), original_recurrence);
        assert!(attentions
            .iter()
            .all(|state| state.speculative_checkpoint.is_none()));
        assert!(convolutions.iter().all(|state| !state.checkpoint_valid));
        assert!(recurrences.iter().all(|state| !state.checkpoint_valid));

        runtime
            .begin_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .expect("begin committed graph branch");
        workspace
            .write_f32(MetalBufferSlot::Normalized, &speculative_hidden)
            .expect("rewrite committed target hidden");
        runtime
            .append_and_dispatch_paged_gqa(&mut attentions[0], &query, &key, &value)
            .expect("advance committed attention branch");
        convolutions[0]
            .write_input(&conv_input)
            .expect("rewrite committed convolution input");
        runtime
            .dispatch_mapped_causal_conv_f16(&mut convolutions[0])
            .expect("advance committed convolution branch");
        recurrences[0]
            .write_step(
                &recurrent_qk,
                &recurrent_qk,
                &recurrent_value,
                &[-0.02],
                &[0.55],
            )
            .expect("rewrite committed recurrent inputs");
        runtime
            .dispatch_gated_delta_f16(&mut recurrences[0])
            .expect("advance committed recurrent branch");
        runtime
            .commit_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .expect("commit complete speculative graph transaction");
        assert!(!transaction.is_active());
        assert_eq!(
            workspace
                .read_f32(MetalBufferSlot::Normalized)
                .expect("read committed target hidden"),
            speculative_hidden
        );
        assert_eq!(attentions[0].tokens(), 1);
        assert_ne!(convolutions[0].verifier_read_state(), original_conv);
        assert_ne!(recurrences[0].verifier_read_state(), original_recurrence);
        assert!(runtime
            .commit_speculative_transaction(
                &mut transaction,
                &workspace,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
            )
            .is_err());

        let projections = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config)
            .expect("complete Metal binding plan");
        let program = workspace
            .bind_decode_program(&bindings)
            .expect("bind guarded decode program");
        assert!(runtime
            .begin_decode_attempt(
                &program,
                &mut transaction,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
                2,
                1,
                16,
            )
            .is_err());
        assert!(!transaction.is_active());

        {
            let mut attempt = runtime
                .begin_decode_attempt(
                    &program,
                    &mut transaction,
                    &mut attentions,
                    &mut convolutions,
                    &mut recurrences,
                    1,
                    1,
                    16,
                )
                .expect("begin guarded partial token");
            let first = attempt.next_step().expect("first guarded decode step");
            let encoded = (
                first.step().schedule_index,
                first.step().layer,
                first.step().operation,
            );
            attempt
                .advance_encoded(encoded.0, encoded.1, encoded.2)
                .expect("advance one guarded decode step");
        }
        assert!(!transaction.is_active());
        assert!(attentions
            .iter()
            .all(|state| state.speculative_checkpoint.is_none()));
        assert!(convolutions.iter().all(|state| !state.checkpoint_valid));
        assert!(recurrences.iter().all(|state| !state.checkpoint_valid));

        let early_commit = runtime
            .begin_decode_attempt(
                &program,
                &mut transaction,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
                1,
                1,
                16,
            )
            .expect("begin guarded early-commit token")
            .commit_after_completion(644);
        assert!(early_commit.is_err());
        assert!(!transaction.is_active());

        let mut attempt = runtime
            .begin_decode_attempt(
                &program,
                &mut transaction,
                &mut attentions,
                &mut convolutions,
                &mut recurrences,
                1,
                1,
                16,
            )
            .expect("begin complete guarded token");
        loop {
            let Some(next) = attempt.next_step() else {
                panic!("guarded token omitted its final barrier");
            };
            let encoded = (
                next.step().schedule_index,
                next.step().layer,
                next.step().operation,
            );
            if encoded.2 == MetalDecodeOperation::TokenCommandBufferCommit {
                break;
            }
            attempt
                .advance_encoded(encoded.0, encoded.1, encoded.2)
                .expect("advance exact guarded decode step");
        }
        assert_eq!(
            attempt
                .commit_after_completion(644)
                .expect("commit guarded token"),
            2
        );
        assert!(!transaction.is_active());
    }

    #[test]
    fn device_argmax_matches_full_vocab_oracle_reuses_buffers_and_rejects_nonfinite() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let values = crate::tokenizer::TOKENIZER_VOCAB_SIZE;
        let mut logits = vec![-7.0_f32; values];
        logits[17] = 9.5;
        logits[values - 1] = 9.5;
        let mut prepared = runtime
            .prepare_argmax_f32(&logits)
            .expect("prepare full-vocabulary Metal argmax");
        assert_eq!(prepared.values(), values);
        assert_eq!(prepared.groups(), 32);
        assert_eq!(
            prepared.resident_bytes(),
            values * std::mem::size_of::<f32>()
                + 32 * 4 * std::mem::size_of::<u32>()
                + 2 * std::mem::size_of::<u32>()
                + MetalArgMaxParams::BYTE_LEN
        );
        assert_eq!(
            runtime
                .dispatch_argmax_f32(&prepared)
                .expect("dispatch tied full-vocabulary Metal argmax"),
            (values - 1) as u32
        );

        logits.fill(-3.0);
        logits[123_456] = 4.0;
        prepared.write_input(&logits).expect("rewrite argmax input");
        assert_eq!(
            runtime
                .dispatch_argmax_f32_repeated(&prepared, 4)
                .expect("dispatch reused Metal argmax"),
            123_456
        );
        assert!(runtime.dispatch_argmax_f32_repeated(&prepared, 0).is_err());

        logits[88] = f32::NAN;
        prepared
            .write_input(&logits)
            .expect("write non-finite verifier input");
        assert!(matches!(
            runtime.dispatch_argmax_f32(&prepared),
            Err(EngineError::InvalidArtifact(_))
        ));
        assert!(prepared.write_input(&[0.0; 3]).is_err());
        assert!(runtime.prepare_argmax_f32_with_groups(&logits, 3).is_err());
    }

    #[test]
    fn q2_and_q4_device_results_match_scalar_oracle() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows = 11;
        let columns = 192;
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.021).cos() * 0.8)
            .collect();
        let s_in_values: Vec<f32> = (0..columns)
            .map(|index| 0.85 + (index % 17) as f32 * 0.015)
            .collect();
        let s_out_values: Vec<f32> = (0..rows).map(|index| 0.9 + index as f32 * 0.025).collect();
        let s_in = f16_bytes(&s_in_values);
        let s_out = f16_bytes(&s_out_values);
        let bias: Vec<f32> = (0..rows).map(|index| index as f32 * 0.013 - 0.04).collect();

        for dtype in [TensorDType::Q2B64, TensorDType::Q4B64] {
            let weights = packed_weights(dtype, rows, columns);
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
            let expected = cpu.fused_matvec(&operation).expect("scalar oracle");
            for simdgroups in [1, 2, 4, 8] {
                let prepared = runtime
                    .prepare_fused_matvec_with_simdgroups(&operation, simdgroups)
                    .expect("prepare Metal geometry");
                let actual = runtime
                    .dispatch_prepared(&prepared)
                    .expect("Metal candidate dispatch");
                for (row, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                    let absolute = (expected - actual).abs();
                    let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
                    assert!(
                        absolute <= tolerance,
                        "{dtype:?} simdgroups {simdgroups} row {row}: expected {expected}, got {actual}, absolute error {absolute}, tolerance {tolerance}"
                    );
                }
            }
        }
    }

    #[test]
    fn shared_fanout_matches_oracles_reuses_residency_and_rejects_other_s_in() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let columns = 2 * BLOCK_LEN;
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.017).cos())
            .collect();
        let s_in = f16_bytes(&vec![1.125; columns]);
        let different_s_in = f16_bytes(&vec![0.875; columns]);
        let q2_rows = 5;
        let q4_rows = 7;
        let q2_weights = packed_weights(TensorDType::Q2B64, q2_rows, columns);
        let q4_weights = packed_weights(TensorDType::Q4B64, q4_rows, columns);
        let q2_s_out = f16_bytes(&vec![0.9; q2_rows]);
        let q4_s_out = f16_bytes(&vec![1.1; q4_rows]);
        let q2 = FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &q2_weights,
            segments: &[],
            rows: q2_rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F16Le(&s_in)),
            s_out: Some(ScaleSlice::F16Le(&q2_s_out)),
            bias: None,
            activation: Activation::Identity,
        };
        let q4 = FusedMatVec {
            dtype: TensorDType::Q4B64,
            weights: &q4_weights,
            rows: q4_rows,
            s_out: Some(ScaleSlice::F16Le(&q4_s_out)),
            ..q2
        };
        let expected = [
            cpu.fused_matvec(&q2).expect("Q2 oracle"),
            cpu.fused_matvec(&q4).expect("Q4 oracle"),
        ];
        let isolated_bytes = runtime
            .prepare_fused_matvec(&q2)
            .expect("isolated Q2")
            .resident_bytes()
            + runtime
                .prepare_fused_matvec(&q4)
                .expect("isolated Q4")
                .resident_bytes();
        let activation = runtime
            .prepare_shared_activation(&q2)
            .expect("shared activation");
        let q2_projection = runtime
            .prepare_shared_projection(&q2)
            .expect("shared Q2 projection");
        let q4_projection = runtime
            .prepare_shared_projection(&q4)
            .expect("shared Q4 projection");
        let shared_bytes = activation.resident_bytes()
            + q2_projection.resident_bytes()
            + q4_projection.resident_bytes();
        assert_eq!(activation.columns(), columns);
        assert_eq!(q2_projection.dtype(), TensorDType::Q2B64);
        assert_eq!(q4_projection.rows(), q4_rows);
        assert_eq!(q4_projection.columns(), columns);
        assert_eq!(
            isolated_bytes - shared_bytes,
            size_of_val(input.as_slice()) + s_in.len()
        );
        let actual = runtime
            .dispatch_shared_fanout(&activation, &[&q2_projection, &q4_projection])
            .expect("shared Metal fan-out");
        for (expected, actual) in expected.iter().zip(actual) {
            for (expected, actual) in expected.iter().zip(actual) {
                let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
                assert!((expected - actual).abs() <= tolerance);
            }
        }

        let mismatched = FusedMatVec {
            s_in: Some(ScaleSlice::F16Le(&different_s_in)),
            ..q2
        };
        let mismatched_projection = runtime
            .prepare_shared_projection(&mismatched)
            .expect("mismatched projection can be prepared independently");
        assert!(runtime
            .dispatch_shared_fanout(&activation, &[&mismatched_projection])
            .is_err());
        assert!(runtime.dispatch_shared_fanout(&activation, &[]).is_err());

        activation
            .write_input(&vec![0.0; columns])
            .expect("shared input update");
        let zero = runtime
            .dispatch_shared_fanout(&activation, &[&q2_projection, &q4_projection])
            .expect("zero shared fan-out");
        assert!(zero
            .iter()
            .flatten()
            .all(|value| value.abs() <= f32::EPSILON));
    }

    #[test]
    fn prepared_projection_reuses_resident_buffers_and_updates_only_input() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let rows = 3;
        let columns = BLOCK_LEN;
        let weights = packed_weights(TensorDType::Q4B64, rows, columns);
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.05).sin())
            .collect();
        let operation = FusedMatVec {
            dtype: TensorDType::Q4B64,
            weights: &weights,
            segments: &[],
            rows,
            columns,
            input: &input,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        let prepared = runtime
            .prepare_fused_matvec(&operation)
            .expect("prepare resident projection");
        assert!(runtime
            .prepare_fused_matvec_with_simdgroups(&operation, 3)
            .is_err());
        assert!(runtime.dispatch_prepared_repeated(&prepared, 0).is_err());
        assert_eq!(prepared.dtype(), TensorDType::Q4B64);
        assert_eq!(prepared.rows(), rows);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(
            prepared.resident_bytes(),
            weights.len()
                + size_of_val(input.as_slice())
                + 2
                + 2
                + size_of_val(&[0.0_f32])
                + rows * std::mem::size_of::<f32>()
                + 32
        );
        let first = runtime
            .dispatch_prepared(&prepared)
            .expect("first resident dispatch");
        assert!(first.iter().any(|value| value.abs() > 1.0e-4));

        let zero_input = vec![0.0; columns];
        prepared
            .write_input(&zero_input)
            .expect("update resident input");
        let second = runtime
            .dispatch_prepared_repeated(&prepared, 3)
            .expect("second resident dispatch");
        assert!(second.iter().all(|value| value.abs() <= f32::EPSILON));
        assert!(prepared.write_input(&[0.0; 3]).is_err());
    }

    #[test]
    fn mmap_artifact_is_shared_without_copy_and_outlives_original_owner() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows = 7;
        let columns = 3 * BLOCK_LEN;
        let weights = packed_weights(TensorDType::Q4B64, rows, columns);
        let s_in = f16_bytes(&vec![1.125; columns]);
        let s_out = f16_bytes(&vec![0.875; rows]);
        let directory = tempdir().expect("temporary artifact directory");
        let path = directory.path().join("mapped.ctoxq");
        ArtifactBuilder {
            model: "test/qwen38".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors: vec![
                PackedTensor {
                    name: "matrix.weight".into(),
                    dtype: TensorDType::Q4B64,
                    shape: vec![rows as u64, columns as u64],
                    bytes: weights,
                },
                PackedTensor {
                    name: "matrix.weight.s_in".into(),
                    dtype: TensorDType::F16,
                    shape: vec![columns as u64],
                    bytes: s_in,
                },
                PackedTensor {
                    name: "matrix.weight.s_out".into(),
                    dtype: TensorDType::F16,
                    shape: vec![rows as u64],
                    bytes: s_out,
                },
            ],
        }
        .write_new(&path)
        .expect("write mmap fixture");
        let artifact =
            ModelArtifact::open(&path, ChecksumPolicy::AllTensors).expect("open mmap fixture");
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.023).sin())
            .collect();
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve recovered fixture matrix");
        let operation = matrix
            .operation(&input, Activation::Silu)
            .expect("construct mmap operation");
        let expected = cpu.fused_matvec(&operation).expect("scalar mmap oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import mmap without copy");
        let prepared = runtime
            .prepare_mapped_fused_matvec(&mapping, &operation)
            .expect("prepare mmap projection");
        assert_eq!(prepared.dtype(), TensorDType::Q4B64);
        assert_eq!(prepared.rows(), rows);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.mapped_file_bytes(), artifact.file_bytes());
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            size_of_val(input.as_slice())
                + size_of_val(&[0.0_f32])
                + rows * std::mem::size_of::<f32>()
                + 32
        );
        drop(mapping);
        drop(artifact);
        let actual = runtime
            .dispatch_mapped(&prepared)
            .expect("dispatch after original artifact owner drops");
        for (expected, actual) in expected.iter().zip(actual) {
            let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
            assert!((expected - actual).abs() <= tolerance);
        }

        prepared
            .write_input(&vec![0.0; columns])
            .expect("update mmap projection input");
        let zero = runtime
            .dispatch_mapped(&prepared)
            .expect("dispatch mmap projection again");
        assert!(zero.iter().all(|value| value.abs() <= f32::EPSILON));

        let copied_weights = packed_weights(TensorDType::Q4B64, rows, columns);
        let copied_s_in = f16_bytes(&vec![1.125; columns]);
        let copied_s_out = f16_bytes(&vec![0.875; rows]);
        let copied_operation = FusedMatVec {
            dtype: TensorDType::Q4B64,
            weights: &copied_weights,
            segments: &[],
            rows,
            columns,
            input: &input,
            s_in: Some(ScaleSlice::F16Le(&copied_s_in)),
            s_out: Some(ScaleSlice::F16Le(&copied_s_out)),
            bias: None,
            activation: Activation::Identity,
        };
        assert!(runtime
            .prepare_mapped_fused_matvec(&prepared.mapping, &copied_operation)
            .is_err());
    }

    #[test]
    fn mixed_mmap_segments_dispatch_without_repacking_or_duplicate_weights() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let rows = rows_q2 + rows_q4;
        let columns = 3 * BLOCK_LEN;
        let directory = tempdir().expect("temporary mixed artifact directory");
        let path = directory.path().join("mixed.ctoxq");
        let expected_segments = write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open mixed mmap fixture");
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.019).cos())
            .collect();
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve mixed recovered matrix");
        assert_eq!(matrix.matrix.segments, expected_segments);
        let operation = matrix
            .operation(&input, Activation::Silu)
            .expect("construct mixed mmap operation");
        let expected = cpu.fused_matvec(&operation).expect("mixed scalar oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import mixed mmap without copy");
        let prepared = runtime
            .prepare_mapped_fused_matvec(&mapping, &operation)
            .expect("prepare mixed mmap projection");
        assert_eq!(prepared.dtype(), TensorDType::MixedQ2Q4B64);
        assert_eq!(prepared.rows(), rows);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.dispatches.len(), 2);
        assert_eq!(prepared.dispatches[0].dtype, TensorDType::Q2B64);
        assert_eq!(prepared.dispatches[0].rows, rows_q2);
        assert_eq!(prepared.dispatches[1].dtype, TensorDType::Q4B64);
        assert_eq!(prepared.dispatches[1].rows, rows_q4);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            size_of_val(input.as_slice())
                + size_of_val(&[0.0_f32])
                + rows * std::mem::size_of::<f32>()
                + 2 * MetalFusedMatVecParams::BYTE_LEN
        );
        let actual = runtime
            .dispatch_mapped(&prepared)
            .expect("dispatch mixed mmap row groups");
        for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "mixed row {row}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn mixed_gathered_lm_head_batches_canonical_rows_from_one_mapping() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let columns = 3 * BLOCK_LEN;
        let directory = tempdir().expect("temporary gathered artifact directory");
        let path = directory.path().join("gathered.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open gathered mmap fixture");
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.027).sin())
            .collect();
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve gathered recovered matrix");
        let operation = matrix
            .operation(&input, Activation::Identity)
            .expect("construct gathered oracle operation");
        let full = cpu
            .fused_matvec(&operation)
            .expect("full mixed scalar oracle");
        let row_ids = [0_u32, 2, 3, 6, 7];
        let expected: Vec<f32> = row_ids.iter().map(|row| full[*row as usize]).collect();
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import gathered mmap without copy");
        let prepared = runtime
            .prepare_mapped_gathered_matvec(&mapping, matrix, &input, &row_ids)
            .expect("prepare gathered LM head");
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.requested_rows(), row_ids.len());
        assert_eq!(prepared.dispatches.len(), 2);
        assert_eq!(prepared.dispatches[0].dtype, TensorDType::Q2B64);
        assert_eq!(prepared.dispatches[0].requested_rows, 2);
        assert_eq!(prepared.dispatches[1].dtype, TensorDType::Q4B64);
        assert_eq!(prepared.dispatches[1].requested_rows, 3);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            size_of_val(input.as_slice())
                + std::mem::size_of::<f32>()
                + row_ids.len() * std::mem::size_of::<f32>()
                + row_ids.len() * std::mem::size_of::<u32>()
                + 2 * MetalFusedMatVecParams::BYTE_LEN
        );
        assert!(runtime
            .prepare_mapped_gathered_matvec(&mapping, matrix, &input, &[2, 1])
            .is_err());
        assert!(runtime
            .prepare_mapped_gathered_matvec(&mapping, matrix, &input, &[8])
            .is_err());
        drop(mapping);
        drop(artifact);
        let actual = runtime
            .dispatch_mapped_gathered(&prepared)
            .expect("dispatch gathered LM head after loader drop");
        for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "gathered row {row}: expected {expected}, got {actual}"
            );
        }
        prepared
            .write_input(&vec![0.0; columns])
            .expect("update gathered input");
        let zero = runtime
            .dispatch_mapped_gathered(&prepared)
            .expect("dispatch zero gathered input");
        assert!(zero.iter().all(|value| value.abs() <= f32::EPSILON));
    }

    #[test]
    fn mixed_embedding_rows_decode_from_one_mapping_without_model_copies() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let columns = 3 * BLOCK_LEN;
        let directory = tempdir().expect("temporary embedding artifact directory");
        let path = directory.path().join("embedding.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open embedding mmap fixture");
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve recovered embedding matrix");
        let q2_row = 1;
        let q4_row = rows_q2 + 2;
        let expected_q2 = cpu
            .recovered_row(&matrix.row_operation(q2_row).expect("resolve Q2 row"))
            .expect("decode Q2 embedding oracle");
        let expected_q4 = cpu
            .recovered_row(&matrix.row_operation(q4_row).expect("resolve Q4 row"))
            .expect("decode Q4 embedding oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import embedding mmap without copy");
        let prepared_q2 = runtime
            .prepare_mapped_recovered_row(&mapping, matrix, q2_row)
            .expect("prepare Q2 embedding row");
        let prepared_q4 = runtime
            .prepare_mapped_recovered_row(&mapping, matrix, q4_row)
            .expect("prepare Q4 embedding row");
        assert_eq!(prepared_q2.dtype(), TensorDType::Q2B64);
        assert_eq!(prepared_q4.dtype(), TensorDType::Q4B64);
        assert_eq!(prepared_q2.columns(), columns);
        assert_eq!(prepared_q4.columns(), columns);
        assert_eq!(prepared_q2.copied_model_bytes(), 0);
        assert_eq!(prepared_q4.copied_model_bytes(), 0);
        assert_eq!(
            prepared_q2.transient_bytes(),
            columns * std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
        );
        assert_eq!(prepared_q4.transient_bytes(), prepared_q2.transient_bytes());
        assert!(runtime
            .prepare_mapped_recovered_row(&mapping, matrix, rows_q2 + rows_q4)
            .is_err());
        assert!(runtime
            .dispatch_mapped_recovered_row_repeated(&prepared_q2, 0)
            .is_err());
        drop(mapping);
        drop(artifact);
        let actual_q2 = runtime
            .dispatch_mapped_recovered_row_repeated(&prepared_q2, 3)
            .expect("dispatch Q2 embedding after loader drop");
        let actual_q4 = runtime
            .dispatch_mapped_recovered_row(&prepared_q4)
            .expect("dispatch Q4 embedding after loader drop");
        for (dtype, expected, actual) in [
            (TensorDType::Q2B64, expected_q2, actual_q2),
            (TensorDType::Q4B64, expected_q4, actual_q4),
        ] {
            for (column, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 2.0e-5_f32.max(expected.abs() * 3.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{dtype:?} embedding column {column}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn complete_embedding_table_selects_q2_and_q4_rows_without_reprepare() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let columns = 3 * BLOCK_LEN;
        let directory = tempdir().expect("temporary resident embedding directory");
        let path = directory.path().join("resident-embedding.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open resident embedding fixture");
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve resident embedding matrix");
        let selected_rows = [0_usize, rows_q2 - 1, rows_q2, rows_q2 + rows_q4 - 1];
        let expected = selected_rows
            .iter()
            .map(|row| {
                cpu.recovered_row(&matrix.row_operation(*row).expect("resolve embedding row"))
                    .expect("decode resident embedding oracle")
            })
            .collect::<Vec<_>>();
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import resident embedding mapping");
        let prepared = runtime
            .prepare_mapped_embedding(&mapping, matrix)
            .expect("prepare complete resident embedding");
        assert_eq!(prepared.rows(), rows_q2 + rows_q4);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.segments.len(), 2);
        assert_eq!(prepared.segments[0].dtype, TensorDType::Q2B64);
        assert_eq!(prepared.segments[1].dtype, TensorDType::Q4B64);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            columns * std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
        );
        assert!(runtime
            .dispatch_mapped_embedding(&prepared, rows_q2 + rows_q4)
            .is_err());
        drop(mapping);
        drop(artifact);

        for (row, expected) in selected_rows.into_iter().zip(expected) {
            let actual = runtime
                .dispatch_mapped_embedding(&prepared, row)
                .expect("dispatch resident embedding after loader drop");
            for (column, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 2.0e-5_f32.max(expected.abs() * 3.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "resident embedding row {row} column {column}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn resident_embedding_norm_projection_chain_stays_on_device() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let columns = 3 * BLOCK_LEN;
        let epsilon = 1.0e-6;
        let directory = tempdir().expect("temporary embedding chain directory");
        let path = directory.path().join("embedding-chain.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open embedding chain fixture");
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve embedding chain matrix");
        let norm_weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve embedding chain norm weight");
        let norm_weight_f32 = norm_weight
            .to_f32_vec()
            .expect("widen embedding chain norm weight");
        let placeholder = vec![0.0_f32; columns];
        let projection_contract = matrix
            .operation(&placeholder, Activation::Identity)
            .expect("construct embedding chain projection contract");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import embedding chain mapping");
        let embedding = runtime
            .prepare_mapped_embedding(&mapping, matrix)
            .expect("prepare resident embedding chain table");
        let norm = runtime
            .prepare_mapped_rms_norm_1p(&mapping, norm_weight, &placeholder, 1, columns, epsilon)
            .expect("prepare embedding chain norm");
        let projection = runtime
            .prepare_mapped_fused_matvec_external_input(&mapping, &projection_contract)
            .expect("prepare embedding chain projection");
        let projection_b = runtime
            .prepare_mapped_fused_matvec_external_input(&mapping, &projection_contract)
            .expect("prepare second embedding fan-out projection");
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_fanout(&embedding, 0, &norm, &[])
            .is_err());
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_projection(
                &embedding,
                rows_q2 + rows_q4,
                &norm,
                &projection,
            )
            .is_err());
        let selected_rows = [1_usize, rows_q2 + 2];
        let expected = selected_rows
            .iter()
            .map(|token| {
                let hidden = cpu
                    .recovered_row(&matrix.row_operation(*token).expect("resolve chain row"))
                    .expect("decode chain embedding oracle");
                let normalized = crate::reference::rms_norm_1p_weight(
                    &hidden,
                    1,
                    columns,
                    &norm_weight_f32,
                    epsilon,
                )
                .expect("normalize chain embedding oracle");
                cpu.fused_matvec(
                    &matrix
                        .operation(&normalized, Activation::Identity)
                        .expect("construct chain projection oracle"),
                )
                .expect("execute chain projection oracle")
            })
            .collect::<Vec<_>>();
        drop(mapping);
        drop(artifact);

        for (token, expected) in selected_rows.into_iter().zip(expected) {
            let actual = runtime
                .dispatch_mapped_embedding_rms_norm_projection(
                    &embedding,
                    token,
                    &norm,
                    &projection,
                )
                .expect("dispatch resident embedding chain after loader drop");
            for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 3.0e-4_f32.max(expected.abs() * 5.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "embedding chain token {token} row {row}: expected {expected}, got {actual}"
                );
            }
            let fanout = runtime
                .dispatch_mapped_embedding_rms_norm_fanout(
                    &embedding,
                    token,
                    &norm,
                    &[&projection, &projection_b],
                )
                .expect("dispatch resident embedding fan-out after loader drop");
            assert_eq!(fanout.len(), 2);
            for (branch, actual) in fanout.into_iter().enumerate() {
                for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                    let tolerance = 3.0e-4_f32.max(expected.abs() * 5.0e-5);
                    assert!(
                        (expected - actual).abs() <= tolerance,
                        "embedding fan-out branch {branch} token {token} row {row}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn mapped_q2_q4_swiglu_down_matches_scalar_oracle_without_product_buffer() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows = 11;
        let columns = 3 * BLOCK_LEN;
        let s_in = f16_bytes(
            &(0..columns)
                .map(|index| 0.875 + 0.015625 * (index % 9) as f32)
                .collect::<Vec<_>>(),
        );
        let directory = tempdir().expect("temporary SwiGLU artifact directory");
        let path = directory.path().join("swiglu-down.ctoxq");
        let mut tensors = Vec::new();
        for (name, dtype, s_out) in [
            ("q2.down.weight", TensorDType::Q2B64, 0.9375),
            ("q4.down.weight", TensorDType::Q4B64, 1.0625),
        ] {
            tensors.extend(repeated_recovered_tensors(
                name, dtype, rows, columns, &s_in, s_out,
            ));
        }
        ArtifactBuilder {
            model: "test/qwen38-swiglu-down".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .expect("write SwiGLU artifact fixture");
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open SwiGLU artifact fixture");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map SwiGLU artifact fixture");
        let gate: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.019).sin() * 1.3 - 0.2)
            .collect();
        let up: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.023).cos() * 0.9 + 0.1)
            .collect();
        let product = crate::reference::swiglu(&gate, &up).expect("execute SwiGLU oracle");
        let validation = vec![0.0; columns];

        for name in ["q2.down.weight", "q4.down.weight"] {
            let matrix = artifact
                .recovered_matrix(name)
                .expect("resolve SwiGLU down matrix");
            let operation = matrix
                .operation(&validation, Activation::Identity)
                .expect("build SwiGLU down operation");
            let prepared = runtime
                .prepare_mapped_fused_matvec_graph_io(&mapping, &operation)
                .expect("prepare graph-I/O SwiGLU down projection");
            let expected = cpu
                .fused_matvec(
                    &matrix
                        .operation(&product, Activation::Identity)
                        .expect("build scalar SwiGLU down oracle"),
                )
                .expect("execute scalar SwiGLU down oracle");
            let actual = runtime
                .dispatch_mapped_swiglu_projection(&prepared, &gate, &up)
                .expect("dispatch mapped SwiGLU down verifier");
            for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 5.0e-4_f32.max(expected.abs() * 6.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{name} SwiGLU row {row}: expected {expected}, got {actual}"
                );
            }
            assert!(runtime
                .dispatch_mapped_swiglu_projection(&prepared, &gate[..columns - 1], &up)
                .is_err());
        }
    }

    #[test]
    fn graph_embedding_norm_linear_fanout_uses_only_shared_arena_views() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let config = Qwen38Config::default();
        let columns = config.hidden_size;
        let embedding_rows = 8;
        let epsilon = config.rms_norm_epsilon;
        let s_in_values = vec![0.9375_f32; columns];
        let s_in = f16_bytes(&s_in_values);
        let norm_values: Vec<f32> = (0..columns)
            .map(|index| 0.05 * (index % 11) as f32 - 0.2)
            .collect();
        let post_attention_norm_values: Vec<f32> = (0..columns)
            .map(|index| 0.025 * (index % 13) as f32 - 0.15)
            .collect();
        let next_layer_norm_values: Vec<f32> = (0..columns)
            .map(|index| 0.01875 * (index % 17) as f32 - 0.125)
            .collect();
        let directory = tempdir().expect("temporary shared-arena graph directory");
        let path = directory.path().join("shared-arena-graph.ctoxq");
        let mut tensors = repeated_recovered_tensors(
            "embedding.weight",
            TensorDType::Q2B64,
            embedding_rows,
            columns,
            &s_in,
            0.875,
        );
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.input_layernorm.weight".into(),
            dtype: TensorDType::F16,
            shape: vec![columns as u64],
            bytes: f16_bytes(&norm_values),
        });
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.post_attention_layernorm.weight".into(),
            dtype: TensorDType::F16,
            shape: vec![columns as u64],
            bytes: f16_bytes(&post_attention_norm_values),
        });
        tensors.push(PackedTensor {
            name: "model.language_model.layers.1.input_layernorm.weight".into(),
            dtype: TensorDType::F16,
            shape: vec![columns as u64],
            bytes: f16_bytes(&next_layer_norm_values),
        });
        let projection_specs = [
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                TensorDType::Q2B64,
                2 * config.linear_num_key_heads * config.linear_key_head_dim
                    + config.linear_num_value_heads * config.linear_value_head_dim,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z.weight",
                TensorDType::Q4B64,
                config.linear_num_value_heads * config.linear_value_head_dim,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                TensorDType::Q2B64,
                config.linear_num_value_heads,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                TensorDType::Q4B64,
                config.linear_num_value_heads,
            ),
        ];
        let convolution_channels = projection_specs[0].2;
        let convolution_kernel = config.linear_conv_kernel_dim;
        let convolution_weight_values: Vec<f32> = (0..convolution_channels * convolution_kernel)
            .map(|index| 0.08 + 0.01 * (index % convolution_kernel) as f32)
            .collect();
        let a_log_values: Vec<f32> = (0..config.linear_num_value_heads)
            .map(|index| -0.4 + index as f32 * 0.003)
            .collect();
        let dt_bias_values: Vec<f32> = (0..config.linear_num_value_heads)
            .map(|index| -0.2 + index as f32 * 0.002)
            .collect();
        let gated_norm_values: Vec<f32> = (0..config.linear_value_head_dim)
            .map(|index| 0.85 + index as f32 * 0.001)
            .collect();
        let linear_output_columns = config.linear_num_value_heads * config.linear_value_head_dim;
        let linear_output_s_in_values = vec![0.96875_f32; linear_output_columns];
        let linear_output_s_in = f16_bytes(&linear_output_s_in_values);
        for (name, dtype, rows) in projection_specs {
            tensors.extend(repeated_recovered_tensors(
                name, dtype, rows, columns, &s_in, 1.0625,
            ));
        }
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.linear_attn.conv1d.weight".into(),
            dtype: TensorDType::F16,
            shape: vec![convolution_channels as u64, 1, convolution_kernel as u64],
            bytes: f16_bytes(&convolution_weight_values),
        });
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.linear_attn.A_log".into(),
            dtype: TensorDType::F32,
            shape: vec![config.linear_num_value_heads as u64],
            bytes: f32_bytes(&a_log_values),
        });
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.linear_attn.dt_bias".into(),
            dtype: TensorDType::F32,
            shape: vec![config.linear_num_value_heads as u64],
            bytes: f32_bytes(&dt_bias_values),
        });
        tensors.push(PackedTensor {
            name: "model.language_model.layers.0.linear_attn.norm.weight".into(),
            dtype: TensorDType::F16,
            shape: vec![config.linear_value_head_dim as u64],
            bytes: f16_bytes(&gated_norm_values),
        });
        tensors.extend(repeated_recovered_tensors(
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            TensorDType::Q2B64,
            columns,
            linear_output_columns,
            &linear_output_s_in,
            1.03125,
        ));
        let ffn_specs = [
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                TensorDType::Q2B64,
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                TensorDType::Q4B64,
            ),
        ];
        for (name, dtype) in ffn_specs {
            tensors.extend(repeated_recovered_tensors(
                name,
                dtype,
                config.intermediate_size,
                columns,
                &s_in,
                0.984375,
            ));
        }
        let ffn_down_s_in_values = vec![0.953125_f32; config.intermediate_size];
        let ffn_down_s_in = f16_bytes(&ffn_down_s_in_values);
        tensors.extend(repeated_recovered_tensors(
            "model.language_model.layers.0.mlp.down_proj.weight",
            TensorDType::Q2B64,
            columns,
            config.intermediate_size,
            &ffn_down_s_in,
            1.015625,
        ));
        ArtifactBuilder {
            model: "test/qwen38-shared-arena".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .expect("write shared-arena graph fixture");
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open shared-arena graph fixture");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map shared-arena graph fixture");
        let mut reusable_layer = runtime
            .prepare_mapped_linear_attention_layer(&mapping, 0)
            .expect("prepare complete canonical layer-0 resources");
        assert_eq!(reusable_layer.layer(), 0);
        assert_eq!(reusable_layer.copied_model_bytes(), 0);
        assert!(reusable_layer.resident_state_bytes() > 0);
        assert!(runtime
            .prepare_mapped_linear_attention_layer(&mapping, 3)
            .is_err());
        assert!(runtime
            .prepare_all_mapped_linear_attention_layers(&mapping)
            .is_err());
        assert!(runtime
            .prepare_all_mapped_target_layers(
                &mapping,
                0,
                MetalPagedGqaConfig {
                    query_heads: config.num_attention_heads,
                    key_value_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    maximum_tokens: 8,
                    page_tokens: 2,
                    sink_tokens: 2,
                    recent_tokens: 2,
                },
            )
            .is_err());
        assert!(runtime
            .prepare_mapped_target_core(
                &mapping,
                0,
                MetalPagedGqaConfig {
                    query_heads: config.num_attention_heads,
                    key_value_heads: config.num_key_value_heads,
                    head_dim: config.head_dim,
                    maximum_tokens: 8,
                    page_tokens: 2,
                    sink_tokens: 2,
                    recent_tokens: 2,
                },
            )
            .is_err());
        let embedding_matrix = artifact
            .recovered_matrix("embedding.weight")
            .expect("resolve graph embedding");
        let norm_weight = artifact
            .float_tensor("model.language_model.layers.0.input_layernorm.weight")
            .expect("resolve graph RMSNorm");
        let validation_input = vec![0.0_f32; columns];
        let embedding = runtime
            .prepare_mapped_embedding_graph_output(&mapping, embedding_matrix)
            .expect("prepare graph embedding without output buffer");
        let norm = runtime
            .prepare_mapped_rms_norm_1p_graph_io(
                &mapping,
                norm_weight,
                &validation_input,
                1,
                columns,
                epsilon,
            )
            .expect("prepare graph RMSNorm without activation buffers");
        let residual_norm = runtime
            .prepare_mapped_rms_norm_1p_graph_io(
                &mapping,
                artifact
                    .float_tensor("model.language_model.layers.0.post_attention_layernorm.weight")
                    .expect("resolve graph post-attention RMSNorm"),
                &validation_input,
                1,
                columns,
                epsilon,
            )
            .expect("prepare graph residual RMSNorm without activation buffers");
        let post_ffn_residual_norm = runtime
            .prepare_mapped_rms_norm_1p_graph_io(
                &mapping,
                artifact
                    .float_tensor("model.language_model.layers.1.input_layernorm.weight")
                    .expect("resolve graph next-layer RMSNorm"),
                &validation_input,
                1,
                columns,
                epsilon,
            )
            .expect("prepare graph post-FFN residual RMSNorm without activation buffers");
        let projection_matrices = projection_specs.map(|(name, _, _)| {
            artifact
                .recovered_matrix(name)
                .expect("resolve graph projection")
        });
        let projection_contracts = projection_matrices.map(|matrix| {
            matrix
                .operation(&validation_input, Activation::Identity)
                .expect("build graph projection contract")
        });
        let prepared_projections = projection_contracts.map(|operation| {
            runtime
                .prepare_mapped_fused_matvec_graph_io(&mapping, &operation)
                .expect("prepare graph projection without activation buffers")
        });
        let convolution_weight = artifact
            .float_tensor("model.language_model.layers.0.linear_attn.conv1d.weight")
            .expect("resolve graph convolution weight");
        let convolution_weight_f32 = convolution_weight
            .to_f32_vec()
            .expect("widen graph convolution oracle weight");
        let mut convolution = runtime
            .prepare_mapped_causal_conv_f16_graph_io(
                &mapping,
                convolution_weight,
                &vec![0.0; convolution_channels],
                convolution_channels,
                convolution_kernel,
            )
            .expect("prepare graph convolution without activation buffers");
        let delta_prepare = runtime
            .prepare_mapped_gated_delta_prepare_graph_io(
                &mapping,
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.A_log")
                    .expect("resolve graph A_log"),
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.dt_bias")
                    .expect("resolve graph dt_bias"),
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
            )
            .expect("prepare graph GatedDelta inputs without activation buffers");
        let recurrence_config = MetalGatedDeltaConfig {
            heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            epsilon: config.rms_norm_epsilon,
        };
        let mut recurrence = runtime
            .prepare_gated_delta_f16_graph_io(recurrence_config, 0)
            .expect("prepare graph recurrence without activation buffers");
        let gated_norm = runtime
            .prepare_mapped_rms_norm_gated_graph_io(
                &mapping,
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.norm.weight")
                    .expect("resolve graph gated RMSNorm weight"),
                &vec![0.0; recurrence_config.heads * recurrence_config.value_dim],
                &vec![0.0; recurrence_config.heads * recurrence_config.value_dim],
                recurrence_config.heads,
                recurrence_config.value_dim,
                recurrence_config.epsilon,
            )
            .expect("prepare graph gated RMSNorm without activation buffers");
        let linear_output_matrix = artifact
            .recovered_matrix("model.language_model.layers.0.linear_attn.out_proj.weight")
            .expect("resolve graph linear output projection");
        let linear_output_validation = vec![0.0_f32; linear_output_columns];
        let linear_output_operation = linear_output_matrix
            .operation(&linear_output_validation, Activation::Identity)
            .expect("build graph linear output projection contract");
        let linear_output_projection = runtime
            .prepare_mapped_fused_matvec_graph_io(&mapping, &linear_output_operation)
            .expect("prepare graph linear output projection without activation buffers");
        let ffn_matrices = ffn_specs.map(|(name, _)| {
            artifact
                .recovered_matrix(name)
                .expect("resolve graph FFN gate/up projection")
        });
        let ffn_contracts = ffn_matrices.map(|matrix| {
            matrix
                .operation(&validation_input, Activation::Identity)
                .expect("build graph FFN gate/up projection contract")
        });
        let ffn_projections = ffn_contracts.map(|operation| {
            runtime
                .prepare_mapped_fused_matvec_graph_io(&mapping, &operation)
                .expect("prepare graph FFN gate/up projection without activation buffers")
        });
        let ffn_down_matrix = artifact
            .recovered_matrix("model.language_model.layers.0.mlp.down_proj.weight")
            .expect("resolve graph FFN down projection");
        let ffn_down_validation = vec![0.0; config.intermediate_size];
        let ffn_down_operation = ffn_down_matrix
            .operation(&ffn_down_validation, Activation::Identity)
            .expect("build graph FFN down projection contract");
        let ffn_down_projection = runtime
            .prepare_mapped_fused_matvec_graph_io(&mapping, &ffn_down_operation)
            .expect("prepare graph FFN down projection without activation buffers");
        assert!(runtime
            .prepare_mapped_gated_delta_prepare_graph_io(
                &mapping,
                norm_weight,
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.dt_bias")
                    .expect("resolve graph dt_bias"),
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
            )
            .is_err());
        assert!(runtime
            .prepare_mapped_gated_delta_prepare_graph_io(
                &mapping,
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.A_log")
                    .expect("resolve graph A_log"),
                artifact
                    .float_tensor("model.language_model.layers.0.linear_attn.dt_bias")
                    .expect("resolve graph dt_bias"),
                8,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
            )
            .is_err());
        assert_eq!(
            embedding.transient_bytes(),
            MetalFusedMatVecParams::BYTE_LEN
        );
        assert_eq!(norm.transient_bytes(), MetalRmsNormParams::BYTE_LEN);
        assert_eq!(
            residual_norm.transient_bytes(),
            MetalRmsNormParams::BYTE_LEN
        );
        assert_eq!(
            post_ffn_residual_norm.transient_bytes(),
            MetalRmsNormParams::BYTE_LEN
        );
        for projection in &prepared_projections {
            assert_eq!(
                projection.transient_bytes(),
                std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
            );
            assert!(projection.write_input(&validation_input).is_err());
        }
        assert!(norm.write_input(&validation_input).is_err());
        assert_eq!(
            convolution.transient_bytes(),
            MetalCausalConvParams::BYTE_LEN
        );
        assert_eq!(delta_prepare.copied_model_bytes(), 0);
        assert_eq!(
            delta_prepare.transient_bytes(),
            MetalGatedDeltaPrepareParams::BYTE_LEN
        );
        assert_eq!(
            recurrence.transient_bytes(),
            MetalGatedDeltaParams::BYTE_LEN
        );
        assert_eq!(gated_norm.transient_bytes(), MetalRmsNormParams::BYTE_LEN);
        assert_eq!(
            linear_output_projection.transient_bytes(),
            std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
        );
        assert!(linear_output_projection
            .write_input(&linear_output_validation)
            .is_err());
        for projection in &ffn_projections {
            assert_eq!(
                projection.transient_bytes(),
                std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
            );
            assert!(projection.write_input(&validation_input).is_err());
        }
        assert_eq!(
            ffn_down_projection.transient_bytes(),
            std::mem::size_of::<f32>() + MetalFusedMatVecParams::BYTE_LEN
        );
        assert!(ffn_down_projection
            .write_input(&vec![0.0; config.intermediate_size])
            .is_err());
        assert!(!gated_norm.has_owned_io());
        assert!(gated_norm
            .write_inputs(
                &vec![0.0; recurrence_config.heads * recurrence_config.value_dim],
                &vec![0.0; recurrence_config.heads * recurrence_config.value_dim],
            )
            .is_err());
        assert!(runtime.dispatch_mapped_rms_norm_gated(&gated_norm).is_err());
        assert!(!recurrence.has_owned_io());
        assert!(recurrence
            .write_step(&[0.0; 1], &[0.0; 1], &[0.0; 1], &[0.0; 1], &[0.0; 1])
            .is_err());
        assert!(runtime.dispatch_gated_delta_f16(&mut recurrence).is_err());
        assert!(convolution
            .write_input(&vec![0.0; convolution_channels])
            .is_err());
        assert!(runtime.dispatch_mapped_embedding(&embedding, 0).is_err());

        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("shared decode workspace plan");
        let workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate shared decode arena");
        let resource_plan = MetalProjectionPlan::qwen38(&config).expect("projection resource plan");
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &resource_plan, &config)
            .expect("decode binding plan");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("bind real shared-arena views");
        let embedding_output = &program.steps()[0].writes()[0];
        let normalized_output = &program.steps()[1].writes()[0];
        let linear_outputs = [
            &program.steps()[2].writes()[0],
            &program.steps()[2].writes()[1],
            &program.steps()[2].writes()[2],
            &program.steps()[2].writes()[3],
        ];
        let delta_outputs = [
            &program.steps()[4].writes()[0],
            &program.steps()[4].writes()[1],
            &program.steps()[4].writes()[2],
            &program.steps()[4].writes()[3],
            &program.steps()[4].writes()[4],
        ];
        let recurrence_step = &program.steps()[5];
        let gated_norm_step = &program.steps()[6];
        let linear_output_step = &program.steps()[7];
        let residual_norm_step = &program.steps()[8];
        let ffn_gate_up_step = &program.steps()[9];
        let swiglu_down_step = &program.steps()[10];
        let post_ffn_residual_step = &program.steps()[11];
        let projection_refs = [
            &prepared_projections[0],
            &prepared_projections[1],
            &prepared_projections[2],
            &prepared_projections[3],
        ];
        let ffn_projection_refs = [&ffn_projections[0], &ffn_projections[1]];
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                [
                    linear_outputs[1],
                    linear_outputs[0],
                    linear_outputs[2],
                    linear_outputs[3],
                ],
            )
            .is_err());
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((
                    &delta_prepare,
                    [
                        delta_outputs[1],
                        delta_outputs[0],
                        delta_outputs[2],
                        delta_outputs[3],
                        delta_outputs[4],
                    ],
                )),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, recurrence_step)),
                None,
                None,
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, gated_norm_step)),
                None,
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, linear_output_step)),
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, residual_norm_step)),
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, ffn_gate_up_step)),
                Some((&ffn_down_projection, ffn_gate_up_step)),
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);
        assert!(runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, ffn_gate_up_step)),
                Some((&ffn_down_projection, swiglu_down_step)),
                Some((&post_ffn_residual_norm, swiglu_down_step)),
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .is_err());
        assert!(!convolution.poisoned);
        assert!(!recurrence.poisoned);

        let hidden = cpu
            .recovered_row(
                &embedding_matrix
                    .row_operation(1)
                    .expect("resolve oracle embedding row"),
            )
            .expect("decode oracle embedding row");
        let normalized =
            crate::reference::rms_norm_1p_weight(&hidden, 1, columns, &norm_values, epsilon)
                .expect("normalize oracle embedding");
        let corrected_input: Vec<f32> = normalized
            .iter()
            .zip(&s_in_values)
            .map(|(value, scale)| value * scale)
            .collect();
        let mut expected = projection_matrices
            .iter()
            .map(|matrix| {
                let row = matrix
                    .row_operation(0)
                    .expect("resolve oracle projection row");
                let value = cpu
                    .recovered_row_matvec(&RecoveredRowMatVec {
                        dtype: row.dtype,
                        weights: row.weights,
                        corrected_input: &corrected_input,
                        s_out: row.s_out,
                    })
                    .expect("execute oracle projection row");
                vec![value; matrix.matrix.rows]
            })
            .collect::<Vec<_>>();
        let mut expected_convolution_state =
            vec![f16::ZERO; convolution_channels * convolution_kernel];
        expected[0] = crate::reference::causal_conv_silu_update_f16_state(
            &expected[0],
            &mut expected_convolution_state,
            &convolution_weight_f32,
            convolution_channels,
            convolution_kernel,
        )
        .expect("execute graph convolution oracle");
        convolution
            .begin_speculative(&runtime)
            .expect("snapshot graph convolution state");
        recurrence
            .begin_speculative(&runtime)
            .expect("snapshot graph recurrence state");
        let actual = runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .expect("dispatch first nine decode steps through shared arena");
        assert_eq!(
            convolution.verifier_read_state(),
            expected_convolution_state
        );
        assert!(actual.is_empty());
        let compact_qk_values = config.linear_num_key_heads * config.linear_key_head_dim;
        let expanded_qk_values = config.linear_num_value_heads * config.linear_key_head_dim;
        let expected_query: Vec<f32> = (0..expanded_qk_values)
            .map(|index| {
                let head = index / config.linear_key_head_dim;
                let column = index % config.linear_key_head_dim;
                expected[0][(head / 3) * config.linear_key_head_dim + column]
            })
            .collect();
        let expected_key: Vec<f32> = (0..expanded_qk_values)
            .map(|index| {
                let head = index / config.linear_key_head_dim;
                let column = index % config.linear_key_head_dim;
                expected[0][compact_qk_values + (head / 3) * config.linear_key_head_dim + column]
            })
            .collect();
        let expected_value =
            expected[0][2 * compact_qk_values..2 * compact_qk_values + expanded_qk_values].to_vec();
        let expected_log_decay: Vec<f32> = expected[2]
            .iter()
            .zip(&a_log_values)
            .zip(&dt_bias_values)
            .map(|((raw_a, a_log), dt_bias)| {
                let a = raw_a + dt_bias;
                let softplus = if a > 20.0 { a } else { a.exp().ln_1p() };
                -a_log.exp() * softplus
            })
            .collect();
        let expected_beta: Vec<f32> = expected[3]
            .iter()
            .map(|raw_b| 1.0 / (1.0 + (-raw_b).exp()))
            .collect();
        let mut expected_recurrence_state =
            vec![
                f16::ZERO;
                recurrence_config.heads * recurrence_config.key_dim * recurrence_config.value_dim
            ];
        let expected_attention = crate::reference::recurrent_gated_delta_step_f16_state(
            &expected_query,
            &expected_key,
            &expected_value,
            &expected_log_decay,
            &expected_beta,
            &mut expected_recurrence_state,
            recurrence_config.heads,
            recurrence_config.key_dim,
            recurrence_config.value_dim,
        )
        .expect("execute graph recurrence oracle");
        let expected_gated_attention = crate::reference::rms_norm_gated(
            &expected_attention,
            &expected[1],
            recurrence_config.heads,
            recurrence_config.value_dim,
            &gated_norm_values,
            recurrence_config.epsilon,
        )
        .expect("execute graph gated RMSNorm oracle");
        let corrected_linear_output: Vec<f32> = expected_gated_attention
            .iter()
            .zip(&linear_output_s_in_values)
            .map(|(value, scale)| value * scale)
            .collect();
        let linear_output_row = linear_output_matrix
            .row_operation(0)
            .expect("resolve graph linear output row");
        let expected_linear_output_row = cpu
            .recovered_row_matvec(&RecoveredRowMatVec {
                dtype: linear_output_row.dtype,
                weights: linear_output_row.weights,
                corrected_input: &corrected_linear_output,
                s_out: linear_output_row.s_out,
            })
            .expect("execute graph linear output projection oracle");
        let actual_attention = workspace
            .read_f32(MetalBufferSlot::AttentionOutput)
            .expect("read graph gated RMSNorm output");
        for (index, (expected, actual)) in expected_gated_attention
            .iter()
            .zip(actual_attention)
            .enumerate()
        {
            let tolerance = 6.0e-4_f32.max(expected.abs() * 4.0e-4);
            assert!(
                (expected - actual).abs() <= tolerance,
                "graph gated RMSNorm output {index}: expected {expected}, got {actual}"
            );
        }
        let actual_mixer = workspace
            .read_f32(MetalBufferSlot::MixerOutput)
            .expect("read graph linear output projection");
        for (index, actual) in actual_mixer.iter().enumerate() {
            let tolerance = 6.0e-4_f32.max(expected_linear_output_row.abs() * 4.0e-4);
            assert!(
                (expected_linear_output_row - *actual).abs() <= tolerance,
                "graph linear output projection {index}: expected {expected_linear_output_row}, got {actual}"
            );
        }
        let actual_hidden = workspace
            .read_f32(MetalBufferSlot::HiddenA)
            .expect("read graph residual input");
        let actual_residual = workspace
            .read_f32(MetalBufferSlot::HiddenB)
            .expect("read graph residual output");
        let actual_post_attention_norm = workspace
            .read_f32(MetalBufferSlot::Normalized)
            .expect("read graph post-attention normalized output");
        let expected_residual: Vec<f32> = actual_hidden
            .iter()
            .zip(&actual_mixer)
            .map(|(residual, update)| residual + update)
            .collect();
        let expected_post_attention_norm = crate::reference::rms_norm_1p_weight(
            &expected_residual,
            1,
            columns,
            &post_attention_norm_values,
            epsilon,
        )
        .expect("execute graph post-attention residual RMSNorm oracle");
        for (index, ((expected_residual, actual_residual), (expected_norm, actual_norm))) in
            expected_residual
                .iter()
                .zip(&actual_residual)
                .zip(
                    expected_post_attention_norm
                        .iter()
                        .zip(&actual_post_attention_norm),
                )
                .enumerate()
        {
            let residual_tolerance = 5.0e-5_f32.max(expected_residual.abs() * 1.0e-6);
            let norm_tolerance = 6.0e-4_f32.max(expected_norm.abs() * 4.0e-4);
            assert!(
                (expected_residual - actual_residual).abs() <= residual_tolerance,
                "graph residual output {index}: expected {expected_residual}, got {actual_residual}"
            );
            assert!(
                (expected_norm - actual_norm).abs() <= norm_tolerance,
                "graph post-attention norm {index}: expected {expected_norm}, got {actual_norm}"
            );
        }
        assert_eq!(recurrence.verifier_read_state(), expected_recurrence_state);
        recurrence
            .restore_speculative(&runtime)
            .expect("restore graph recurrence checkpoint");
        convolution
            .restore_speculative(&runtime)
            .expect("restore graph convolution checkpoint");
        assert!(recurrence
            .verifier_read_state()
            .iter()
            .all(|value| *value == f16::ZERO));
        assert!(convolution
            .verifier_read_state()
            .iter()
            .all(|value| *value == f16::ZERO));

        convolution
            .begin_speculative(&runtime)
            .expect("snapshot graph convolution state for FFN prefix");
        recurrence
            .begin_speculative(&runtime)
            .expect("snapshot graph recurrence state for FFN prefix");
        runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, ffn_gate_up_step)),
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .expect("dispatch first ten decode steps through shared arena");
        let ffn_input = workspace
            .read_f32(MetalBufferSlot::Normalized)
            .expect("read graph FFN normalized input");
        let actual_ffn = [
            workspace
                .read_f32(MetalBufferSlot::FfnGate)
                .expect("read graph FFN gate projection"),
            workspace
                .read_f32(MetalBufferSlot::FfnUp)
                .expect("read graph FFN up projection"),
        ];
        for (branch, (matrix, actual)) in ffn_matrices.iter().zip(&actual_ffn).enumerate() {
            let expected = cpu
                .fused_matvec(
                    &matrix
                        .operation(&ffn_input, Activation::Identity)
                        .expect("build graph FFN gate/up oracle operation"),
                )
                .expect("execute graph FFN gate/up oracle");
            for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 8.0e-4_f32.max(expected.abs() * 5.0e-4);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "graph FFN branch {branch} row {row}: expected {expected}, got {actual}"
                );
            }
        }
        recurrence
            .restore_speculative(&runtime)
            .expect("restore graph recurrence after FFN prefix");
        convolution
            .restore_speculative(&runtime)
            .expect("restore graph convolution after FFN prefix");

        let expected_swiglu = crate::reference::swiglu(&actual_ffn[0], &actual_ffn[1])
            .expect("execute graph SwiGLU oracle");
        let expected_down = cpu
            .fused_matvec(
                &ffn_down_matrix
                    .operation(&expected_swiglu, Activation::Identity)
                    .expect("build graph fused SwiGLU down oracle operation"),
            )
            .expect("execute graph fused SwiGLU down oracle");
        convolution
            .begin_speculative(&runtime)
            .expect("snapshot graph convolution state for SwiGLU prefix");
        recurrence
            .begin_speculative(&runtime)
            .expect("snapshot graph recurrence state for SwiGLU prefix");
        runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, ffn_gate_up_step)),
                Some((&ffn_down_projection, swiglu_down_step)),
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .expect("dispatch first eleven decode steps through shared arena");
        let actual_down = workspace
            .read_f32(MetalBufferSlot::FfnDown)
            .expect("read graph fused SwiGLU down projection");
        for (row, (expected, actual)) in expected_down.iter().zip(&actual_down).enumerate() {
            let tolerance = 1.2e-3_f32.max(expected.abs() * 8.0e-4);
            assert!(
                (expected - actual).abs() <= tolerance,
                "graph fused SwiGLU down row {row}: expected {expected}, got {actual}"
            );
        }
        recurrence
            .restore_speculative(&runtime)
            .expect("restore graph recurrence after SwiGLU prefix");
        convolution
            .restore_speculative(&runtime)
            .expect("restore graph convolution after SwiGLU prefix");

        let expected_post_ffn_residual: Vec<f32> = actual_residual
            .iter()
            .zip(&actual_down)
            .map(|(residual, update)| residual + update)
            .collect();
        let expected_next_norm = crate::reference::rms_norm_1p_weight(
            &expected_post_ffn_residual,
            1,
            columns,
            &next_layer_norm_values,
            epsilon,
        )
        .expect("execute graph post-FFN residual RMSNorm oracle");
        convolution
            .begin_speculative(&runtime)
            .expect("snapshot graph convolution state for complete layer");
        recurrence
            .begin_speculative(&runtime)
            .expect("snapshot graph recurrence state for complete layer");
        runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                Some(&mut convolution),
                Some((&delta_prepare, delta_outputs)),
                Some((&mut recurrence, recurrence_step)),
                Some((&gated_norm, gated_norm_step)),
                Some((&linear_output_projection, linear_output_step)),
                Some((&residual_norm, residual_norm_step)),
                Some((ffn_projection_refs, ffn_gate_up_step)),
                Some((&ffn_down_projection, swiglu_down_step)),
                Some((&post_ffn_residual_norm, post_ffn_residual_step)),
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .expect("dispatch complete first transformer layer through shared arena");
        let actual_post_ffn_residual = workspace
            .read_f32(MetalBufferSlot::HiddenA)
            .expect("read graph post-FFN residual output");
        let actual_next_norm = workspace
            .read_f32(MetalBufferSlot::Normalized)
            .expect("read graph next-layer normalized output");
        for (index, ((expected_residual, actual_residual), (expected_norm, actual_norm))) in
            expected_post_ffn_residual
                .iter()
                .zip(actual_post_ffn_residual)
                .zip(expected_next_norm.iter().zip(actual_next_norm))
                .enumerate()
        {
            let residual_tolerance = 8.0e-5_f32.max(expected_residual.abs() * 2.0e-6);
            let norm_tolerance = 8.0e-4_f32.max(expected_norm.abs() * 5.0e-4);
            assert!(
                (expected_residual - actual_residual).abs() <= residual_tolerance,
                "graph post-FFN residual {index}: expected {expected_residual}, got {actual_residual}"
            );
            assert!(
                (expected_norm - actual_norm).abs() <= norm_tolerance,
                "graph next-layer norm {index}: expected {expected_norm}, got {actual_norm}"
            );
        }
        recurrence
            .restore_speculative(&runtime)
            .expect("restore graph recurrence after complete layer");
        convolution
            .restore_speculative(&runtime)
            .expect("restore graph convolution after complete layer");

        runtime
            .dispatch_mapped_embedding_rms_norm_linear_fanout_views(
                &embedding,
                1,
                &norm,
                projection_refs,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                embedding_output,
                normalized_output,
                linear_outputs,
            )
            .expect("reinitialize shared-arena input for reusable layer encoder");
        assert!(runtime
            .dispatch_prepared_mapped_linear_attention_layer(
                program
                    .linear_attention_layer_steps(1)
                    .expect("bind layer-1 schedule slice"),
                &mut reusable_layer,
            )
            .is_err());
        assert!(!reusable_layer.convolution.poisoned);
        assert!(!reusable_layer.recurrence.poisoned);
        reusable_layer
            .convolution
            .begin_speculative(&runtime)
            .expect("snapshot convolution for reusable layer encoder");
        reusable_layer
            .recurrence
            .begin_speculative(&runtime)
            .expect("snapshot recurrence for reusable layer encoder");
        runtime
            .dispatch_prepared_mapped_linear_attention_layer(
                program
                    .linear_attention_layer_steps(0)
                    .expect("bind layer-0 schedule slice"),
                &mut reusable_layer,
            )
            .expect("dispatch reusable complete linear-attention layer encoder");
        let reusable_residual = workspace
            .read_f32(MetalBufferSlot::HiddenA)
            .expect("read reusable layer residual");
        let reusable_norm = workspace
            .read_f32(MetalBufferSlot::Normalized)
            .expect("read reusable layer next norm");
        for (index, ((expected_residual, actual_residual), (expected_norm, actual_norm))) in
            expected_post_ffn_residual
                .iter()
                .zip(reusable_residual)
                .zip(expected_next_norm.iter().zip(reusable_norm))
                .enumerate()
        {
            let residual_tolerance = 8.0e-5_f32.max(expected_residual.abs() * 2.0e-6);
            let norm_tolerance = 8.0e-4_f32.max(expected_norm.abs() * 5.0e-4);
            assert!(
                (expected_residual - actual_residual).abs() <= residual_tolerance,
                "reusable Metal layer residual {index}: expected {expected_residual}, got {actual_residual}"
            );
            assert!(
                (expected_norm - actual_norm).abs() <= norm_tolerance,
                "reusable Metal layer norm {index}: expected {expected_norm}, got {actual_norm}"
            );
        }
        reusable_layer
            .recurrence
            .restore_speculative(&runtime)
            .expect("restore reusable layer recurrence");
        reusable_layer
            .convolution
            .restore_speculative(&runtime)
            .expect("restore reusable layer convolution");
    }

    #[test]
    fn mapped_qwen_rms_norm_matches_oracle_and_reuses_input_buffer() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let rows_q2 = 3;
        let rows_q4 = 5;
        let rows = 3;
        let columns = 3 * BLOCK_LEN;
        let epsilon = 1.0e-6;
        let directory = tempdir().expect("temporary RMSNorm artifact directory");
        let path = directory.path().join("rms-norm.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open RMSNorm mmap fixture");
        let weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve mmap-backed FP16 norm weight");
        let weight_f32 = weight.to_f32_vec().expect("widen norm oracle weight");
        let input: Vec<f32> = (0..rows * columns)
            .map(|index| (index as f32 * 0.017).sin() * 0.7 + 0.1)
            .collect();
        let expected =
            crate::reference::rms_norm_1p_weight(&input, rows, columns, &weight_f32, epsilon)
                .expect("RMSNorm scalar oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import RMSNorm mmap without copy");
        let prepared = runtime
            .prepare_mapped_rms_norm_1p(&mapping, weight, &input, rows, columns, epsilon)
            .expect("prepare mmap-backed RMSNorm");
        assert_eq!(prepared.rows(), rows);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            2 * size_of_val(input.as_slice()) + MetalRmsNormParams::BYTE_LEN
        );
        assert!(runtime
            .dispatch_mapped_rms_norm_1p_repeated(&prepared, 0)
            .is_err());
        assert!(runtime
            .prepare_mapped_rms_norm_1p(&mapping, weight, &input, rows, columns, 0.0)
            .is_err());
        let copied_weight = f16_bytes(&vec![1.125; columns]);
        assert!(runtime
            .prepare_mapped_rms_norm_1p(
                &mapping,
                FloatTensorView::F16Le(&copied_weight),
                &input,
                rows,
                columns,
                epsilon,
            )
            .is_err());
        drop(mapping);
        drop(artifact);
        let actual = runtime
            .dispatch_mapped_rms_norm_1p_repeated(&prepared, 3)
            .expect("dispatch RMSNorm after loader drop");
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 3.0e-5_f32.max(expected.abs() * 4.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "RMSNorm value {index}: expected {expected}, got {actual}"
            );
        }
        prepared
            .write_input(&vec![0.0; rows * columns])
            .expect("update RMSNorm input in place");
        let zero = runtime
            .dispatch_mapped_rms_norm_1p(&prepared)
            .expect("dispatch zero RMSNorm input");
        assert!(zero.iter().all(|value| value.abs() <= f32::EPSILON));
    }

    #[test]
    fn mapped_gated_rms_norm_matches_oracle_and_reuses_both_inputs() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let rows_q2 = 3;
        let rows_q4 = 5;
        let rows = 48;
        let columns = 2 * BLOCK_LEN;
        let epsilon = 1.0e-6;
        let directory = tempdir().expect("temporary gated RMSNorm artifact directory");
        let path = directory.path().join("gated-rms-norm.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open gated RMSNorm mmap fixture");
        let weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve mmap-backed FP16 gated norm weight");
        let weight_f32 = weight.to_f32_vec().expect("widen gated norm oracle weight");
        let input: Vec<f32> = (0..rows * columns)
            .map(|index| (index as f32 * 0.013).sin() * 0.75 - 0.05)
            .collect();
        let gate: Vec<f32> = (0..rows * columns)
            .map(|index| (index as f32 * 0.019).cos() * 1.2 + 0.1)
            .collect();
        let expected =
            crate::reference::rms_norm_gated(&input, &gate, rows, columns, &weight_f32, epsilon)
                .expect("gated RMSNorm scalar oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import gated RMSNorm mmap without copy");
        let prepared = runtime
            .prepare_mapped_rms_norm_gated(&mapping, weight, &input, &gate, rows, columns, epsilon)
            .expect("prepare mmap-backed gated RMSNorm");
        assert_eq!(prepared.rows(), rows);
        assert_eq!(prepared.columns(), columns);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            3 * size_of_val(input.as_slice()) + MetalRmsNormParams::BYTE_LEN
        );
        assert!(runtime
            .prepare_mapped_rms_norm_gated(
                &mapping,
                weight,
                &input,
                &gate[..gate.len() - 1],
                rows,
                columns,
                epsilon,
            )
            .is_err());
        assert!(runtime
            .prepare_mapped_rms_norm_gated(&mapping, weight, &input, &gate, rows, columns, 0.0,)
            .is_err());
        let copied_weight = f16_bytes(&vec![1.0; columns]);
        assert!(runtime
            .prepare_mapped_rms_norm_gated(
                &mapping,
                FloatTensorView::F16Le(&copied_weight),
                &input,
                &gate,
                rows,
                columns,
                epsilon,
            )
            .is_err());
        drop(mapping);
        drop(artifact);
        let actual = runtime
            .dispatch_mapped_rms_norm_gated(&prepared)
            .expect("dispatch gated RMSNorm after loader drop");
        for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 4.0e-5_f32.max(expected.abs() * 5.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "gated RMSNorm value {index}: expected {expected}, got {actual}"
            );
        }
        let zero = vec![0.0; rows * columns];
        prepared
            .write_inputs(&zero, &gate)
            .expect("update gated RMSNorm inputs in place");
        let zero_output = runtime
            .dispatch_mapped_rms_norm_gated(&prepared)
            .expect("dispatch zero gated RMSNorm input");
        assert!(zero_output.iter().all(|value| value.abs() <= f32::EPSILON));
    }

    #[test]
    fn mapped_rms_norm_lm_head_argmax_chain_stays_on_device() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let rows_q2 = 3;
        let rows_q4 = 5;
        let columns = 3 * BLOCK_LEN;
        let epsilon = 1.0e-6;
        let directory = tempdir().expect("temporary chained artifact directory");
        let path = directory.path().join("norm-projection.ctoxq");
        write_mixed_fixture(&path, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open chained mmap fixture");
        let matrix = artifact
            .recovered_matrix("matrix.weight")
            .expect("resolve chained recovered matrix");
        let norm_weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve chained norm weight");
        let norm_weight_f32 = norm_weight
            .to_f32_vec()
            .expect("widen chained norm oracle weight");
        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.023).cos() * 0.6 - 0.05)
            .collect();
        let normalized =
            crate::reference::rms_norm_1p_weight(&input, 1, columns, &norm_weight_f32, epsilon)
                .expect("chained RMSNorm oracle");
        let expected_operation = matrix
            .operation(&normalized, Activation::Identity)
            .expect("construct chained projection oracle");
        let expected = cpu
            .fused_matvec(&expected_operation)
            .expect("execute chained projection oracle");
        let placeholder_operation = matrix
            .operation(&input, Activation::Identity)
            .expect("construct external-input projection contract");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import chained mmap without copy");
        let norm = runtime
            .prepare_mapped_rms_norm_1p(&mapping, norm_weight, &input, 1, columns, epsilon)
            .expect("prepare chained RMSNorm");
        let owned_projection = runtime
            .prepare_mapped_fused_matvec(&mapping, &placeholder_operation)
            .expect("prepare owned-input comparison projection");
        let projection = runtime
            .prepare_mapped_fused_matvec_external_input(&mapping, &placeholder_operation)
            .expect("prepare external-input projection");
        assert_eq!(norm.copied_model_bytes(), 0);
        assert_eq!(projection.copied_model_bytes(), 0);
        assert_eq!(
            owned_projection.transient_bytes() - projection.transient_bytes(),
            size_of_val(input.as_slice())
        );
        assert!(projection.write_input(&input).is_err());
        assert!(runtime.dispatch_mapped(&projection).is_err());
        assert!(runtime
            .dispatch_mapped_rms_norm_then_projection(&norm, &owned_projection)
            .is_err());
        drop(owned_projection);
        drop(mapping);
        drop(artifact);
        let actual = runtime
            .dispatch_mapped_rms_norm_then_projection(&norm, &projection)
            .expect("dispatch chained norm/projection after loader drop");
        for (row, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            let tolerance = 3.0e-4_f32.max(expected.abs() * 5.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "chained projection row {row}: expected {expected}, got {actual}"
            );
        }
        let expected_token = expected
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(row, _)| row as u32)
            .expect("non-empty LM-head oracle");
        let selector = runtime
            .prepare_argmax_f32_scratch(rows_q2 + rows_q4)
            .expect("prepare graph-resident selector scratch");
        assert_eq!(selector.values(), rows_q2 + rows_q4);
        assert_eq!(selector.groups(), 32);
        assert_eq!(
            selector.transient_bytes(),
            32 * 4 * std::mem::size_of::<u32>()
                + 2 * std::mem::size_of::<u32>()
                + MetalArgMaxParams::BYTE_LEN
        );
        assert_eq!(
            runtime
                .dispatch_mapped_rms_norm_projection_argmax(&norm, &projection, &selector)
                .expect("select directly from mapped LM-head output"),
            expected_token
        );
        let oversized_selector = runtime
            .prepare_argmax_f32_scratch(rows_q2 + rows_q4 + 1)
            .expect("prepare oversized selector contract");
        assert!(runtime
            .dispatch_mapped_rms_norm_projection_argmax(&norm, &projection, &oversized_selector,)
            .is_err());
        norm.write_input(&vec![0.0; columns])
            .expect("update chained norm input");
        let zero = runtime
            .dispatch_mapped_rms_norm_then_projection(&norm, &projection)
            .expect("dispatch zero chained input");
        assert!(zero.iter().all(|value| value.abs() <= f32::EPSILON));
        assert_eq!(
            runtime
                .dispatch_mapped_rms_norm_projection_argmax(&norm, &projection, &selector)
                .expect("select tied zero logits without host readback"),
            (rows_q2 + rows_q4 - 1) as u32
        );
    }

    #[test]
    fn partial_rope_pair_matches_qwen_oracle_and_preserves_tail() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let query_heads = 24;
        let key_heads = 4;
        let head_dim = 256;
        let rotary_dim = 64;
        let position = 12_345;
        let theta = 10_000_000.0;
        let query: Vec<f32> = (0..query_heads * head_dim)
            .map(|index| (index as f32 * 0.013).sin() * 0.8)
            .collect();
        let key: Vec<f32> = (0..key_heads * head_dim)
            .map(|index| (index as f32 * 0.019).cos() * 0.7)
            .collect();
        let mut expected_query = query.clone();
        let mut expected_key = key.clone();
        crate::reference::apply_partial_rope(
            &mut expected_query,
            &mut expected_key,
            query_heads,
            key_heads,
            head_dim,
            rotary_dim,
            position,
            theta,
        )
        .expect("partial-RoPE scalar oracle");
        let prepared_query = runtime
            .prepare_partial_rope(&query, query_heads, head_dim, rotary_dim, position, theta)
            .expect("prepare query RoPE");
        let prepared_key = runtime
            .prepare_partial_rope(&key, key_heads, head_dim, rotary_dim, position, theta)
            .expect("prepare key RoPE");
        assert_eq!(prepared_query.heads(), query_heads);
        assert_eq!(prepared_query.head_dim(), head_dim);
        assert_eq!(prepared_query.rotary_dim(), rotary_dim);
        assert_eq!(
            prepared_query.transient_bytes(),
            size_of_val(query.as_slice())
                + rotary_dim * std::mem::size_of::<f32>()
                + MetalPartialRopeParams::BYTE_LEN
        );
        assert!(runtime
            .prepare_partial_rope(&query, query_heads, head_dim, 63, position, theta)
            .is_err());
        assert!(runtime
            .prepare_partial_rope(
                &query,
                query_heads,
                head_dim,
                rotary_dim,
                u64::from(u32::MAX) + 1,
                theta,
            )
            .is_err());
        let (actual_query, actual_key) = runtime
            .dispatch_partial_rope_pair(&prepared_query, &prepared_key)
            .expect("dispatch query/key RoPE pair");
        for (kind, expected, actual) in [
            ("query", &expected_query, &actual_query),
            ("key", &expected_key, &actual_key),
        ] {
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 3.0e-5_f32.max(expected.abs() * 4.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "{kind} RoPE value {index}: expected {expected}, got {actual}"
                );
            }
        }
        for head in 0..query_heads {
            let tail = head * head_dim + rotary_dim..(head + 1) * head_dim;
            assert_eq!(&actual_query[tail.clone()], &query[tail]);
        }
        prepared_query
            .write_values(&query)
            .expect("restore query values");
        prepared_query
            .write_position(0)
            .expect("update RoPE position");
        let identity = runtime
            .dispatch_partial_rope(&prepared_query)
            .expect("dispatch position-zero RoPE");
        assert_eq!(identity, query);
    }

    #[test]
    fn full_attention_fanout_writes_shared_query_gate_key_and_value_views() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let config = Qwen38Config::default();
        let columns = config.hidden_size;
        let query_values = config.num_attention_heads * config.head_dim;
        let key_value_values = config.num_key_value_heads * config.head_dim;
        let s_in_values: Vec<f32> = (0..columns)
            .map(|index| 0.875 + 0.015625 * (index % 7) as f32)
            .collect();
        let s_in = f16_bytes(&s_in_values);
        let specs = [
            (
                "model.language_model.layers.3.self_attn.q_proj.weight",
                TensorDType::Q2B64,
                query_values * 2,
                0.9375,
            ),
            (
                "model.language_model.layers.3.self_attn.k_proj.weight",
                TensorDType::Q4B64,
                key_value_values,
                1.03125,
            ),
            (
                "model.language_model.layers.3.self_attn.v_proj.weight",
                TensorDType::Q2B64,
                key_value_values,
                1.0625,
            ),
        ];
        let directory = tempdir().expect("temporary full-attention fan-out directory");
        let path = directory.path().join("full-attention-fanout.ctoxq");
        let mut tensors = Vec::new();
        for (name, dtype, rows, s_out) in specs {
            tensors.extend(repeated_recovered_tensors(
                name, dtype, rows, columns, &s_in, s_out,
            ));
        }
        ArtifactBuilder {
            model: "test/qwen38-full-attention-fanout".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .expect("write full-attention fan-out fixture");
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open full-attention fan-out fixture");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map full-attention fan-out fixture");
        let prepared = runtime
            .prepare_mapped_full_attention_fanout(&mapping, 3)
            .expect("prepare canonical full-attention fan-out");
        assert_eq!(prepared.layer(), 3);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert!(prepared.transient_bytes() > 0);
        assert!(prepared
            .projections
            .iter()
            .all(|projection| projection.input_buffer.is_none()
                && projection.output_buffer.is_none()));
        assert!(runtime
            .prepare_mapped_full_attention_fanout(&mapping, 0)
            .is_err());

        let input: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.007).sin() * 0.6)
            .collect();
        let corrected_input = input
            .iter()
            .zip(&s_in_values)
            .map(|(value, scale)| value * scale)
            .collect::<Vec<_>>();
        let expected = specs.map(|(name, _, _, _)| {
            let row = artifact
                .recovered_matrix(name)
                .expect("resolve fan-out matrix")
                .row_operation(0)
                .expect("resolve first fan-out row");
            cpu.recovered_row_matvec(&RecoveredRowMatVec {
                dtype: row.dtype,
                weights: row.weights,
                corrected_input: &corrected_input,
                s_out: row.s_out,
            })
            .expect("execute one-row fan-out oracle")
        });

        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projection_plan = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &projection_plan, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate shared decode arena");
        workspace
            .write_f32(MetalBufferSlot::Normalized, &input)
            .expect("seed normalized full-attention input");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("bind shared-arena decode program");
        let layer_three = program
            .full_attention_layer_steps(3)
            .expect("bind layer-3 full attention");
        let wrong_layer = program
            .full_attention_layer_steps(7)
            .expect("bind layer-7 full attention");
        assert!(runtime
            .dispatch_mapped_full_attention_fanout_views(&wrong_layer[0], &prepared)
            .is_err());
        runtime
            .dispatch_mapped_full_attention_fanout_views(&layer_three[0], &prepared)
            .expect("dispatch full-attention fan-out directly in shared arena");
        drop(program);

        for ((slot, logical_values), expected) in [
            (MetalBufferSlot::QueryGate, query_values * 2),
            (MetalBufferSlot::Key, key_value_values),
            (MetalBufferSlot::Value, key_value_values),
        ]
        .into_iter()
        .zip(expected)
        {
            let actual = workspace.read_f32(slot).expect("read fan-out arena slot");
            for (index, value) in actual[..logical_values].iter().enumerate() {
                let tolerance = 4.0e-4_f32.max(expected.abs() * 6.0e-5);
                assert!(
                    (*value - expected).abs() <= tolerance,
                    "full-attention {slot:?} row {index}: expected {expected}, got {value}"
                );
            }
            assert!(actual[logical_values..].iter().all(|value| *value == 0.0));
        }
    }

    #[test]
    fn query_gate_norm_rope_deinterleaves_directly_in_shared_arena() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let config = Qwen38Config::default();
        let heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let query_values = heads * head_dim;
        let position = 12_345;
        let q_norm_values: Vec<f32> = (0..head_dim)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.015625)
            .collect();
        let directory = tempdir().expect("temporary query/gate directory");
        let path = directory.path().join("query-gate.ctoxq");
        ArtifactBuilder {
            model: "test/qwen38-query-gate".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors: vec![PackedTensor {
                name: "model.language_model.layers.3.self_attn.q_norm.weight".into(),
                dtype: TensorDType::F16,
                shape: vec![head_dim as u64],
                bytes: f16_bytes(&q_norm_values),
            }],
        }
        .write_new(&path)
        .expect("write query/gate fixture");
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open query/gate fixture");
        let mapped_q_norm = artifact
            .float_tensor("model.language_model.layers.3.self_attn.q_norm.weight")
            .expect("resolve query norm")
            .to_f32_vec()
            .expect("widen query norm oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map query/gate fixture");
        let prepared = runtime
            .prepare_mapped_query_gate_norm_rope(&mapping, 3, position)
            .expect("prepare mapped query/gate kernel");
        assert_eq!(prepared.layer(), 3);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(
            prepared.transient_bytes(),
            config.rotary_dim * std::mem::size_of::<f32>() + MetalQueryGateParams::BYTE_LEN
        );
        assert!(runtime
            .prepare_mapped_query_gate_norm_rope(&mapping, 0, position)
            .is_err());
        drop(mapping);
        drop(artifact);

        let mut query_gate = Vec::with_capacity(query_values * 2);
        let mut raw_query = Vec::with_capacity(query_values);
        let mut expected_gate = Vec::with_capacity(query_values);
        for head in 0..heads {
            let query = (0..head_dim)
                .map(|column| ((head * 13 + column) as f32 * 0.019).sin() * 0.7)
                .collect::<Vec<_>>();
            let gate = (0..head_dim)
                .map(|column| ((head * 17 + column) as f32 * 0.013).cos() * 0.5)
                .collect::<Vec<_>>();
            query_gate.extend_from_slice(&query);
            query_gate.extend_from_slice(&gate);
            raw_query.extend_from_slice(&query);
            expected_gate.extend_from_slice(&gate);
        }
        let mut expected_query = crate::reference::rms_norm_1p_weight(
            &raw_query,
            heads,
            head_dim,
            &mapped_q_norm,
            config.rms_norm_epsilon,
        )
        .expect("query RMSNorm oracle");
        let mut unused_key = vec![0.0; head_dim];
        crate::reference::apply_partial_rope(
            &mut expected_query,
            &mut unused_key,
            heads,
            1,
            head_dim,
            config.rotary_dim,
            position,
            config.rope_theta,
        )
        .expect("query partial-RoPE oracle");

        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projection_plan = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &projection_plan, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate shared decode arena");
        workspace
            .write_f32(MetalBufferSlot::QueryGate, &query_gate)
            .expect("seed query/gate arena slot");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("bind shared-arena decode program");
        let layer_three = program
            .full_attention_layer_steps(3)
            .expect("bind layer-3 full attention");
        let wrong_layer = program
            .full_attention_layer_steps(7)
            .expect("bind layer-7 full attention");
        assert!(runtime
            .dispatch_mapped_query_gate_norm_rope_view(&wrong_layer[1], &prepared)
            .is_err());
        runtime
            .dispatch_mapped_query_gate_norm_rope_view(&layer_three[1], &prepared)
            .expect("dispatch query/gate norm+RoPE in shared arena");
        drop(program);
        let actual_query = workspace
            .read_f32(MetalBufferSlot::Query)
            .expect("read query arena slot");
        let actual_gate = workspace
            .read_f32(MetalBufferSlot::AttentionGate)
            .expect("read attention gate arena slot");
        for (label, expected, actual) in [
            ("query", &expected_query, &actual_query),
            ("gate", &expected_gate, &actual_gate),
        ] {
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 4.0e-5_f32.max(expected.abs() * 5.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "query/gate {label} value {index}: expected {expected}, got {actual}"
                );
            }
        }

        prepared
            .write_position(0)
            .expect("reuse query/gate tables at position zero");
        workspace
            .write_f32(MetalBufferSlot::QueryGate, &query_gate)
            .expect("restore query/gate arena slot");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("rebind shared-arena decode program");
        let step = &program
            .full_attention_layer_steps(3)
            .expect("rebind layer-3 full attention")[1];
        runtime
            .dispatch_mapped_query_gate_norm_rope_view(step, &prepared)
            .expect("reuse query/gate kernel at position zero");
        drop(program);
        let identity_query = workspace
            .read_f32(MetalBufferSlot::Query)
            .expect("read position-zero query");
        let normalized_query = crate::reference::rms_norm_1p_weight(
            &raw_query,
            heads,
            head_dim,
            &mapped_q_norm,
            config.rms_norm_epsilon,
        )
        .expect("position-zero query oracle");
        for (expected, actual) in normalized_query.iter().zip(identity_query) {
            let tolerance = 4.0e-5_f32.max(expected.abs() * 5.0e-5);
            assert!((expected - actual).abs() <= tolerance);
        }
    }

    #[test]
    fn attention_sigmoid_gate_projects_mixed_q2_q4_without_product_buffer() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let config = Qwen38Config::default();
        let columns = config.num_attention_heads * config.head_dim;
        let rows_q2 = config.hidden_size / 2;
        let rows_q4 = config.hidden_size - rows_q2;
        let name = "model.language_model.layers.3.self_attn.o_proj.weight";
        let directory = tempdir().expect("temporary attention output directory");
        let path = directory.path().join("attention-output.ctoxq");
        write_named_mixed_fixture(&path, name, rows_q2, rows_q4, columns);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open mixed attention output fixture");
        let matrix = artifact
            .recovered_matrix(name)
            .expect("resolve mixed attention output projection");
        let attention: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.011).sin() * 0.55 - 0.03)
            .collect();
        let gate: Vec<f32> = (0..columns)
            .map(|index| (index as f32 * 0.017).cos() * 1.1 + 0.07)
            .collect();
        let mut gated = attention.clone();
        crate::reference::sigmoid_gate(&mut gated, &gate).expect("sigmoid-gate scalar oracle");
        let expected = cpu
            .fused_matvec(
                &matrix
                    .operation(&gated, Activation::Identity)
                    .expect("construct mixed attention output oracle"),
            )
            .expect("execute mixed attention output oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map mixed attention output fixture");
        let prepared = runtime
            .prepare_mapped_attention_gate_output_projection(&mapping, 3)
            .expect("prepare mixed attention output projection");
        assert_eq!(prepared.layer(), 3);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert!(prepared.transient_bytes() > 0);
        assert!(prepared.projection.input_buffer.is_none());
        assert!(prepared.projection.output_buffer.is_none());
        assert_eq!(prepared.projection.dtype, TensorDType::MixedQ2Q4B64);
        assert!(runtime
            .prepare_mapped_attention_gate_output_projection(&mapping, 0)
            .is_err());
        drop(mapping);
        drop(artifact);

        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projection_plan = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &projection_plan, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate shared decode arena");
        workspace
            .write_f32(MetalBufferSlot::AttentionOutput, &attention)
            .expect("seed attention output arena slot");
        workspace
            .write_f32(MetalBufferSlot::AttentionGate, &gate)
            .expect("seed attention gate arena slot");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("bind shared-arena decode program");
        let layer_three = program
            .full_attention_layer_steps(3)
            .expect("bind layer-3 full attention");
        let wrong_layer = program
            .full_attention_layer_steps(7)
            .expect("bind layer-7 full attention");
        assert!(runtime
            .dispatch_mapped_attention_gate_output_projection_view(&wrong_layer[5], &prepared)
            .is_err());
        runtime
            .dispatch_mapped_attention_gate_output_projection_view(&layer_three[5], &prepared)
            .expect("dispatch fused attention output projection in shared arena");
        drop(program);
        let actual = workspace
            .read_f32(MetalBufferSlot::MixerOutput)
            .expect("read mixed attention projection output");
        for (row, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            let tolerance = 5.0e-4_f32.max(expected.abs() * 7.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "attention output row {row}: expected {expected}, got {actual}"
            );
        }

        workspace
            .write_f32(MetalBufferSlot::AttentionOutput, &vec![0.0; columns])
            .expect("zero attention output arena slot");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("rebind shared-arena decode program");
        let step = &program
            .full_attention_layer_steps(3)
            .expect("rebind layer-3 full attention")[5];
        runtime
            .dispatch_mapped_attention_gate_output_projection_view(step, &prepared)
            .expect("reuse fused attention output projection");
        drop(program);
        assert!(workspace
            .read_f32(MetalBufferSlot::MixerOutput)
            .expect("read zero attention projection")
            .iter()
            .all(|value| value.abs() <= f32::EPSILON));
    }

    #[test]
    fn complete_full_attention_layer_reaches_next_normalized_view() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let cpu = CpuBackend::scalar_verifier();
        let config = Qwen38Config::default();
        let hidden = config.hidden_size;
        let query_values = config.num_attention_heads * config.head_dim;
        let key_value_values = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let hidden_s_in_values = vec![0.9375_f32; hidden];
        let hidden_s_in = f16_bytes(&hidden_s_in_values);
        let attention_s_in_values = vec![0.96875_f32; query_values];
        let attention_s_in = f16_bytes(&attention_s_in_values);
        let down_s_in_values = vec![0.953125_f32; intermediate];
        let down_s_in = f16_bytes(&down_s_in_values);
        let post_attention_norm = vec![0.0625_f32; hidden];
        let next_input_norm = vec![-0.03125_f32; hidden];
        let q_norm = vec![0.0_f32; config.head_dim];
        let prefix = "model.language_model.layers.3";
        let attention_prefix = format!("{prefix}.self_attn");
        let mlp_prefix = format!("{prefix}.mlp");
        let projection_specs = [
            (
                format!("{attention_prefix}.q_proj.weight"),
                TensorDType::Q2B64,
                query_values * 2,
                hidden,
                hidden_s_in.as_slice(),
                0.9375,
            ),
            (
                format!("{attention_prefix}.k_proj.weight"),
                TensorDType::Q4B64,
                key_value_values,
                hidden,
                hidden_s_in.as_slice(),
                1.03125,
            ),
            (
                format!("{attention_prefix}.v_proj.weight"),
                TensorDType::Q2B64,
                key_value_values,
                hidden,
                hidden_s_in.as_slice(),
                1.0625,
            ),
            (
                format!("{attention_prefix}.o_proj.weight"),
                TensorDType::Q4B64,
                hidden,
                query_values,
                attention_s_in.as_slice(),
                1.015625,
            ),
            (
                format!("{mlp_prefix}.gate_proj.weight"),
                TensorDType::Q2B64,
                intermediate,
                hidden,
                hidden_s_in.as_slice(),
                0.984375,
            ),
            (
                format!("{mlp_prefix}.up_proj.weight"),
                TensorDType::Q4B64,
                intermediate,
                hidden,
                hidden_s_in.as_slice(),
                1.03125,
            ),
            (
                format!("{mlp_prefix}.down_proj.weight"),
                TensorDType::Q2B64,
                hidden,
                intermediate,
                down_s_in.as_slice(),
                1.015625,
            ),
        ];
        let directory = tempdir().expect("temporary complete full-attention directory");
        let path = directory.path().join("full-attention-layer.ctoxq");
        let mut tensors = Vec::new();
        for (name, dtype, rows, columns, s_in, s_out) in &projection_specs {
            tensors.extend(repeated_recovered_tensors(
                name, *dtype, *rows, *columns, s_in, *s_out,
            ));
        }
        tensors.extend([
            PackedTensor {
                name: format!("{attention_prefix}.q_norm.weight"),
                dtype: TensorDType::F16,
                shape: vec![config.head_dim as u64],
                bytes: f16_bytes(&q_norm),
            },
            PackedTensor {
                name: format!("{prefix}.post_attention_layernorm.weight"),
                dtype: TensorDType::F16,
                shape: vec![hidden as u64],
                bytes: f16_bytes(&post_attention_norm),
            },
            PackedTensor {
                name: "model.language_model.layers.4.input_layernorm.weight".into(),
                dtype: TensorDType::F16,
                shape: vec![hidden as u64],
                bytes: f16_bytes(&next_input_norm),
            },
        ]);
        ArtifactBuilder {
            model: "test/qwen38-complete-full-attention".into(),
            revision: "0123456789abcdef".into(),
            target: "canonical-b64".into(),
            alignment: DEFAULT_ALIGNMENT,
            tensors,
        }
        .write_new(&path)
        .expect("write complete full-attention fixture");
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open complete full-attention fixture");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("map complete full-attention fixture");
        let cache = MetalPagedGqaConfig {
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            maximum_tokens: 8,
            page_tokens: 2,
            sink_tokens: 2,
            recent_tokens: 2,
        };
        let mut prepared = runtime
            .prepare_mapped_full_attention_layer(&mapping, 3, 0, cache)
            .expect("prepare complete full-attention layer");
        assert_eq!(prepared.layer(), 3);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert!(prepared.resident_state_bytes() > 0);
        assert_eq!(prepared.cached_tokens(), 0);
        assert!(runtime
            .prepare_mapped_full_attention_layer(&mapping, 0, 0, cache)
            .is_err());
        assert!(runtime
            .prepare_mapped_full_attention_layer(
                &mapping,
                3,
                0,
                MetalPagedGqaConfig {
                    query_heads: 12,
                    ..cache
                },
            )
            .is_err());
        assert!(runtime
            .prepare_all_mapped_full_attention_layers(&mapping, 0, cache)
            .is_err());

        let residual_input: Vec<f32> = (0..hidden)
            .map(|index| (index as f32 * 0.007).cos() * 0.4)
            .collect();
        let normalized_input: Vec<f32> = (0..hidden)
            .map(|index| (index as f32 * 0.011).sin() * 0.55 - 0.03)
            .collect();
        let corrected_hidden: Vec<f32> = normalized_input
            .iter()
            .zip(&hidden_s_in_values)
            .map(|(value, scale)| value * scale)
            .collect();
        let recovered_scalar = |name: &str, corrected_input: &[f32]| {
            let row = artifact
                .recovered_matrix(name)
                .expect("resolve complete-layer oracle matrix")
                .row_operation(0)
                .expect("resolve complete-layer oracle row");
            cpu.recovered_row_matvec(&RecoveredRowMatVec {
                dtype: row.dtype,
                weights: row.weights,
                corrected_input,
                s_out: row.s_out,
            })
            .expect("execute complete-layer row oracle")
        };
        let query_gate_scalar = recovered_scalar(
            &format!("{attention_prefix}.q_proj.weight"),
            &corrected_hidden,
        );
        let value_scalar = recovered_scalar(
            &format!("{attention_prefix}.v_proj.weight"),
            &corrected_hidden,
        );
        let quantized_value = Q4Block64::quantize(&[value_scalar; BLOCK_LEN])
            .expect("quantize one-token value oracle")
            .value(0);
        let gated_attention = quantized_value / (1.0 + (-query_gate_scalar).exp());
        let corrected_attention = vec![gated_attention * attention_s_in_values[0]; query_values];
        let mixer_scalar = recovered_scalar(
            &format!("{attention_prefix}.o_proj.weight"),
            &corrected_attention,
        );
        let first_residual: Vec<f32> = residual_input
            .iter()
            .map(|value| value + mixer_scalar)
            .collect();
        let first_normalized = crate::reference::rms_norm_1p_weight(
            &first_residual,
            1,
            hidden,
            &post_attention_norm,
            config.rms_norm_epsilon,
        )
        .expect("post-attention norm oracle");
        let corrected_ffn: Vec<f32> = first_normalized
            .iter()
            .zip(&hidden_s_in_values)
            .map(|(value, scale)| value * scale)
            .collect();
        let gate_scalar =
            recovered_scalar(&format!("{mlp_prefix}.gate_proj.weight"), &corrected_ffn);
        let up_scalar = recovered_scalar(&format!("{mlp_prefix}.up_proj.weight"), &corrected_ffn);
        let swiglu_scalar = gate_scalar / (1.0 + (-gate_scalar).exp()) * up_scalar;
        let corrected_down = vec![swiglu_scalar * down_s_in_values[0]; intermediate];
        let down_scalar =
            recovered_scalar(&format!("{mlp_prefix}.down_proj.weight"), &corrected_down);
        let expected_hidden: Vec<f32> = first_residual
            .iter()
            .map(|value| value + down_scalar)
            .collect();
        let expected_normalized = crate::reference::rms_norm_1p_weight(
            &expected_hidden,
            1,
            hidden,
            &next_input_norm,
            config.rms_norm_epsilon,
        )
        .expect("next-layer norm oracle");

        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projection_plan = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let binding_plan = MetalDecodeBindingPlan::qwen38(&schedule, &projection_plan, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate shared decode arena");
        workspace
            .write_f32(MetalBufferSlot::HiddenA, &residual_input)
            .expect("seed residual input");
        workspace
            .write_f32(MetalBufferSlot::Normalized, &normalized_input)
            .expect("seed normalized input");
        let program = workspace
            .bind_decode_program(&binding_plan)
            .expect("bind complete decode program");
        let wrong_layer = program
            .full_attention_layer_steps(7)
            .expect("bind wrong full-attention layer");
        assert!(runtime
            .dispatch_prepared_mapped_full_attention_layer(wrong_layer, &mut prepared)
            .is_err());
        assert_eq!(prepared.cached_tokens(), 0);
        let layer_three = program
            .full_attention_layer_steps(3)
            .expect("bind layer-3 full attention");
        runtime
            .dispatch_prepared_mapped_full_attention_layer(layer_three, &mut prepared)
            .expect("dispatch complete full-attention layer");
        drop(program);
        assert_eq!(prepared.cached_tokens(), 1);
        drop(mapping);
        drop(artifact);

        let actual_hidden = workspace
            .read_f32(MetalBufferSlot::HiddenA)
            .expect("read complete-layer residual");
        let actual_normalized = workspace
            .read_f32(MetalBufferSlot::Normalized)
            .expect("read complete-layer normalized output");
        for (label, expected, actual) in [
            ("hidden", &expected_hidden, &actual_hidden),
            ("normalized", &expected_normalized, &actual_normalized),
        ] {
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 3.0e-3_f32.max(expected.abs() * 8.0e-4);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "complete full-attention {label} {index}: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn target_layer_validator_rejects_partial_graph_before_state_mutation() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projections = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config)
            .expect("Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("Metal workspace plan");
        let workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate Metal workspace");
        let program = workspace
            .bind_decode_program(&bindings)
            .expect("bind Metal decode program");
        let mut partial = PreparedMappedMetalTargetLayers {
            layers: Vec::new(),
            transaction_active: false,
            poisoned: false,
        };
        assert!(runtime
            .validate_prepared_mapped_target_layers(&program, &partial)
            .is_err());
        assert!(partial.begin_speculative(&runtime).is_err());
        assert!(!partial.transaction_active());
        assert!(partial.is_empty());
        assert_eq!(partial.copied_model_bytes(), 0);
    }

    #[test]
    fn partial_rope_updates_full_attention_key_in_shared_decode_arena() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projections = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate one decode arena");
        let key_heads = 4;
        let query_heads = 24;
        let head_dim = 256;
        let rotary_dim = 64;
        let position = 12_345;
        let theta = 10_000_000.0;
        let key: Vec<f32> = (0..key_heads * head_dim)
            .map(|index| (index as f32 * 0.019).cos() * 0.7)
            .collect();
        let mut expected_key = key.clone();
        let mut unused_query = vec![0.0; query_heads * head_dim];
        crate::reference::apply_partial_rope(
            &mut unused_query,
            &mut expected_key,
            query_heads,
            key_heads,
            head_dim,
            rotary_dim,
            position,
            theta,
        )
        .expect("partial-RoPE scalar oracle");
        let key_slot_values = workspace
            .binding(MetalBufferSlot::Key)
            .expect("key arena binding")
            .values;
        let mut key_slot = vec![-17.0; key_slot_values];
        key_slot[..key.len()].copy_from_slice(&key);
        workspace
            .write_f32(MetalBufferSlot::Key, &key_slot)
            .expect("seed shared-arena key");
        let prepared = runtime
            .prepare_partial_rope_graph(key_heads, head_dim, rotary_dim, position, theta)
            .expect("prepare graph-only key RoPE");
        assert!(!prepared.has_owned_values());
        assert_eq!(
            prepared.transient_bytes(),
            rotary_dim * std::mem::size_of::<f32>() + MetalPartialRopeParams::BYTE_LEN
        );
        assert!(prepared.write_values(&key).is_err());
        let program = workspace
            .bind_decode_program(&bindings)
            .expect("bind shared-arena decode program");
        let full_attention = program
            .full_attention_layer_steps(3)
            .expect("bind layer-3 full-attention schedule");
        let key_rope = &full_attention[2];
        assert_eq!(key_rope.step().operation, MetalDecodeOperation::KeyRope);
        assert_eq!(key_rope.reads()[0].offset(), key_rope.writes()[0].offset());
        runtime
            .dispatch_partial_rope_view(&prepared, &key_rope.writes()[0])
            .expect("dispatch key RoPE directly in shared arena");
        drop(program);
        let actual = workspace
            .read_f32(MetalBufferSlot::Key)
            .expect("read shared-arena key");
        for (index, (expected, actual)) in expected_key.iter().zip(&actual).enumerate() {
            let tolerance = 3.0e-5_f32.max(expected.abs() * 4.0e-5);
            assert!(
                (expected - actual).abs() <= tolerance,
                "graph key RoPE value {index}: expected {expected}, got {actual}"
            );
        }
        assert!(actual[key.len()..].iter().all(|value| *value == -17.0));
    }

    #[test]
    fn paged_gqa_consumes_full_attention_shared_arena_without_owned_io() {
        let config = Qwen38Config::default();
        let schedule = MetalDecodeSchedule::qwen38(&config).expect("frozen Metal schedule");
        let projections = MetalProjectionPlan::qwen38(&config).expect("Metal projection plan");
        let bindings = MetalDecodeBindingPlan::qwen38(&schedule, &projections, &config)
            .expect("complete Metal binding plan");
        let workspace_plan = MetalDecodeWorkspacePlan::qwen38(&schedule, &config, 40_000)
            .expect("decode workspace plan");
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let mut workspace = runtime
            .prepare_decode_workspace(&workspace_plan)
            .expect("allocate one decode arena");
        let query_heads = 24;
        let key_value_heads = 4;
        let head_dim = 256;
        let mut prepared = runtime
            .prepare_paged_gqa_decode_graph(
                3,
                MetalPagedGqaConfig {
                    query_heads,
                    key_value_heads,
                    head_dim,
                    maximum_tokens: 8,
                    page_tokens: 2,
                    sink_tokens: 2,
                    recent_tokens: 2,
                },
            )
            .expect("prepare layer-owned graph GQA");
        assert_eq!(prepared.owner_layer, Some(3));
        assert!(prepared.query_buffer.is_none());
        assert!(prepared.output_buffer.is_none());
        assert_eq!(
            prepared.transient_bytes(),
            2 * MetalKvPackParams::BYTE_LEN + MetalPagedGqaParams::BYTE_LEN
        );
        assert!(runtime
            .prepare_paged_gqa_decode_graph(
                4,
                MetalPagedGqaConfig {
                    query_heads,
                    key_value_heads,
                    head_dim,
                    maximum_tokens: 8,
                    page_tokens: 2,
                    sink_tokens: 2,
                    recent_tokens: 2,
                },
            )
            .is_err());

        for token in 0..7 {
            let query: Vec<f32> = (0..query_heads * head_dim)
                .map(|index| ((index + token * 7) as f32 * 0.017).sin() * 0.35)
                .collect();
            let key: Vec<f32> = (0..key_value_heads * head_dim)
                .map(|index| ((index + token * 11) as f32 * 0.021).cos() * 0.45)
                .collect();
            let value: Vec<f32> = (0..key_value_heads * head_dim)
                .map(|index| ((index + token * 13) as f32 * 0.015).sin() * 0.55)
                .collect();
            for (slot, logical) in [
                (MetalBufferSlot::Query, query.as_slice()),
                (MetalBufferSlot::Key, key.as_slice()),
                (MetalBufferSlot::Value, value.as_slice()),
            ] {
                let slot_values = workspace.binding(slot).expect("arena slot binding").values;
                let mut contents = vec![-19.0; slot_values];
                contents[..logical.len()].copy_from_slice(logical);
                workspace
                    .write_f32(slot, &contents)
                    .expect("seed graph attention input");
            }

            let program = workspace
                .bind_decode_program(&bindings)
                .expect("bind shared-arena decode program");
            let layer_three = program
                .full_attention_layer_steps(3)
                .expect("bind layer-3 full attention");
            if token == 0 {
                let wrong_layer = program
                    .full_attention_layer_steps(7)
                    .expect("bind different full-attention layer");
                assert!(runtime
                    .append_and_dispatch_paged_gqa_views(
                        &mut prepared,
                        &wrong_layer[3],
                        &wrong_layer[4],
                    )
                    .is_err());
                assert!(!prepared.poisoned);
                assert_eq!(prepared.tokens(), 0);
            }
            runtime
                .append_and_dispatch_paged_gqa_views(
                    &mut prepared,
                    &layer_three[3],
                    &layer_three[4],
                )
                .expect("append and attend directly in shared arena");
            let cached_key = prepared
                .verifier_cache
                .flattened_key(key_value_heads, head_dim)
                .expect("flatten quantized graph keys");
            let cached_value = prepared
                .verifier_cache
                .flattened_value(key_value_heads, head_dim)
                .expect("flatten quantized graph values");
            let tokens = token + 1;
            let expected = crate::reference::grouped_query_attention(
                &query,
                &cached_key,
                &cached_value,
                query_heads,
                key_value_heads,
                1,
                tokens,
                head_dim,
                tokens - 1,
            )
            .expect("quantized graph GQA scalar oracle");
            drop(program);
            let actual = workspace
                .read_f32(MetalBufferSlot::AttentionOutput)
                .expect("read graph GQA output");
            for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                let tolerance = 4.0e-4_f32.max(expected.abs() * 8.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "graph GQA token {token} value {index}: expected {expected}, got {actual}"
                );
            }
        }
        assert_eq!(prepared.tokens(), 7);
        assert_eq!(prepared.cache.q2_tokens(), 2);
        assert_eq!(
            prepared
                .cache
                .pages
                .iter()
                .map(|page| page.precision)
                .collect::<Vec<_>>(),
            vec![
                KvPrecision::Q4,
                KvPrecision::Q2,
                KvPrecision::Q4,
                KvPrecision::Q4,
            ]
        );
    }

    #[test]
    fn device_kv_pack_and_demotion_match_canonical_q4_q2_bytes() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let component_values = 2 * BLOCK_LEN;
        let key: Vec<f32> = (0..component_values)
            .map(|index| (index as f32 * 0.071).sin() * 2.3)
            .collect();
        let value: Vec<f32> = (0..component_values)
            .map(|index| (index as f32 * 0.113).cos() * 1.7)
            .collect();
        let combined = key.iter().chain(&value).copied().collect::<Vec<_>>();
        let mut expected_q4 = Vec::new();
        let mut expected_q2 = Vec::new();
        for values in combined.chunks_exact(BLOCK_LEN) {
            let q4 = crate::quant::Q4Block64::quantize(values).expect("quantize Q4 oracle");
            expected_q4.extend_from_slice(&q4.encode());
            expected_q2.extend_from_slice(
                &crate::quant::Q2Block64::quantize(&q4.dequantize())
                    .expect("demote Q4 oracle to Q2")
                    .encode(),
            );
        }
        let (actual_q4, actual_q2) = runtime
            .dispatch_kv_q4_pack_and_demote(&key, &value)
            .expect("pack and demote KV entirely on Metal");
        assert_eq!(actual_q4, expected_q4);
        assert_eq!(actual_q2, expected_q2);
        assert!(runtime
            .dispatch_kv_q4_pack_and_demote(&key[..BLOCK_LEN], &value)
            .is_err());
        let mut non_finite = key.clone();
        non_finite[3] = f32::NAN;
        assert!(runtime
            .dispatch_kv_q4_pack_and_demote(&non_finite, &value)
            .is_err());
    }

    #[test]
    fn paged_q2q4_gqa_decode_matches_quantized_oracle_and_demotes_pages() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let query_heads = 4;
        let key_value_heads = 2;
        let head_dim = 64;
        let maximum_tokens = 8;
        let page_tokens = 2;
        let sink_tokens = 2;
        let recent_tokens = 2;
        let mut prepared = runtime
            .prepare_paged_gqa_decode(MetalPagedGqaConfig {
                query_heads,
                key_value_heads,
                head_dim,
                maximum_tokens,
                page_tokens,
                sink_tokens,
                recent_tokens,
            })
            .expect("prepare packed paged GQA");
        assert_eq!(prepared.tokens(), 0);
        assert_eq!(prepared.maximum_tokens(), maximum_tokens);
        assert_eq!(prepared.q2_arena_bytes(), 4 * 2 * 4 * Q2_BLOCK_BYTES);
        assert_eq!(prepared.q4_arena_bytes(), 3 * 2 * 4 * Q4_BLOCK_BYTES);
        assert_eq!(
            prepared.packed_device_bytes(),
            prepared.q2_arena_bytes()
                + prepared.q4_arena_bytes()
                + 4 * METAL_PAGED_KV_DESCRIPTOR_BYTES
        );
        assert_eq!(
            prepared.transient_bytes(),
            2 * query_heads * head_dim * std::mem::size_of::<f32>()
                + 2 * MetalKvPackParams::BYTE_LEN
                + MetalPagedGqaParams::BYTE_LEN
        );

        for token in 0..7 {
            let query: Vec<f32> = (0..query_heads * head_dim)
                .map(|index| ((index + token * 7) as f32 * 0.017).sin() * 0.35)
                .collect();
            let key: Vec<f32> = (0..key_value_heads * head_dim)
                .map(|index| ((index + token * 11) as f32 * 0.021).cos() * 0.45)
                .collect();
            let value: Vec<f32> = (0..key_value_heads * head_dim)
                .map(|index| ((index + token * 13) as f32 * 0.015).sin() * 0.55)
                .collect();
            let actual = runtime
                .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
                .expect("append packed K/V and dispatch GQA");
            let cached_key = prepared
                .verifier_cache
                .flattened_key(key_value_heads, head_dim)
                .expect("flatten quantized keys");
            let cached_value = prepared
                .verifier_cache
                .flattened_value(key_value_heads, head_dim)
                .expect("flatten quantized values");
            let tokens = token + 1;
            let expected = crate::reference::grouped_query_attention(
                &query,
                &cached_key,
                &cached_value,
                query_heads,
                key_value_heads,
                1,
                tokens,
                head_dim,
                tokens - 1,
            )
            .expect("quantized paged GQA scalar oracle");
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 4.0e-4_f32.max(expected.abs() * 8.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "paged GQA token {token} value {index}: expected {expected}, got {actual}"
                );
            }
        }

        let precisions = prepared
            .cache
            .pages
            .iter()
            .map(|page| page.precision)
            .collect::<Vec<_>>();
        assert_eq!(
            precisions,
            vec![
                KvPrecision::Q4,
                KvPrecision::Q2,
                KvPrecision::Q4,
                KvPrecision::Q4
            ]
        );
        assert!(prepared.verifier_cpu_packed_bytes() > 0);
        assert_eq!(prepared.free_q4_slots.len(), 0);
        for page in prepared.verifier_cache.page_views() {
            let (buffer, offset, token_bytes) = match page.precision {
                KvPrecision::Q2 => (
                    &prepared.q2_pages_buffer,
                    page.page_index * prepared.q2_page_bytes,
                    prepared.q2_token_bytes,
                ),
                KvPrecision::Q4 => (
                    &prepared.q4_pages_buffer,
                    prepared.page_to_q4_slot[page.page_index].expect("Q4 verifier page slot")
                        * prepared.q4_page_bytes,
                    prepared.q4_token_bytes,
                ),
            };
            let bytes = page.tokens * token_bytes;
            let actual =
                unsafe { slice::from_raw_parts(buffer.contents().cast::<u8>().add(offset), bytes) };
            assert_eq!(
                actual, page.bytes,
                "device-packed page {} differs from canonical CPU oracle",
                page.page_index
            );
        }

        prepared.reset();
        assert_eq!(prepared.tokens(), 0);
        assert_eq!(prepared.verifier_cpu_packed_bytes(), 0);
        assert_eq!(prepared.free_q4_slots.len(), 3);
        let query = vec![0.1; query_heads * head_dim];
        let key = vec![0.2; key_value_heads * head_dim];
        let value = vec![0.3; key_value_heads * head_dim];
        let output = runtime
            .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
            .expect("reuse reset paged GQA");
        assert!(output.iter().all(|actual| (*actual - 0.3).abs() < 0.02));
    }

    #[test]
    fn paged_gqa_speculative_branch_restores_without_full_arena_copy() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let query_heads = 4;
        let key_value_heads = 2;
        let head_dim = 64;
        let mut prepared = runtime
            .prepare_paged_gqa_decode(MetalPagedGqaConfig {
                query_heads,
                key_value_heads,
                head_dim,
                maximum_tokens: 16,
                page_tokens: 4,
                sink_tokens: 4,
                recent_tokens: 4,
            })
            .expect("prepare speculative paged GQA");

        for token in 0..8 {
            let query = vec![0.01 * (token + 1) as f32; query_heads * head_dim];
            let key = vec![0.02 * (token + 1) as f32; key_value_heads * head_dim];
            let value = vec![-0.015 * (token + 1) as f32; key_value_heads * head_dim];
            runtime
                .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
                .expect("build committed KV prefix");
        }
        assert_eq!(prepared.tokens(), 8);
        assert_eq!(prepared.free_q4_slots.len(), 1);
        let committed_pages = prepared
            .verifier_cache
            .page_views()
            .map(|page| (page.tokens, page.precision, page.bytes.to_vec()))
            .collect::<Vec<_>>();

        prepared
            .begin_speculative()
            .expect("begin bounded KV branch");
        assert!(prepared.begin_speculative().is_err());
        let mut first_branch = Vec::new();
        for token in 8..12 {
            let query = vec![0.01 * (token + 1) as f32; query_heads * head_dim];
            let key = vec![0.02 * (token + 1) as f32; key_value_heads * head_dim];
            let value = vec![-0.015 * (token + 1) as f32; key_value_heads * head_dim];
            first_branch.push(
                runtime
                    .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
                    .expect("advance speculative KV branch"),
            );
        }
        assert_eq!(prepared.tokens(), 12);
        assert_eq!(prepared.free_q4_slots.len(), 0);
        prepared
            .restore_speculative()
            .expect("restore bounded KV metadata");
        assert_eq!(prepared.tokens(), 8);
        assert_eq!(prepared.free_q4_slots.len(), 1);
        assert_eq!(
            prepared
                .verifier_cache
                .page_views()
                .map(|page| (page.tokens, page.precision, page.bytes.to_vec()))
                .collect::<Vec<_>>(),
            committed_pages
        );
        assert!(prepared.restore_speculative().is_err());

        prepared
            .begin_speculative()
            .expect("begin replayed KV branch");
        let mut replayed_branch = Vec::new();
        for token in 8..12 {
            let query = vec![0.01 * (token + 1) as f32; query_heads * head_dim];
            let key = vec![0.02 * (token + 1) as f32; key_value_heads * head_dim];
            let value = vec![-0.015 * (token + 1) as f32; key_value_heads * head_dim];
            replayed_branch.push(
                runtime
                    .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
                    .expect("replay speculative KV branch"),
            );
        }
        assert_eq!(replayed_branch, first_branch);
        prepared
            .commit_speculative()
            .expect("commit replayed KV branch");
        assert_eq!(prepared.tokens(), 12);
        assert!(prepared.commit_speculative().is_err());

        let query = vec![0.13; query_heads * head_dim];
        let key = vec![0.26; key_value_heads * head_dim];
        let value = vec![-0.195; key_value_heads * head_dim];
        runtime
            .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
            .expect("resume ordinary KV demotion after commit");
        assert_eq!(prepared.tokens(), 13);
        assert_eq!(prepared.cache.q2_tokens(), 4);
    }

    #[test]
    fn paged_gqa_rejects_invalid_shapes_without_cpu_fallback() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        assert!(runtime
            .prepare_paged_gqa_decode(MetalPagedGqaConfig {
                query_heads: 3,
                key_value_heads: 2,
                head_dim: 64,
                maximum_tokens: 8,
                page_tokens: 2,
                sink_tokens: 2,
                recent_tokens: 2,
            })
            .is_err());
        assert!(runtime
            .prepare_paged_gqa_decode(MetalPagedGqaConfig {
                query_heads: 4,
                key_value_heads: 2,
                head_dim: 80,
                maximum_tokens: 8,
                page_tokens: 2,
                sink_tokens: 2,
                recent_tokens: 2,
            })
            .is_err());
        let mut prepared = runtime
            .prepare_paged_gqa_decode(MetalPagedGqaConfig {
                query_heads: 4,
                key_value_heads: 2,
                head_dim: 64,
                maximum_tokens: 1,
                page_tokens: 1,
                sink_tokens: 0,
                recent_tokens: 1,
            })
            .expect("prepare one-token GQA");
        let query = vec![0.1; 4 * 64];
        let key = vec![0.2; 2 * 64];
        let value = vec![0.3; 2 * 64];
        assert!(runtime
            .append_and_dispatch_paged_gqa(&mut prepared, &query[..64], &key, &value)
            .is_err());
        runtime
            .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
            .expect("one permitted GQA token");
        assert!(runtime
            .append_and_dispatch_paged_gqa(&mut prepared, &query, &key, &value)
            .is_err());
        assert!(!prepared.poisoned);
    }

    #[test]
    fn gated_delta_f16_matches_recurrent_oracle_and_reuses_state() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let config = MetalGatedDeltaConfig {
            heads: 2,
            key_dim: 64,
            value_dim: 64,
            epsilon: 1.0e-6,
        };
        let mut prepared = runtime
            .prepare_gated_delta_f16(config)
            .expect("prepare FP16 gated-delta state");
        assert_eq!(prepared.config(), config);
        assert_eq!(prepared.resident_state_bytes(), 2 * 64 * 64 * 2);
        assert_eq!(prepared.transient_bytes(), 2_080);
        let mut expected_state = vec![f16::ZERO; config.heads * config.key_dim * config.value_dim];

        for token in 0..6 {
            let query: Vec<f32> = (0..config.heads * config.key_dim)
                .map(|index| ((index + token * 5) as f32 * 0.023).sin() * 0.4)
                .collect();
            let key: Vec<f32> = (0..config.heads * config.key_dim)
                .map(|index| ((index + token * 7) as f32 * 0.019).cos() * 0.35)
                .collect();
            let value: Vec<f32> = (0..config.heads * config.value_dim)
                .map(|index| ((index + token * 11) as f32 * 0.017).sin() * 0.5)
                .collect();
            let log_decay = vec![-0.015 - token as f32 * 0.002, -0.025];
            let beta = vec![0.45, 0.62];
            let expected = crate::reference::recurrent_gated_delta_step_f16_state(
                &query,
                &key,
                &value,
                &log_decay,
                &beta,
                &mut expected_state,
                config.heads,
                config.key_dim,
                config.value_dim,
            )
            .expect("FP16 recurrent scalar oracle");
            prepared
                .write_step(&query, &key, &value, &log_decay, &beta)
                .expect("update gated-delta inputs");
            let actual = runtime
                .dispatch_gated_delta_f16(&mut prepared)
                .expect("dispatch persistent gated-delta state");
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 3.0e-4_f32.max(expected.abs() * 2.0e-4);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "gated-delta token {token} output {index}: expected {expected}, got {actual}"
                );
            }
            let actual_state = prepared.verifier_read_state();
            for (index, (expected, actual)) in expected_state.iter().zip(actual_state).enumerate() {
                assert!(
                    (expected.to_f32() - actual.to_f32()).abs() <= 5.0e-4,
                    "gated-delta token {token} state {index}: expected {expected}, got {actual}"
                );
            }
        }

        assert_eq!(
            prepared.speculative_checkpoint_bytes(),
            prepared.resident_state_bytes()
        );
        let committed_state = prepared.verifier_read_state();
        let speculative_query = vec![0.17_f32; config.heads * config.key_dim];
        let speculative_key = vec![-0.11_f32; config.heads * config.key_dim];
        let speculative_value = vec![0.23_f32; config.heads * config.value_dim];
        let speculative_decay = vec![-0.02_f32; config.heads];
        let speculative_beta = vec![0.55_f32; config.heads];
        prepared
            .begin_speculative(&runtime)
            .expect("snapshot gated-delta state on device");
        assert!(prepared.begin_speculative(&runtime).is_err());
        prepared
            .write_step(
                &speculative_query,
                &speculative_key,
                &speculative_value,
                &speculative_decay,
                &speculative_beta,
            )
            .expect("write speculative gated-delta step");
        runtime
            .dispatch_gated_delta_f16(&mut prepared)
            .expect("advance speculative gated-delta state");
        assert_ne!(prepared.verifier_read_state(), committed_state);
        prepared
            .restore_speculative(&runtime)
            .expect("restore gated-delta state on device");
        assert_eq!(prepared.verifier_read_state(), committed_state);
        assert!(prepared.restore_speculative(&runtime).is_err());

        prepared
            .begin_speculative(&runtime)
            .expect("snapshot committed gated-delta branch");
        prepared
            .write_step(
                &speculative_query,
                &speculative_key,
                &speculative_value,
                &speculative_decay,
                &speculative_beta,
            )
            .expect("rewrite speculative gated-delta step");
        runtime
            .dispatch_gated_delta_f16(&mut prepared)
            .expect("advance committed gated-delta branch");
        prepared
            .commit_speculative()
            .expect("commit gated-delta state");
        assert_ne!(prepared.verifier_read_state(), committed_state);
        assert!(prepared.commit_speculative().is_err());

        assert!(prepared
            .write_step(&[0.0; 64], &[0.0; 128], &[0.0; 128], &[0.0; 2], &[0.0; 2])
            .is_err());
        prepared.reset();
        assert!(prepared
            .verifier_read_state()
            .iter()
            .all(|value| *value == f16::ZERO));
        assert!(!prepared.poisoned);
    }

    #[test]
    fn gated_delta_f16_rejects_invalid_geometry() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        for config in [
            MetalGatedDeltaConfig {
                heads: 0,
                key_dim: 128,
                value_dim: 128,
                epsilon: 1.0e-6,
            },
            MetalGatedDeltaConfig {
                heads: 48,
                key_dim: 128,
                value_dim: 80,
                epsilon: 1.0e-6,
            },
            MetalGatedDeltaConfig {
                heads: 48,
                key_dim: 128,
                value_dim: 128,
                epsilon: 0.0,
            },
        ] {
            assert!(runtime.prepare_gated_delta_f16(config).is_err());
        }
    }

    #[test]
    fn mapped_causal_conv_f16_matches_oracle_and_reuses_state() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let channels = 48;
        let kernel = 4;
        let directory = tempdir().expect("temporary convolution artifact directory");
        let path = directory.path().join("causal-conv.ctoxq");
        write_mixed_fixture(&path, 3, 5, channels * kernel);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open convolution mmap fixture");
        let weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve mmap-backed convolution weight");
        let weight_f32 = weight
            .to_f32_vec()
            .expect("widen convolution weight for oracle");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import convolution mmap without copy");
        let initial = vec![0.0; channels];
        let mut prepared = runtime
            .prepare_mapped_causal_conv_f16(&mapping, weight, &initial, channels, kernel)
            .expect("prepare mmap-backed FP16 convolution");
        assert_eq!(prepared.channels(), channels);
        assert_eq!(prepared.kernel(), kernel);
        assert_eq!(prepared.copied_model_bytes(), 0);
        assert_eq!(prepared.resident_state_bytes(), channels * kernel * 2);
        assert_eq!(
            prepared.transient_bytes(),
            2 * channels * std::mem::size_of::<f32>() + MetalCausalConvParams::BYTE_LEN
        );
        let mut expected_state = vec![f16::ZERO; channels * kernel];
        drop(mapping);
        drop(artifact);

        for token in 0..6 {
            let input: Vec<f32> = (0..channels)
                .map(|channel| ((channel + token * 7) as f32 * 0.031).sin() * 0.65)
                .collect();
            let expected = crate::reference::causal_conv_silu_update_f16_state(
                &input,
                &mut expected_state,
                &weight_f32,
                channels,
                kernel,
            )
            .expect("FP16 convolution scalar oracle");
            prepared
                .write_input(&input)
                .expect("update convolution input");
            let actual = runtime
                .dispatch_mapped_causal_conv_f16(&mut prepared)
                .expect("dispatch mmap-backed convolution");
            for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
                let tolerance = 2.0e-5_f32.max(expected.abs() * 4.0e-5);
                assert!(
                    (expected - actual).abs() <= tolerance,
                    "convolution token {token} output {index}: expected {expected}, got {actual}"
                );
            }
            assert_eq!(prepared.verifier_read_state(), expected_state);
        }

        assert_eq!(
            prepared.speculative_checkpoint_bytes(),
            prepared.resident_state_bytes()
        );
        let committed_state = prepared.verifier_read_state();
        let speculative_input: Vec<f32> = (0..channels)
            .map(|channel| (channel as f32 * 0.047).cos() * 0.52)
            .collect();
        prepared
            .begin_speculative(&runtime)
            .expect("snapshot convolution state on device");
        assert!(prepared.begin_speculative(&runtime).is_err());
        prepared
            .write_input(&speculative_input)
            .expect("write speculative convolution input");
        runtime
            .dispatch_mapped_causal_conv_f16(&mut prepared)
            .expect("advance speculative convolution state");
        assert_ne!(prepared.verifier_read_state(), committed_state);
        prepared
            .restore_speculative(&runtime)
            .expect("restore convolution state on device");
        assert_eq!(prepared.verifier_read_state(), committed_state);
        assert!(prepared.restore_speculative(&runtime).is_err());

        prepared
            .begin_speculative(&runtime)
            .expect("snapshot committed convolution branch");
        prepared
            .write_input(&speculative_input)
            .expect("rewrite speculative convolution input");
        runtime
            .dispatch_mapped_causal_conv_f16(&mut prepared)
            .expect("advance committed convolution branch");
        prepared
            .commit_speculative()
            .expect("commit convolution state");
        assert_ne!(prepared.verifier_read_state(), committed_state);
        assert!(prepared.commit_speculative().is_err());

        prepared.reset();
        assert!(prepared
            .verifier_read_state()
            .iter()
            .all(|value| *value == f16::ZERO));
        assert!(prepared.write_input(&[0.0; 3]).is_err());
    }

    #[test]
    fn mapped_causal_conv_rejects_copied_or_wrong_weight() {
        let runtime = MetalCandidateRuntime::new().expect("Metal device runtime");
        let channels = 48;
        let kernel = 4;
        let directory = tempdir().expect("temporary convolution contract directory");
        let path = directory.path().join("causal-conv-contract.ctoxq");
        write_mixed_fixture(&path, 3, 5, channels * kernel);
        let artifact = ModelArtifact::open(&path, ChecksumPolicy::AllTensors)
            .expect("open convolution contract fixture");
        let weight = artifact
            .float_tensor("matrix.weight.s_in")
            .expect("resolve convolution contract weight");
        let mapping = runtime
            .map_artifact_no_copy(&artifact)
            .expect("import convolution contract mmap");
        let copied = match weight {
            FloatTensorView::F16Le(bytes) => bytes.to_vec(),
            FloatTensorView::F32Le(_) => unreachable!(),
        };
        assert!(runtime
            .prepare_mapped_causal_conv_f16(
                &mapping,
                FloatTensorView::F16Le(&copied),
                &[0.0; 48],
                channels,
                kernel,
            )
            .is_err());
        assert!(runtime
            .prepare_mapped_causal_conv_f16(&mapping, weight, &[0.0; 48], channels, 3)
            .is_err());
    }
}
