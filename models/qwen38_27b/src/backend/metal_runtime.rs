//! Direct Metal execution for the unpromoted Q2/Q4 fused-matvec candidate.
//!
//! This module deliberately does not implement [`super::Backend`]. It exists
//! to generate same-device verifier and benchmark evidence while the public
//! Metal backend remains fail-closed at `PromotionState::Contract`.

use std::ffi::c_void;
use std::mem::size_of_val;
use std::rc::Rc;
use std::slice;

use metal_driver::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};
use sha2::{Digest, Sha256};

use super::metal::{
    validate_mixed_operation, validate_operation, validate_recovered_row,
    MetalArgMaxFinalBufferAbi, MetalArgMaxParams, MetalArgMaxPartialBufferAbi, MetalBufferAbi,
    MetalCausalConvBufferAbi, MetalCausalConvParams, MetalFusedMatVecParams,
    MetalGatedDeltaBufferAbi, MetalGatedDeltaParams, MetalGatedRmsNormBufferAbi,
    MetalPagedGqaBufferAbi, MetalPagedGqaParams, MetalPartialRopeBufferAbi, MetalPartialRopeParams,
    MetalRmsNormBufferAbi, MetalRmsNormParams, ARGMAX_F32_FINAL_KERNEL_NAME,
    ARGMAX_F32_PARTIAL_KERNEL_NAME, CAUSAL_CONV_F16_KERNEL_NAME, GATED_DELTA_F16_KERNEL_NAME,
    MAX_SIMDGROUPS_PER_THREADGROUP, PAGED_GQA_DECODE_KERNEL_NAME, PARTIAL_ROPE_KERNEL_NAME,
    Q2_GATHERED_KERNEL_NAME, Q2_KERNEL_NAME, Q2_RECOVERED_ROW_KERNEL_NAME, Q4_GATHERED_KERNEL_NAME,
    Q4_KERNEL_NAME, Q4_RECOVERED_ROW_KERNEL_NAME, RMS_NORM_1P_KERNEL_NAME,
    RMS_NORM_GATED_KERNEL_NAME,
};
use super::{Activation, FusedMatVec, ScaleSlice};
use crate::format::TensorDType;
use crate::kv_cache::{KvPrecision, PagedKvCache};
use crate::loader::{FloatTensorView, ModelArtifact, RecoveredMatrixView};
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::{EngineError, Result};

const KERNEL_SOURCE: &str = include_str!("../../kernels/metal/q2q4_fused_matvec.metal");
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
    q2_gathered_pipeline: ComputePipelineState,
    q4_gathered_pipeline: ComputePipelineState,
    q2_recovered_row_pipeline: ComputePipelineState,
    q4_recovered_row_pipeline: ComputePipelineState,
    rms_norm_1p_pipeline: ComputePipelineState,
    rms_norm_gated_pipeline: ComputePipelineState,
    partial_rope_pipeline: ComputePipelineState,
    paged_gqa_decode_pipeline: ComputePipelineState,
    gated_delta_f16_pipeline: ComputePipelineState,
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
/// offsets into one shared no-copy CTOXQ mapping. A standalone projection owns
/// an input buffer; a graph projection consumes an upstream device buffer and
/// allocates only bias, output, and small parameter blocks.
pub struct PreparedMappedMetalMatVec {
    dtype: TensorDType,
    rows: usize,
    columns: usize,
    s_in_offset: u64,
    dispatches: Vec<MappedMetalDispatch>,
    mapping: MappedMetalArtifact,
    input_buffer: Option<Buffer>,
    bias_buffer: Buffer,
    output_buffer: Buffer,
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
/// only the resulting f32 hidden vector and 32-byte parameter block allocate
/// transient Metal storage.
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

/// Qwen `(1 + weight)` RMSNorm with an mmap-backed FP16 weight vector.
/// Input/output remain reusable f32 graph buffers; no expanded f32 weight
/// copy is created at load time.
pub struct PreparedMappedMetalRmsNorm {
    rows: usize,
    columns: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    transient_bytes: usize,
}

/// GatedDeltaNet's direct-weight RMSNorm fused with `SiLU(z)`. The FP16 norm
/// weight remains an mmap offset; core, gate, and output are reusable f32
/// graph buffers.
pub struct PreparedMappedMetalGatedRmsNorm {
    rows: usize,
    columns: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Buffer,
    gate_buffer: Buffer,
    output_buffer: Buffer,
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
    values_buffer: Buffer,
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

/// Persistent FP16 GatedDeltaNet recurrence state plus reusable f32 inputs and
/// output. State never has an f32 device duplicate.
pub struct PreparedMetalGatedDelta {
    config: MetalGatedDeltaConfig,
    query_buffer: Buffer,
    key_buffer: Buffer,
    value_buffer: Buffer,
    log_decay_buffer: Buffer,
    beta_buffer: Buffer,
    state_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
}

/// Mmap-backed FP16 convolution weight, persistent FP16 history, and reusable
/// f32 input/output buffers for one linear-attention layer.
pub struct PreparedMappedMetalCausalConv {
    channels: usize,
    kernel: usize,
    mapping: MappedMetalArtifact,
    weight_offset: u64,
    input_buffer: Buffer,
    state_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
}

/// Standalone verifier owner for deterministic device-resident target
/// selection. Production graph assembly will bind the argmax pipeline directly
/// to the mapped LM-head output instead of owning this input buffer.
pub struct PreparedMetalArgMax {
    values: usize,
    groups: usize,
    input_buffer: Buffer,
    partials_buffer: Buffer,
    result_buffer: Buffer,
    params_buffer: Buffer,
    resident_bytes: usize,
}

impl PreparedMetalArgMax {
    pub fn values(&self) -> usize {
        self.values
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    pub fn write_input(&mut self, input: &[f32]) -> Result<()> {
        if input.len() != self.values {
            return Err(EngineError::Shape(format!(
                "Metal argmax input has {} values, expected {}",
                input.len(),
                self.values
            )));
        }
        write_buffer_range(
            &self.input_buffer,
            0,
            as_bytes(input),
            self.values * std::mem::size_of::<f32>(),
        )
    }
}

/// Decode-only grouped-query attention retaining K/V pages in their packed
/// Q2/Q4 representation. The Q2 arena has one deterministic slot per logical
/// page; the bounded Q4 arena retains only sink, recent, and one boundary page.
pub struct PreparedMetalPagedGqa {
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
    cache: PagedKvCache,
    page_to_q4_slot: Vec<Option<usize>>,
    free_q4_slots: Vec<usize>,
    q2_pages_buffer: Buffer,
    q4_pages_buffer: Buffer,
    descriptors_buffer: Buffer,
    query_buffer: Buffer,
    output_buffer: Buffer,
    params_buffer: Buffer,
    packed_device_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
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

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        let expected = self
            .rows
            .checked_mul(self.columns)
            .ok_or_else(|| EngineError::Shape("Metal RMSNorm input shape overflows".into()))?;
        validate_metal_input(input, expected)?;
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

    pub fn write_inputs(&self, input: &[f32], gate: &[f32]) -> Result<()> {
        let expected = self.rows.checked_mul(self.columns).ok_or_else(|| {
            EngineError::Shape("Metal gated RMSNorm input shape overflows".into())
        })?;
        validate_metal_input(input, expected)?;
        validate_metal_input(gate, expected)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buffer.contents().cast::<f32>(),
                input.len(),
            );
            std::ptr::copy_nonoverlapping(
                gate.as_ptr(),
                self.gate_buffer.contents().cast::<f32>(),
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

    pub fn write_values(&self, values: &[f32]) -> Result<()> {
        let expected = self
            .heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| EngineError::Shape("Metal RoPE value shape overflows".into()))?;
        validate_metal_input(values, expected)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.values_buffer.contents().cast::<f32>(),
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

    /// Verifier-only CPU packed mirror used to produce deterministic page
    /// transitions. A production graph must replace this with GPU packing and
    /// demotion before this candidate can be promoted.
    pub fn verifier_cpu_packed_bytes(&self) -> usize {
        self.cache.packed_bytes()
    }

    pub fn reset(&mut self) {
        self.cache.reset();
        self.page_to_q4_slot.fill(None);
        self.free_q4_slots = (0..self.q4_slots).rev().collect();
        zero_buffer(&self.q2_pages_buffer, self.q2_arena_bytes());
        zero_buffer(&self.q4_pages_buffer, self.q4_arena_bytes());
        zero_buffer(
            &self.descriptors_buffer,
            self.page_to_q4_slot.len() * METAL_PAGED_KV_DESCRIPTOR_BYTES,
        );
        zero_buffer(
            &self.query_buffer,
            self.query_heads * self.head_dim * std::mem::size_of::<f32>(),
        );
        zero_buffer(
            &self.output_buffer,
            self.query_heads * self.head_dim * std::mem::size_of::<f32>(),
        );
        zero_buffer(&self.params_buffer, MetalPagedGqaParams::BYTE_LEN);
        self.poisoned = false;
    }
}

impl PreparedMetalGatedDelta {
    pub fn config(&self) -> MetalGatedDeltaConfig {
        self.config
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
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
            (query, &self.query_buffer),
            (key, &self.key_buffer),
            (value, &self.value_buffer),
            (log_decay, &self.log_decay_buffer),
            (beta, &self.beta_buffer),
        ] {
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
        let output_bytes = self.config.heads * self.config.value_dim * size_of_val(&[0.0_f32]);
        zero_buffer(&self.output_buffer, output_bytes);
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

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        validate_metal_input(input, self.channels)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.input_buffer.contents().cast::<f32>(),
                input.len(),
            );
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        zero_buffer(&self.state_buffer, self.resident_state_bytes);
        zero_buffer(
            &self.output_buffer,
            self.channels * std::mem::size_of::<f32>(),
        );
        self.poisoned = false;
    }

    pub fn verifier_read_state(&self) -> Vec<half::f16> {
        let values = self.resident_state_bytes / std::mem::size_of::<half::f16>();
        unsafe {
            slice::from_raw_parts(self.state_buffer.contents().cast::<half::f16>(), values).to_vec()
        }
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
            q2_gathered_pipeline,
            q4_gathered_pipeline,
            q2_recovered_row_pipeline,
            q4_recovered_row_pipeline,
            rms_norm_1p_pipeline,
            rms_norm_gated_pipeline,
            partial_rope_pipeline,
            paged_gqa_decode_pipeline,
            gated_delta_f16_pipeline,
            causal_conv_f16_pipeline,
            argmax_f32_partial_pipeline,
            argmax_f32_final_pipeline,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    pub fn prepare_argmax_f32(&self, input: &[f32]) -> Result<PreparedMetalArgMax> {
        self.prepare_argmax_f32_with_groups(input, 32)
    }

    pub fn prepare_argmax_f32_with_groups(
        &self,
        input: &[f32],
        groups: usize,
    ) -> Result<PreparedMetalArgMax> {
        if input.is_empty() {
            return Err(EngineError::Shape(
                "Metal argmax input must be non-empty".into(),
            ));
        }
        let values = u32::try_from(input.len())
            .map_err(|_| EngineError::Shape("Metal argmax input exceeds u32".into()))?;
        if groups == 0 || groups > 256 || !groups.is_power_of_two() {
            return Err(EngineError::Shape(format!(
                "Metal argmax group count must be a power of two from 1 through 256, got {groups}"
            )));
        }
        let threads = 256_u32;
        let params = MetalArgMaxParams {
            values,
            threads,
            groups: groups as u32,
            reserved1: 0,
        };
        let input_bytes = input
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("Metal argmax input bytes overflow".into()))?;
        let result_bytes = 2 * std::mem::size_of::<u32>();
        let partials_bytes = groups * 4 * std::mem::size_of::<u32>();
        let input_buffer = buffer_with_data(&self.device, as_bytes(input));
        let partials_buffer = new_zeroed_buffer(&self.device, partials_bytes)?;
        let result_buffer = new_zeroed_buffer(&self.device, result_bytes)?;
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let resident_bytes = input_bytes
            .checked_add(result_bytes)
            .and_then(|bytes| bytes.checked_add(partials_bytes))
            .and_then(|bytes| bytes.checked_add(MetalArgMaxParams::BYTE_LEN))
            .ok_or_else(|| EngineError::MemoryBudget("Metal argmax bytes overflow".into()))?;
        Ok(PreparedMetalArgMax {
            values: input.len(),
            groups,
            input_buffer,
            partials_buffer,
            result_buffer,
            params_buffer,
            resident_bytes,
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
        zero_buffer(&prepared.result_buffer, 2 * std::mem::size_of::<u32>());
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-argmax-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        for _ in 0..dispatches {
            encoder.set_compute_pipeline_state(&self.argmax_f32_partial_pipeline);
            encoder.set_buffer(
                MetalArgMaxPartialBufferAbi::INPUT as u64,
                Some(&prepared.input_buffer),
                0,
            );
            encoder.set_buffer(
                MetalArgMaxPartialBufferAbi::PARTIALS as u64,
                Some(&prepared.partials_buffer),
                0,
            );
            encoder.set_buffer(
                MetalArgMaxPartialBufferAbi::PARAMS as u64,
                Some(&prepared.params_buffer),
                0,
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: prepared.groups as u64,
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
                Some(&prepared.partials_buffer),
                0,
            );
            encoder.set_buffer(
                MetalArgMaxFinalBufferAbi::RESULT as u64,
                Some(&prepared.result_buffer),
                0,
            );
            encoder.set_buffer(
                MetalArgMaxFinalBufferAbi::PARAMS as u64,
                Some(&prepared.params_buffer),
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
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::InvalidState(format!(
                "Metal argmax command ended with {:?}",
                command_buffer.status()
            )));
        }
        let result =
            unsafe { slice::from_raw_parts(prepared.result_buffer.contents().cast::<u32>(), 2) };
        if result[1] != 0 {
            return Err(EngineError::InvalidArtifact(format!(
                "Metal argmax rejected {} non-finite logits",
                result[1]
            )));
        }
        if result[0] as usize >= prepared.values {
            return Err(EngineError::InvalidState(format!(
                "Metal argmax selected {}, input has {} values",
                result[0], prepared.values
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
        self.prepare_mapped_fused_matvec_internal(mapping, operation, DEFAULT_SIMDGROUPS, true)
    }

    /// Prepare a projection whose input will be supplied by an upstream Metal
    /// graph buffer. No duplicate activation buffer is allocated.
    pub fn prepare_mapped_fused_matvec_external_input(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(mapping, operation, DEFAULT_SIMDGROUPS, false)
    }

    pub fn prepare_mapped_fused_matvec_with_simdgroups(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
    ) -> Result<PreparedMappedMetalMatVec> {
        self.prepare_mapped_fused_matvec_internal(mapping, operation, simdgroups, true)
    }

    fn prepare_mapped_fused_matvec_internal(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
        own_input: bool,
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
        let output_buffer = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
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
            .and_then(|total| total.checked_add(output_bytes))
            .and_then(|total| total.checked_add(parameter_bytes))
            .ok_or_else(|| EngineError::Shape("Metal transient byte count overflows".into()))?;
        Ok(PreparedMappedMetalMatVec {
            dtype: operation.dtype,
            rows: operation.rows,
            columns: operation.columns,
            s_in_offset,
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
        let input_buffer = buffer_with_data(&self.device, as_bytes(input));
        let output_buffer = self
            .device
            .new_buffer(value_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params_buffer = buffer_with_data(&self.device, &params.encode());
        let transient_bytes = value_bytes
            .checked_mul(2)
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
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(MetalRmsNormParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::Shape("Metal gated RMSNorm transient bytes overflow".into())
            })?;
        Ok(PreparedMappedMetalGatedRmsNorm {
            rows,
            columns,
            mapping: mapping.clone(),
            weight_offset,
            input_buffer: buffer_with_data(&self.device, as_bytes(input)),
            gate_buffer: buffer_with_data(&self.device, as_bytes(gate)),
            output_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            transient_bytes,
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
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(MetalCausalConvParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal convolution transient bytes overflow".into())
            })?;
        Ok(PreparedMappedMetalCausalConv {
            channels,
            kernel,
            mapping: mapping.clone(),
            weight_offset,
            input_buffer: buffer_with_data(&self.device, as_bytes(input)),
            state_buffer: new_zeroed_buffer(&self.device, weight_bytes_expected)?,
            output_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            resident_state_bytes: weight_bytes_expected,
            transient_bytes,
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
            values_buffer,
            cosine_buffer,
            sine_buffer,
            params_buffer,
            transient_bytes,
        })
    }

    /// Allocate a verifier-only packed paged GQA cache. Device residency is
    /// fixed at preparation time so decode never grows or expands K/V tensors
    /// to f32. The CPU `PagedKvCache` mirror is temporary correctness plumbing
    /// and remains an explicit promotion blocker.
    pub fn prepare_paged_gqa_decode(
        &self,
        config: MetalPagedGqaConfig,
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
        let cache = PagedKvCache::new(
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
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(MetalPagedGqaParams::BYTE_LEN))
            .ok_or_else(|| {
                EngineError::MemoryBudget("Metal GQA transient bytes overflow".into())
            })?;

        Ok(PreparedMetalPagedGqa {
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
            page_to_q4_slot: vec![None; maximum_pages],
            free_q4_slots: (0..q4_slots).rev().collect(),
            q2_pages_buffer: new_zeroed_buffer(&self.device, q2_arena_bytes)?,
            q4_pages_buffer: new_zeroed_buffer(&self.device, q4_arena_bytes)?,
            descriptors_buffer: new_zeroed_buffer(&self.device, descriptor_bytes)?,
            query_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            output_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            params_buffer: new_zeroed_buffer(&self.device, MetalPagedGqaParams::BYTE_LEN)?,
            packed_device_bytes,
            transient_bytes,
            poisoned: false,
        })
    }

    /// Allocate one persistent FP16 GatedDelta recurrence state. Inputs and
    /// output are reusable f32 buffers; no f32 state shadow is retained.
    pub fn prepare_gated_delta_f16(
        &self,
        config: MetalGatedDeltaConfig,
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
        let transient_bytes = qk_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(value_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(head_bytes.checked_mul(2)?))
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
            query_buffer: new_zeroed_buffer(&self.device, qk_bytes)?,
            key_buffer: new_zeroed_buffer(&self.device, qk_bytes)?,
            value_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            log_decay_buffer: new_zeroed_buffer(&self.device, head_bytes)?,
            beta_buffer: new_zeroed_buffer(&self.device, head_bytes)?,
            state_buffer: new_zeroed_buffer(&self.device, resident_state_bytes)?,
            output_buffer: new_zeroed_buffer(&self.device, value_bytes)?,
            params_buffer: buffer_with_data(&self.device, &params.encode()),
            resident_state_bytes,
            transient_bytes,
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
                Some(&prepared.output_buffer),
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
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                prepared.rows,
            )
            .to_vec()
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
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-rms-norm-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_1p_pipeline);
        encoder.set_buffer(
            MetalRmsNormBufferAbi::INPUT as u64,
            Some(&prepared.input_buffer),
            0,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::WEIGHT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weight_offset,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::OUTPUT as u64,
            Some(&prepared.output_buffer),
            0,
        );
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
            slice::from_raw_parts(prepared.output_buffer.contents().cast::<f32>(), value_count)
                .to_vec()
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
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-gated-rms-norm-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_gated_pipeline);
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::INPUT as u64,
            Some(&prepared.input_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::GATE as u64,
            Some(&prepared.gate_buffer),
            0,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::WEIGHT as u64,
            Some(&prepared.mapping.inner.buffer),
            prepared.weight_offset,
        );
        encoder.set_buffer(
            MetalGatedRmsNormBufferAbi::OUTPUT as u64,
            Some(&prepared.output_buffer),
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
            slice::from_raw_parts(prepared.output_buffer.contents().cast::<f32>(), value_count)
                .to_vec()
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
        encoder.set_compute_pipeline_state(&self.partial_rope_pipeline);
        for operation in prepared {
            encoder.set_buffer(
                MetalPartialRopeBufferAbi::VALUES as u64,
                Some(&operation.values_buffer),
                0,
            );
            encoder.set_buffer(
                MetalPartialRopeBufferAbi::COSINE as u64,
                Some(&operation.cosine_buffer),
                0,
            );
            encoder.set_buffer(
                MetalPartialRopeBufferAbi::SINE as u64,
                Some(&operation.sine_buffer),
                0,
            );
            encoder.set_buffer(
                MetalPartialRopeBufferAbi::PARAMS as u64,
                Some(&operation.params_buffer),
                0,
            );
            let pair_count = operation
                .heads
                .checked_mul(operation.rotary_dim / 2)
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
                        operation.values_buffer.contents().cast::<f32>(),
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
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "Metal paged GQA state is poisoned; reset is required".into(),
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

    fn append_and_dispatch_paged_gqa_inner(
        &self,
        prepared: &mut PreparedMetalPagedGqa,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
        let update = prepared.cache.push(key, value)?;

        for page_index in update.demoted_pages {
            let page = metal_kv_page_snapshot(&prepared.cache, page_index)?;
            if page.precision != KvPrecision::Q2 {
                return Err(EngineError::InvalidState(
                    "Metal demoted KV page did not become Q2".into(),
                ));
            }
            write_buffer_range(
                &prepared.q2_pages_buffer,
                page_index
                    .checked_mul(prepared.q2_page_bytes)
                    .ok_or_else(|| EngineError::MemoryBudget("Metal Q2 slot overflows".into()))?,
                &page.bytes,
                prepared.q2_arena_bytes(),
            )?;
            let slot = prepared.page_to_q4_slot[page_index].take().ok_or_else(|| {
                EngineError::InvalidState("Metal demoted page has no Q4 arena slot".into())
            })?;
            zero_buffer_range(
                &prepared.q4_pages_buffer,
                slot.checked_mul(prepared.q4_page_bytes)
                    .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 slot overflows".into()))?,
                prepared.q4_page_bytes,
                prepared.q4_arena_bytes(),
            )?;
            prepared.free_q4_slots.push(slot);
        }

        let current = metal_kv_page_snapshot(&prepared.cache, update.page_index)?;
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
        write_buffer_range(
            &prepared.q4_pages_buffer,
            q4_slot
                .checked_mul(prepared.q4_page_bytes)
                .ok_or_else(|| EngineError::MemoryBudget("Metal Q4 slot overflows".into()))?,
            &current.bytes,
            prepared.q4_arena_bytes(),
        )?;

        let mut descriptor_words = Vec::with_capacity(
            prepared.cache.page_views().len() * (METAL_PAGED_KV_DESCRIPTOR_BYTES / 4),
        );
        for page in prepared.cache.page_views() {
            let (precision, slot) = match page.precision {
                KvPrecision::Q2 => (0_u32, page.page_index),
                KvPrecision::Q4 => (
                    1_u32,
                    prepared.page_to_q4_slot[page.page_index].ok_or_else(|| {
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
                u32::try_from(page.first_token)
                    .map_err(|_| EngineError::Shape("Metal KV token index exceeds u32".into()))?,
            ]);
        }
        let descriptor_bytes = as_bytes(&descriptor_words);
        write_buffer_range(
            &prepared.descriptors_buffer,
            0,
            descriptor_bytes,
            prepared.page_to_q4_slot.len() * METAL_PAGED_KV_DESCRIPTOR_BYTES,
        )?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                query.as_ptr(),
                prepared.query_buffer.contents().cast::<f32>(),
                query.len(),
            );
        }
        let page_count = prepared.cache.page_views().len();
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
            page_count: usize_to_u32(page_count, "Metal GQA page count")?,
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
        )?;

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-paged-q2q4-gqa-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.paged_gqa_decode_pipeline);
        encoder.set_buffer(
            MetalPagedGqaBufferAbi::QUERY as u64,
            Some(&prepared.query_buffer),
            0,
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
            Some(&prepared.output_buffer),
            0,
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
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                output_values,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal paged GQA produced a non-finite output".into(),
            ));
        }
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
        prepared.poisoned = true;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-gated-delta-f16-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gated_delta_f16_pipeline);
        for (binding, buffer) in [
            (MetalGatedDeltaBufferAbi::QUERY, &prepared.query_buffer),
            (MetalGatedDeltaBufferAbi::KEY, &prepared.key_buffer),
            (MetalGatedDeltaBufferAbi::VALUE, &prepared.value_buffer),
            (
                MetalGatedDeltaBufferAbi::LOG_DECAY,
                &prepared.log_decay_buffer,
            ),
            (MetalGatedDeltaBufferAbi::BETA, &prepared.beta_buffer),
            (MetalGatedDeltaBufferAbi::STATE, &prepared.state_buffer),
            (MetalGatedDeltaBufferAbi::OUTPUT, &prepared.output_buffer),
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
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                output_values,
            )
            .to_vec()
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
        prepared.poisoned = true;
        let thread_width = dispatch_width(&self.causal_conv_f16_pipeline, DEFAULT_SIMDGROUPS)?;
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-causal-conv-f16-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.causal_conv_f16_pipeline);
        encoder.set_buffer(
            MetalCausalConvBufferAbi::INPUT as u64,
            Some(&prepared.input_buffer),
            0,
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
            Some(&prepared.output_buffer),
            0,
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
            slice::from_raw_parts(
                prepared.output_buffer.contents().cast::<f32>(),
                prepared.channels,
            )
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

    /// Encode one decode-row RMSNorm followed by a recovered Q2/Q4
    /// projection in a single command encoder. The projection consumes the
    /// RMSNorm output buffer directly and therefore owns no second activation
    /// allocation and performs no host readback between operations.
    pub fn dispatch_mapped_rms_norm_then_projection(
        &self,
        norm: &PreparedMappedMetalRmsNorm,
        projection: &PreparedMappedMetalMatVec,
    ) -> Result<Vec<f32>> {
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

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-rmsnorm-projection-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_1p_pipeline);
        encoder.set_buffer(
            MetalRmsNormBufferAbi::INPUT as u64,
            Some(&norm.input_buffer),
            0,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::WEIGHT as u64,
            Some(&norm.mapping.inner.buffer),
            norm.weight_offset,
        );
        encoder.set_buffer(
            MetalRmsNormBufferAbi::OUTPUT as u64,
            Some(&norm.output_buffer),
            0,
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

        encoder.set_buffer(MetalBufferAbi::INPUT as u64, Some(&norm.output_buffer), 0);
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
                Some(&projection.output_buffer),
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
                "Metal norm/projection command ended with {:?}",
                command_buffer.status()
            )));
        }
        let output = unsafe {
            slice::from_raw_parts(
                projection.output_buffer.contents().cast::<f32>(),
                projection.rows,
            )
            .to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal norm/projection chain produced a non-finite output".into(),
            ));
        }
        Ok(output)
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

#[derive(Debug)]
struct MetalKvPageSnapshot {
    precision: KvPrecision,
    bytes: Vec<u8>,
}

fn metal_kv_page_snapshot(cache: &PagedKvCache, page_index: usize) -> Result<MetalKvPageSnapshot> {
    let page = cache
        .page_views()
        .find(|page| page.page_index == page_index)
        .ok_or_else(|| EngineError::InvalidState("Metal KV page is missing".into()))?;
    Ok(MetalKvPageSnapshot {
        precision: page.precision,
        bytes: page.bytes.to_vec(),
    })
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

fn zero_buffer_range(buffer: &Buffer, offset: usize, bytes: usize, capacity: usize) -> Result<()> {
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| EngineError::MemoryBudget("Metal zero range overflows".into()))?;
    if end > capacity {
        return Err(EngineError::MemoryBudget(format!(
            "Metal zero range ends at {end}, capacity is {capacity}"
        )));
    }
    unsafe {
        std::ptr::write_bytes(buffer.contents().cast::<u8>().add(offset), 0, bytes);
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
    use crate::backend::{Activation, Backend};
    use crate::format::{
        ArtifactBuilder, FileHeader, ModelManifest, PackedTensor, QuantSegment, TensorEntry,
        DEFAULT_ALIGNMENT, HEADER_BYTES,
    };
    use crate::loader::{ChecksumPolicy, ModelArtifact};
    use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
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

    fn aligned(value: usize) -> usize {
        value.div_ceil(DEFAULT_ALIGNMENT as usize) * DEFAULT_ALIGNMENT as usize
    }

    fn write_mixed_fixture(
        path: &std::path::Path,
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
                name: "matrix.weight".into(),
                dtype: TensorDType::MixedQ2Q4B64,
                shape: vec![(rows_q2 + rows_q4) as u64, columns as u64],
                offset: 0,
                length: weights.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&weights)),
                segments: segments.clone(),
            },
            TensorEntry {
                name: "matrix.weight.s_in".into(),
                dtype: TensorDType::F16,
                shape: vec![columns as u64],
                offset: s_in_offset as u64,
                length: s_in.len() as u64,
                sha256: format!("{:x}", Sha256::digest(&s_in)),
                segments: Vec::new(),
            },
            TensorEntry {
                name: "matrix.weight.s_out".into(),
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
    fn mapped_rms_norm_feeds_mixed_projection_without_host_intermediate() {
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
        norm.write_input(&vec![0.0; columns])
            .expect("update chained norm input");
        let zero = runtime
            .dispatch_mapped_rms_norm_then_projection(&norm, &projection)
            .expect("dispatch zero chained input");
        assert!(zero.iter().all(|value| value.abs() <= f32::EPSILON));
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
            2 * query_heads * head_dim * std::mem::size_of::<f32>() + MetalPagedGqaParams::BYTE_LEN
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
                .cache
                .flattened_key(key_value_heads, head_dim)
                .expect("flatten quantized keys");
            let cached_value = prepared
                .cache
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
            .page_views()
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
