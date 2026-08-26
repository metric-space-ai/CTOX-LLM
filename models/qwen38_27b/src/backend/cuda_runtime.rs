//! Direct CUDA Driver API execution for the unpromoted SM86 Q2/Q4 candidate.
//!
//! This verifier runtime deliberately does not implement [`super::Backend`].
//! It loads an explicitly supplied cubin, resolves the ABI symbols pinned in
//! [`super::cuda`], owns all device allocations, and fails closed on any
//! driver/shape/profile mismatch. There is no CUDA Runtime API, framework, or
//! CPU fallback in this path.

use std::cell::Cell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::size_of_val;
use std::ptr;
use std::rc::Rc;
use std::slice;
use std::time::Instant;

use libloading::Library;
use sha2::{Digest, Sha256};

use super::cuda::{
    validate_mixed_operation, validate_operation, validate_recovered_row, CudaMixedRowSegment,
    A8_BATCHED_QUANTIZE_SYMBOL, A8_QUANTIZE_SYMBOL, ARGMAX_F32_SYMBOL, CAUSAL_CONV_F16_SYMBOL,
    CAUSAL_CONV_SCAN_F16_SYMBOL, CUDA_SAMPLER_MAX_TOP_K, DEMOTE_PAGED_KV_Q4_TO_Q2_SYMBOL,
    GATED_DELTA_F16_SYMBOL, GATED_DELTA_HEADS, GATED_DELTA_KEY_DIM, GATED_DELTA_KEY_HEADS,
    GATED_DELTA_PREP_F32_SYMBOL, GATED_DELTA_PREP_SCAN_F32_SYMBOL, GATED_DELTA_SCAN_F16_SYMBOL,
    GATED_DELTA_STATE_BYTES, GATED_DELTA_VALUE_DIM, GATED_RMS_NORM_COLUMNS,
    GATED_RMS_NORM_F16_SYMBOL, GATED_RMS_NORM_ROWS, LINEAR_CONV_CHANNELS, LINEAR_CONV_KERNEL_WIDTH,
    LINEAR_CONV_STATE_BYTES, PACK_PAGED_KV_Q4_BATCH_F32_SYMBOL, PACK_PAGED_KV_Q4_F32_SYMBOL,
    PAGED_GQA_DESCRIPTOR_BYTES, PAGED_GQA_PARAMS_BYTES, PAGED_GQA_SPLIT_MAX_QUERY_TOKENS,
    PAGED_GQA_SPLIT_SEGMENTS, PAGED_Q2Q4_GQA_F32_SYMBOL, PAGED_Q2Q4_GQA_PREFILL_F32_SYMBOL,
    PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_SYMBOL, PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_SYMBOL,
    PARTIAL_ROPE_BATCH_F32_SYMBOL, PARTIAL_ROPE_F32_SYMBOL, Q2_B64_A8_BATCHED_MATMUL_SYMBOL,
    Q2_B64_A8_BATCHED_MMQ_SYMBOL, Q2_B64_A8_GATHERED_MATVEC_SYMBOL, Q2_B64_A8_MATVEC_SYMBOL,
    Q2_B64_FUSED_MATVEC, Q2_B64_RECOVERED_ROWS_SYMBOL, Q2_B64_RECOVERED_ROW_SYMBOL,
    Q4_B64_A8_BATCHED_MATMUL_SYMBOL, Q4_B64_A8_BATCHED_MMQ_SYMBOL,
    Q4_B64_A8_GATHERED_MATVEC_SYMBOL, Q4_B64_A8_MATVEC_SYMBOL, Q4_B64_FUSED_MATVEC,
    Q4_B64_RECOVERED_ROWS_SYMBOL, Q4_B64_RECOVERED_ROW_SYMBOL,
    QUERY_GATE_NORM_ROPE_BATCH_F32_SYMBOL, QUERY_GATE_NORM_ROPE_F32_SYMBOL,
    QWEN_RMS_NORM_F16_SYMBOL, RESIDUAL_RMS_NORM_F16_SYMBOL, ROPE_TABLE_BATCH_F32_SYMBOL,
    SIGMOID_GATE_A8_QUANTIZE_SYMBOL, SWIGLU_A8_QUANTIZE_SYMBOL, TOPK_TOPP_SAMPLE_F32_SYMBOL,
};
use super::{Activation, FusedMatVec, RecoveredRow, ScaleSlice};
use crate::format::TensorDType;
use crate::kv_cache::KvPrecision;
use crate::loader::RecoveredMatrixView;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
use crate::sampler::SamplerConfig;
use crate::{EngineError, Result};

type CuResult = i32;
type CuDevice = i32;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuDevicePtr = u64;

const CUDA_SUCCESS: CuResult = 0;
const THREADS_PER_BLOCK: u32 = 128;
const LINEAR_THREADS_PER_BLOCK: u32 = 256;
const WARP_SIZE: u32 = 32;
const A8_ROWS_PER_BLOCK: u32 = (THREADS_PER_BLOCK / WARP_SIZE) * 2;
const MMQ_THREADS_PER_BLOCK: u32 = 256;
const MMQ_ROWS_PER_BLOCK: u32 = 128;
const MMQ_BATCH_ROWS_PER_BLOCK: u32 = 64;
const CUDA_GRID_Y_MAX: u32 = 65_535;

type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, i32, CuDevice) -> CuResult;
type CuDeviceComputeCapability = unsafe extern "C" fn(*mut i32, *mut i32, CuDevice) -> CuResult;
type CuCtxCreate = unsafe extern "C" fn(*mut CuContext, u32, CuDevice) -> CuResult;
type CuCtxSetCurrent = unsafe extern "C" fn(CuContext) -> CuResult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CuResult;
type CuCtxDestroy = unsafe extern "C" fn(CuContext) -> CuResult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult;
type CuModuleGetFunction =
    unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult;
type CuModuleUnload = unsafe extern "C" fn(CuModule) -> CuResult;
type CuMemAlloc = unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult;
type CuMemFree = unsafe extern "C" fn(CuDevicePtr) -> CuResult;
type CuMemsetD8 = unsafe extern "C" fn(CuDevicePtr, u8, usize) -> CuResult;
type CuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CuResult;
type CuMemcpyHtoD = unsafe extern "C" fn(CuDevicePtr, *const c_void, usize) -> CuResult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize) -> CuResult;
type CuMemcpyDtoD = unsafe extern "C" fn(CuDevicePtr, CuDevicePtr, usize) -> CuResult;
type CuMemcpy2D = unsafe extern "C" fn(*const CudaMemcpy2DDescriptor) -> CuResult;
type CuLaunchKernel = unsafe extern "C" fn(
    CuFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CuStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CuResult;
type CuGetErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuGetErrorString = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;

const CU_MEMORYTYPE_DEVICE: u32 = 2;

/// CUDA Driver API `CUDA_MEMCPY2D`. All unused host/array fields remain null;
/// the model runtime admits device-to-device row copies only.
#[repr(C)]
struct CudaMemcpy2DDescriptor {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: u32,
    src_host: *const c_void,
    src_device: CuDevicePtr,
    src_array: *mut c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: u32,
    dst_host: *mut c_void,
    dst_device: CuDevicePtr,
    dst_array: *mut c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct F32Copy2DGeometry {
    source_row_values: usize,
    destination_row_values: usize,
    destination_column: usize,
    rows: usize,
    columns: usize,
}

impl F32Copy2DGeometry {
    fn validate(self, source_values: usize, destination_values: usize) -> Result<()> {
        if self.rows == 0
            || self.columns == 0
            || self.columns > self.source_row_values
            || self
                .destination_column
                .checked_add(self.columns)
                .is_none_or(|end| end > self.destination_row_values)
        {
            return Err(EngineError::Shape(
                "CUDA f32 2-D copy has empty or invalid row geometry".into(),
            ));
        }
        let source_end = self
            .rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(self.source_row_values))
            .and_then(|offset| offset.checked_add(self.columns))
            .ok_or_else(|| EngineError::Shape("CUDA f32 2-D source shape overflows".into()))?;
        let destination_end = self
            .rows
            .checked_sub(1)
            .and_then(|row| row.checked_mul(self.destination_row_values))
            .and_then(|offset| offset.checked_add(self.destination_column))
            .and_then(|offset| offset.checked_add(self.columns))
            .ok_or_else(|| EngineError::Shape("CUDA f32 2-D destination shape overflows".into()))?;
        if source_end > source_values || destination_end > destination_values {
            return Err(EngineError::Shape(format!(
                "CUDA f32 2-D copy touches source/destination {source_end}/{destination_end} values, available {source_values}/{destination_values}"
            )));
        }
        Ok(())
    }
}

struct CudaDriver {
    _library: Library,
    init: CuInit,
    device_get: CuDeviceGet,
    device_get_name: CuDeviceGetName,
    device_compute_capability: CuDeviceComputeCapability,
    ctx_create: CuCtxCreate,
    ctx_set_current: CuCtxSetCurrent,
    ctx_synchronize: CuCtxSynchronize,
    ctx_destroy: CuCtxDestroy,
    module_load_data: CuModuleLoadData,
    module_get_function: CuModuleGetFunction,
    module_unload: CuModuleUnload,
    mem_alloc: CuMemAlloc,
    mem_free: CuMemFree,
    memset_d8: CuMemsetD8,
    mem_get_info: CuMemGetInfo,
    memcpy_htod: CuMemcpyHtoD,
    memcpy_dtoh: CuMemcpyDtoH,
    memcpy_dtod: CuMemcpyDtoD,
    memcpy_2d: CuMemcpy2D,
    launch_kernel: CuLaunchKernel,
    get_error_name: CuGetErrorName,
    get_error_string: CuGetErrorString,
}

impl CudaDriver {
    fn load() -> Result<Self> {
        let library = load_driver_library()?;
        // SAFETY: every copied pointer is resolved from the process CUDA
        // driver and the Library is retained in the returned struct.
        unsafe {
            Ok(Self {
                init: symbol(&library, b"cuInit\0")?,
                device_get: symbol(&library, b"cuDeviceGet\0")?,
                device_get_name: symbol(&library, b"cuDeviceGetName\0")?,
                device_compute_capability: symbol(&library, b"cuDeviceComputeCapability\0")?,
                ctx_create: symbol(&library, b"cuCtxCreate_v2\0")?,
                ctx_set_current: symbol(&library, b"cuCtxSetCurrent\0")?,
                ctx_synchronize: symbol(&library, b"cuCtxSynchronize\0")?,
                ctx_destroy: symbol(&library, b"cuCtxDestroy_v2\0")?,
                module_load_data: symbol(&library, b"cuModuleLoadData\0")?,
                module_get_function: symbol(&library, b"cuModuleGetFunction\0")?,
                module_unload: symbol(&library, b"cuModuleUnload\0")?,
                mem_alloc: symbol(&library, b"cuMemAlloc_v2\0")?,
                mem_free: symbol(&library, b"cuMemFree_v2\0")?,
                memset_d8: symbol(&library, b"cuMemsetD8_v2\0")?,
                mem_get_info: symbol(&library, b"cuMemGetInfo_v2\0")?,
                memcpy_htod: symbol(&library, b"cuMemcpyHtoD_v2\0")?,
                memcpy_dtoh: symbol(&library, b"cuMemcpyDtoH_v2\0")?,
                memcpy_dtod: symbol(&library, b"cuMemcpyDtoD_v2\0")?,
                memcpy_2d: symbol(&library, b"cuMemcpy2D_v2\0")?,
                launch_kernel: symbol(&library, b"cuLaunchKernel\0")?,
                get_error_name: symbol(&library, b"cuGetErrorName\0")?,
                get_error_string: symbol(&library, b"cuGetErrorString\0")?,
                _library: library,
            })
        }
    }

    fn check(&self, result: CuResult, operation: &'static str) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name = ptr::null();
        let mut description = ptr::null();
        // SAFETY: CUDA owns both returned static strings. Failed diagnostic
        // lookups leave null pointers, which are handled below.
        unsafe {
            (self.get_error_name)(result, &mut name);
            (self.get_error_string)(result, &mut description);
        }
        let name = unsafe { nullable_cstr(name) }.unwrap_or("CUDA_ERROR_UNKNOWN");
        let description = unsafe { nullable_cstr(description) }.unwrap_or("no driver message");
        Err(EngineError::InvalidState(format!(
            "CUDA {operation} failed with {name} ({result}): {description}"
        )))
    }
}

/// Verifier-only CUDA context and loaded SM86 module.
pub struct CudaCandidateRuntime {
    inner: Rc<CudaContextInner>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaSubmissionStats {
    pub token_submission_attempts: u64,
    pub token_submission_commits: u64,
    pub deferred_operator_synchronizations: u64,
    pub context_synchronizations: u64,
    pub device_argmax_launches: u64,
    pub device_sampling_launches: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaSampledToken {
    pub token: u32,
    pub nucleus_len: u32,
    pub nucleus_total: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaSplitGqaBenchmark {
    pub iterations: usize,
    pub sequential_full_context_microseconds: f64,
    pub split_causal_microseconds: f64,
    pub speedup: f64,
}

struct CudaContextInner {
    driver: CudaDriver,
    context: CuContext,
    module: CuModule,
    q2_function: CuFunction,
    q4_function: CuFunction,
    a8_quantize_function: CuFunction,
    swiglu_a8_quantize_function: CuFunction,
    sigmoid_gate_a8_quantize_function: CuFunction,
    q2_a8_function: CuFunction,
    q4_a8_function: CuFunction,
    a8_batched_quantize_function: CuFunction,
    q2_a8_batched_function: CuFunction,
    q4_a8_batched_function: CuFunction,
    q2_a8_batched_mmq_function: CuFunction,
    q4_a8_batched_mmq_function: CuFunction,
    q2_a8_gathered_function: CuFunction,
    q4_a8_gathered_function: CuFunction,
    q2_recovered_row_function: CuFunction,
    q4_recovered_row_function: CuFunction,
    q2_recovered_rows_function: CuFunction,
    q4_recovered_rows_function: CuFunction,
    gated_delta_prep_f32_function: CuFunction,
    gated_delta_prep_scan_f32_function: CuFunction,
    gated_delta_f16_function: CuFunction,
    gated_delta_scan_f16_function: CuFunction,
    causal_conv_f16_function: CuFunction,
    causal_conv_scan_f16_function: CuFunction,
    gated_rms_norm_f16_function: CuFunction,
    qwen_rms_norm_f16_function: CuFunction,
    residual_rms_norm_f16_function: CuFunction,
    partial_rope_f32_function: CuFunction,
    rope_table_batch_f32_function: CuFunction,
    partial_rope_batch_f32_function: CuFunction,
    query_gate_norm_rope_f32_function: CuFunction,
    query_gate_norm_rope_batch_f32_function: CuFunction,
    pack_paged_kv_q4_f32_function: CuFunction,
    pack_paged_kv_q4_batch_f32_function: CuFunction,
    demote_paged_kv_q4_to_q2_function: CuFunction,
    paged_q2q4_gqa_f32_function: CuFunction,
    paged_q2q4_gqa_prefill_f32_function: CuFunction,
    paged_q2q4_gqa_split_partial_f32_function: CuFunction,
    paged_q2q4_gqa_split_combine_f32_function: CuFunction,
    argmax_f32_function: CuFunction,
    topk_topp_sample_f32_function: CuFunction,
    device_name: String,
    compute_capability: (u32, u32),
    token_submission_active: Cell<bool>,
    token_submission_attempts: Cell<u64>,
    token_submission_commits: Cell<u64>,
    deferred_operator_synchronizations: Cell<u64>,
    context_synchronizations: Cell<u64>,
    device_argmax_launches: Cell<u64>,
    device_sampling_launches: Cell<u64>,
}

/// Borrowed, context-bound f32 device tensor. The private buffer reference
/// prevents a raw CUDA pointer from outliving its allocation while allowing
/// model-specific prepared operators to feed one another without host copies.
#[derive(Clone, Copy)]
pub struct CudaDeviceF32View<'a> {
    context: &'a Rc<CudaContextInner>,
    buffer: &'a DeviceBuffer,
    offset_values: usize,
    values: usize,
}

impl<'a> CudaDeviceF32View<'a> {
    pub fn values(&self) -> usize {
        self.values
    }

    pub fn slice(&self, offset_values: usize, values: usize) -> Result<Self> {
        let absolute_offset = self
            .offset_values
            .checked_add(offset_values)
            .ok_or_else(|| EngineError::Shape("CUDA f32 subview offset overflows".into()))?;
        let end = offset_values
            .checked_add(values)
            .ok_or_else(|| EngineError::Shape("CUDA f32 subview shape overflows".into()))?;
        if values == 0 || end > self.values {
            return Err(EngineError::Shape(format!(
                "CUDA f32 subview ends at {end} values, parent has {}",
                self.values
            )));
        }
        Ok(Self {
            context: self.context,
            buffer: self.buffer,
            offset_values: absolute_offset,
            values,
        })
    }

    fn ptr(&self) -> Result<CuDevicePtr> {
        device_ptr_offset(
            self.buffer.ptr(),
            self.offset_values
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| EngineError::Shape("CUDA f32 view offset overflows".into()))?,
        )
    }
}

/// Explicit host-upload staging used only by CUDA numerical verifiers. The
/// production graph obtains [`CudaDeviceF32View`] values from prepared model
/// operators and never constructs this owner.
pub struct CudaVerifierF32Tensor {
    buffer: DeviceBuffer,
    values: usize,
}

/// Reusable device-only concatenation buffer for the exact MTP
/// `[normalized_embedding, normalized_hidden]` input. It introduces no host
/// tensor and owns no model data.
pub struct PreparedCudaF32Concat {
    context: Rc<CudaContextInner>,
    left_values: usize,
    right_values: usize,
    output: DeviceBuffer,
}

/// One bounded device-side snapshot used by replay-on-reject. The snapshot
/// never crosses host memory and is valid for exactly one speculative branch.
pub struct PreparedCudaF32Checkpoint {
    context: Rc<CudaContextInner>,
    values: usize,
    snapshot: DeviceBuffer,
    valid: bool,
}

/// Device-resident buffers for one pure Q2 or Q4 projection. Immutable model
/// and recovery buffers remain allocated across repeated token dispatches.
pub struct PreparedCudaMatVec {
    context: Rc<CudaContextInner>,
    dtype: TensorDType,
    rows: u32,
    columns: u32,
    activation: u32,
    weights: DeviceBuffer,
    input: DeviceBuffer,
    s_in: Option<DeviceBuffer>,
    s_out: Option<DeviceBuffer>,
    bias: Option<DeviceBuffer>,
    output: DeviceBuffer,
    resident_bytes: usize,
}

/// Explicit A8 activation buffers paired with the same immutable logical
/// Q2/Q4 weights. The activation codes are transient per input and never
/// become part of a backend-specific model artifact.
pub struct PreparedCudaA8MatVec {
    base: PreparedCudaMatVec,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
}

/// One canonical mixed Q2/Q4 payload with shared transient A8 activation.
/// Segment metadata points into `weights`; no segment is copied or repacked.
pub struct PreparedCudaMixedA8MatVec {
    context: Rc<CudaContextInner>,
    rows: u32,
    columns: u32,
    activation: u32,
    segments: Vec<CudaMixedRowSegment>,
    weights: DeviceBuffer,
    input: DeviceBuffer,
    s_in: Option<DeviceBuffer>,
    s_out: Option<DeviceBuffer>,
    bias: Option<DeviceBuffer>,
    output: DeviceBuffer,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
}

/// Multi-token projection owner for the CUDA prefill verifier baseline. One
/// row-major activation batch is quantized in a single 2-D launch and consumes
/// the same immutable pure or mixed Q2/Q4 payload as decode. This establishes
/// the graph/storage ABI; production promotion still requires the SM86 MMQ
/// tile implementation and roofline evidence.
pub struct PreparedCudaBatchedA8MatMul {
    context: Rc<CudaContextInner>,
    batch_rows: u32,
    rows: u32,
    columns: u32,
    activation: u32,
    layout: CudaA8ProjectionLayout,
    weights: DeviceBuffer,
    input: DeviceBuffer,
    s_in: Option<DeviceBuffer>,
    s_out: Option<DeviceBuffer>,
    bias: Option<DeviceBuffer>,
    output: DeviceBuffer,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct A8CorrectionIdentity {
    columns: u32,
    s_in_sha256: [u8; 32],
}

/// One corrected input and transient A8 encoding shared by an exact Qwen
/// fan-out group. It owns no matrix weights or output buffers.
pub struct PreparedCudaA8Activation {
    context: Rc<CudaContextInner>,
    columns: u32,
    correction_identity: A8CorrectionIdentity,
    input: Option<DeviceBuffer>,
    s_in: Option<DeviceBuffer>,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
}

/// Device-only A8 workspace for one row-major prefill chunk. It owns no model
/// weights and can therefore be shared by every projection whose packed
/// `s_in` identity matches. Only the active prefix of `batch_capacity` is
/// touched by a dispatch.
pub struct PreparedCudaBatchedA8Activation {
    context: Rc<CudaContextInner>,
    batch_capacity: u32,
    columns: u32,
    correction_identity: A8CorrectionIdentity,
    s_in: Option<DeviceBuffer>,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
}

/// Maximum-width A8 scratch shared across recovery groups. Immutable `s_in`
/// remains in [`PreparedCudaA8Activation`]; dispatch validates that owner
/// against every projection before overwriting this arena.
pub struct PreparedCudaBatchedA8Workspace {
    context: Rc<CudaContextInner>,
    batch_capacity: u32,
    column_capacity: u32,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    transient_bytes: usize,
}

enum CudaA8ProjectionLayout {
    Pure(TensorDType),
    Mixed(Vec<CudaMixedRowSegment>),
}

/// Matrix-local state for a projection that consumes a separately owned,
/// identity-checked [`PreparedCudaA8Activation`].
pub struct PreparedCudaA8Projection {
    context: Rc<CudaContextInner>,
    dtype: TensorDType,
    rows: u32,
    columns: u32,
    activation: u32,
    correction_identity: A8CorrectionIdentity,
    layout: CudaA8ProjectionLayout,
    weights: DeviceBuffer,
    s_out: Option<DeviceBuffer>,
    bias: Option<DeviceBuffer>,
    output: DeviceBuffer,
    resident_bytes: usize,
}

/// Transient row-major output for a batched projection. Immutable weights and
/// recovery scales remain in [`PreparedCudaA8Projection`], so growing or
/// replacing a prefill workspace never duplicates model state.
pub struct PreparedCudaBatchedA8Output {
    context: Rc<CudaContextInner>,
    batch_capacity: u32,
    rows: u32,
    output: DeviceBuffer,
    resident_bytes: usize,
}

/// Four fixed-address row-major output slots for the frozen Qwen prefill
/// fan-outs. A slot may host a narrower projection; only its active compact
/// prefix is exposed to downstream operators.
pub struct PreparedCudaBatchedA8OutputArena {
    context: Rc<CudaContextInner>,
    batch_capacity: u32,
    slot_rows: [u32; 4],
    slot_offsets: [usize; 4],
    output: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaGatheredRowGroup {
    dtype: TensorDType,
    weight_offset: usize,
    row_id_offset: usize,
    output_offset: usize,
    scale_row_offset: usize,
    row_count: u32,
}

fn build_gathered_row_plan(
    layout: &CudaA8ProjectionLayout,
    projection_rows: u32,
    row_ids: &[u32],
) -> Result<(Vec<u32>, Vec<CudaGatheredRowGroup>)> {
    if row_ids.is_empty()
        || row_ids.windows(2).any(|pair| pair[0] >= pair[1])
        || row_ids.last().is_some_and(|row| *row >= projection_rows)
    {
        return Err(EngineError::Shape(
            "gathered CUDA row IDs must be non-empty, canonical, and in range".into(),
        ));
    }
    let mut local_ids = Vec::with_capacity(row_ids.len());
    let mut groups = Vec::new();
    match layout {
        CudaA8ProjectionLayout::Pure(dtype) => {
            local_ids.extend_from_slice(row_ids);
            groups.push(CudaGatheredRowGroup {
                dtype: *dtype,
                weight_offset: 0,
                row_id_offset: 0,
                output_offset: 0,
                scale_row_offset: 0,
                row_count: u32::try_from(row_ids.len()).map_err(|_| {
                    EngineError::Shape("gathered CUDA row count exceeds u32".into())
                })?,
            });
        }
        CudaA8ProjectionLayout::Mixed(segments) => {
            let mut requested = 0_usize;
            for segment in segments {
                let segment_end = segment
                    .row_start
                    .checked_add(segment.row_count)
                    .ok_or_else(|| {
                        EngineError::Shape("gathered CUDA segment row range overflows".into())
                    })?;
                let first = local_ids.len();
                while requested < row_ids.len() && row_ids[requested] < segment_end {
                    let row = row_ids[requested];
                    if row < segment.row_start {
                        return Err(EngineError::Shape(
                            "gathered CUDA row precedes its mixed segment".into(),
                        ));
                    }
                    local_ids.push(row - segment.row_start);
                    requested += 1;
                }
                let count = local_ids.len() - first;
                if count != 0 {
                    groups.push(CudaGatheredRowGroup {
                        dtype: segment.descriptor.dtype,
                        weight_offset: segment.weight_offset,
                        row_id_offset: first * std::mem::size_of::<u32>(),
                        output_offset: first * std::mem::size_of::<f32>(),
                        scale_row_offset: segment.row_start as usize,
                        row_count: u32::try_from(count).map_err(|_| {
                            EngineError::Shape("gathered CUDA segment count exceeds u32".into())
                        })?,
                    });
                }
            }
            if requested != row_ids.len() {
                return Err(EngineError::Shape(
                    "gathered CUDA mixed segments do not cover every row ID".into(),
                ));
            }
        }
    }
    Ok((local_ids, groups))
}

/// Release-bound subset of one resident LM head. It owns only canonical local
/// row IDs plus compact logits; weights and A8 activation stay shared with the
/// complete target projection.
pub struct PreparedCudaGatheredA8Projection {
    context: Rc<CudaContextInner>,
    rows: u32,
    columns: u32,
    groups: Vec<CudaGatheredRowGroup>,
    row_ids: DeviceBuffer,
    output: DeviceBuffer,
    resident_bytes: usize,
}

/// Eight-byte result buffer for deterministic device-side greedy selection.
pub struct PreparedCudaArgmax {
    context: Rc<CudaContextInner>,
    result: DeviceBuffer,
}

/// Bounded top-k/top-p workspace and result. It never owns logits or RNG
/// state; both remain part of the caller's graph/session contract.
pub struct PreparedCudaTopKTopPSampler {
    context: Rc<CudaContextInner>,
    max_values: usize,
    scratch: DeviceBuffer,
    result: DeviceBuffer,
}

/// One loader-resolved embedding row with both recovery corrections and its
/// decoded activation resident on the device.
pub struct PreparedCudaRecoveredRow {
    context: Rc<CudaContextInner>,
    dtype: TensorDType,
    columns: u32,
    s_out: f32,
    weights: DeviceBuffer,
    s_in: DeviceBuffer,
    output: DeviceBuffer,
    resident_bytes: usize,
}

/// Complete resident embedding table using the canonical packed Q2/Q4 codes.
/// A token dispatch selects one row inside this allocation and reuses the
/// recovered-row kernel; no per-token weight upload or backend repacking is
/// performed.
pub struct PreparedCudaEmbedding {
    context: Rc<CudaContextInner>,
    rows: u32,
    columns: u32,
    layout: CudaA8ProjectionLayout,
    weights: DeviceBuffer,
    s_in: DeviceBuffer,
    s_out: DeviceBuffer,
    output: DeviceBuffer,
    model_bytes: usize,
    graph_bytes: usize,
}

/// Token-major embedding output and device row-ID list shared by every
/// bounded prefill chunk. Immutable table data stays in
/// [`PreparedCudaEmbedding`].
pub struct PreparedCudaBatchedEmbeddingWorkspace {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    columns: u32,
    row_ids: DeviceBuffer,
    output: DeviceBuffer,
    transient_bytes: usize,
}

/// Exact Qwen3.8-27B recurrence geometry accepted by the CUDA verifier.
/// Dynamic shapes remain rejected until their own kernel profile exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaGatedDeltaConfig {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub epsilon: f32,
}

impl CudaGatedDeltaConfig {
    pub const QWEN38_27B: Self = Self {
        heads: GATED_DELTA_HEADS,
        key_dim: GATED_DELTA_KEY_DIM,
        value_dim: GATED_DELTA_VALUE_DIM,
        epsilon: 1.0e-6,
    };
}

/// Persistent FP16 GatedDeltaNet state with reusable f32 step buffers. No
/// host or device FP32 state shadow is retained.
pub struct PreparedCudaGatedDelta {
    context: Rc<CudaContextInner>,
    config: CudaGatedDeltaConfig,
    query: DeviceBuffer,
    key: DeviceBuffer,
    value: DeviceBuffer,
    log_decay: DeviceBuffer,
    beta: DeviceBuffer,
    state: DeviceBuffer,
    checkpoint: DeviceBuffer,
    output: DeviceBuffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
    checkpoint_valid: bool,
}

/// Bounded token-major output for a multi-token GatedDelta prefill scan. The
/// persistent FP16 recurrence and speculative checkpoint remain owned exactly
/// once by [`PreparedCudaGatedDelta`].
pub struct PreparedCudaGatedDeltaScanOutput {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    heads: u32,
    value_dim: u32,
    output: DeviceBuffer,
    transient_bytes: usize,
}

/// Resident model parameters and reusable outputs for the Qwen-specific
/// compact-Q/K and A/B preparation stage preceding GatedDeltaNet.
pub struct PreparedCudaGatedDeltaInputs {
    context: Rc<CudaContextInner>,
    a_log: DeviceBuffer,
    dt_bias: DeviceBuffer,
    query: DeviceBuffer,
    key: DeviceBuffer,
    log_decay: DeviceBuffer,
    beta: DeviceBuffer,
    model_bytes: usize,
    transient_bytes: usize,
}

/// Bounded token-major Q/K and decay/beta workspace sharing the immutable
/// parameters owned by [`PreparedCudaGatedDeltaInputs`].
pub struct PreparedCudaGatedDeltaScanInputs {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    query: DeviceBuffer,
    key: DeviceBuffer,
    value: DeviceBuffer,
    log_decay: DeviceBuffer,
    beta: DeviceBuffer,
    transient_bytes: usize,
}

pub struct CudaGatedDeltaInputViews<'a> {
    pub query: CudaDeviceF32View<'a>,
    pub key: CudaDeviceF32View<'a>,
    pub log_decay: CudaDeviceF32View<'a>,
    pub beta: CudaDeviceF32View<'a>,
}

pub struct CudaGatedDeltaScanInputViews<'a> {
    pub query: CudaDeviceF32View<'a>,
    pub key: CudaDeviceF32View<'a>,
    pub value: CudaDeviceF32View<'a>,
    pub log_decay: CudaDeviceF32View<'a>,
    pub beta: CudaDeviceF32View<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaCausalConvConfig {
    pub channels: usize,
    pub kernel_width: usize,
}

impl CudaCausalConvConfig {
    pub const QWEN38_27B: Self = Self {
        channels: LINEAR_CONV_CHANNELS,
        kernel_width: LINEAR_CONV_KERNEL_WIDTH,
    };
}

pub struct PreparedCudaCausalConv {
    context: Rc<CudaContextInner>,
    config: CudaCausalConvConfig,
    input: DeviceBuffer,
    weight: DeviceBuffer,
    state: DeviceBuffer,
    checkpoint: DeviceBuffer,
    output: DeviceBuffer,
    model_bytes: usize,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
    checkpoint_valid: bool,
}

/// Bounded token-major output for a causal-convolution prefill scan. The
/// resident FP16 weight, state, and speculative checkpoint remain owned once
/// by [`PreparedCudaCausalConv`].
pub struct PreparedCudaCausalConvScanOutput {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    channels: u32,
    output: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaGatedRmsNormConfig {
    pub rows: usize,
    pub columns: usize,
    pub epsilon: f32,
}

impl CudaGatedRmsNormConfig {
    pub const QWEN38_27B: Self = Self {
        rows: GATED_RMS_NORM_ROWS,
        columns: GATED_RMS_NORM_COLUMNS,
        epsilon: 1.0e-6,
    };
}

pub struct PreparedCudaGatedRmsNorm {
    context: Rc<CudaContextInner>,
    config: CudaGatedRmsNormConfig,
    input: DeviceBuffer,
    gate: DeviceBuffer,
    weight: DeviceBuffer,
    output: DeviceBuffer,
    model_bytes: usize,
    transient_bytes: usize,
}

/// Token-major output for batched GatedDelta RMSNorm. The immutable FP16
/// weight remains in [`PreparedCudaGatedRmsNorm`].
pub struct PreparedCudaBatchedGatedRmsNormOutput {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    heads: u32,
    columns: u32,
    output: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaRmsNormConfig {
    pub rows: usize,
    pub columns: usize,
    pub epsilon: f32,
}

pub struct PreparedCudaRmsNorm {
    context: Rc<CudaContextInner>,
    config: CudaRmsNormConfig,
    input: DeviceBuffer,
    weight: DeviceBuffer,
    output: DeviceBuffer,
    model_bytes: usize,
    transient_bytes: usize,
}

pub struct PreparedCudaResidualRmsNorm {
    context: Rc<CudaContextInner>,
    config: CudaRmsNormConfig,
    weight: DeviceBuffer,
    residual_output: DeviceBuffer,
    normalized_output: DeviceBuffer,
    model_bytes: usize,
    transient_bytes: usize,
}

/// Two reusable f32 buffers for layer-major prefill normalization. Norm
/// weights remain in their resident per-layer owners; this workspace is
/// overwritten as the schedule advances and therefore scales with chunk size
/// only once rather than once per layer.
pub struct PreparedCudaBatchedRmsNormWorkspace {
    context: Rc<CudaContextInner>,
    batch_capacity: u32,
    columns: u32,
    residual_output: DeviceBuffer,
    normalized_output: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaPartialRopeConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
}

pub struct PreparedCudaPartialRope {
    context: Rc<CudaContextInner>,
    config: CudaPartialRopeConfig,
    values: DeviceBuffer,
    cosine: DeviceBuffer,
    sine: DeviceBuffer,
    transient_bytes: usize,
}

/// Shared prompt-chunk RoPE table. Query and key operators borrow the same
/// position/frequency values; only the active token prefix is initialized.
pub struct PreparedCudaBatchedRopeWorkspace {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    rotary_dim: u32,
    theta: f32,
    cosine: DeviceBuffer,
    sine: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaQueryGateConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
    pub epsilon: f32,
}

impl CudaQueryGateConfig {
    pub const QWEN38_27B: Self = Self {
        heads: 24,
        head_dim: 256,
        rotary_dim: 64,
        theta: 10_000_000.0,
        epsilon: 1.0e-6,
    };
}

pub struct PreparedCudaQueryGate {
    context: Rc<CudaContextInner>,
    config: CudaQueryGateConfig,
    q_norm_weight: DeviceBuffer,
    cosine: DeviceBuffer,
    sine: DeviceBuffer,
    query: DeviceBuffer,
    gate: DeviceBuffer,
    model_bytes: usize,
    transient_bytes: usize,
}

/// Shared token-major outputs for one full-attention prompt chunk. Q-norm
/// weights remain in the resident per-layer [`PreparedCudaQueryGate`] owner.
pub struct PreparedCudaBatchedQueryGateOutput {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    heads: u32,
    head_dim: u32,
    query: DeviceBuffer,
    gate: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPagedGqaConfig {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub maximum_tokens: usize,
    pub page_tokens: usize,
    pub sink_tokens: usize,
    pub recent_tokens: usize,
}

pub struct PreparedCudaPagedGqa {
    context: Rc<CudaContextInner>,
    config: CudaPagedGqaConfig,
    q2_token_bytes: usize,
    q4_token_bytes: usize,
    q2_page_bytes: usize,
    q4_page_bytes: usize,
    q4_slots: usize,
    component_values: usize,
    combined_values: usize,
    blocks_per_token: usize,
    tokens: usize,
    pages: Vec<CudaPagedKvPage>,
    free_q4_slots: Vec<usize>,
    q2_pages: DeviceBuffer,
    q4_pages: DeviceBuffer,
    descriptors: DeviceBuffer,
    query: DeviceBuffer,
    key: DeviceBuffer,
    value: DeviceBuffer,
    output: DeviceBuffer,
    params: DeviceBuffer,
    packed_device_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
    speculative_checkpoint: Option<CudaPagedGqaCheckpoint>,
}

/// Fixed-address scratch for the unpromoted five-query split-KV verifier.
/// It borrows the canonical cache at dispatch time and never owns or widens
/// persistent K/V pages.
pub struct PreparedCudaSplitPagedGqa {
    context: Rc<CudaContextInner>,
    query: DeviceBuffer,
    output: DeviceBuffer,
    partial_output: DeviceBuffer,
    partial_maximum: DeviceBuffer,
    partial_denominator: DeviceBuffer,
    transient_bytes: usize,
}

/// Bounded token-major output for the direct causal paged-GQA prefill
/// candidate. Persistent mixed Q2/Q4 pages remain owned by
/// [`PreparedCudaPagedGqa`].
pub struct PreparedCudaPagedGqaPrefillOutput {
    context: Rc<CudaContextInner>,
    token_capacity: u32,
    query_heads: u32,
    head_dim: u32,
    output: DeviceBuffer,
    transient_bytes: usize,
}

#[derive(Clone)]
struct CudaPagedGqaCheckpoint {
    tokens: usize,
    pages: Vec<CudaPagedKvPage>,
    free_q4_slots: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
struct CudaPagedKvPage {
    precision: KvPrecision,
    physical_slot: usize,
    tokens: usize,
    first_token: usize,
}

impl CudaCandidateRuntime {
    /// Creates a private driver context on one exact device and loads the
    /// caller-supplied cubin. Only compute capability 8.6 is accepted.
    pub fn new(cubin: &[u8], device_ordinal: i32) -> Result<Self> {
        if cubin.is_empty() {
            return Err(EngineError::InvalidArtifact(
                "CUDA candidate cubin is empty".into(),
            ));
        }
        let driver = CudaDriver::load()?;
        unsafe {
            driver.check((driver.init)(0), "initialization")?;
        }
        let mut device = 0;
        unsafe {
            driver.check(
                (driver.device_get)(&mut device, device_ordinal),
                "device selection",
            )?;
        }
        let mut major = 0;
        let mut minor = 0;
        unsafe {
            driver.check(
                (driver.device_compute_capability)(&mut major, &mut minor, device),
                "compute-capability query",
            )?;
        }
        let compute_capability = (
            u32::try_from(major).unwrap_or(0),
            u32::try_from(minor).unwrap_or(0),
        );
        if compute_capability != (8, 6) {
            return Err(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "create SM86 candidate runtime",
                reason: format!(
                    "device {device_ordinal} reports compute capability {major}.{minor}, expected 8.6"
                ),
            });
        }
        let mut name = [0_i8; 256];
        unsafe {
            driver.check(
                (driver.device_get_name)(name.as_mut_ptr(), name.len() as i32, device),
                "device-name query",
            )?;
        }
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let mut context = ptr::null_mut();
        unsafe {
            driver.check(
                (driver.ctx_create)(&mut context, 0, device),
                "context creation",
            )?;
        }
        Self::load_module(driver, context, cubin, device_name, compute_capability)
    }

    fn load_module(
        driver: CudaDriver,
        context: CuContext,
        cubin: &[u8],
        device_name: String,
        compute_capability: (u32, u32),
    ) -> Result<Self> {
        let mut module = ptr::null_mut();
        let load_result = unsafe { (driver.module_load_data)(&mut module, cubin.as_ptr().cast()) };
        if load_result != CUDA_SUCCESS {
            let error = driver
                .check(load_result, "module load")
                .expect_err("non-success CUDA result must be an error");
            unsafe {
                let _ = (driver.ctx_destroy)(context);
            }
            return Err(error);
        }
        let q2_function = match resolve_function(&driver, module, Q2_B64_FUSED_MATVEC.symbol) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let q4_function = match resolve_function(&driver, module, Q4_B64_FUSED_MATVEC.symbol) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let a8_quantize_function = match resolve_function(&driver, module, A8_QUANTIZE_SYMBOL) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let swiglu_a8_quantize_function =
            match resolve_function(&driver, module, SWIGLU_A8_QUANTIZE_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let sigmoid_gate_a8_quantize_function =
            match resolve_function(&driver, module, SIGMOID_GATE_A8_QUANTIZE_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_a8_function = match resolve_function(&driver, module, Q2_B64_A8_MATVEC_SYMBOL) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let q4_a8_function = match resolve_function(&driver, module, Q4_B64_A8_MATVEC_SYMBOL) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let a8_batched_quantize_function =
            match resolve_function(&driver, module, A8_BATCHED_QUANTIZE_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_a8_batched_function =
            match resolve_function(&driver, module, Q2_B64_A8_BATCHED_MATMUL_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q4_a8_batched_function =
            match resolve_function(&driver, module, Q4_B64_A8_BATCHED_MATMUL_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_a8_batched_mmq_function =
            match resolve_function(&driver, module, Q2_B64_A8_BATCHED_MMQ_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q4_a8_batched_mmq_function =
            match resolve_function(&driver, module, Q4_B64_A8_BATCHED_MMQ_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_a8_gathered_function =
            match resolve_function(&driver, module, Q2_B64_A8_GATHERED_MATVEC_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q4_a8_gathered_function =
            match resolve_function(&driver, module, Q4_B64_A8_GATHERED_MATVEC_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_recovered_row_function =
            match resolve_function(&driver, module, Q2_B64_RECOVERED_ROW_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q4_recovered_row_function =
            match resolve_function(&driver, module, Q4_B64_RECOVERED_ROW_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q2_recovered_rows_function =
            match resolve_function(&driver, module, Q2_B64_RECOVERED_ROWS_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let q4_recovered_rows_function =
            match resolve_function(&driver, module, Q4_B64_RECOVERED_ROWS_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let gated_delta_prep_f32_function =
            match resolve_function(&driver, module, GATED_DELTA_PREP_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let gated_delta_prep_scan_f32_function =
            match resolve_function(&driver, module, GATED_DELTA_PREP_SCAN_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let gated_delta_f16_function =
            match resolve_function(&driver, module, GATED_DELTA_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let gated_delta_scan_f16_function =
            match resolve_function(&driver, module, GATED_DELTA_SCAN_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let causal_conv_f16_function =
            match resolve_function(&driver, module, CAUSAL_CONV_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let causal_conv_scan_f16_function =
            match resolve_function(&driver, module, CAUSAL_CONV_SCAN_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let gated_rms_norm_f16_function =
            match resolve_function(&driver, module, GATED_RMS_NORM_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let qwen_rms_norm_f16_function =
            match resolve_function(&driver, module, QWEN_RMS_NORM_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let residual_rms_norm_f16_function =
            match resolve_function(&driver, module, RESIDUAL_RMS_NORM_F16_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let partial_rope_f32_function =
            match resolve_function(&driver, module, PARTIAL_ROPE_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let rope_table_batch_f32_function =
            match resolve_function(&driver, module, ROPE_TABLE_BATCH_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let partial_rope_batch_f32_function =
            match resolve_function(&driver, module, PARTIAL_ROPE_BATCH_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let query_gate_norm_rope_f32_function =
            match resolve_function(&driver, module, QUERY_GATE_NORM_ROPE_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let query_gate_norm_rope_batch_f32_function =
            match resolve_function(&driver, module, QUERY_GATE_NORM_ROPE_BATCH_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let pack_paged_kv_q4_f32_function =
            match resolve_function(&driver, module, PACK_PAGED_KV_Q4_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let pack_paged_kv_q4_batch_f32_function =
            match resolve_function(&driver, module, PACK_PAGED_KV_Q4_BATCH_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let demote_paged_kv_q4_to_q2_function =
            match resolve_function(&driver, module, DEMOTE_PAGED_KV_Q4_TO_Q2_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let paged_q2q4_gqa_f32_function =
            match resolve_function(&driver, module, PAGED_Q2Q4_GQA_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let paged_q2q4_gqa_prefill_f32_function =
            match resolve_function(&driver, module, PAGED_Q2Q4_GQA_PREFILL_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let paged_q2q4_gqa_split_partial_f32_function =
            match resolve_function(&driver, module, PAGED_Q2Q4_GQA_SPLIT_PARTIAL_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let paged_q2q4_gqa_split_combine_f32_function =
            match resolve_function(&driver, module, PAGED_Q2Q4_GQA_SPLIT_COMBINE_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        let argmax_f32_function = match resolve_function(&driver, module, ARGMAX_F32_SYMBOL) {
            Ok(function) => function,
            Err(error) => {
                unsafe {
                    let _ = (driver.module_unload)(module);
                    let _ = (driver.ctx_destroy)(context);
                }
                return Err(error);
            }
        };
        let topk_topp_sample_f32_function =
            match resolve_function(&driver, module, TOPK_TOPP_SAMPLE_F32_SYMBOL) {
                Ok(function) => function,
                Err(error) => {
                    unsafe {
                        let _ = (driver.module_unload)(module);
                        let _ = (driver.ctx_destroy)(context);
                    }
                    return Err(error);
                }
            };
        Ok(Self {
            inner: Rc::new(CudaContextInner {
                driver,
                context,
                module,
                q2_function,
                q4_function,
                a8_quantize_function,
                swiglu_a8_quantize_function,
                sigmoid_gate_a8_quantize_function,
                q2_a8_function,
                q4_a8_function,
                a8_batched_quantize_function,
                q2_a8_batched_function,
                q4_a8_batched_function,
                q2_a8_batched_mmq_function,
                q4_a8_batched_mmq_function,
                q2_a8_gathered_function,
                q4_a8_gathered_function,
                q2_recovered_row_function,
                q4_recovered_row_function,
                q2_recovered_rows_function,
                q4_recovered_rows_function,
                gated_delta_prep_f32_function,
                gated_delta_prep_scan_f32_function,
                gated_delta_f16_function,
                gated_delta_scan_f16_function,
                causal_conv_f16_function,
                causal_conv_scan_f16_function,
                gated_rms_norm_f16_function,
                qwen_rms_norm_f16_function,
                residual_rms_norm_f16_function,
                partial_rope_f32_function,
                rope_table_batch_f32_function,
                partial_rope_batch_f32_function,
                query_gate_norm_rope_f32_function,
                query_gate_norm_rope_batch_f32_function,
                pack_paged_kv_q4_f32_function,
                pack_paged_kv_q4_batch_f32_function,
                demote_paged_kv_q4_to_q2_function,
                paged_q2q4_gqa_f32_function,
                paged_q2q4_gqa_prefill_f32_function,
                paged_q2q4_gqa_split_partial_f32_function,
                paged_q2q4_gqa_split_combine_f32_function,
                argmax_f32_function,
                topk_topp_sample_f32_function,
                device_name,
                compute_capability,
                token_submission_active: Cell::new(false),
                token_submission_attempts: Cell::new(0),
                token_submission_commits: Cell::new(0),
                deferred_operator_synchronizations: Cell::new(0),
                context_synchronizations: Cell::new(0),
                device_argmax_launches: Cell::new(0),
                device_sampling_launches: Cell::new(0),
            }),
        })
    }

    pub fn submission_stats(&self) -> CudaSubmissionStats {
        CudaSubmissionStats {
            token_submission_attempts: self.inner.token_submission_attempts.get(),
            token_submission_commits: self.inner.token_submission_commits.get(),
            deferred_operator_synchronizations: self.inner.deferred_operator_synchronizations.get(),
            context_synchronizations: self.inner.context_synchronizations.get(),
            device_argmax_launches: self.inner.device_argmax_launches.get(),
            device_sampling_launches: self.inner.device_sampling_launches.get(),
        }
    }

    /// Submit one complete target or MTP token as an ordered default-stream
    /// transaction. Per-operator helpers keep their standalone synchronous
    /// behavior, but suppress intermediate context barriers while this scope
    /// is active. The final synchronization is the commit point at which
    /// asynchronous launch failures become visible to the graph owner.
    pub(crate) fn run_token_submission<T>(
        &self,
        operation: &'static str,
        execute: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if self.inner.token_submission_active.replace(true) {
            return Err(EngineError::InvalidState(
                "nested CUDA token submissions are forbidden".into(),
            ));
        }
        self.inner
            .token_submission_attempts
            .set(self.inner.token_submission_attempts.get().saturating_add(1));
        let launch_result = execute();
        self.inner.token_submission_active.set(false);
        let synchronize_result = self.synchronize_context(operation);
        match (launch_result, synchronize_result) {
            (Ok(value), Ok(())) => {
                self.inner
                    .token_submission_commits
                    .set(self.inner.token_submission_commits.get().saturating_add(1));
                Ok(value)
            }
            (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
            (Err(launch_error), Err(synchronize_error)) => Err(EngineError::InvalidState(format!(
                "CUDA token submission failed before and at its commit barrier: \
                     launch={launch_error}; synchronize={synchronize_error}"
            ))),
        }
    }

    fn synchronize_context(&self, operation: &'static str) -> Result<()> {
        self.make_current()?;
        self.inner
            .context_synchronizations
            .set(self.inner.context_synchronizations.get().saturating_add(1));
        unsafe {
            self.inner
                .driver
                .check((self.inner.driver.ctx_synchronize)(), operation)
        }
    }

    fn synchronize_after_launch(&self, operation: &'static str) -> Result<()> {
        if self.inner.token_submission_active.get() {
            self.inner.deferred_operator_synchronizations.set(
                self.inner
                    .deferred_operator_synchronizations
                    .get()
                    .saturating_add(1),
            );
            Ok(())
        } else {
            self.synchronize_context(operation)
        }
    }

    pub fn device_name(&self) -> &str {
        &self.inner.device_name
    }

    pub fn compute_capability(&self) -> (u32, u32) {
        self.inner.compute_capability
    }

    /// Driver-reported memory in the private context. Verifiers use this to
    /// prove that dropping prepared graph objects returns every owned device
    /// allocation without waiting for process exit or a global cache trim.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        self.make_current()?;
        let mut free = 0_usize;
        let mut total = 0_usize;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.mem_get_info)(&mut free, &mut total),
                "device memory query",
            )?;
        }
        Ok((free, total))
    }

    pub fn prepare_f32_concat(
        &self,
        left_values: usize,
        right_values: usize,
    ) -> Result<PreparedCudaF32Concat> {
        let values = left_values
            .checked_add(right_values)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA concat width overflows".into()))?;
        if left_values == 0 || right_values == 0 {
            return Err(EngineError::Shape(
                "CUDA concat inputs must both be non-empty".into(),
            ));
        }
        let bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA concat bytes overflow".into()))?;
        Ok(PreparedCudaF32Concat {
            context: Rc::clone(&self.inner),
            left_values,
            right_values,
            output: DeviceBuffer::allocate(self, bytes)?,
        })
    }

    pub fn prepare_argmax_f32(&self) -> Result<PreparedCudaArgmax> {
        Ok(PreparedCudaArgmax {
            context: Rc::clone(&self.inner),
            result: DeviceBuffer::allocate(self, 2 * std::mem::size_of::<u32>())?,
        })
    }

    pub fn prepare_topk_topp_sampler(
        &self,
        max_values: usize,
    ) -> Result<PreparedCudaTopKTopPSampler> {
        if max_values == 0 || u32::try_from(max_values).is_err() {
            return Err(EngineError::Shape(
                "CUDA sampler capacity must be positive and fit u32".into(),
            ));
        }
        let scratch_bytes = max_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA sampler scratch overflows".into()))?;
        Ok(PreparedCudaTopKTopPSampler {
            context: Rc::clone(&self.inner),
            max_values,
            scratch: DeviceBuffer::allocate(self, scratch_bytes)?,
            result: DeviceBuffer::allocate(self, 4 * std::mem::size_of::<u32>())?,
        })
    }

    /// Sampling-adjacent greedy selection over an already resident finite f32
    /// distribution. Only the selected local index crosses the device boundary.
    pub fn dispatch_argmax_f32_device(
        &self,
        prepared: &PreparedCudaArgmax,
        values: CudaDeviceF32View<'_>,
    ) -> Result<u32> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, values.context) {
            return Err(EngineError::InvalidState(
                "CUDA argmax crosses driver contexts".into(),
            ));
        }
        let mut input = values.ptr()?;
        let mut result = prepared.result.ptr();
        let mut count = cuda_u32(values.values(), "CUDA argmax value count")?;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut result as *mut CuDevicePtr).cast::<c_void>(),
            (&mut count as *mut u32).cast::<c_void>(),
        ];
        self.make_current()?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.argmax_f32_function,
                    1,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "f32 argmax launch",
            )?;
        }
        self.synchronize_after_launch("f32 argmax context synchronization")?;
        let mut host = [0_u32; 2];
        prepared.result.copy_to(as_bytes_mut(&mut host))?;
        if host[1] != 0 || host[0] >= count {
            return Err(EngineError::InvalidState(
                "CUDA argmax observed non-finite logits or an invalid index".into(),
            ));
        }
        self.inner
            .device_argmax_launches
            .set(self.inner.device_argmax_launches.get().saturating_add(1));
        Ok(host[0])
    }

    /// Verifier-only bounded top-k/top-p selection over an already resident
    /// finite f32 distribution. The caller supplies the canonical PCG draw;
    /// only the selected token and compact evidence cross the boundary.
    pub fn dispatch_topk_topp_sample_f32_device(
        &self,
        prepared: &PreparedCudaTopKTopPSampler,
        values: CudaDeviceF32View<'_>,
        config: SamplerConfig,
        draw: f32,
    ) -> Result<CudaSampledToken> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, values.context) {
            return Err(EngineError::InvalidState(
                "CUDA sampler crosses driver contexts".into(),
            ));
        }
        if values.values() == 0 || values.values() > prepared.max_values {
            return Err(EngineError::Shape(format!(
                "CUDA sampler received {} logits, capacity is {}",
                values.values(),
                prepared.max_values
            )));
        }
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(EngineError::InvalidArtifact(
                "CUDA stochastic sampling requires a finite positive temperature".into(),
            ));
        }
        if config.top_k == 0 || config.top_k > CUDA_SAMPLER_MAX_TOP_K {
            return Err(EngineError::UnsupportedOperation {
                backend: "cuda",
                operation: "top-k/top-p sampling",
                reason: format!(
                    "verifier candidate requires top_k in 1..={CUDA_SAMPLER_MAX_TOP_K}"
                ),
            });
        }
        if !config.top_p.is_finite() || !(0.0..=1.0).contains(&config.top_p) || config.top_p == 0.0
        {
            return Err(EngineError::InvalidArtifact(
                "CUDA top_p must be in (0, 1]".into(),
            ));
        }
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(EngineError::InvalidArtifact(
                "CUDA sampler draw must be finite and in [0, 1)".into(),
            ));
        }
        let inverse_temperature = config.temperature.recip();
        if !inverse_temperature.is_finite() {
            return Err(EngineError::InvalidArtifact(
                "CUDA inverse temperature must be finite".into(),
            ));
        }

        let mut input = values.ptr()?;
        let mut scratch = prepared.scratch.ptr();
        let mut result = prepared.result.ptr();
        let mut count = cuda_u32(values.values(), "CUDA sampler value count")?;
        let mut top_k = cuda_u32(config.top_k, "CUDA sampler top_k")?;
        let mut top_p = config.top_p;
        let mut draw = draw;
        let mut inverse_temperature = inverse_temperature;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut scratch as *mut CuDevicePtr).cast::<c_void>(),
            (&mut result as *mut CuDevicePtr).cast::<c_void>(),
            (&mut count as *mut u32).cast::<c_void>(),
            (&mut top_k as *mut u32).cast::<c_void>(),
            (&mut inverse_temperature as *mut f32).cast::<c_void>(),
            (&mut top_p as *mut f32).cast::<c_void>(),
            (&mut draw as *mut f32).cast::<c_void>(),
        ];
        self.make_current()?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.topk_topp_sample_f32_function,
                    1,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "top-k/top-p sampler launch",
            )?;
        }
        self.synchronize_after_launch("top-k/top-p sampler context synchronization")?;
        let mut host = [0_u32; 4];
        prepared.result.copy_to(as_bytes_mut(&mut host))?;
        let expected_top_k = top_k.min(count);
        let nucleus_total = f32::from_bits(host[3]);
        if host[1] != 0
            || host[0] >= count
            || host[2] == 0
            || host[2] > expected_top_k
            || !nucleus_total.is_finite()
            || nucleus_total <= 0.0
        {
            return Err(EngineError::InvalidState(format!(
                "CUDA sampler rejected logits or produced invalid evidence (status={}, token={}, nucleus_len={}, nucleus_total={nucleus_total})",
                host[1], host[0], host[2]
            )));
        }
        self.inner
            .device_sampling_launches
            .set(self.inner.device_sampling_launches.get().saturating_add(1));
        Ok(CudaSampledToken {
            token: host[0],
            nucleus_len: host[2],
            nucleus_total,
        })
    }

    pub fn prepare_f32_checkpoint(&self, values: usize) -> Result<PreparedCudaF32Checkpoint> {
        let bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA checkpoint bytes overflow".into()))?;
        if values == 0 {
            return Err(EngineError::Shape(
                "CUDA checkpoint must contain at least one value".into(),
            ));
        }
        let snapshot = DeviceBuffer::allocate(self, bytes)?;
        snapshot.zero()?;
        Ok(PreparedCudaF32Checkpoint {
            context: Rc::clone(&self.inner),
            values,
            snapshot,
            valid: false,
        })
    }

    pub fn snapshot_f32_device(
        &self,
        prepared: &mut PreparedCudaF32Checkpoint,
        source: CudaDeviceF32View<'_>,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, source.context) {
            return Err(EngineError::InvalidState(
                "CUDA checkpoint source belongs to another context".into(),
            ));
        }
        if prepared.valid || source.values() != prepared.values {
            return Err(EngineError::InvalidState(format!(
                "CUDA checkpoint is already active or has {} values for a {}-value source",
                prepared.values,
                source.values()
            )));
        }
        prepared
            .snapshot
            .copy_from_view(source, "target-hidden checkpoint copy")?;
        prepared.valid = true;
        Ok(())
    }

    pub fn restore_f32_device(
        &self,
        prepared: &mut PreparedCudaF32Checkpoint,
        destination: CudaDeviceF32View<'_>,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, destination.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA checkpoint destination belongs to another context".into(),
            ));
        }
        if !prepared.valid || destination.values() != prepared.values {
            return Err(EngineError::InvalidState(
                "CUDA checkpoint is absent or has the wrong destination width".into(),
            ));
        }
        destination
            .buffer
            .copy_from_buffer(&prepared.snapshot, "target-hidden checkpoint restore")?;
        prepared.valid = false;
        Ok(())
    }

    /// Copies one compact f32 rectangle between device views using the CUDA
    /// Driver API. This is the no-kernel primitive used to assemble row-major
    /// MTP `[embedding, previous_target_hidden]` inputs with two strided
    /// copies; it never stages values through the host.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_f32_rows_device(
        &self,
        source: CudaDeviceF32View<'_>,
        source_row_values: usize,
        destination: CudaDeviceF32View<'_>,
        destination_row_values: usize,
        destination_column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, source.context) || !Rc::ptr_eq(&self.inner, destination.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA f32 2-D copy crosses driver contexts".into(),
            ));
        }
        let geometry = F32Copy2DGeometry {
            source_row_values,
            destination_row_values,
            destination_column,
            rows,
            columns,
        };
        geometry.validate(source.values(), destination.values())?;
        let value_bytes = std::mem::size_of::<f32>();
        let descriptor = CudaMemcpy2DDescriptor {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CU_MEMORYTYPE_DEVICE,
            src_host: ptr::null(),
            src_device: source.ptr()?,
            src_array: ptr::null_mut(),
            src_pitch: source_row_values.checked_mul(value_bytes).ok_or_else(|| {
                EngineError::MemoryBudget("CUDA f32 2-D source pitch overflows".into())
            })?,
            dst_x_in_bytes: destination_column.checked_mul(value_bytes).ok_or_else(|| {
                EngineError::MemoryBudget("CUDA f32 2-D destination offset overflows".into())
            })?,
            dst_y: 0,
            dst_memory_type: CU_MEMORYTYPE_DEVICE,
            dst_host: ptr::null_mut(),
            dst_device: destination.ptr()?,
            dst_array: ptr::null_mut(),
            dst_pitch: destination_row_values
                .checked_mul(value_bytes)
                .ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA f32 2-D destination pitch overflows".into())
                })?,
            width_in_bytes: columns
                .checked_mul(value_bytes)
                .ok_or_else(|| EngineError::MemoryBudget("CUDA f32 2-D width overflows".into()))?,
            height: rows,
        };
        self.make_current()?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.memcpy_2d)(&descriptor),
                "f32 device row copy",
            )?;
        }
        self.synchronize_after_launch("f32 device row-copy context synchronization")
    }

    pub fn dispatch_f32_concat_device<'a>(
        &self,
        prepared: &'a PreparedCudaF32Concat,
        left: CudaDeviceF32View<'_>,
        right: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, left.context)
            || !Rc::ptr_eq(&self.inner, right.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA concat input belongs to another context".into(),
            ));
        }
        if left.values() != prepared.left_values || right.values() != prepared.right_values {
            return Err(EngineError::Shape(format!(
                "CUDA concat has {}+{} values, expected {}+{}",
                left.values(),
                right.values(),
                prepared.left_values,
                prepared.right_values
            )));
        }
        self.make_current()?;
        let left_bytes = prepared
            .left_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA concat left bytes overflow".into()))?;
        let right_bytes = prepared
            .right_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA concat right bytes overflow".into()))?;
        let right_target = device_ptr_offset(prepared.output.ptr(), left_bytes)?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.memcpy_dtod)(prepared.output.ptr(), left.ptr()?, left_bytes),
                "MTP left device concatenation",
            )?;
            self.inner.driver.check(
                (self.inner.driver.memcpy_dtod)(right_target, right.ptr()?, right_bytes),
                "MTP right device concatenation",
            )?;
        }
        prepared
            .output
            .f32_view(0, prepared.left_values + prepared.right_values)
    }

    /// Explicit token-boundary readback for verifier binaries. Production
    /// graph execution keeps intermediate tensors device-resident and may
    /// expose only sampler-selected results at this boundary.
    pub fn verifier_read_f32_device(&self, view: CudaDeviceF32View<'_>) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, view.context) {
            return Err(EngineError::InvalidState(
                "CUDA verifier readback belongs to another context".into(),
            ));
        }
        self.synchronize_context("verifier token-boundary synchronization")?;
        let mut values = vec![0.0_f32; view.values()];
        let byte_offset = view
            .offset_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA verifier offset overflows".into()))?;
        view.buffer
            .copy_range_to(byte_offset, as_bytes_mut(&mut values))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA verifier readback contains non-finite values".into(),
            ));
        }
        Ok(values)
    }

    pub fn prepare_verifier_f32_tensor(&self, values: &[f32]) -> Result<CudaVerifierF32Tensor> {
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "CUDA verifier tensor requires finite non-empty f32 values".into(),
            ));
        }
        Ok(CudaVerifierF32Tensor {
            buffer: DeviceBuffer::from_bytes(self, as_bytes(values))?,
            values: values.len(),
        })
    }

    /// Device-to-host readback is deliberately named and scoped as verifier
    /// functionality so production graph code cannot mistake it for an
    /// admitted tensor edge.
    pub fn verifier_read_f32(&self, view: CudaDeviceF32View<'_>) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, view.context) {
            return Err(EngineError::InvalidState(
                "CUDA verifier read belongs to another context".into(),
            ));
        }
        let offset_bytes = view
            .offset_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA verifier offset overflows".into()))?;
        let mut result = vec![0.0_f32; view.values];
        view.buffer
            .copy_range_to(offset_bytes, as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA verifier read produced a non-finite value".into(),
            ));
        }
        Ok(result)
    }

    pub fn prepare_gated_delta_inputs_f32(
        &self,
        a_log: &[f32],
        dt_bias: &[f32],
    ) -> Result<PreparedCudaGatedDeltaInputs> {
        for (name, values) in [("A_log", a_log), ("dt_bias", dt_bias)] {
            if values.len() != GATED_DELTA_HEADS || values.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta {name} must contain {GATED_DELTA_HEADS} finite values"
                )));
            }
        }
        self.prepare_gated_delta_inputs_f32_le(as_bytes(a_log), as_bytes(dt_bias))
    }

    /// Prepares immutable GatedDelta parameters directly from their canonical
    /// little-endian F32 artifact representation. No widened or duplicate host
    /// vectors are constructed by the production loader.
    pub fn prepare_gated_delta_inputs_f32_le(
        &self,
        a_log_f32_le: &[u8],
        dt_bias_f32_le: &[u8],
    ) -> Result<PreparedCudaGatedDeltaInputs> {
        let head_bytes = GATED_DELTA_HEADS
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta head bytes overflow".into()))?;
        validate_f32_buffer(a_log_f32_le, head_bytes, "GatedDelta A_log")?;
        validate_f32_buffer(dt_bias_f32_le, head_bytes, "GatedDelta dt_bias")?;
        let qk_values = GATED_DELTA_HEADS * GATED_DELTA_KEY_DIM;
        let qk_bytes = qk_values * std::mem::size_of::<f32>();
        Ok(PreparedCudaGatedDeltaInputs {
            context: Rc::clone(&self.inner),
            a_log: DeviceBuffer::from_bytes(self, a_log_f32_le)?,
            dt_bias: DeviceBuffer::from_bytes(self, dt_bias_f32_le)?,
            query: DeviceBuffer::allocate(self, qk_bytes)?,
            key: DeviceBuffer::allocate(self, qk_bytes)?,
            log_decay: DeviceBuffer::allocate(self, head_bytes)?,
            beta: DeviceBuffer::allocate(self, head_bytes)?,
            model_bytes: head_bytes * 2,
            transient_bytes: qk_bytes * 2 + head_bytes * 2,
        })
    }

    pub fn dispatch_gated_delta_inputs_device<'a>(
        &self,
        prepared: &'a mut PreparedCudaGatedDeltaInputs,
        convolved_qkv: CudaDeviceF32View<'_>,
        raw_a: CudaDeviceF32View<'_>,
        raw_b: CudaDeviceF32View<'_>,
    ) -> Result<CudaGatedDeltaInputViews<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA gated-delta inputs belong to another context".into(),
            ));
        }
        for (name, view, expected) in [
            ("convolved_qkv", convolved_qkv, LINEAR_CONV_CHANNELS),
            ("raw_a", raw_a, GATED_DELTA_HEADS),
            ("raw_b", raw_b, GATED_DELTA_HEADS),
        ] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA gated-delta {name} belongs to another context"
                )));
            }
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }

        self.make_current()?;
        let mut convolved_qkv_ptr = convolved_qkv.ptr()?;
        let mut raw_a_ptr = raw_a.ptr()?;
        let mut raw_b_ptr = raw_b.ptr()?;
        let mut a_log_ptr = prepared.a_log.ptr();
        let mut dt_bias_ptr = prepared.dt_bias.ptr();
        let mut query_ptr = prepared.query.ptr();
        let mut key_ptr = prepared.key.ptr();
        let mut log_decay_ptr = prepared.log_decay.ptr();
        let mut beta_ptr = prepared.beta.ptr();
        let mut key_heads = GATED_DELTA_KEY_HEADS as u32;
        let mut value_heads = GATED_DELTA_HEADS as u32;
        let mut key_dim = GATED_DELTA_KEY_DIM as u32;
        let mut params = [
            (&mut convolved_qkv_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut raw_a_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut raw_b_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut a_log_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut dt_bias_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut log_decay_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut beta_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key_heads as *mut u32).cast::<c_void>(),
            (&mut value_heads as *mut u32).cast::<c_void>(),
            (&mut key_dim as *mut u32).cast::<c_void>(),
        ];
        let qk_values = GATED_DELTA_HEADS * GATED_DELTA_KEY_DIM;
        let blocks = u32::try_from(qk_values.div_ceil(LINEAR_THREADS_PER_BLOCK as usize))
            .map_err(|_| EngineError::Shape("CUDA gated-delta prep grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.gated_delta_prep_f32_function,
                    blocks,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gated-delta input preparation kernel launch",
            )?;
        }
        self.synchronize_after_launch("gated-delta input preparation context synchronization")?;
        Ok(CudaGatedDeltaInputViews {
            query: prepared.query.f32_view(0, qk_values)?,
            key: prepared.key.f32_view(0, qk_values)?,
            log_decay: prepared.log_decay.f32_view(0, GATED_DELTA_HEADS)?,
            beta: prepared.beta.f32_view(0, GATED_DELTA_HEADS)?,
        })
    }

    pub fn prepare_gated_delta_scan_inputs(
        &self,
        token_capacity: usize,
    ) -> Result<PreparedCudaGatedDeltaScanInputs> {
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let qk_values = (token_capacity as usize)
            .checked_mul(GATED_DELTA_HEADS)
            .and_then(|values| values.checked_mul(GATED_DELTA_KEY_DIM))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan Q/K capacity overflows".into())
            })?;
        let head_values = (token_capacity as usize)
            .checked_mul(GATED_DELTA_HEADS)
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan head capacity overflows".into())
            })?;
        let qk_bytes = qk_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan Q/K bytes overflow".into())
            })?;
        let head_bytes = head_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan head bytes overflow".into())
            })?;
        let transient_bytes = qk_bytes
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_add(head_bytes.checked_mul(2)?))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan workspace overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaGatedDeltaScanInputs {
            context: Rc::clone(&self.inner),
            token_capacity,
            query: DeviceBuffer::allocate(self, qk_bytes)?,
            key: DeviceBuffer::allocate(self, qk_bytes)?,
            value: DeviceBuffer::allocate(self, qk_bytes)?,
            log_decay: DeviceBuffer::allocate(self, head_bytes)?,
            beta: DeviceBuffer::allocate(self, head_bytes)?,
            transient_bytes,
        })
    }

    /// Prepares all Qwen-specific GatedDelta chunk inputs in one flattened
    /// device launch while sharing the immutable A_log/dt_bias allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gated_delta_scan_inputs_device<'a>(
        &self,
        prepared: &PreparedCudaGatedDeltaInputs,
        workspace: &'a PreparedCudaGatedDeltaScanInputs,
        convolved_qkv: CudaDeviceF32View<'_>,
        raw_a: CudaDeviceF32View<'_>,
        raw_b: CudaDeviceF32View<'_>,
        tokens: usize,
    ) -> Result<CudaGatedDeltaScanInputViews<'a>> {
        let tokens = validate_a8_batch_capacity(tokens)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &workspace.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta scan preparation crosses driver contexts".into(),
            ));
        }
        if tokens > workspace.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA gated-delta preparation requests {tokens} tokens, capacity is {}",
                workspace.token_capacity
            )));
        }
        let convolved_values = (tokens as usize)
            .checked_mul(LINEAR_CONV_CHANNELS)
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta convolved chunk shape overflows".into())
            })?;
        let head_values = (tokens as usize)
            .checked_mul(GATED_DELTA_HEADS)
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta head chunk shape overflows".into())
            })?;
        for (name, view, expected) in [
            ("convolved_qkv", convolved_qkv, convolved_values),
            ("raw_a", raw_a, head_values),
            ("raw_b", raw_b, head_values),
        ] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA gated-delta scan {name} belongs to another context"
                )));
            }
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta scan {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }

        self.make_current()?;
        let mut convolved_qkv_ptr = convolved_qkv.ptr()?;
        let mut raw_a_ptr = raw_a.ptr()?;
        let mut raw_b_ptr = raw_b.ptr()?;
        let mut a_log_ptr = prepared.a_log.ptr();
        let mut dt_bias_ptr = prepared.dt_bias.ptr();
        let mut query_ptr = workspace.query.ptr();
        let mut key_ptr = workspace.key.ptr();
        let mut value_ptr = workspace.value.ptr();
        let mut log_decay_ptr = workspace.log_decay.ptr();
        let mut beta_ptr = workspace.beta.ptr();
        let mut token_count = tokens;
        let mut key_heads = GATED_DELTA_KEY_HEADS as u32;
        let mut value_heads = GATED_DELTA_HEADS as u32;
        let mut key_dim = GATED_DELTA_KEY_DIM as u32;
        let mut params = [
            (&mut convolved_qkv_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut raw_a_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut raw_b_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut a_log_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut dt_bias_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut value_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut log_decay_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut beta_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut token_count as *mut u32).cast::<c_void>(),
            (&mut key_heads as *mut u32).cast::<c_void>(),
            (&mut value_heads as *mut u32).cast::<c_void>(),
            (&mut key_dim as *mut u32).cast::<c_void>(),
        ];
        let qk_values = (tokens as usize)
            .checked_mul(GATED_DELTA_HEADS)
            .and_then(|values| values.checked_mul(GATED_DELTA_KEY_DIM))
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta prepared Q/K shape overflows".into())
            })?;
        let blocks = u32::try_from(qk_values.div_ceil(LINEAR_THREADS_PER_BLOCK as usize))
            .map_err(|_| EngineError::Shape("CUDA gated-delta scan prep grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.gated_delta_prep_scan_f32_function,
                    blocks,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gated-delta scan input preparation kernel launch",
            )?;
        }
        self.synchronize_after_launch(
            "gated-delta scan input preparation context synchronization",
        )?;
        Ok(CudaGatedDeltaScanInputViews {
            query: workspace.query.f32_view(0, qk_values)?,
            key: workspace.key.f32_view(0, qk_values)?,
            value: workspace.value.f32_view(0, qk_values)?,
            log_decay: workspace.log_decay.f32_view(0, head_values)?,
            beta: workspace.beta.f32_view(0, head_values)?,
        })
    }

    /// Allocate the exact Qwen3.8-27B persistent FP16 recurrent state and
    /// reusable step buffers. The kernel profile is intentionally fixed at
    /// 48x128x128; unsupported shapes fail before any device allocation.
    pub fn prepare_gated_delta_f16(
        &self,
        config: CudaGatedDeltaConfig,
    ) -> Result<PreparedCudaGatedDelta> {
        if config.heads != GATED_DELTA_HEADS
            || config.key_dim != GATED_DELTA_KEY_DIM
            || config.value_dim != GATED_DELTA_VALUE_DIM
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
        {
            return Err(EngineError::Shape(
                "CUDA gated-delta requires the exact Qwen3.8-27B 48x128x128 profile".into(),
            ));
        }
        let qk_values = config
            .heads
            .checked_mul(config.key_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta Q/K values overflow".into()))?;
        let value_values = config
            .heads
            .checked_mul(config.value_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta values overflow".into()))?;
        let qk_bytes = qk_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta Q/K bytes overflow".into()))?;
        let value_bytes = value_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta value bytes overflow".into()))?;
        let head_bytes = config
            .heads
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA delta head bytes overflow".into()))?;
        let transient_bytes = qk_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(value_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(head_bytes.checked_mul(2)?))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA delta transient bytes overflow".into())
            })?;

        let query = DeviceBuffer::allocate(self, qk_bytes)?;
        let key = DeviceBuffer::allocate(self, qk_bytes)?;
        let value = DeviceBuffer::allocate(self, value_bytes)?;
        let log_decay = DeviceBuffer::allocate(self, head_bytes)?;
        let beta = DeviceBuffer::allocate(self, head_bytes)?;
        let state = DeviceBuffer::allocate(self, GATED_DELTA_STATE_BYTES)?;
        let checkpoint = DeviceBuffer::allocate(self, GATED_DELTA_STATE_BYTES)?;
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        state.zero()?;
        checkpoint.zero()?;
        output.zero()?;
        Ok(PreparedCudaGatedDelta {
            context: Rc::clone(&self.inner),
            config,
            query,
            key,
            value,
            log_decay,
            beta,
            state,
            checkpoint,
            output,
            resident_state_bytes: GATED_DELTA_STATE_BYTES,
            transient_bytes,
            poisoned: false,
            checkpoint_valid: false,
        })
    }

    /// Execute one token against the persistent FP16 recurrence. A failed
    /// state-mutating launch poisons the object until the caller resets it;
    /// there is no CPU or alternate-kernel fallback.
    pub fn dispatch_gated_delta_f16(
        &self,
        prepared: &mut PreparedCudaGatedDelta,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA gated-delta belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta state is poisoned; reset is required".into(),
            ));
        }
        prepared.poisoned = true;
        self.dispatch_gated_delta_f16_inner(
            prepared,
            prepared.query.ptr(),
            prepared.key.ptr(),
            prepared.value.ptr(),
            prepared.log_decay.ptr(),
            prepared.beta.ptr(),
        )?;
        let output_values = prepared
            .config
            .heads
            .checked_mul(prepared.config.value_dim)
            .ok_or_else(|| EngineError::Shape("CUDA delta output shape overflows".into()))?;
        let mut result = vec![0.0_f32; output_values];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta produced a non-finite output".into(),
            ));
        }
        prepared.poisoned = false;
        Ok(result)
    }

    pub fn dispatch_gated_delta_f16_device<'a>(
        &self,
        prepared: &'a mut PreparedCudaGatedDelta,
        query: CudaDeviceF32View<'_>,
        key: CudaDeviceF32View<'_>,
        value: CudaDeviceF32View<'_>,
        log_decay: CudaDeviceF32View<'_>,
        beta: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA gated-delta belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta state is poisoned; reset is required".into(),
            ));
        }
        let qk_values = prepared.config.heads * prepared.config.key_dim;
        let value_values = prepared.config.heads * prepared.config.value_dim;
        for (name, view, expected) in [
            ("query", query, qk_values),
            ("key", key, qk_values),
            ("value", value, value_values),
            ("log_decay", log_decay, prepared.config.heads),
            ("beta", beta, prepared.config.heads),
        ] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA gated-delta {name} belongs to another context"
                )));
            }
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }
        prepared.poisoned = true;
        self.dispatch_gated_delta_f16_inner(
            prepared,
            query.ptr()?,
            key.ptr()?,
            value.ptr()?,
            log_decay.ptr()?,
            beta.ptr()?,
        )?;
        prepared.poisoned = false;
        prepared.output.f32_view(0, value_values)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gated_delta_f16_inner(
        &self,
        prepared: &PreparedCudaGatedDelta,
        mut query: CuDevicePtr,
        mut key: CuDevicePtr,
        mut value: CuDevicePtr,
        mut log_decay: CuDevicePtr,
        mut beta: CuDevicePtr,
    ) -> Result<()> {
        self.make_current()?;
        let mut state = prepared.state.ptr();
        let mut output = prepared.output.ptr();
        let mut heads = prepared.config.heads as u32;
        let mut key_dim = prepared.config.key_dim as u32;
        let mut value_dim = prepared.config.value_dim as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut query as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key as *mut CuDevicePtr).cast::<c_void>(),
            (&mut value as *mut CuDevicePtr).cast::<c_void>(),
            (&mut log_decay as *mut CuDevicePtr).cast::<c_void>(),
            (&mut beta as *mut CuDevicePtr).cast::<c_void>(),
            (&mut state as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut key_dim as *mut u32).cast::<c_void>(),
            (&mut value_dim as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.gated_delta_f16_function,
                    heads,
                    1,
                    1,
                    value_dim,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gated-delta kernel launch",
            )?;
        }
        self.synchronize_after_launch("gated-delta context synchronization")
    }

    pub fn prepare_gated_delta_scan_output(
        &self,
        config: CudaGatedDeltaConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaGatedDeltaScanOutput> {
        if config != CudaGatedDeltaConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "CUDA gated-delta scan requires the exact Qwen3.8-27B profile".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(config.heads)
            .and_then(|values| values.checked_mul(config.value_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated-delta scan output overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaGatedDeltaScanOutput {
            context: Rc::clone(&self.inner),
            token_capacity,
            heads: config.heads as u32,
            value_dim: config.value_dim as u32,
            output: DeviceBuffer::allocate(self, output_bytes)?,
            transient_bytes: output_bytes,
        })
    }

    /// Advances the exact decode recurrence through one token-major prompt
    /// chunk in a single causal launch. Failure leaves the state poisoned so a
    /// caller cannot unknowingly resume from a partially advanced recurrence.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_gated_delta_f16_scan_device<'a>(
        &self,
        prepared: &mut PreparedCudaGatedDelta,
        output: &'a PreparedCudaGatedDeltaScanOutput,
        query: CudaDeviceF32View<'_>,
        key: CudaDeviceF32View<'_>,
        value: CudaDeviceF32View<'_>,
        log_decay: CudaDeviceF32View<'_>,
        beta: CudaDeviceF32View<'_>,
        tokens: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let tokens = validate_a8_batch_capacity(tokens)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, &output.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta scan crosses driver contexts".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta state is poisoned; reset is required".into(),
            ));
        }
        if output.heads != prepared.config.heads as u32
            || output.value_dim != prepared.config.value_dim as u32
            || tokens > output.token_capacity
        {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA gated-delta scan {tokens}x{}x{} exceeds output capacity {}x{}x{}",
                prepared.config.heads,
                prepared.config.value_dim,
                output.token_capacity,
                output.heads,
                output.value_dim
            )));
        }
        let qk_values = (tokens as usize)
            .checked_mul(prepared.config.heads)
            .and_then(|values| values.checked_mul(prepared.config.key_dim))
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta scan Q/K shape overflows".into())
            })?;
        let value_values = (tokens as usize)
            .checked_mul(prepared.config.heads)
            .and_then(|values| values.checked_mul(prepared.config.value_dim))
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta scan value shape overflows".into())
            })?;
        let head_values = (tokens as usize)
            .checked_mul(prepared.config.heads)
            .ok_or_else(|| {
                EngineError::Shape("CUDA gated-delta scan head shape overflows".into())
            })?;
        for (name, view, expected) in [
            ("query", query, qk_values),
            ("key", key, qk_values),
            ("value", value, value_values),
            ("log_decay", log_decay, head_values),
            ("beta", beta, head_values),
        ] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA gated-delta scan {name} belongs to another context"
                )));
            }
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta scan {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }

        prepared.poisoned = true;
        self.dispatch_gated_delta_f16_scan_inner(
            prepared,
            output,
            query.ptr()?,
            key.ptr()?,
            value.ptr()?,
            log_decay.ptr()?,
            beta.ptr()?,
            tokens,
        )?;
        prepared.poisoned = false;
        output.device_output(tokens as usize)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_gated_delta_f16_scan_inner(
        &self,
        prepared: &PreparedCudaGatedDelta,
        output: &PreparedCudaGatedDeltaScanOutput,
        mut query: CuDevicePtr,
        mut key: CuDevicePtr,
        mut value: CuDevicePtr,
        mut log_decay: CuDevicePtr,
        mut beta: CuDevicePtr,
        mut tokens: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut state = prepared.state.ptr();
        let mut output_ptr = output.output.ptr();
        let mut heads = prepared.config.heads as u32;
        let mut key_dim = prepared.config.key_dim as u32;
        let mut value_dim = prepared.config.value_dim as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut query as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key as *mut CuDevicePtr).cast::<c_void>(),
            (&mut value as *mut CuDevicePtr).cast::<c_void>(),
            (&mut log_decay as *mut CuDevicePtr).cast::<c_void>(),
            (&mut beta as *mut CuDevicePtr).cast::<c_void>(),
            (&mut state as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut tokens as *mut u32).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut key_dim as *mut u32).cast::<c_void>(),
            (&mut value_dim as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.gated_delta_scan_f16_function,
                    heads,
                    1,
                    1,
                    value_dim,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gated-delta scan kernel launch",
            )?;
        }
        self.synchronize_after_launch("gated-delta scan context synchronization")
    }

    pub fn prepare_causal_conv_f16(
        &self,
        config: CudaCausalConvConfig,
        weight_f16_le: &[u8],
    ) -> Result<PreparedCudaCausalConv> {
        if config != CudaCausalConvConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "CUDA causal convolution requires the exact Qwen3.8-27B 10240x4 profile".into(),
            ));
        }
        validate_f16_buffer(weight_f16_le, LINEAR_CONV_STATE_BYTES, "convolution weight")?;
        let value_bytes = config
            .channels
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA convolution values overflow".into()))?;
        let input = DeviceBuffer::allocate(self, value_bytes)?;
        let weight = DeviceBuffer::from_bytes(self, weight_f16_le)?;
        let state = DeviceBuffer::allocate(self, LINEAR_CONV_STATE_BYTES)?;
        let checkpoint = DeviceBuffer::allocate(self, LINEAR_CONV_STATE_BYTES)?;
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        input.zero()?;
        state.zero()?;
        checkpoint.zero()?;
        output.zero()?;
        Ok(PreparedCudaCausalConv {
            context: Rc::clone(&self.inner),
            config,
            input,
            weight,
            state,
            checkpoint,
            output,
            model_bytes: weight_f16_le.len(),
            resident_state_bytes: LINEAR_CONV_STATE_BYTES,
            transient_bytes: value_bytes * 2,
            poisoned: false,
            checkpoint_valid: false,
        })
    }

    pub fn dispatch_causal_conv_f16(
        &self,
        prepared: &mut PreparedCudaCausalConv,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA convolution belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA convolution state is poisoned; reset is required".into(),
            ));
        }
        prepared.poisoned = true;
        self.dispatch_causal_conv_f16_inner(prepared, prepared.input.ptr())?;
        let mut result = vec![0.0_f32; prepared.config.channels];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA causal convolution produced a non-finite output".into(),
            ));
        }
        prepared.poisoned = false;
        Ok(result)
    }

    pub fn dispatch_causal_conv_f16_device<'a>(
        &self,
        prepared: &'a mut PreparedCudaCausalConv,
        input: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, input.context) {
            return Err(EngineError::InvalidState(
                "CUDA causal convolution device input belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA convolution state is poisoned; reset is required".into(),
            ));
        }
        if input.values() != prepared.config.channels {
            return Err(EngineError::Shape(format!(
                "CUDA convolution device input has {} values, expected {}",
                input.values(),
                prepared.config.channels
            )));
        }
        prepared.poisoned = true;
        self.dispatch_causal_conv_f16_inner(prepared, input.ptr()?)?;
        prepared.poisoned = false;
        prepared.output.f32_view(0, prepared.config.channels)
    }

    fn dispatch_causal_conv_f16_inner(
        &self,
        prepared: &PreparedCudaCausalConv,
        mut input: CuDevicePtr,
    ) -> Result<()> {
        self.make_current()?;
        let mut weight = prepared.weight.ptr();
        let mut state = prepared.state.ptr();
        let mut output = prepared.output.ptr();
        let mut channels = prepared.config.channels as u32;
        let mut kernel_width = prepared.config.kernel_width as u32;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut state as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut channels as *mut u32).cast::<c_void>(),
            (&mut kernel_width as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.causal_conv_f16_function,
                    channels.div_ceil(LINEAR_THREADS_PER_BLOCK),
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "causal-convolution kernel launch",
            )?;
        }
        self.synchronize_after_launch("causal-convolution context synchronization")
    }

    pub fn prepare_causal_conv_scan_output(
        &self,
        config: CudaCausalConvConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaCausalConvScanOutput> {
        if config != CudaCausalConvConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "CUDA convolution scan requires the exact Qwen3.8-27B profile".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(config.channels)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA convolution scan output overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaCausalConvScanOutput {
            context: Rc::clone(&self.inner),
            token_capacity,
            channels: config.channels as u32,
            output: DeviceBuffer::allocate(self, output_bytes)?,
            transient_bytes: output_bytes,
        })
    }

    /// Advances the exact decode state through a token-major prompt chunk in
    /// one causal device scan. A failed launch leaves the state owner poisoned
    /// so callers must reset rather than silently continuing from ambiguity.
    pub fn dispatch_causal_conv_f16_scan_device<'a>(
        &self,
        prepared: &mut PreparedCudaCausalConv,
        output: &'a PreparedCudaCausalConvScanOutput,
        input: CudaDeviceF32View<'_>,
        tokens: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let tokens = validate_a8_batch_capacity(tokens)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &output.context)
            || !Rc::ptr_eq(&self.inner, input.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA convolution scan crosses driver contexts".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA convolution state is poisoned; reset is required".into(),
            ));
        }
        if output.channels != prepared.config.channels as u32 || tokens > output.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA convolution scan {tokens}x{} exceeds output capacity {}x{}",
                prepared.config.channels, output.token_capacity, output.channels
            )));
        }
        let expected = (tokens as usize)
            .checked_mul(prepared.config.channels)
            .ok_or_else(|| EngineError::Shape("CUDA convolution scan shape overflows".into()))?;
        if input.values() != expected {
            return Err(EngineError::Shape(format!(
                "CUDA convolution scan input has {} values, expected {expected}",
                input.values()
            )));
        }
        prepared.poisoned = true;
        self.dispatch_causal_conv_f16_scan_inner(prepared, output, input.ptr()?, tokens)?;
        prepared.poisoned = false;
        output.device_output(tokens as usize)
    }

    fn dispatch_causal_conv_f16_scan_inner(
        &self,
        prepared: &PreparedCudaCausalConv,
        output: &PreparedCudaCausalConvScanOutput,
        mut input: CuDevicePtr,
        mut tokens: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut weight = prepared.weight.ptr();
        let mut state = prepared.state.ptr();
        let mut output_ptr = output.output.ptr();
        let mut channels = prepared.config.channels as u32;
        let mut kernel_width = prepared.config.kernel_width as u32;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut state as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut tokens as *mut u32).cast::<c_void>(),
            (&mut channels as *mut u32).cast::<c_void>(),
            (&mut kernel_width as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.causal_conv_scan_f16_function,
                    channels.div_ceil(LINEAR_THREADS_PER_BLOCK),
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "causal-convolution scan kernel launch",
            )?;
        }
        self.synchronize_after_launch("causal-convolution scan context synchronization")
    }

    pub fn prepare_gated_rms_norm_f16(
        &self,
        config: CudaGatedRmsNormConfig,
        weight_f16_le: &[u8],
    ) -> Result<PreparedCudaGatedRmsNorm> {
        if config.rows != GATED_RMS_NORM_ROWS
            || config.columns != GATED_RMS_NORM_COLUMNS
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
        {
            return Err(EngineError::Shape(
                "CUDA gated RMSNorm requires the exact Qwen3.8-27B 48x128 profile".into(),
            ));
        }
        let weight_bytes = config.columns * std::mem::size_of::<half::f16>();
        validate_f16_buffer(weight_f16_le, weight_bytes, "gated RMSNorm weight")?;
        let values = config.rows.checked_mul(config.columns).ok_or_else(|| {
            EngineError::MemoryBudget("CUDA gated RMSNorm shape overflows".into())
        })?;
        let value_bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA gated RMSNorm values overflow".into())
            })?;
        let input = DeviceBuffer::allocate(self, value_bytes)?;
        let gate = DeviceBuffer::allocate(self, value_bytes)?;
        let weight = DeviceBuffer::from_bytes(self, weight_f16_le)?;
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        input.zero()?;
        gate.zero()?;
        output.zero()?;
        Ok(PreparedCudaGatedRmsNorm {
            context: Rc::clone(&self.inner),
            config,
            input,
            gate,
            weight,
            output,
            model_bytes: weight_f16_le.len(),
            transient_bytes: value_bytes * 3,
        })
    }

    pub fn prepare_batched_gated_rms_norm_output(
        &self,
        config: CudaGatedRmsNormConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaBatchedGatedRmsNormOutput> {
        if config != CudaGatedRmsNormConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "CUDA batched gated RMSNorm requires the exact Qwen3.8-27B profile".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(config.rows)
            .and_then(|values| values.checked_mul(config.columns))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA batched gated RMSNorm output overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaBatchedGatedRmsNormOutput {
            context: Rc::clone(&self.inner),
            token_capacity,
            heads: config.rows as u32,
            columns: config.columns as u32,
            output: DeviceBuffer::allocate(self, output_bytes)?,
            transient_bytes: output_bytes,
        })
    }

    pub fn dispatch_gated_rms_norm_f16(
        &self,
        prepared: &PreparedCudaGatedRmsNorm,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA gated RMSNorm belongs to another context".into(),
            ));
        }
        self.dispatch_gated_rms_norm_f16_inner(
            prepared,
            prepared.input.ptr(),
            prepared.gate.ptr(),
        )?;
        let mut result = vec![0.0_f32; prepared.config.rows * prepared.config.columns];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA gated RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(result)
    }

    pub fn dispatch_gated_rms_norm_f16_device<'a>(
        &self,
        prepared: &'a PreparedCudaGatedRmsNorm,
        input: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, input.context)
            || !Rc::ptr_eq(&self.inner, gate.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA gated RMSNorm device input belongs to another context".into(),
            ));
        }
        let expected = prepared.config.rows * prepared.config.columns;
        for (name, view) in [("input", input), ("gate", gate)] {
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA gated RMSNorm device {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }
        self.dispatch_gated_rms_norm_f16_inner(prepared, input.ptr()?, gate.ptr()?)?;
        prepared.output.f32_view(0, expected)
    }

    pub fn dispatch_batched_gated_rms_norm_f16_device<'a>(
        &self,
        prepared: &PreparedCudaGatedRmsNorm,
        output: &'a PreparedCudaBatchedGatedRmsNormOutput,
        input: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        token_count: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let token_count = validate_a8_batch_capacity(token_count)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &output.context)
            || !Rc::ptr_eq(&self.inner, input.context)
            || !Rc::ptr_eq(&self.inner, gate.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA batched gated RMSNorm crosses driver contexts".into(),
            ));
        }
        if output.heads != prepared.config.rows as u32
            || output.columns != prepared.config.columns as u32
            || token_count > output.token_capacity
        {
            return Err(EngineError::MemoryBudget(
                "CUDA batched gated RMSNorm output does not admit the request".into(),
            ));
        }
        let rows = (token_count as usize)
            .checked_mul(prepared.config.rows)
            .ok_or_else(|| EngineError::Shape("CUDA gated RMSNorm rows overflow".into()))?;
        let expected = rows
            .checked_mul(prepared.config.columns)
            .ok_or_else(|| EngineError::Shape("CUDA gated RMSNorm values overflow".into()))?;
        for (name, view) in [("input", input), ("gate", gate)] {
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA batched gated RMSNorm {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }
        self.dispatch_gated_rms_norm_f16_buffers(
            prepared,
            input.ptr()?,
            gate.ptr()?,
            output.output.ptr(),
            rows,
        )?;
        output.output.f32_view(0, expected)
    }

    fn dispatch_gated_rms_norm_f16_inner(
        &self,
        prepared: &PreparedCudaGatedRmsNorm,
        input: CuDevicePtr,
        gate: CuDevicePtr,
    ) -> Result<()> {
        self.dispatch_gated_rms_norm_f16_buffers(
            prepared,
            input,
            gate,
            prepared.output.ptr(),
            prepared.config.rows,
        )
    }

    fn dispatch_gated_rms_norm_f16_buffers(
        &self,
        prepared: &PreparedCudaGatedRmsNorm,
        mut input: CuDevicePtr,
        mut gate: CuDevicePtr,
        mut output: CuDevicePtr,
        rows: usize,
    ) -> Result<()> {
        self.make_current()?;
        let mut weight = prepared.weight.ptr();
        let mut rows = u32::try_from(rows)
            .map_err(|_| EngineError::Shape("CUDA gated RMSNorm rows exceed u32".into()))?;
        let mut columns = prepared.config.columns as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut gate as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        let rows_per_block = LINEAR_THREADS_PER_BLOCK / WARP_SIZE;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.gated_rms_norm_f16_function,
                    rows.div_ceil(rows_per_block),
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gated RMSNorm kernel launch",
            )?;
        }
        self.synchronize_after_launch("gated RMSNorm context synchronization")
    }

    pub fn prepare_qwen_rms_norm_f16(
        &self,
        config: CudaRmsNormConfig,
        weight_f16_le: &[u8],
    ) -> Result<PreparedCudaRmsNorm> {
        if config.rows == 0
            || config.columns == 0
            || !config.columns.is_multiple_of(WARP_SIZE as usize)
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
            || u32::try_from(config.rows).is_err()
            || u32::try_from(config.columns).is_err()
        {
            return Err(EngineError::Shape(
                "CUDA Qwen RMSNorm requires positive u32 geometry, 32-aligned columns, and positive epsilon"
                    .into(),
            ));
        }
        let weight_bytes = config
            .columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RMSNorm weight overflows".into()))?;
        validate_f16_buffer(weight_f16_le, weight_bytes, "Qwen RMSNorm weight")?;
        let values = config
            .rows
            .checked_mul(config.columns)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RMSNorm shape overflows".into()))?;
        let value_bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RMSNorm values overflow".into()))?;
        let input = DeviceBuffer::allocate(self, value_bytes)?;
        let weight = DeviceBuffer::from_bytes(self, weight_f16_le)?;
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        input.zero()?;
        output.zero()?;
        Ok(PreparedCudaRmsNorm {
            context: Rc::clone(&self.inner),
            config,
            input,
            weight,
            output,
            model_bytes: weight_f16_le.len(),
            transient_bytes: value_bytes * 2,
        })
    }

    pub fn dispatch_qwen_rms_norm_f16(&self, prepared: &PreparedCudaRmsNorm) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA Qwen RMSNorm belongs to another context".into(),
            ));
        }
        self.dispatch_qwen_rms_norm_f16_inner(prepared, prepared.input.ptr())?;
        let mut result = vec![0.0_f32; prepared.config.rows * prepared.config.columns];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA Qwen RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(result)
    }

    pub fn dispatch_qwen_rms_norm_f16_device<'a>(
        &self,
        prepared: &'a PreparedCudaRmsNorm,
        input: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, input.context) {
            return Err(EngineError::InvalidState(
                "CUDA Qwen RMSNorm device input belongs to another context".into(),
            ));
        }
        let expected = prepared.config.rows * prepared.config.columns;
        if input.values() != expected {
            return Err(EngineError::Shape(format!(
                "CUDA Qwen RMSNorm device input has {} values, expected {expected}",
                input.values()
            )));
        }
        self.dispatch_qwen_rms_norm_f16_inner(prepared, input.ptr()?)?;
        prepared.output.f32_view(0, expected)
    }

    fn dispatch_qwen_rms_norm_f16_inner(
        &self,
        prepared: &PreparedCudaRmsNorm,
        mut input: CuDevicePtr,
    ) -> Result<()> {
        self.make_current()?;
        let mut weight = prepared.weight.ptr();
        let mut output = prepared.output.ptr();
        let mut rows = prepared.config.rows as u32;
        let mut columns = prepared.config.columns as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.qwen_rms_norm_f16_function,
                    rows,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "Qwen RMSNorm kernel launch",
            )?;
        }
        self.synchronize_after_launch("Qwen RMSNorm context synchronization")
    }

    pub fn prepare_residual_rms_norm_f16(
        &self,
        config: CudaRmsNormConfig,
        weight_f16_le: &[u8],
    ) -> Result<PreparedCudaResidualRmsNorm> {
        if config.rows == 0
            || config.columns == 0
            || !config.columns.is_multiple_of(WARP_SIZE as usize)
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
            || u32::try_from(config.rows).is_err()
            || u32::try_from(config.columns).is_err()
        {
            return Err(EngineError::Shape(
                "CUDA residual RMSNorm requires positive u32 geometry, 32-aligned columns, and positive epsilon"
                    .into(),
            ));
        }
        let weight_bytes = config
            .columns
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA residual RMSNorm weight overflows".into())
            })?;
        validate_f16_buffer(weight_f16_le, weight_bytes, "residual RMSNorm weight")?;
        let values = config.rows.checked_mul(config.columns).ok_or_else(|| {
            EngineError::MemoryBudget("CUDA residual RMSNorm shape overflows".into())
        })?;
        let value_bytes = values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA residual RMSNorm values overflow".into())
            })?;
        Ok(PreparedCudaResidualRmsNorm {
            context: Rc::clone(&self.inner),
            config,
            weight: DeviceBuffer::from_bytes(self, weight_f16_le)?,
            residual_output: DeviceBuffer::allocate(self, value_bytes)?,
            normalized_output: DeviceBuffer::allocate(self, value_bytes)?,
            model_bytes: weight_f16_le.len(),
            transient_bytes: value_bytes * 2,
        })
    }

    pub fn dispatch_residual_rms_norm_f16_device<'a>(
        &self,
        prepared: &'a PreparedCudaResidualRmsNorm,
        residual: CudaDeviceF32View<'_>,
        update: CudaDeviceF32View<'_>,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, residual.context)
            || !Rc::ptr_eq(&self.inner, update.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA residual RMSNorm device input belongs to another context".into(),
            ));
        }
        let expected = prepared.config.rows * prepared.config.columns;
        for (name, view) in [("residual", residual), ("update", update)] {
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA residual RMSNorm {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }
        self.make_current()?;
        let mut residual_ptr = residual.ptr()?;
        let mut update_ptr = update.ptr()?;
        let mut weight_ptr = prepared.weight.ptr();
        let mut residual_output_ptr = prepared.residual_output.ptr();
        let mut normalized_output_ptr = prepared.normalized_output.ptr();
        let mut rows = prepared.config.rows as u32;
        let mut columns = prepared.config.columns as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut residual_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut update_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut residual_output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut normalized_output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.residual_rms_norm_f16_function,
                    rows,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "residual RMSNorm kernel launch",
            )?;
        }
        self.synchronize_after_launch("residual RMSNorm context synchronization")?;
        Ok((
            prepared.residual_output.f32_view(0, expected)?,
            prepared.normalized_output.f32_view(0, expected)?,
        ))
    }

    /// Allocates the two normalization buffers shared by every layer while a
    /// bounded prompt chunk advances through the layer-major schedule.
    pub fn prepare_batched_rms_norm_workspace(
        &self,
        batch_capacity: usize,
        columns: usize,
    ) -> Result<PreparedCudaBatchedRmsNormWorkspace> {
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        if columns == 0
            || !columns.is_multiple_of(WARP_SIZE as usize)
            || u32::try_from(columns).is_err()
        {
            return Err(EngineError::Shape(
                "CUDA batched RMSNorm workspace requires positive u32 32-aligned columns".into(),
            ));
        }
        let value_bytes = (batch_capacity as usize)
            .checked_mul(columns)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA batched RMSNorm bytes overflow".into())
            })?;
        self.make_current()?;
        let residual_output = DeviceBuffer::allocate(self, value_bytes)?;
        let normalized_output = DeviceBuffer::allocate(self, value_bytes)?;
        Ok(PreparedCudaBatchedRmsNormWorkspace {
            context: Rc::clone(&self.inner),
            batch_capacity,
            columns: columns as u32,
            transient_bytes: value_bytes * 2,
            residual_output,
            normalized_output,
        })
    }

    /// Applies one resident layer norm weight to an active prompt-chunk
    /// prefix. The prepared operator contributes only immutable weight and
    /// epsilon; output storage comes from the shared workspace.
    pub fn dispatch_batched_qwen_rms_norm_f16_device<'a>(
        &self,
        prepared: &PreparedCudaRmsNorm,
        workspace: &'a PreparedCudaBatchedRmsNormWorkspace,
        input: CudaDeviceF32View<'_>,
        batch_rows: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let batch_rows = validate_batched_norm_inputs(
            &self.inner,
            &prepared.context,
            workspace,
            &[input],
            prepared.config.columns,
            batch_rows,
        )?;
        self.make_current()?;
        let mut input_ptr = input.ptr()?;
        let mut weight_ptr = prepared.weight.ptr();
        let mut output_ptr = workspace.normalized_output.ptr();
        let mut rows = batch_rows;
        let mut columns = workspace.columns;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut input_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.qwen_rms_norm_f16_function,
                    rows,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched Qwen RMSNorm kernel launch",
            )?;
        }
        self.synchronize_after_launch("batched Qwen RMSNorm context synchronization")?;
        workspace.normalized_output(batch_rows as usize)
    }

    /// Fuses a batched residual edge with the following resident layer norm
    /// using the same two shared chunk buffers.
    pub fn dispatch_batched_residual_rms_norm_f16_device<'a>(
        &self,
        prepared: &PreparedCudaResidualRmsNorm,
        workspace: &'a PreparedCudaBatchedRmsNormWorkspace,
        residual: CudaDeviceF32View<'_>,
        update: CudaDeviceF32View<'_>,
        batch_rows: usize,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        let batch_rows = validate_batched_norm_inputs(
            &self.inner,
            &prepared.context,
            workspace,
            &[residual, update],
            prepared.config.columns,
            batch_rows,
        )?;
        self.make_current()?;
        let mut residual_ptr = residual.ptr()?;
        let mut update_ptr = update.ptr()?;
        let mut weight_ptr = prepared.weight.ptr();
        let mut residual_output_ptr = workspace.residual_output.ptr();
        let mut normalized_output_ptr = workspace.normalized_output.ptr();
        let mut rows = batch_rows;
        let mut columns = workspace.columns;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut residual_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut update_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut weight_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut residual_output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut normalized_output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.residual_rms_norm_f16_function,
                    rows,
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched residual RMSNorm kernel launch",
            )?;
        }
        self.synchronize_after_launch("batched residual RMSNorm context synchronization")?;
        Ok((
            workspace.residual_output(batch_rows as usize)?,
            workspace.normalized_output(batch_rows as usize)?,
        ))
    }

    pub fn prepare_partial_rope_f32(
        &self,
        config: CudaPartialRopeConfig,
    ) -> Result<PreparedCudaPartialRope> {
        if config.heads == 0
            || config.head_dim == 0
            || config.rotary_dim == 0
            || !config.rotary_dim.is_multiple_of(2)
            || config.rotary_dim > config.head_dim
            || !config.theta.is_finite()
            || config.theta <= 0.0
            || u32::try_from(config.heads).is_err()
            || u32::try_from(config.head_dim).is_err()
            || u32::try_from(config.rotary_dim).is_err()
        {
            return Err(EngineError::Shape(
                "invalid CUDA partial-RoPE geometry or theta".into(),
            ));
        }
        let value_count = config
            .heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RoPE shape overflows".into()))?;
        let value_bytes = value_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RoPE values overflow".into()))?;
        let table_bytes = (config.rotary_dim / 2)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA RoPE tables overflow".into()))?;
        let values = DeviceBuffer::allocate(self, value_bytes)?;
        let cosine = DeviceBuffer::allocate(self, table_bytes)?;
        let sine = DeviceBuffer::allocate(self, table_bytes)?;
        values.zero()?;
        let prepared = PreparedCudaPartialRope {
            context: Rc::clone(&self.inner),
            config,
            values,
            cosine,
            sine,
            transient_bytes: value_bytes + table_bytes * 2,
        };
        prepared.write_position(0)?;
        Ok(prepared)
    }

    pub fn prepare_batched_query_gate_output(
        &self,
        config: CudaQueryGateConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaBatchedQueryGateOutput> {
        if config != CudaQueryGateConfig::QWEN38_27B {
            return Err(EngineError::Shape(
                "CUDA batched query/gate output requires the exact Qwen3.8-27B profile".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(config.heads)
            .and_then(|values| values.checked_mul(config.head_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA batched query/gate output overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaBatchedQueryGateOutput {
            context: Rc::clone(&self.inner),
            token_capacity,
            heads: config.heads as u32,
            head_dim: config.head_dim as u32,
            query: DeviceBuffer::allocate(self, output_bytes)?,
            gate: DeviceBuffer::allocate(self, output_bytes)?,
            transient_bytes: output_bytes * 2,
        })
    }

    pub fn prepare_batched_rope_workspace(
        &self,
        config: CudaPartialRopeConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaBatchedRopeWorkspace> {
        if config.heads == 0
            || config.head_dim == 0
            || config.rotary_dim == 0
            || !config.rotary_dim.is_multiple_of(2)
            || config.rotary_dim > config.head_dim
            || !config.theta.is_finite()
            || config.theta <= 0.0
            || u32::try_from(config.rotary_dim).is_err()
        {
            return Err(EngineError::Shape(
                "invalid CUDA batched RoPE geometry or theta".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let table_bytes = (token_capacity as usize)
            .checked_mul(config.rotary_dim / 2)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA batched RoPE table overflows".into()))?;
        self.make_current()?;
        Ok(PreparedCudaBatchedRopeWorkspace {
            context: Rc::clone(&self.inner),
            token_capacity,
            rotary_dim: config.rotary_dim as u32,
            theta: config.theta,
            cosine: DeviceBuffer::allocate(self, table_bytes)?,
            sine: DeviceBuffer::allocate(self, table_bytes)?,
            transient_bytes: table_bytes * 2,
        })
    }

    /// Builds the compact prompt position table once. Query and key operators
    /// may then consume the same workspace without repeating transcendental
    /// work per head or per projection.
    pub fn write_batched_rope_positions(
        &self,
        workspace: &PreparedCudaBatchedRopeWorkspace,
        start_position: u64,
        token_count: usize,
    ) -> Result<()> {
        let token_count = validate_a8_batch_capacity(token_count)?;
        if !Rc::ptr_eq(&self.inner, &workspace.context) {
            return Err(EngineError::InvalidState(
                "CUDA batched RoPE workspace belongs to another context".into(),
            ));
        }
        if token_count > workspace.token_capacity
            || workspace.rotary_dim == 0
            || !workspace.rotary_dim.is_multiple_of(2)
        {
            return Err(EngineError::Shape(
                "CUDA batched RoPE token count or workspace is invalid".into(),
            ));
        }
        start_position
            .checked_add(token_count as u64)
            .ok_or_else(|| EngineError::Shape("CUDA batched RoPE position overflows".into()))?;
        self.make_current()?;
        let mut cosine = workspace.cosine.ptr();
        let mut sine = workspace.sine.ptr();
        let mut first_position = start_position;
        let mut table_tokens = token_count;
        let mut rotary_dim = workspace.rotary_dim;
        let mut theta = workspace.theta;
        let mut table_params = [
            (&mut cosine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut sine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut first_position as *mut u64).cast::<c_void>(),
            (&mut table_tokens as *mut u32).cast::<c_void>(),
            (&mut rotary_dim as *mut u32).cast::<c_void>(),
            (&mut theta as *mut f32).cast::<c_void>(),
        ];
        let table_values = token_count
            .checked_mul(rotary_dim / 2)
            .ok_or_else(|| EngineError::Shape("CUDA batched RoPE table grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.rope_table_batch_f32_function,
                    table_values.div_ceil(LINEAR_THREADS_PER_BLOCK),
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    table_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched RoPE table kernel launch",
            )?;
        }
        Ok(())
    }

    /// Applies a previously built prompt table in place to token-major Q or K.
    pub fn dispatch_batched_partial_rope_with_table_f32_device<'a>(
        &self,
        workspace: &PreparedCudaBatchedRopeWorkspace,
        config: CudaPartialRopeConfig,
        values: CudaDeviceF32View<'a>,
        token_count: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let token_count = validate_a8_batch_capacity(token_count)?;
        if !Rc::ptr_eq(&self.inner, &workspace.context) || !Rc::ptr_eq(&self.inner, values.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA batched partial RoPE crosses driver contexts".into(),
            ));
        }
        if token_count > workspace.token_capacity
            || config.rotary_dim as u32 != workspace.rotary_dim
            || config.theta.to_bits() != workspace.theta.to_bits()
            || config.heads == 0
            || config.head_dim == 0
            || config.rotary_dim == 0
            || !config.rotary_dim.is_multiple_of(2)
            || config.rotary_dim > config.head_dim
            || u32::try_from(config.heads).is_err()
            || u32::try_from(config.head_dim).is_err()
        {
            return Err(EngineError::Shape(
                "CUDA batched partial RoPE workspace/config mismatch".into(),
            ));
        }
        let expected_values = (token_count as usize)
            .checked_mul(config.heads)
            .and_then(|values| values.checked_mul(config.head_dim))
            .ok_or_else(|| EngineError::Shape("CUDA batched RoPE tensor overflows".into()))?;
        if values.values() != expected_values {
            return Err(EngineError::Shape(format!(
                "CUDA batched partial RoPE has {} values, expected {expected_values}",
                values.values()
            )));
        }
        self.make_current()?;
        let mut values_ptr = values.ptr()?;
        let mut cosine = workspace.cosine.ptr();
        let mut sine = workspace.sine.ptr();
        let mut table_tokens = token_count;
        let mut heads = config.heads as u32;
        let mut head_dim = config.head_dim as u32;
        let mut rotary_dim = config.rotary_dim as u32;
        let mut apply_params = [
            (&mut values_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut cosine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut sine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut table_tokens as *mut u32).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut rotary_dim as *mut u32).cast::<c_void>(),
        ];
        let pair_count = heads
            .checked_mul(rotary_dim / 2)
            .ok_or_else(|| EngineError::Shape("CUDA batched RoPE pair grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.partial_rope_batch_f32_function,
                    pair_count.div_ceil(LINEAR_THREADS_PER_BLOCK),
                    token_count,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    apply_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched partial-RoPE kernel launch",
            )?;
        }
        self.synchronize_after_launch("batched partial-RoPE context synchronization")?;
        Ok(values)
    }

    /// Convenience path for one consumer. Full-attention prefill should call
    /// `write_batched_rope_positions` once and apply the table to both Q/K.
    pub fn dispatch_batched_partial_rope_f32_device<'a>(
        &self,
        workspace: &PreparedCudaBatchedRopeWorkspace,
        config: CudaPartialRopeConfig,
        values: CudaDeviceF32View<'a>,
        start_position: u64,
        token_count: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.write_batched_rope_positions(workspace, start_position, token_count)?;
        self.dispatch_batched_partial_rope_with_table_f32_device(
            workspace,
            config,
            values,
            token_count,
        )
    }

    pub fn dispatch_partial_rope_f32(
        &self,
        prepared: &PreparedCudaPartialRope,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA partial RoPE belongs to another context".into(),
            ));
        }
        self.dispatch_partial_rope_f32_inner(prepared, prepared.values.ptr())?;
        let mut result = vec![0.0_f32; prepared.config.heads * prepared.config.head_dim];
        prepared.values.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA partial RoPE produced a non-finite output".into(),
            ));
        }
        Ok(result)
    }

    /// Applies Qwen partial RoPE in place to a producer-owned device tensor.
    /// The returned view aliases the same allocation; no staging allocation or
    /// host transfer is performed.
    pub fn dispatch_partial_rope_f32_device<'a>(
        &self,
        prepared: &PreparedCudaPartialRope,
        values: CudaDeviceF32View<'a>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) || !Rc::ptr_eq(&self.inner, values.context) {
            return Err(EngineError::InvalidState(
                "CUDA partial RoPE device input belongs to another context".into(),
            ));
        }
        let expected = prepared.config.heads * prepared.config.head_dim;
        if values.values() != expected {
            return Err(EngineError::Shape(format!(
                "CUDA partial RoPE device input has {} values, expected {expected}",
                values.values()
            )));
        }
        self.dispatch_partial_rope_f32_inner(prepared, values.ptr()?)?;
        Ok(values)
    }

    fn dispatch_partial_rope_f32_inner(
        &self,
        prepared: &PreparedCudaPartialRope,
        mut values: CuDevicePtr,
    ) -> Result<()> {
        self.make_current()?;
        let mut cosine = prepared.cosine.ptr();
        let mut sine = prepared.sine.ptr();
        let mut heads = prepared.config.heads as u32;
        let mut head_dim = prepared.config.head_dim as u32;
        let mut rotary_dim = prepared.config.rotary_dim as u32;
        let mut params = [
            (&mut values as *mut CuDevicePtr).cast::<c_void>(),
            (&mut cosine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut sine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut rotary_dim as *mut u32).cast::<c_void>(),
        ];
        let pair_count = heads
            .checked_mul(rotary_dim / 2)
            .ok_or_else(|| EngineError::Shape("CUDA RoPE pair count overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.partial_rope_f32_function,
                    pair_count.div_ceil(LINEAR_THREADS_PER_BLOCK),
                    1,
                    1,
                    LINEAR_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "partial-RoPE kernel launch",
            )?;
        }
        self.synchronize_after_launch("partial-RoPE context synchronization")
    }

    pub fn prepare_query_gate_norm_rope_f32(
        &self,
        config: CudaQueryGateConfig,
        q_norm_weight_f16_le: &[u8],
    ) -> Result<PreparedCudaQueryGate> {
        if config.heads != 24
            || config.head_dim != 256
            || config.rotary_dim != 64
            || !config.theta.is_finite()
            || config.theta <= 0.0
            || !config.epsilon.is_finite()
            || config.epsilon <= 0.0
        {
            return Err(EngineError::Shape(
                "CUDA query/gate fusion requires the exact Qwen3.8-27B 24x256/64 profile".into(),
            ));
        }
        let weight_bytes = config
            .head_dim
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q norm weight overflows".into()))?;
        validate_f16_buffer(
            q_norm_weight_f16_le,
            weight_bytes,
            "query/gate Q norm weight",
        )?;
        let output_values = config
            .heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA query/gate width overflows".into()))?;
        let output_bytes = output_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA query/gate output overflows".into()))?;
        let table_bytes = (config.rotary_dim / 2)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA query/gate table overflows".into()))?;
        let q_norm_weight = DeviceBuffer::from_bytes(self, q_norm_weight_f16_le)?;
        let cosine = DeviceBuffer::allocate(self, table_bytes)?;
        let sine = DeviceBuffer::allocate(self, table_bytes)?;
        let query = DeviceBuffer::allocate(self, output_bytes)?;
        let gate = DeviceBuffer::allocate(self, output_bytes)?;
        query.zero()?;
        gate.zero()?;
        let prepared = PreparedCudaQueryGate {
            context: Rc::clone(&self.inner),
            config,
            q_norm_weight,
            cosine,
            sine,
            query,
            gate,
            model_bytes: weight_bytes,
            transient_bytes: output_bytes * 2 + table_bytes * 2,
        };
        prepared.write_position(0)?;
        Ok(prepared)
    }

    pub fn dispatch_query_gate_norm_rope_device<'a>(
        &self,
        prepared: &'a PreparedCudaQueryGate,
        query_gate: CudaDeviceF32View<'_>,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, query_gate.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA query/gate device input belongs to another context".into(),
            ));
        }
        let output_values = prepared.config.heads * prepared.config.head_dim;
        let expected_input = output_values * 2;
        if query_gate.values() != expected_input {
            return Err(EngineError::Shape(format!(
                "CUDA query/gate device input has {} values, expected {expected_input}",
                query_gate.values()
            )));
        }
        self.make_current()?;
        let mut query_gate_ptr = query_gate.ptr()?;
        let mut q_norm_weight = prepared.q_norm_weight.ptr();
        let mut cosine = prepared.cosine.ptr();
        let mut sine = prepared.sine.ptr();
        let mut query = prepared.query.ptr();
        let mut gate = prepared.gate.ptr();
        let mut heads = prepared.config.heads as u32;
        let mut head_dim = prepared.config.head_dim as u32;
        let mut rotary_dim = prepared.config.rotary_dim as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut query_gate_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q_norm_weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut cosine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut sine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query as *mut CuDevicePtr).cast::<c_void>(),
            (&mut gate as *mut CuDevicePtr).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut rotary_dim as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.query_gate_norm_rope_f32_function,
                    heads,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "query/gate norm+RoPE kernel launch",
            )?;
        }
        self.synchronize_after_launch("query/gate norm+RoPE context synchronization")?;
        Ok((
            prepared.query.f32_view(0, output_values)?,
            prepared.gate.f32_view(0, output_values)?,
        ))
    }

    /// Deinterleaves token-major Q/Gate rows, applies the resident per-layer
    /// Q RMSNorm, and consumes the same device RoPE table used by batched K.
    pub fn dispatch_batched_query_gate_norm_rope_with_table_device<'a>(
        &self,
        prepared: &PreparedCudaQueryGate,
        rope: &PreparedCudaBatchedRopeWorkspace,
        output: &'a PreparedCudaBatchedQueryGateOutput,
        query_gate: CudaDeviceF32View<'_>,
        token_count: usize,
    ) -> Result<(CudaDeviceF32View<'a>, CudaDeviceF32View<'a>)> {
        let token_count = validate_a8_batch_capacity(token_count)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &rope.context)
            || !Rc::ptr_eq(&self.inner, &output.context)
            || !Rc::ptr_eq(&self.inner, query_gate.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA batched query/gate path crosses driver contexts".into(),
            ));
        }
        if prepared.config != CudaQueryGateConfig::QWEN38_27B
            || output.heads != prepared.config.heads as u32
            || output.head_dim != prepared.config.head_dim as u32
            || rope.rotary_dim != prepared.config.rotary_dim as u32
            || rope.theta.to_bits() != prepared.config.theta.to_bits()
            || token_count > output.token_capacity
            || token_count > rope.token_capacity
        {
            return Err(EngineError::Shape(
                "CUDA batched query/gate workspace/config mismatch".into(),
            ));
        }
        let output_values = (token_count as usize)
            .checked_mul(prepared.config.heads)
            .and_then(|values| values.checked_mul(prepared.config.head_dim))
            .ok_or_else(|| EngineError::Shape("CUDA batched query/gate output overflows".into()))?;
        let expected_input = output_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::Shape("CUDA batched query/gate input overflows".into()))?;
        if query_gate.values() != expected_input {
            return Err(EngineError::Shape(format!(
                "CUDA batched query/gate input has {} values, expected {expected_input}",
                query_gate.values()
            )));
        }

        self.make_current()?;
        let mut query_gate_ptr = query_gate.ptr()?;
        let mut q_norm_weight = prepared.q_norm_weight.ptr();
        let mut cosine = rope.cosine.ptr();
        let mut sine = rope.sine.ptr();
        let mut query = output.query.ptr();
        let mut gate = output.gate.ptr();
        let mut tokens = token_count;
        let mut heads = prepared.config.heads as u32;
        let mut head_dim = prepared.config.head_dim as u32;
        let mut rotary_dim = prepared.config.rotary_dim as u32;
        let mut epsilon = prepared.config.epsilon;
        let mut params = [
            (&mut query_gate_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q_norm_weight as *mut CuDevicePtr).cast::<c_void>(),
            (&mut cosine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut sine as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query as *mut CuDevicePtr).cast::<c_void>(),
            (&mut gate as *mut CuDevicePtr).cast::<c_void>(),
            (&mut tokens as *mut u32).cast::<c_void>(),
            (&mut heads as *mut u32).cast::<c_void>(),
            (&mut head_dim as *mut u32).cast::<c_void>(),
            (&mut rotary_dim as *mut u32).cast::<c_void>(),
            (&mut epsilon as *mut f32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.query_gate_norm_rope_batch_f32_function,
                    heads,
                    token_count,
                    1,
                    256,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched query/gate norm+RoPE kernel launch",
            )?;
        }
        self.synchronize_after_launch("batched query/gate norm+RoPE context synchronization")?;
        Ok((
            output.query.f32_view(0, output_values)?,
            output.gate.f32_view(0, output_values)?,
        ))
    }

    pub fn prepare_paged_q2q4_gqa(
        &self,
        config: CudaPagedGqaConfig,
    ) -> Result<PreparedCudaPagedGqa> {
        if config.query_heads != 24
            || config.key_value_heads != 4
            || config.head_dim != 256
            || config.maximum_tokens == 0
            || config.page_tokens == 0
            || !config.sink_tokens.is_multiple_of(config.page_tokens)
            || config.sink_tokens > config.maximum_tokens
            || config.recent_tokens > config.maximum_tokens
        {
            return Err(EngineError::Shape(
                "CUDA paged GQA requires Qwen's 24/4/256 heads and a valid page policy".into(),
            ));
        }
        let component_values = config
            .key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA KV width overflows".into()))?;
        let combined_values = component_values
            .checked_mul(2)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA combined KV width overflows".into()))?;
        if !combined_values.is_multiple_of(BLOCK_LEN) {
            return Err(EngineError::Shape(
                "CUDA combined KV width must be divisible by 64".into(),
            ));
        }
        let maximum_pages = config.maximum_tokens.div_ceil(config.page_tokens);
        let blocks_per_token = combined_values / BLOCK_LEN;
        let q2_token_bytes = blocks_per_token
            .checked_mul(Q2_BLOCK_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q2 token bytes overflow".into()))?;
        let q4_token_bytes = blocks_per_token
            .checked_mul(Q4_BLOCK_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q4 token bytes overflow".into()))?;
        let q2_page_bytes = config
            .page_tokens
            .checked_mul(q2_token_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q2 page bytes overflow".into()))?;
        let q4_page_bytes = config
            .page_tokens
            .checked_mul(q4_token_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q4 page bytes overflow".into()))?;
        let q4_slots = config
            .sink_tokens
            .checked_div(config.page_tokens)
            .and_then(|slots| slots.checked_add(config.recent_tokens.div_ceil(config.page_tokens)))
            .and_then(|slots| slots.checked_add(1))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q4 slot count overflows".into()))?
            .clamp(1, maximum_pages);
        let q2_arena_bytes = maximum_pages
            .checked_mul(q2_page_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q2 arena overflows".into()))?;
        let q4_arena_bytes = q4_slots
            .checked_mul(q4_page_bytes)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA Q4 arena overflows".into()))?;
        let descriptor_bytes = maximum_pages
            .checked_mul(PAGED_GQA_DESCRIPTOR_BYTES)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA KV descriptors overflow".into()))?;
        let value_bytes = config
            .query_heads
            .checked_mul(config.head_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA GQA values overflow".into()))?;
        let component_bytes = component_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA K/V staging overflows".into()))?;
        let packed_device_bytes = q2_arena_bytes
            .checked_add(q4_arena_bytes)
            .and_then(|bytes| bytes.checked_add(descriptor_bytes))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA packed KV residency overflows".into())
            })?;
        let transient_bytes = value_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(component_bytes.checked_mul(2)?))
            .and_then(|bytes| bytes.checked_add(PAGED_GQA_PARAMS_BYTES))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA GQA transient bytes overflow".into()))?;
        let q2_pages = DeviceBuffer::allocate(self, q2_arena_bytes)?;
        let q4_pages = DeviceBuffer::allocate(self, q4_arena_bytes)?;
        let descriptors = DeviceBuffer::allocate(self, descriptor_bytes)?;
        let query = DeviceBuffer::allocate(self, value_bytes)?;
        let key = DeviceBuffer::allocate(self, component_bytes)?;
        let value = DeviceBuffer::allocate(self, component_bytes)?;
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        let params = DeviceBuffer::allocate(self, PAGED_GQA_PARAMS_BYTES)?;
        q2_pages.zero()?;
        q4_pages.zero()?;
        descriptors.zero()?;
        query.zero()?;
        key.zero()?;
        value.zero()?;
        output.zero()?;
        params.zero()?;
        Ok(PreparedCudaPagedGqa {
            context: Rc::clone(&self.inner),
            config,
            q2_token_bytes,
            q4_token_bytes,
            q2_page_bytes,
            q4_page_bytes,
            q4_slots,
            component_values,
            combined_values,
            blocks_per_token,
            tokens: 0,
            pages: Vec::with_capacity(maximum_pages),
            free_q4_slots: (0..q4_slots).rev().collect(),
            q2_pages,
            q4_pages,
            descriptors,
            query,
            key,
            value,
            output,
            params,
            packed_device_bytes,
            transient_bytes,
            poisoned: false,
            speculative_checkpoint: None,
        })
    }

    pub fn append_and_dispatch_paged_q2q4_gqa(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA paged GQA belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA state is poisoned; reset is required".into(),
            ));
        }
        let query_values = prepared.config.query_heads * prepared.config.head_dim;
        let component_values = prepared.config.key_value_heads * prepared.config.head_dim;
        for (name, values, expected) in [
            ("query", query, query_values),
            ("key", key, component_values),
            ("value", value, component_values),
        ] {
            if values.len() != expected || values.iter().any(|item| !item.is_finite()) {
                return Err(EngineError::Shape(format!(
                    "CUDA paged GQA {name} has invalid length or values"
                )));
            }
        }
        if prepared.tokens >= prepared.config.maximum_tokens {
            return Err(EngineError::MemoryBudget(
                "CUDA paged GQA reached its token capacity".into(),
            ));
        }
        prepared.query.write(as_bytes(query))?;
        prepared.key.write(as_bytes(key))?;
        prepared.value.write(as_bytes(value))?;
        let query_ptr = prepared.query.ptr();
        let key_ptr = prepared.key.ptr();
        let value_ptr = prepared.value.ptr();
        prepared.poisoned = true;
        self.append_and_dispatch_paged_q2q4_gqa_inner(prepared, query_ptr, key_ptr, value_ptr)?;
        prepared.poisoned = false;
        match self.read_paged_q2q4_gqa_output(prepared) {
            Ok(output) => Ok(output),
            Err(error) => {
                prepared.poisoned = true;
                Err(error)
            }
        }
    }

    /// Appends one exact Qwen GQA step without copying Q, K, V, or the
    /// attention result through host memory. Every view is tied to its owning
    /// allocation and must originate from this CUDA context. This is the
    /// production graph entry point; the slice-based method above is retained
    /// only for the standalone numerical verifier.
    pub fn append_and_dispatch_paged_q2q4_gqa_device<'a>(
        &self,
        prepared: &'a mut PreparedCudaPagedGqa,
        query: CudaDeviceF32View<'_>,
        key: CudaDeviceF32View<'_>,
        value: CudaDeviceF32View<'_>,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA paged GQA belongs to another context".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA state is poisoned; reset is required".into(),
            ));
        }
        let query_values = prepared.config.query_heads * prepared.config.head_dim;
        let component_values = prepared.config.key_value_heads * prepared.config.head_dim;
        for (name, view, expected) in [
            ("query", query, query_values),
            ("key", key, component_values),
            ("value", value, component_values),
        ] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA paged GQA {name} belongs to another context"
                )));
            }
            if view.values() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA paged GQA {name} has {} values, expected {expected}",
                    view.values()
                )));
            }
        }
        if prepared.tokens >= prepared.config.maximum_tokens {
            return Err(EngineError::MemoryBudget(
                "CUDA paged GQA reached its token capacity".into(),
            ));
        }
        let query_ptr = query.ptr()?;
        let key_ptr = key.ptr()?;
        let value_ptr = value.ptr()?;
        prepared.poisoned = true;
        self.append_and_dispatch_paged_q2q4_gqa_inner(prepared, query_ptr, key_ptr, value_ptr)?;
        prepared.poisoned = false;
        prepared.output.f32_view(0, query_values)
    }

    fn append_and_dispatch_paged_q2q4_gqa_inner(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        query_ptr: CuDevicePtr,
        key_ptr: CuDevicePtr,
        value_ptr: CuDevicePtr,
    ) -> Result<()> {
        self.append_paged_q2q4_kv_inner(prepared, key_ptr, value_ptr)?;
        self.launch_paged_q2q4_gqa(
            prepared,
            query_ptr,
            prepared.output.ptr(),
            "paged Q2/Q4 GQA context synchronization",
        )
    }

    fn append_paged_q2q4_kv_inner(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        mut key_ptr: CuDevicePtr,
        mut value_ptr: CuDevicePtr,
    ) -> Result<()> {
        self.make_current()?;
        let page_index = prepared.tokens / prepared.config.page_tokens;
        let token_in_page = prepared.tokens % prepared.config.page_tokens;
        if token_in_page == 0 {
            let slot = prepared.free_q4_slots.pop().ok_or_else(|| {
                EngineError::MemoryBudget("CUDA Q4 arena has no free slot".into())
            })?;
            prepared.pages.push(CudaPagedKvPage {
                precision: KvPrecision::Q4,
                physical_slot: slot,
                tokens: 0,
                first_token: prepared.tokens,
            });
        }
        if page_index + 1 != prepared.pages.len()
            || prepared.pages[page_index].precision != KvPrecision::Q4
        {
            return Err(EngineError::InvalidState(
                "CUDA current KV page metadata is inconsistent".into(),
            ));
        }
        let mut q4_pages_ptr = prepared.q4_pages.ptr();
        let mut physical_slot = cuda_u32(
            prepared.pages[page_index].physical_slot,
            "CUDA Q4 physical slot",
        )?;
        let mut token_in_page_u32 = cuda_u32(token_in_page, "CUDA token in page")?;
        let mut component_values = cuda_u32(prepared.component_values, "CUDA KV width")?;
        let mut q4_token_bytes = cuda_u32(prepared.q4_token_bytes, "CUDA Q4 token bytes")?;
        let mut q4_page_bytes = cuda_u32(prepared.q4_page_bytes, "CUDA Q4 page bytes")?;
        let mut pack_params = [
            (&mut q4_pages_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut key_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut value_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut physical_slot as *mut u32).cast::<c_void>(),
            (&mut token_in_page_u32 as *mut u32).cast::<c_void>(),
            (&mut component_values as *mut u32).cast::<c_void>(),
            (&mut q4_token_bytes as *mut u32).cast::<c_void>(),
            (&mut q4_page_bytes as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.pack_paged_kv_q4_f32_function,
                    cuda_u32(prepared.blocks_per_token, "CUDA KV quant blocks")?,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    pack_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "paged Q4 KV pack kernel launch",
            )?;
        }
        prepared.pages[page_index].tokens += 1;
        prepared.tokens += 1;
        self.demote_stale_paged_q2q4_pages(prepared)?;
        self.write_paged_q2q4_metadata(prepared)
    }

    /// Appends a token-major prompt chunk to the canonical persistent cache.
    /// One two-dimensional pack launch is submitted per crossed page; there
    /// is no host loop over tokens and no host-staged K/V copy.
    pub fn append_paged_q2q4_kv_batch_device(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        keys: CudaDeviceF32View<'_>,
        values: CudaDeviceF32View<'_>,
        token_count: usize,
    ) -> Result<usize> {
        self.append_paged_q2q4_kv_batch_device_inner(prepared, keys, values, token_count, true)
    }

    fn append_paged_q2q4_kv_batch_device_inner(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        keys: CudaDeviceF32View<'_>,
        values: CudaDeviceF32View<'_>,
        token_count: usize,
        demote_after_page: bool,
    ) -> Result<usize> {
        validate_a8_batch_capacity(token_count)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, keys.context)
            || !Rc::ptr_eq(&self.inner, values.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA batched paged-KV append crosses driver contexts".into(),
            ));
        }
        if prepared.poisoned || prepared.speculative_checkpoint.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA batched paged-KV append requires healthy non-speculative state".into(),
            ));
        }
        let expected_values = token_count
            .checked_mul(prepared.component_values)
            .ok_or_else(|| EngineError::Shape("CUDA batched KV shape overflows".into()))?;
        if keys.values() != expected_values || values.values() != expected_values {
            return Err(EngineError::Shape(format!(
                "CUDA batched paged-KV append has K/V {}/{} values, expected {expected_values}",
                keys.values(),
                values.values()
            )));
        }
        let final_tokens = prepared
            .tokens
            .checked_add(token_count)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA KV token count overflows".into()))?;
        if final_tokens > prepared.config.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched paged-KV append reaches {final_tokens} tokens, capacity is {}",
                prepared.config.maximum_tokens
            )));
        }

        self.make_current()?;
        prepared.poisoned = true;
        let result = (|| {
            let mut input_token = 0_usize;
            let mut page_launches = 0_usize;
            while input_token < token_count {
                let page_index = prepared.tokens / prepared.config.page_tokens;
                let first_token_in_page = prepared.tokens % prepared.config.page_tokens;
                if first_token_in_page == 0 {
                    let slot = prepared.free_q4_slots.pop().ok_or_else(|| {
                        EngineError::MemoryBudget("CUDA Q4 arena has no free slot".into())
                    })?;
                    prepared.pages.push(CudaPagedKvPage {
                        precision: KvPrecision::Q4,
                        physical_slot: slot,
                        tokens: 0,
                        first_token: prepared.tokens,
                    });
                }
                if page_index + 1 != prepared.pages.len()
                    || prepared.pages[page_index].precision != KvPrecision::Q4
                {
                    return Err(EngineError::InvalidState(
                        "CUDA current batched KV page metadata is inconsistent".into(),
                    ));
                }
                let page_tokens = (token_count - input_token)
                    .min(prepared.config.page_tokens - first_token_in_page);
                let mut q4_pages_ptr = prepared.q4_pages.ptr();
                let mut keys_ptr = keys.ptr()?;
                let mut values_ptr = values.ptr()?;
                let mut physical_slot = cuda_u32(
                    prepared.pages[page_index].physical_slot,
                    "CUDA Q4 physical slot",
                )?;
                let mut first_token_in_page_u32 =
                    cuda_u32(first_token_in_page, "CUDA first token in page")?;
                let mut first_input_token = cuda_u32(input_token, "CUDA first input token")?;
                let mut page_token_count = cuda_u32(page_tokens, "CUDA batch page tokens")?;
                let mut component_values = cuda_u32(prepared.component_values, "CUDA KV width")?;
                let mut q4_token_bytes = cuda_u32(prepared.q4_token_bytes, "CUDA Q4 token bytes")?;
                let mut q4_page_bytes = cuda_u32(prepared.q4_page_bytes, "CUDA Q4 page bytes")?;
                let mut pack_params = [
                    (&mut q4_pages_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut keys_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut values_ptr as *mut CuDevicePtr).cast::<c_void>(),
                    (&mut physical_slot as *mut u32).cast::<c_void>(),
                    (&mut first_token_in_page_u32 as *mut u32).cast::<c_void>(),
                    (&mut first_input_token as *mut u32).cast::<c_void>(),
                    (&mut page_token_count as *mut u32).cast::<c_void>(),
                    (&mut component_values as *mut u32).cast::<c_void>(),
                    (&mut q4_token_bytes as *mut u32).cast::<c_void>(),
                    (&mut q4_page_bytes as *mut u32).cast::<c_void>(),
                ];
                unsafe {
                    self.inner.driver.check(
                        (self.inner.driver.launch_kernel)(
                            self.inner.pack_paged_kv_q4_batch_f32_function,
                            cuda_u32(prepared.blocks_per_token, "CUDA KV quant blocks")?,
                            page_token_count,
                            1,
                            64,
                            1,
                            1,
                            0,
                            ptr::null_mut(),
                            pack_params.as_mut_ptr(),
                            ptr::null_mut(),
                        ),
                        "batched paged Q4 KV pack kernel launch",
                    )?;
                }
                prepared.pages[page_index].tokens += page_tokens;
                prepared.tokens += page_tokens;
                input_token += page_tokens;
                page_launches += 1;
                if demote_after_page {
                    self.demote_stale_paged_q2q4_pages(prepared)?;
                }
            }
            self.write_paged_q2q4_metadata(prepared)?;
            Ok(page_launches)
        })();
        if result.is_ok() {
            debug_assert_eq!(prepared.tokens, final_tokens);
            prepared.poisoned = false;
        }
        result
    }

    fn demote_stale_paged_q2q4_pages(&self, prepared: &mut PreparedCudaPagedGqa) -> Result<()> {
        let recent_start = prepared
            .tokens
            .saturating_sub(prepared.config.recent_tokens);
        // A speculative block is at most four tokens and can cross at most one
        // 64-token page boundary. Keep all pre-branch Q4 slots intact until
        // commit/restore so metadata rollback never points at a demoted or
        // reused physical page. The extra boundary slot is admitted by the
        // memory plan; normal demotion resumes on the first committed replay.
        let demoted_pages = if prepared.speculative_checkpoint.is_some() {
            Vec::new()
        } else {
            prepared
                .pages
                .iter()
                .enumerate()
                .filter_map(|(index, page)| {
                    let end = page.first_token + page.tokens;
                    (page.precision == KvPrecision::Q4
                        && page.first_token >= prepared.config.sink_tokens
                        && end <= recent_start)
                        .then_some(index)
                })
                .collect::<Vec<_>>()
        };
        for index in demoted_pages {
            let page = prepared.pages[index];
            let q4_offset = page
                .physical_slot
                .checked_mul(prepared.q4_page_bytes)
                .ok_or_else(|| EngineError::MemoryBudget("CUDA Q4 page offset overflows".into()))?;
            let q2_offset = index
                .checked_mul(prepared.q2_page_bytes)
                .ok_or_else(|| EngineError::MemoryBudget("CUDA Q2 page offset overflows".into()))?;
            let mut q4_page_ptr = device_ptr_offset(prepared.q4_pages.ptr(), q4_offset)?;
            let mut q2_page_ptr = device_ptr_offset(prepared.q2_pages.ptr(), q2_offset)?;
            let mut page_tokens = cuda_u32(page.tokens, "CUDA demoted page tokens")?;
            let mut blocks_per_token =
                cuda_u32(prepared.blocks_per_token, "CUDA KV blocks per token")?;
            let mut demote_params = [
                (&mut q4_page_ptr as *mut CuDevicePtr).cast::<c_void>(),
                (&mut q2_page_ptr as *mut CuDevicePtr).cast::<c_void>(),
                (&mut page_tokens as *mut u32).cast::<c_void>(),
                (&mut blocks_per_token as *mut u32).cast::<c_void>(),
            ];
            unsafe {
                self.inner.driver.check(
                    (self.inner.driver.launch_kernel)(
                        self.inner.demote_paged_kv_q4_to_q2_function,
                        cuda_u32(
                            page.tokens
                                .checked_mul(prepared.blocks_per_token)
                                .ok_or_else(|| {
                                    EngineError::MemoryBudget("CUDA demotion grid overflows".into())
                                })?,
                            "CUDA demotion grid",
                        )?,
                        1,
                        1,
                        16,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        demote_params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "paged Q4-to-Q2 KV demotion kernel launch",
                )?;
            }
            prepared.pages[index].precision = KvPrecision::Q2;
            prepared.pages[index].physical_slot = index;
            prepared.free_q4_slots.push(page.physical_slot);
        }
        Ok(())
    }

    fn write_paged_q2q4_metadata(&self, prepared: &PreparedCudaPagedGqa) -> Result<()> {
        let mut descriptor_words =
            Vec::with_capacity(prepared.pages.len() * (PAGED_GQA_DESCRIPTOR_BYTES / 4));
        for page in &prepared.pages {
            descriptor_words.extend_from_slice(&[
                match page.precision {
                    KvPrecision::Q2 => 0,
                    KvPrecision::Q4 => 1,
                },
                cuda_u32(page.physical_slot, "CUDA KV physical slot")?,
                cuda_u32(page.tokens, "CUDA page tokens")?,
                cuda_u32(page.first_token, "CUDA first token")?,
            ]);
        }
        prepared
            .descriptors
            .write_range(0, as_bytes(&descriptor_words))?;
        let page_count = prepared.pages.len();
        let params_words = [
            cuda_u32(prepared.config.query_heads, "CUDA query heads")?,
            cuda_u32(prepared.config.key_value_heads, "CUDA KV heads")?,
            cuda_u32(prepared.config.head_dim, "CUDA head dimension")?,
            cuda_u32(prepared.tokens, "CUDA GQA tokens")?,
            cuda_u32(prepared.config.page_tokens, "CUDA page tokens")?,
            cuda_u32(page_count, "CUDA page count")?,
            cuda_u32(prepared.combined_values, "CUDA combined KV width")?,
            cuda_u32(prepared.q2_token_bytes, "CUDA Q2 token bytes")?,
            cuda_u32(prepared.q4_token_bytes, "CUDA Q4 token bytes")?,
            cuda_u32(prepared.q2_page_bytes, "CUDA Q2 page bytes")?,
            cuda_u32(prepared.q4_page_bytes, "CUDA Q4 page bytes")?,
            (1.0 / (prepared.config.head_dim as f32).sqrt()).to_bits(),
        ];
        prepared.params.write(as_bytes(&params_words))
    }

    fn launch_paged_q2q4_gqa(
        &self,
        prepared: &PreparedCudaPagedGqa,
        mut query_ptr: CuDevicePtr,
        mut output: CuDevicePtr,
        synchronization: &'static str,
    ) -> Result<()> {
        self.make_current()?;
        let mut q2_pages = prepared.q2_pages.ptr();
        let mut q4_pages = prepared.q4_pages.ptr();
        let mut descriptors = prepared.descriptors.ptr();
        let mut params_ptr = prepared.params.ptr();
        let mut kernel_params = [
            (&mut query_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q2_pages as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q4_pages as *mut CuDevicePtr).cast::<c_void>(),
            (&mut descriptors as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut params_ptr as *mut CuDevicePtr).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.paged_q2q4_gqa_f32_function,
                    prepared.config.query_heads as u32,
                    1,
                    1,
                    WARP_SIZE,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "paged Q2/Q4 GQA kernel launch",
            )?;
        }
        self.synchronize_after_launch(synchronization)
    }

    fn read_paged_q2q4_gqa_output(&self, prepared: &PreparedCudaPagedGqa) -> Result<Vec<f32>> {
        let mut result = vec![0.0_f32; prepared.config.query_heads * prepared.config.head_dim];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|item| !item.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA produced a non-finite output".into(),
            ));
        }
        Ok(result)
    }

    pub fn prepare_paged_q2q4_gqa_prefill_output(
        &self,
        config: CudaPagedGqaConfig,
        token_capacity: usize,
    ) -> Result<PreparedCudaPagedGqaPrefillOutput> {
        if config.query_heads != 24 || config.key_value_heads != 4 || config.head_dim != 256 {
            return Err(EngineError::Shape(
                "CUDA paged-GQA prefill requires Qwen's 24/4/256 profile".into(),
            ));
        }
        let token_capacity = validate_a8_batch_capacity(token_capacity)?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(config.query_heads)
            .and_then(|values| values.checked_mul(config.head_dim))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA paged-GQA prefill output overflows".into())
            })?;
        self.make_current()?;
        Ok(PreparedCudaPagedGqaPrefillOutput {
            context: Rc::clone(&self.inner),
            token_capacity,
            query_heads: config.query_heads as u32,
            head_dim: config.head_dim as u32,
            output: DeviceBuffer::allocate(self, output_bytes)?,
            transient_bytes: output_bytes,
        })
    }

    /// Executes causal attention for the newest token-major prompt chunk over
    /// an already populated canonical mixed Q2/Q4 cache. The cache is borrowed
    /// in place; only bounded output scratch belongs to `output`.
    pub fn dispatch_paged_q2q4_gqa_prefill_device<'a>(
        &self,
        prepared: &PreparedCudaPagedGqa,
        output: &'a PreparedCudaPagedGqaPrefillOutput,
        query: CudaDeviceF32View<'_>,
        query_tokens: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.dispatch_paged_q2q4_gqa_prefill_segment_device(
            prepared,
            output,
            query,
            0,
            query_tokens,
        )?;
        output.device_output(query_tokens)
    }

    fn dispatch_paged_q2q4_gqa_prefill_segment_device(
        &self,
        prepared: &PreparedCudaPagedGqa,
        output: &PreparedCudaPagedGqaPrefillOutput,
        query: CudaDeviceF32View<'_>,
        output_token_offset: usize,
        query_tokens: usize,
    ) -> Result<()> {
        let query_tokens = validate_a8_batch_capacity(query_tokens)?;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &output.context)
            || !Rc::ptr_eq(&self.inner, query.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA paged-GQA prefill crosses driver contexts".into(),
            ));
        }
        if prepared.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA state is poisoned; reset is required".into(),
            ));
        }
        if output.query_heads != prepared.config.query_heads as u32
            || output.head_dim != prepared.config.head_dim as u32
            || output_token_offset
                .checked_add(query_tokens as usize)
                .is_none_or(|end| end > output.token_capacity as usize)
            || query_tokens as usize > prepared.tokens
        {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA paged-GQA prefill requests {query_tokens} tokens with capacity {} and {} cached tokens",
                output.token_capacity, prepared.tokens
            )));
        }
        let query_values = (query_tokens as usize)
            .checked_mul(prepared.config.query_heads)
            .and_then(|values| values.checked_mul(prepared.config.head_dim))
            .ok_or_else(|| EngineError::Shape("CUDA paged-GQA query shape overflows".into()))?;
        if query.values() != query_values {
            return Err(EngineError::Shape(format!(
                "CUDA paged-GQA prefill query has {} values, expected {query_values}",
                query.values()
            )));
        }

        self.make_current()?;
        let mut query_ptr = query.ptr()?;
        let mut q2_pages_ptr = prepared.q2_pages.ptr();
        let mut q4_pages_ptr = prepared.q4_pages.ptr();
        let mut descriptors_ptr = prepared.descriptors.ptr();
        let output_value_offset = output_token_offset
            .checked_mul(prepared.config.query_heads)
            .and_then(|values| values.checked_mul(prepared.config.head_dim))
            .ok_or_else(|| EngineError::Shape("CUDA paged-GQA output offset overflows".into()))?;
        let output_byte_offset = output_value_offset
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::Shape("CUDA paged-GQA output byte offset overflows".into())
            })?;
        let mut output_ptr = device_ptr_offset(output.output.ptr(), output_byte_offset)?;
        let mut params_ptr = prepared.params.ptr();
        let mut query_token_count = query_tokens;
        let mut params = [
            (&mut query_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q2_pages_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q4_pages_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut descriptors_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut params_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query_token_count as *mut u32).cast::<c_void>(),
        ];
        let blocks = query_tokens
            .checked_mul(prepared.config.query_heads as u32)
            .ok_or_else(|| EngineError::Shape("CUDA paged-GQA prefill grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.paged_q2q4_gqa_prefill_f32_function,
                    blocks,
                    1,
                    1,
                    WARP_SIZE,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "paged Q2/Q4 GQA prefill kernel launch",
            )?;
        }
        self.synchronize_after_launch("paged Q2/Q4 GQA prefill context synchronization")?;
        Ok(())
    }

    /// Appends and attends one prompt chunk in page-bounded segments. A page
    /// that becomes stale at a boundary remains Q4 through the queries that
    /// precede that boundary and is demoted only afterwards. This preserves
    /// token-wise cache precision semantics without a host loop over tokens.
    pub fn append_and_dispatch_paged_q2q4_gqa_prefill_device<'a>(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        output: &'a PreparedCudaPagedGqaPrefillOutput,
        query: CudaDeviceF32View<'_>,
        keys: CudaDeviceF32View<'_>,
        values: CudaDeviceF32View<'_>,
        token_count: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let token_count = validate_a8_batch_capacity(token_count)? as usize;
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &output.context)
            || !Rc::ptr_eq(&self.inner, query.context)
            || !Rc::ptr_eq(&self.inner, keys.context)
            || !Rc::ptr_eq(&self.inner, values.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA page-segmented prefill crosses driver contexts".into(),
            ));
        }
        if prepared.poisoned || prepared.speculative_checkpoint.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA page-segmented prefill requires healthy non-speculative state".into(),
            ));
        }
        if token_count > output.token_capacity as usize {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA page-segmented prefill requests {token_count} tokens, output capacity is {}",
                output.token_capacity
            )));
        }
        let query_width = prepared
            .config
            .query_heads
            .checked_mul(prepared.config.head_dim)
            .ok_or_else(|| EngineError::Shape("CUDA prefill query width overflows".into()))?;
        let query_values = token_count
            .checked_mul(query_width)
            .ok_or_else(|| EngineError::Shape("CUDA prefill query shape overflows".into()))?;
        let component_values = token_count
            .checked_mul(prepared.component_values)
            .ok_or_else(|| EngineError::Shape("CUDA prefill K/V shape overflows".into()))?;
        if query.values() != query_values
            || keys.values() != component_values
            || values.values() != component_values
        {
            return Err(EngineError::Shape(format!(
                "CUDA page-segmented prefill has Q/K/V {}/{}/{}, expected {query_values}/{component_values}/{component_values}",
                query.values(),
                keys.values(),
                values.values()
            )));
        }
        let final_tokens = prepared.tokens.checked_add(token_count).ok_or_else(|| {
            EngineError::MemoryBudget("CUDA prefill token count overflows".into())
        })?;
        if final_tokens > prepared.config.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA page-segmented prefill reaches {final_tokens} tokens, capacity is {}",
                prepared.config.maximum_tokens
            )));
        }

        let result = (|| {
            let mut processed = 0_usize;
            while processed < token_count {
                let (segment_tokens, demote_before_attention) = paged_prefill_segment(
                    prepared.tokens,
                    token_count - processed,
                    &prepared.pages,
                    prepared.config.page_tokens,
                    prepared.config.sink_tokens,
                    prepared.config.recent_tokens,
                )?;
                let segment_query = query.slice(
                    processed
                        .checked_mul(query_width)
                        .ok_or_else(|| EngineError::Shape("CUDA query offset overflows".into()))?,
                    segment_tokens.checked_mul(query_width).ok_or_else(|| {
                        EngineError::Shape("CUDA query segment shape overflows".into())
                    })?,
                )?;
                let kv_offset = processed
                    .checked_mul(prepared.component_values)
                    .ok_or_else(|| EngineError::Shape("CUDA K/V offset overflows".into()))?;
                let kv_values = segment_tokens
                    .checked_mul(prepared.component_values)
                    .ok_or_else(|| EngineError::Shape("CUDA K/V segment shape overflows".into()))?;
                let segment_keys = keys.slice(kv_offset, kv_values)?;
                let segment_values = values.slice(kv_offset, kv_values)?;
                self.append_paged_q2q4_kv_batch_device_inner(
                    prepared,
                    segment_keys,
                    segment_values,
                    segment_tokens,
                    demote_before_attention,
                )?;
                self.dispatch_paged_q2q4_gqa_prefill_segment_device(
                    prepared,
                    output,
                    segment_query,
                    processed,
                    segment_tokens,
                )?;
                if !demote_before_attention {
                    self.demote_stale_paged_q2q4_pages(prepared)?;
                    self.write_paged_q2q4_metadata(prepared)?;
                }
                processed += segment_tokens;
            }
            Ok(())
        })();
        if let Err(error) = result {
            prepared.poisoned = true;
            return Err(error);
        }
        debug_assert_eq!(prepared.tokens, final_tokens);
        output.device_output(token_count)
    }

    /// Seeds a verifier cache from finite f32 K/V rows without running the
    /// quadratic attention scan after every append. Quantization, Q4-to-Q2
    /// demotion, descriptors, and persistent storage are the same device path
    /// used by decode. The f32 inputs are benchmark fixtures only and are not
    /// retained by the prepared cache.
    pub fn seed_paged_q2q4_gqa_verifier(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        keys: &[f32],
        values: &[f32],
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA seed belongs to another context".into(),
            ));
        }
        if prepared.poisoned || prepared.tokens != 0 || prepared.speculative_checkpoint.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA seed requires a reset, healthy cache".into(),
            ));
        }
        let component_values = prepared.component_values;
        if keys.is_empty()
            || keys.len() != values.len()
            || !keys.len().is_multiple_of(component_values)
            || keys.iter().chain(values).any(|item| !item.is_finite())
        {
            return Err(EngineError::Shape(
                "CUDA paged GQA seed requires equal finite complete K/V rows".into(),
            ));
        }
        let tokens = keys.len() / component_values;
        if tokens > prepared.config.maximum_tokens {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA paged GQA seed has {tokens} tokens but capacity is {}",
                prepared.config.maximum_tokens
            )));
        }

        prepared.poisoned = true;
        let result = self.run_token_submission("CUDA paged GQA verifier seed", || {
            for token in 0..tokens {
                let start = token * component_values;
                let end = start + component_values;
                prepared.key.write(as_bytes(&keys[start..end]))?;
                prepared.value.write(as_bytes(&values[start..end]))?;
                self.append_paged_q2q4_kv_inner(
                    prepared,
                    prepared.key.ptr(),
                    prepared.value.ptr(),
                )?;
            }
            Ok(())
        });
        if result.is_ok() {
            prepared.poisoned = false;
        }
        result
    }

    /// Verifier fixture for the production-shaped batched page packer. Host
    /// inputs are uploaded once into temporary device buffers, then the same
    /// device-view API consumed by projection outputs is exercised.
    pub fn seed_paged_q2q4_gqa_batch_verifier(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        keys: &[f32],
        values: &[f32],
    ) -> Result<usize> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "CUDA batched paged GQA seed belongs to another context".into(),
            ));
        }
        let component_values = prepared.component_values;
        if keys.is_empty()
            || keys.len() != values.len()
            || !keys.len().is_multiple_of(component_values)
            || keys.iter().chain(values).any(|item| !item.is_finite())
        {
            return Err(EngineError::Shape(
                "CUDA batched paged GQA seed requires equal finite complete K/V rows".into(),
            ));
        }
        let token_count = keys.len() / component_values;
        self.make_current()?;
        let key_staging = DeviceBuffer::from_bytes(self, as_bytes(keys))?;
        let value_staging = DeviceBuffer::from_bytes(self, as_bytes(values))?;
        let key_view = key_staging.f32_view(0, keys.len())?;
        let value_view = value_staging.f32_view(0, values.len())?;
        let page_launches =
            self.append_paged_q2q4_kv_batch_device(prepared, key_view, value_view, token_count)?;
        self.synchronize_after_launch("batched paged GQA verifier seed synchronization")?;
        Ok(page_launches)
    }

    pub fn prepare_paged_q2q4_gqa_split(
        &self,
        paged: &PreparedCudaPagedGqa,
    ) -> Result<PreparedCudaSplitPagedGqa> {
        if !Rc::ptr_eq(&self.inner, &paged.context) {
            return Err(EngineError::InvalidState(
                "CUDA split GQA cache belongs to another context".into(),
            ));
        }
        let row_values = paged
            .config
            .query_heads
            .checked_mul(paged.config.head_dim)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA split GQA row overflows".into()))?;
        let query_values = row_values
            .checked_mul(PAGED_GQA_SPLIT_MAX_QUERY_TOKENS)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA split GQA queries overflow".into()))?;
        let partial_rows = paged
            .config
            .query_heads
            .checked_mul(PAGED_GQA_SPLIT_MAX_QUERY_TOKENS)
            .and_then(|rows| rows.checked_mul(PAGED_GQA_SPLIT_SEGMENTS))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA split GQA partials overflow".into()))?;
        let query_bytes = query_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA split GQA query bytes overflow".into())
            })?;
        let partial_output_bytes = partial_rows
            .checked_mul(paged.config.head_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA split GQA partial output overflows".into())
            })?;
        let partial_scalar_bytes = partial_rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA split GQA partial scalars overflow".into())
            })?;
        let transient_bytes = query_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(partial_output_bytes))
            .and_then(|bytes| bytes.checked_add(partial_scalar_bytes.checked_mul(2)?))
            .ok_or_else(|| EngineError::MemoryBudget("CUDA split GQA scratch overflows".into()))?;
        let query = DeviceBuffer::allocate(self, query_bytes)?;
        let output = DeviceBuffer::allocate(self, query_bytes)?;
        let partial_output = DeviceBuffer::allocate(self, partial_output_bytes)?;
        let partial_maximum = DeviceBuffer::allocate(self, partial_scalar_bytes)?;
        let partial_denominator = DeviceBuffer::allocate(self, partial_scalar_bytes)?;
        query.zero()?;
        output.zero()?;
        partial_output.zero()?;
        partial_maximum.zero()?;
        partial_denominator.zero()?;
        Ok(PreparedCudaSplitPagedGqa {
            context: Rc::clone(&self.inner),
            query,
            output,
            partial_output,
            partial_maximum,
            partial_denominator,
            transient_bytes,
        })
    }

    pub fn dispatch_paged_q2q4_gqa_split(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &mut PreparedCudaSplitPagedGqa,
        query: &[f32],
        query_tokens: usize,
    ) -> Result<Vec<f32>> {
        let query_values =
            self.validate_paged_q2q4_gqa_split(paged, prepared, query.len(), query_tokens)?;
        if query.iter().any(|item| !item.is_finite()) {
            return Err(EngineError::Shape(
                "CUDA split GQA query contains non-finite values".into(),
            ));
        }
        prepared.query.write_range(0, as_bytes(query))?;
        self.dispatch_paged_q2q4_gqa_split_inner(
            paged,
            prepared,
            prepared.query.ptr(),
            query_tokens,
        )?;
        let mut result = vec![0.0_f32; query_values];
        prepared
            .output
            .copy_range_to(0, as_bytes_mut(&mut result))?;
        if result.iter().any(|item| !item.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA split GQA produced a non-finite output".into(),
            ));
        }
        Ok(result)
    }

    pub fn dispatch_paged_q2q4_gqa_split_device<'a>(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &'a mut PreparedCudaSplitPagedGqa,
        query: CudaDeviceF32View<'_>,
        query_tokens: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, query.context) {
            return Err(EngineError::InvalidState(
                "CUDA split GQA query belongs to another context".into(),
            ));
        }
        let query_values =
            self.validate_paged_q2q4_gqa_split(paged, prepared, query.values(), query_tokens)?;
        self.dispatch_paged_q2q4_gqa_split_inner(paged, prepared, query.ptr()?, query_tokens)?;
        prepared.output.f32_view(0, query_values)
    }

    fn validate_paged_q2q4_gqa_split(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &PreparedCudaSplitPagedGqa,
        query_values: usize,
        query_tokens: usize,
    ) -> Result<usize> {
        if !Rc::ptr_eq(&self.inner, &paged.context) || !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "CUDA split GQA operands belong to another context".into(),
            ));
        }
        if paged.poisoned {
            return Err(EngineError::InvalidState(
                "CUDA split GQA cannot read a poisoned cache".into(),
            ));
        }
        if !(2..=PAGED_GQA_SPLIT_MAX_QUERY_TOKENS).contains(&query_tokens)
            || query_tokens > paged.tokens
        {
            return Err(EngineError::Shape(format!(
                "CUDA split GQA requires 2..={PAGED_GQA_SPLIT_MAX_QUERY_TOKENS} tail queries already present in the cache"
            )));
        }
        let expected_values = query_tokens
            .checked_mul(paged.config.query_heads)
            .and_then(|values| values.checked_mul(paged.config.head_dim))
            .ok_or_else(|| EngineError::Shape("CUDA split GQA query shape overflows".into()))?;
        if query_values != expected_values {
            return Err(EngineError::Shape(format!(
                "CUDA split GQA query has {query_values} values, expected {expected_values}"
            )));
        }
        Ok(expected_values)
    }

    fn dispatch_paged_q2q4_gqa_split_inner(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &mut PreparedCudaSplitPagedGqa,
        mut query_ptr: CuDevicePtr,
        query_tokens: usize,
    ) -> Result<()> {
        self.make_current()?;
        let mut q2_pages = paged.q2_pages.ptr();
        let mut q4_pages = paged.q4_pages.ptr();
        let mut descriptors = paged.descriptors.ptr();
        let mut partial_output = prepared.partial_output.ptr();
        let mut partial_maximum = prepared.partial_maximum.ptr();
        let mut partial_denominator = prepared.partial_denominator.ptr();
        let mut params = paged.params.ptr();
        let mut query_tokens_u32 = cuda_u32(query_tokens, "CUDA split GQA query tokens")?;
        let mut segments_u32 = cuda_u32(PAGED_GQA_SPLIT_SEGMENTS, "CUDA split GQA segments")?;
        let mut partial_params = [
            (&mut query_ptr as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q2_pages as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q4_pages as *mut CuDevicePtr).cast::<c_void>(),
            (&mut descriptors as *mut CuDevicePtr).cast::<c_void>(),
            (&mut partial_output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut partial_maximum as *mut CuDevicePtr).cast::<c_void>(),
            (&mut partial_denominator as *mut CuDevicePtr).cast::<c_void>(),
            (&mut params as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query_tokens_u32 as *mut u32).cast::<c_void>(),
            (&mut segments_u32 as *mut u32).cast::<c_void>(),
        ];
        let partial_blocks = query_tokens
            .checked_mul(paged.config.query_heads)
            .and_then(|blocks| blocks.checked_mul(PAGED_GQA_SPLIT_SEGMENTS))
            .ok_or_else(|| EngineError::Shape("CUDA split GQA grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.paged_q2q4_gqa_split_partial_f32_function,
                    cuda_u32(partial_blocks, "CUDA split GQA partial grid")?,
                    1,
                    1,
                    WARP_SIZE,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    partial_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "paged Q2/Q4 split GQA partial kernel launch",
            )?;
        }

        let mut output = prepared.output.ptr();
        let mut combine_params = [
            (&mut partial_output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut partial_maximum as *mut CuDevicePtr).cast::<c_void>(),
            (&mut partial_denominator as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut params as *mut CuDevicePtr).cast::<c_void>(),
            (&mut query_tokens_u32 as *mut u32).cast::<c_void>(),
            (&mut segments_u32 as *mut u32).cast::<c_void>(),
        ];
        let combine_blocks = query_tokens
            .checked_mul(paged.config.query_heads)
            .ok_or_else(|| EngineError::Shape("CUDA split GQA combine grid overflows".into()))?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.paged_q2q4_gqa_split_combine_f32_function,
                    cuda_u32(combine_blocks, "CUDA split GQA combine grid")?,
                    1,
                    1,
                    WARP_SIZE,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    combine_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "paged Q2/Q4 split GQA combine kernel launch",
            )?;
        }
        self.synchronize_after_launch("paged Q2/Q4 split GQA context synchronization")
    }

    /// Compares one five-query split-KV verification block with five
    /// sequential single-query launches over the same resident cache. The
    /// sequential baseline intentionally exposes the full cache to every
    /// query; it is a conservative traffic/launch baseline rather than a
    /// second causality oracle. Both sides use one final context barrier per
    /// measured block.
    pub fn benchmark_paged_q2q4_gqa_split(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &mut PreparedCudaSplitPagedGqa,
        query: CudaDeviceF32View<'_>,
        query_tokens: usize,
        iterations: usize,
    ) -> Result<CudaSplitGqaBenchmark> {
        if !(1..=10_000).contains(&iterations) {
            return Err(EngineError::Shape(
                "CUDA split GQA benchmark iterations must be within 1..=10000".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, query.context) {
            return Err(EngineError::InvalidState(
                "CUDA split GQA benchmark query belongs to another context".into(),
            ));
        }
        self.validate_paged_q2q4_gqa_split(paged, prepared, query.values(), query_tokens)?;
        let query_ptr = query.ptr()?;

        self.run_token_submission("CUDA sequential GQA benchmark warmup", || {
            self.dispatch_paged_q2q4_gqa_sequential_full_context_inner(
                paged,
                prepared,
                query_ptr,
                query_tokens,
            )
        })?;
        self.run_token_submission("CUDA split GQA benchmark warmup", || {
            self.dispatch_paged_q2q4_gqa_split_inner(paged, prepared, query_ptr, query_tokens)
        })?;

        let mut sequential_seconds = 0.0_f64;
        let mut split_seconds = 0.0_f64;
        for iteration in 0..iterations {
            if iteration.is_multiple_of(2) {
                let started = Instant::now();
                self.run_token_submission("CUDA sequential GQA benchmark", || {
                    self.dispatch_paged_q2q4_gqa_sequential_full_context_inner(
                        paged,
                        prepared,
                        query_ptr,
                        query_tokens,
                    )
                })?;
                sequential_seconds += started.elapsed().as_secs_f64();

                let started = Instant::now();
                self.run_token_submission("CUDA split GQA benchmark", || {
                    self.dispatch_paged_q2q4_gqa_split_inner(
                        paged,
                        prepared,
                        query_ptr,
                        query_tokens,
                    )
                })?;
                split_seconds += started.elapsed().as_secs_f64();
            } else {
                let started = Instant::now();
                self.run_token_submission("CUDA split GQA benchmark", || {
                    self.dispatch_paged_q2q4_gqa_split_inner(
                        paged,
                        prepared,
                        query_ptr,
                        query_tokens,
                    )
                })?;
                split_seconds += started.elapsed().as_secs_f64();

                let started = Instant::now();
                self.run_token_submission("CUDA sequential GQA benchmark", || {
                    self.dispatch_paged_q2q4_gqa_sequential_full_context_inner(
                        paged,
                        prepared,
                        query_ptr,
                        query_tokens,
                    )
                })?;
                sequential_seconds += started.elapsed().as_secs_f64();
            }
        }
        let sequential_full_context_microseconds = sequential_seconds * 1.0e6 / iterations as f64;
        let split_causal_microseconds = split_seconds * 1.0e6 / iterations as f64;
        Ok(CudaSplitGqaBenchmark {
            iterations,
            sequential_full_context_microseconds,
            split_causal_microseconds,
            speedup: sequential_full_context_microseconds / split_causal_microseconds,
        })
    }

    fn dispatch_paged_q2q4_gqa_sequential_full_context_inner(
        &self,
        paged: &PreparedCudaPagedGqa,
        prepared: &PreparedCudaSplitPagedGqa,
        query_ptr: CuDevicePtr,
        query_tokens: usize,
    ) -> Result<()> {
        let row_bytes = paged
            .config
            .query_heads
            .checked_mul(paged.config.head_dim)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| EngineError::Shape("CUDA sequential GQA row overflows".into()))?;
        for query_token in 0..query_tokens {
            let offset = query_token
                .checked_mul(row_bytes)
                .ok_or_else(|| EngineError::Shape("CUDA sequential GQA offset overflows".into()))?;
            self.launch_paged_q2q4_gqa(
                paged,
                device_ptr_offset(query_ptr, offset)?,
                device_ptr_offset(prepared.output.ptr(), offset)?,
                "paged Q2/Q4 sequential GQA benchmark synchronization",
            )?;
        }
        Ok(())
    }

    pub fn prepare_embedding_recovered(
        &self,
        recovered: RecoveredMatrixView<'_>,
    ) -> Result<PreparedCudaEmbedding> {
        let layout = validate_recovered_a8_projection_layout(recovered)?;
        let ScaleSlice::F16Le(s_in_bytes) = recovered.s_in.as_recovery_scales()? else {
            unreachable!("recovered embedding scales reject F32")
        };
        let ScaleSlice::F16Le(s_out_bytes) = recovered.s_out.as_recovery_scales()? else {
            unreachable!("recovered embedding scales reject F32")
        };
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, recovered.matrix.weights)?;
        let s_in = DeviceBuffer::from_bytes(self, s_in_bytes)?;
        let s_out = DeviceBuffer::from_bytes(self, s_out_bytes)?;
        let output_bytes = recovered
            .matrix
            .columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA embedding output overflows".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let model_bytes = weights
            .len()
            .checked_add(s_in.len())
            .and_then(|bytes| bytes.checked_add(s_out.len()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA embedding model bytes overflow".into())
            })?;
        let graph_bytes = output.len();
        Ok(PreparedCudaEmbedding {
            context: Rc::clone(&self.inner),
            rows: recovered.matrix.rows as u32,
            columns: recovered.matrix.columns as u32,
            layout,
            weights,
            s_in,
            s_out,
            output,
            model_bytes,
            graph_bytes,
        })
    }

    pub fn prepare_batched_embedding_workspace(
        &self,
        prepared: &PreparedCudaEmbedding,
        token_capacity: usize,
    ) -> Result<PreparedCudaBatchedEmbeddingWorkspace> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || token_capacity == 0
            || token_capacity > 65_535
        {
            return Err(EngineError::Shape(
                "CUDA batched embedding requires the same context and 1..=65535 tokens".into(),
            ));
        }
        let token_capacity = u32::try_from(token_capacity)
            .map_err(|_| EngineError::MemoryBudget("CUDA embedding capacity exceeds u32".into()))?;
        let row_id_bytes = (token_capacity as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA embedding row IDs overflow".into()))?;
        let output_bytes = (token_capacity as usize)
            .checked_mul(prepared.columns as usize)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                EngineError::MemoryBudget("CUDA batched embedding output overflows".into())
            })?;
        let row_ids = DeviceBuffer::allocate(self, row_id_bytes)?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        Ok(PreparedCudaBatchedEmbeddingWorkspace {
            context: Rc::clone(&self.inner),
            token_capacity,
            columns: prepared.columns,
            transient_bytes: row_ids.len() + output.len(),
            row_ids,
            output,
        })
    }

    pub fn dispatch_embedding_rows_device<'a>(
        &self,
        prepared: &PreparedCudaEmbedding,
        workspace: &'a PreparedCudaBatchedEmbeddingWorkspace,
        row_ids: &[u32],
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context)
            || !Rc::ptr_eq(&self.inner, &workspace.context)
            || workspace.columns != prepared.columns
            || row_ids.is_empty()
            || row_ids.len() > workspace.token_capacity as usize
            || row_ids.iter().any(|row| *row >= prepared.rows)
        {
            return Err(EngineError::Shape(
                "CUDA batched embedding context, shape, or row IDs are invalid".into(),
            ));
        }
        let mut row_bytes = Vec::with_capacity(std::mem::size_of_val(row_ids));
        for row in row_ids {
            row_bytes.extend_from_slice(&row.to_le_bytes());
        }
        workspace.row_ids.write_range(0, &row_bytes)?;
        self.make_current()?;
        let launches: Vec<(TensorDType, u32, u32, usize)> = match &prepared.layout {
            CudaA8ProjectionLayout::Pure(dtype) => vec![(*dtype, 0, prepared.rows, 0)],
            CudaA8ProjectionLayout::Mixed(segments) => segments
                .iter()
                .map(|segment| {
                    (
                        segment.descriptor.dtype,
                        segment.row_start,
                        segment.row_start + segment.row_count,
                        segment.weight_offset,
                    )
                })
                .collect(),
        };
        for (dtype, mut row_start, mut row_end, weight_offset) in launches {
            let function = match dtype {
                TensorDType::Q2B64 => self.inner.q2_recovered_rows_function,
                TensorDType::Q4B64 => self.inner.q4_recovered_rows_function,
                _ => unreachable!("validated CUDA embedding segment dtype"),
            };
            let mut weights = device_ptr_offset(prepared.weights.ptr(), weight_offset)?;
            let mut s_in = prepared.s_in.ptr();
            let mut s_out = prepared.s_out.ptr();
            let mut ids = workspace.row_ids.ptr();
            let mut output = workspace.output.ptr();
            let mut requested_rows = u32::try_from(row_ids.len())
                .map_err(|_| EngineError::Shape("CUDA embedding row count exceeds u32".into()))?;
            let mut columns = prepared.columns;
            let mut params = [
                (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
                (&mut ids as *mut CuDevicePtr).cast::<c_void>(),
                (&mut output as *mut CuDevicePtr).cast::<c_void>(),
                (&mut requested_rows as *mut u32).cast::<c_void>(),
                (&mut columns as *mut u32).cast::<c_void>(),
                (&mut row_start as *mut u32).cast::<c_void>(),
                (&mut row_end as *mut u32).cast::<c_void>(),
            ];
            unsafe {
                self.inner.driver.check(
                    (self.inner.driver.launch_kernel)(
                        function,
                        requested_rows,
                        prepared.columns.div_ceil(THREADS_PER_BLOCK),
                        1,
                        THREADS_PER_BLOCK,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "batched recovered embedding launch",
                )?;
            }
        }
        workspace.output.f32_view(
            0,
            row_ids
                .len()
                .checked_mul(prepared.columns as usize)
                .ok_or_else(|| {
                    EngineError::Shape("CUDA batched embedding output view overflows".into())
                })?,
        )
    }

    /// Selects and decodes one row from the resident embedding table. `s_out`
    /// is the one finite scalar read from the immutable mapped artifact; all
    /// large data and the resulting activation stay on the device.
    pub fn dispatch_embedding_row_device<'a>(
        &self,
        prepared: &'a PreparedCudaEmbedding,
        row: usize,
        s_out: f32,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA embedding belongs to another context".into(),
            ));
        }
        if row >= prepared.rows as usize || !s_out.is_finite() {
            return Err(EngineError::Shape(format!(
                "CUDA embedding row {row} or recovery scale is invalid"
            )));
        }
        let (dtype, offset) =
            embedding_row_location(&prepared.layout, prepared.rows, prepared.columns, row)?;
        self.make_current()?;
        let function = match dtype {
            TensorDType::Q2B64 => self.inner.q2_recovered_row_function,
            TensorDType::Q4B64 => self.inner.q4_recovered_row_function,
            _ => unreachable!("validated CUDA embedding row dtype"),
        };
        let mut weights = device_ptr_offset(prepared.weights.ptr(), offset)?;
        let mut s_in = prepared.s_in.ptr();
        let mut s_out = s_out;
        let mut output = prepared.output.ptr();
        let mut columns = prepared.columns;
        let mut params = [
            (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_out as *mut f32).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    function,
                    prepared.columns.div_ceil(THREADS_PER_BLOCK),
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "resident embedding-row launch",
            )?;
        }
        prepared.output.f32_view(0, prepared.columns as usize)
    }

    pub fn prepare_recovered_row(
        &self,
        operation: &RecoveredRow<'_>,
    ) -> Result<PreparedCudaRecoveredRow> {
        let descriptor = validate_recovered_row(operation)?;
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, operation.weights)?;
        let ScaleSlice::F16Le(s_in_bytes) = operation.s_in else {
            unreachable!("validated CUDA recovered-row s_in is FP16")
        };
        let s_in = DeviceBuffer::from_bytes(self, s_in_bytes)?;
        let output_bytes = operation
            .columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA recovered-row output size overflows".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let resident_bytes = weights
            .len()
            .checked_add(s_in.len())
            .and_then(|total| total.checked_add(output.len()))
            .ok_or_else(|| EngineError::Shape("CUDA recovered-row residency overflows".into()))?;
        Ok(PreparedCudaRecoveredRow {
            context: Rc::clone(&self.inner),
            dtype: descriptor.dtype,
            columns: operation.columns as u32,
            s_out: operation.s_out,
            weights,
            s_in,
            output,
            resident_bytes,
        })
    }

    pub fn dispatch_recovered_row(&self, prepared: &PreparedCudaRecoveredRow) -> Result<Vec<f32>> {
        self.dispatch_prepared_recovered_row_repeated(prepared, 1)
    }

    pub fn dispatch_prepared_recovered_row_repeated(
        &self,
        prepared: &PreparedCudaRecoveredRow,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "CUDA recovered-row dispatch count must be positive".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA recovered row belongs to another context".into(),
            ));
        }
        self.make_current()?;
        let function = match prepared.dtype {
            TensorDType::Q2B64 => self.inner.q2_recovered_row_function,
            TensorDType::Q4B64 => self.inner.q4_recovered_row_function,
            _ => unreachable!("validated CUDA recovered row is Q2 or Q4"),
        };
        for _ in 0..dispatches {
            let mut weights = prepared.weights.ptr();
            let mut s_in = prepared.s_in.ptr();
            let mut s_out = prepared.s_out;
            let mut output = prepared.output.ptr();
            let mut columns = prepared.columns;
            let mut params = [
                (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_out as *mut f32).cast::<c_void>(),
                (&mut output as *mut CuDevicePtr).cast::<c_void>(),
                (&mut columns as *mut u32).cast::<c_void>(),
            ];
            unsafe {
                self.inner.driver.check(
                    (self.inner.driver.launch_kernel)(
                        function,
                        prepared.columns.div_ceil(THREADS_PER_BLOCK),
                        1,
                        1,
                        THREADS_PER_BLOCK,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "recovered-row launch",
                )?;
            }
        }
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "recovered-row context synchronization",
            )?;
        }
        let mut output = vec![0.0_f32; prepared.columns as usize];
        prepared.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA recovered-row candidate produced non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        let prepared = self.prepare_fused_matvec(operation)?;
        self.dispatch_prepared(&prepared)
    }

    pub fn prepare_fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<PreparedCudaMatVec> {
        let descriptor = validate_operation(operation)?;
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, operation.weights)?;
        let input = DeviceBuffer::from_bytes(self, as_bytes(operation.input))?;
        let s_in = optional_scale_buffer(self, operation.s_in)?;
        let s_out = optional_scale_buffer(self, operation.s_out)?;
        let bias = operation
            .bias
            .map(|values| DeviceBuffer::from_bytes(self, as_bytes(values)))
            .transpose()?;
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA output size overflows usize".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let resident_bytes = weights
            .len()
            .checked_add(input.len())
            .and_then(|total| total.checked_add(s_in.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(s_out.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(bias.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(output.len()))
            .ok_or_else(|| EngineError::Shape("CUDA resident byte count overflows".into()))?;
        Ok(PreparedCudaMatVec {
            context: Rc::clone(&self.inner),
            dtype: descriptor.dtype,
            rows: operation.rows as u32,
            columns: operation.columns as u32,
            activation: match operation.activation {
                Activation::Identity => 0,
                Activation::Silu => 1,
            },
            weights,
            input,
            s_in,
            s_out,
            bias,
            output,
            resident_bytes,
        })
    }

    pub fn dispatch_prepared(&self, prepared: &PreparedCudaMatVec) -> Result<Vec<f32>> {
        self.dispatch_prepared_repeated(prepared, 1)
    }

    pub fn prepare_a8_fused_matvec(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedCudaA8MatVec> {
        let base = self.prepare_fused_matvec(operation)?;
        let q8_codes = DeviceBuffer::allocate(self, operation.columns)?;
        let scale_bytes = a8_scale_bytes(operation.columns)?;
        let q8_scales = DeviceBuffer::allocate(self, scale_bytes)?;
        let resident_bytes = base
            .resident_bytes()
            .checked_add(q8_codes.len())
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| EngineError::Shape("CUDA A8 resident byte count overflows".into()))?;
        Ok(PreparedCudaA8MatVec {
            base,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    pub fn prepare_mixed_a8_fused_matvec(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedCudaMixedA8MatVec> {
        let segments = validate_mixed_operation(operation)?;
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, operation.weights)?;
        let input = DeviceBuffer::from_bytes(self, as_bytes(operation.input))?;
        let s_in = optional_scale_buffer(self, operation.s_in)?;
        let s_out = optional_scale_buffer(self, operation.s_out)?;
        let bias = operation
            .bias
            .map(|values| DeviceBuffer::from_bytes(self, as_bytes(values)))
            .transpose()?;
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("mixed CUDA output size overflows usize".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let q8_codes = DeviceBuffer::allocate(self, operation.columns)?;
        let q8_scales = DeviceBuffer::allocate(self, a8_scale_bytes(operation.columns)?)?;
        let resident_bytes = weights
            .len()
            .checked_add(input.len())
            .and_then(|total| total.checked_add(s_in.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(s_out.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(bias.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(output.len()))
            .and_then(|total| total.checked_add(q8_codes.len()))
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| EngineError::Shape("mixed CUDA resident byte count overflows".into()))?;
        Ok(PreparedCudaMixedA8MatVec {
            context: Rc::clone(&self.inner),
            rows: operation.rows as u32,
            columns: operation.columns as u32,
            activation: match operation.activation {
                Activation::Identity => 0,
                Activation::Silu => 1,
            },
            segments,
            weights,
            input,
            s_in,
            s_out,
            bias,
            output,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Prepares a row-major prompt batch against one immutable logical Q2/Q4
    /// matrix. This is the correctness baseline for prefill: it removes the
    /// per-token launch loop without changing or repacking model weights.
    pub fn prepare_batched_a8_matmul(
        &self,
        operation: &FusedMatVec<'_>,
        batch_inputs: &[f32],
        batch_rows: usize,
    ) -> Result<PreparedCudaBatchedA8MatMul> {
        let (layout, batch_rows_u32, input_values) =
            validate_batched_a8_inputs(operation, batch_inputs, batch_rows)?;

        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, operation.weights)?;
        let input = DeviceBuffer::from_bytes(self, as_bytes(batch_inputs))?;
        let s_in = optional_scale_buffer(self, operation.s_in)?;
        let s_out = optional_scale_buffer(self, operation.s_out)?;
        let bias = operation
            .bias
            .map(|values| DeviceBuffer::from_bytes(self, as_bytes(values)))
            .transpose()?;
        let output_values = batch_rows.checked_mul(operation.rows).ok_or_else(|| {
            EngineError::Shape("CUDA batched A8 output value count overflows".into())
        })?;
        let output_bytes = output_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 output bytes overflow".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let q8_codes = DeviceBuffer::allocate(self, input_values)?;
        let q8_scale_bytes = batch_rows
            .checked_mul(a8_scale_bytes(operation.columns)?)
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 scale bytes overflow".into()))?;
        let q8_scales = DeviceBuffer::allocate(self, q8_scale_bytes)?;
        let resident_bytes = weights
            .len()
            .checked_add(input.len())
            .and_then(|total| total.checked_add(s_in.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(s_out.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(bias.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(output.len()))
            .and_then(|total| total.checked_add(q8_codes.len()))
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 residency overflows".into()))?;
        Ok(PreparedCudaBatchedA8MatMul {
            context: Rc::clone(&self.inner),
            batch_rows: batch_rows_u32,
            rows: operation.rows as u32,
            columns: operation.columns as u32,
            activation: match operation.activation {
                Activation::Identity => 0,
                Activation::Silu => 1,
            },
            layout,
            weights,
            input,
            s_in,
            s_out,
            bias,
            output,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Prepares one corrected activation independently of its fan-out
    /// projections. The exact packed FP16 `s_in` bytes form part of the
    /// identity checked by every consuming projection.
    pub fn prepare_shared_a8_activation(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedCudaA8Activation> {
        validate_a8_projection_layout(operation)?;
        let correction_identity = a8_correction_identity(operation.columns, operation.s_in)?;
        self.make_current()?;
        let input = DeviceBuffer::from_bytes(self, as_bytes(operation.input))?;
        let s_in = optional_scale_buffer(self, operation.s_in)?;
        let q8_codes = DeviceBuffer::allocate(self, operation.columns)?;
        let q8_scales = DeviceBuffer::allocate(self, a8_scale_bytes(operation.columns)?)?;
        let resident_bytes = input
            .len()
            .checked_add(s_in.as_ref().map_or(0, DeviceBuffer::len))
            .and_then(|total| total.checked_add(q8_codes.len()))
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| {
                EngineError::Shape("shared CUDA A8 activation byte count overflows".into())
            })?;
        Ok(PreparedCudaA8Activation {
            context: Rc::clone(&self.inner),
            columns: operation.columns as u32,
            correction_identity,
            input: Some(input),
            s_in,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Verifier constructor for a bounded batched activation workspace. It
    /// accepts host-described matrix metadata but deliberately does not stage
    /// the activation values; those arrive through a device view at dispatch.
    pub fn prepare_batched_shared_a8_activation(
        &self,
        operation: &FusedMatVec<'_>,
        batch_capacity: usize,
    ) -> Result<PreparedCudaBatchedA8Activation> {
        validate_a8_projection_layout(operation)?;
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        let correction_identity = a8_correction_identity(operation.columns, operation.s_in)?;
        self.make_current()?;
        let s_in = optional_scale_buffer(self, operation.s_in)?;
        let q8_codes_bytes = (batch_capacity as usize)
            .checked_mul(operation.columns)
            .ok_or_else(|| EngineError::Shape("batched CUDA A8 code bytes overflow".into()))?;
        let q8_scale_bytes = (batch_capacity as usize)
            .checked_mul(a8_scale_bytes(operation.columns)?)
            .ok_or_else(|| EngineError::Shape("batched CUDA A8 scale bytes overflow".into()))?;
        let q8_codes = DeviceBuffer::allocate(self, q8_codes_bytes)?;
        let q8_scales = DeviceBuffer::allocate(self, q8_scale_bytes)?;
        let resident_bytes = s_in
            .as_ref()
            .map_or(0, DeviceBuffer::len)
            .checked_add(q8_codes.len())
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| {
                EngineError::Shape("batched CUDA A8 activation residency overflows".into())
            })?;
        Ok(PreparedCudaBatchedA8Activation {
            context: Rc::clone(&self.inner),
            batch_capacity,
            columns: operation.columns as u32,
            correction_identity,
            s_in,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Prepares the shared corrected activation buffers directly from one
    /// mmap-backed CTOXQ matrix. Model input is supplied later as a device
    /// view, so this path deliberately allocates no host-staged device input.
    pub fn prepare_shared_a8_activation_recovered(
        &self,
        recovered: RecoveredMatrixView<'_>,
    ) -> Result<PreparedCudaA8Activation> {
        validate_recovered_a8_projection_layout(recovered)?;
        let s_in = recovered.s_in.as_recovery_scales()?;
        let correction_identity = a8_correction_identity(recovered.matrix.columns, Some(s_in))?;
        self.make_current()?;
        let s_in = optional_scale_buffer(self, Some(s_in))?;
        let q8_codes = DeviceBuffer::allocate(self, recovered.matrix.columns)?;
        let q8_scales = DeviceBuffer::allocate(self, a8_scale_bytes(recovered.matrix.columns)?)?;
        let resident_bytes = s_in
            .as_ref()
            .map_or(0, DeviceBuffer::len)
            .checked_add(q8_codes.len())
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| {
                EngineError::Shape("mmap CUDA A8 activation byte count overflows".into())
            })?;
        Ok(PreparedCudaA8Activation {
            context: Rc::clone(&self.inner),
            columns: recovered.matrix.columns as u32,
            correction_identity,
            input: None,
            s_in,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Allocates one device-only activation workspace for a bounded prefill
    /// chunk. The immutable `s_in` bytes are copied once from the canonical
    /// CTOXQ tensor; activation values are supplied later by a producer-owned
    /// device view and are never staged through host memory.
    pub fn prepare_batched_shared_a8_activation_recovered(
        &self,
        recovered: RecoveredMatrixView<'_>,
        batch_capacity: usize,
    ) -> Result<PreparedCudaBatchedA8Activation> {
        validate_recovered_a8_projection_layout(recovered)?;
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        let s_in = recovered.s_in.as_recovery_scales()?;
        let correction_identity = a8_correction_identity(recovered.matrix.columns, Some(s_in))?;
        self.make_current()?;
        let s_in = optional_scale_buffer(self, Some(s_in))?;
        let q8_codes_bytes = (batch_capacity as usize)
            .checked_mul(recovered.matrix.columns)
            .ok_or_else(|| EngineError::Shape("batched CUDA A8 code bytes overflow".into()))?;
        let q8_scale_bytes = (batch_capacity as usize)
            .checked_mul(a8_scale_bytes(recovered.matrix.columns)?)
            .ok_or_else(|| EngineError::Shape("batched CUDA A8 scale bytes overflow".into()))?;
        let q8_codes = DeviceBuffer::allocate(self, q8_codes_bytes)?;
        let q8_scales = DeviceBuffer::allocate(self, q8_scale_bytes)?;
        let resident_bytes = s_in
            .as_ref()
            .map_or(0, DeviceBuffer::len)
            .checked_add(q8_codes.len())
            .and_then(|total| total.checked_add(q8_scales.len()))
            .ok_or_else(|| {
                EngineError::Shape("batched CUDA A8 activation residency overflows".into())
            })?;
        Ok(PreparedCudaBatchedA8Activation {
            context: Rc::clone(&self.inner),
            batch_capacity,
            columns: recovered.matrix.columns as u32,
            correction_identity,
            s_in,
            q8_codes,
            q8_scales,
            resident_bytes,
        })
    }

    /// Allocates one maximum-width activation arena. Recovery scale ownership
    /// stays in the resident per-group activation object, so this does not
    /// duplicate 262 `s_in` tensors.
    pub fn prepare_batched_a8_workspace(
        &self,
        batch_capacity: usize,
        column_capacity: usize,
    ) -> Result<PreparedCudaBatchedA8Workspace> {
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        if column_capacity == 0
            || !column_capacity.is_multiple_of(BLOCK_LEN)
            || u32::try_from(column_capacity).is_err()
        {
            return Err(EngineError::Shape(
                "CUDA batched A8 workspace requires positive u32 block-aligned columns".into(),
            ));
        }
        let code_bytes = (batch_capacity as usize)
            .checked_mul(column_capacity)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA A8 arena codes overflow".into()))?;
        let scale_bytes = (batch_capacity as usize)
            .checked_mul(a8_scale_bytes(column_capacity)?)
            .ok_or_else(|| EngineError::MemoryBudget("CUDA A8 arena scales overflow".into()))?;
        self.make_current()?;
        Ok(PreparedCudaBatchedA8Workspace {
            context: Rc::clone(&self.inner),
            batch_capacity,
            column_capacity: column_capacity as u32,
            q8_codes: DeviceBuffer::allocate(self, code_bytes)?,
            q8_scales: DeviceBuffer::allocate(self, scale_bytes)?,
            transient_bytes: code_bytes.checked_add(scale_bytes).ok_or_else(|| {
                EngineError::MemoryBudget("CUDA A8 arena residency overflows".into())
            })?,
        })
    }

    /// Prepares matrix-local state without duplicating the input, `s_in`, or
    /// A8 buffers. Dispatch still fails closed unless its activation carries
    /// the byte-identical correction identity.
    pub fn prepare_shared_a8_projection(
        &self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedCudaA8Projection> {
        let layout = validate_a8_projection_layout(operation)?;
        let correction_identity = a8_correction_identity(operation.columns, operation.s_in)?;
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, operation.weights)?;
        let s_out = optional_scale_buffer(self, operation.s_out)?;
        let bias = operation
            .bias
            .map(|values| DeviceBuffer::from_bytes(self, as_bytes(values)))
            .transpose()?;
        let output_bytes = operation
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::Shape("shared CUDA A8 projection output size overflows".into())
            })?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let resident_bytes = weights
            .len()
            .checked_add(s_out.as_ref().map_or(0, DeviceBuffer::len))
            .and_then(|total| total.checked_add(bias.as_ref().map_or(0, DeviceBuffer::len)))
            .and_then(|total| total.checked_add(output.len()))
            .ok_or_else(|| {
                EngineError::Shape("shared CUDA A8 projection byte count overflows".into())
            })?;
        Ok(PreparedCudaA8Projection {
            context: Rc::clone(&self.inner),
            dtype: operation.dtype,
            rows: operation.rows as u32,
            columns: operation.columns as u32,
            activation: match operation.activation {
                Activation::Identity => 0,
                Activation::Silu => 1,
            },
            correction_identity,
            layout,
            weights,
            s_out,
            bias,
            output,
            resident_bytes,
        })
    }

    /// Uploads immutable packed Q2/Q4 weights and `s_out` directly from the
    /// mmap-backed CTOXQ artifact. The logical codes are neither widened nor
    /// repacked and no matrix-sized host allocation is introduced.
    pub fn prepare_shared_a8_projection_recovered(
        &self,
        recovered: RecoveredMatrixView<'_>,
        activation: Activation,
    ) -> Result<PreparedCudaA8Projection> {
        let layout = validate_recovered_a8_projection_layout(recovered)?;
        let s_in = recovered.s_in.as_recovery_scales()?;
        let s_out = recovered.s_out.as_recovery_scales()?;
        let correction_identity = a8_correction_identity(recovered.matrix.columns, Some(s_in))?;
        self.make_current()?;
        let weights = DeviceBuffer::from_bytes(self, recovered.matrix.weights)?;
        let s_out = optional_scale_buffer(self, Some(s_out))?;
        let output_bytes = recovered
            .matrix
            .rows
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                EngineError::Shape("mmap CUDA A8 projection output size overflows".into())
            })?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let resident_bytes = weights
            .len()
            .checked_add(s_out.as_ref().map_or(0, DeviceBuffer::len))
            .and_then(|total| total.checked_add(output.len()))
            .ok_or_else(|| {
                EngineError::Shape("mmap CUDA A8 projection byte count overflows".into())
            })?;
        Ok(PreparedCudaA8Projection {
            context: Rc::clone(&self.inner),
            dtype: recovered.matrix.dtype,
            rows: recovered.matrix.rows as u32,
            columns: recovered.matrix.columns as u32,
            activation: match activation {
                Activation::Identity => 0,
                Activation::Silu => 1,
            },
            correction_identity,
            layout,
            weights,
            s_out,
            bias: None,
            output,
            resident_bytes,
        })
    }

    /// Allocates only the transient output needed to execute a resident
    /// projection over a bounded prompt chunk. The projection's weight and
    /// recovery buffers remain single-copy and are not repacked.
    pub fn prepare_batched_a8_output(
        &self,
        projection: &PreparedCudaA8Projection,
        batch_capacity: usize,
    ) -> Result<PreparedCudaBatchedA8Output> {
        if !Rc::ptr_eq(&self.inner, &projection.context) {
            return Err(EngineError::InvalidState(
                "batched CUDA output projection belongs to another context".into(),
            ));
        }
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        let output_bytes = (batch_capacity as usize)
            .checked_mul(projection.rows as usize)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| EngineError::Shape("batched CUDA output bytes overflow".into()))?;
        self.make_current()?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        Ok(PreparedCudaBatchedA8Output {
            context: Rc::clone(&self.inner),
            batch_capacity,
            rows: projection.rows,
            resident_bytes: output.len(),
            output,
        })
    }

    /// Allocates the fixed four-slot output arena selected by the model graph.
    pub fn prepare_batched_a8_output_arena(
        &self,
        batch_capacity: usize,
        slot_rows: [usize; 4],
    ) -> Result<PreparedCudaBatchedA8OutputArena> {
        let batch_capacity = validate_a8_batch_capacity(batch_capacity)?;
        if slot_rows
            .into_iter()
            .any(|rows| rows == 0 || u32::try_from(rows).is_err())
        {
            return Err(EngineError::Shape(
                "CUDA A8 output arena slots must be positive u32 row counts".into(),
            ));
        }
        let mut slot_offsets = [0_usize; 4];
        let mut total_values = 0_usize;
        for (slot, rows) in slot_rows.into_iter().enumerate() {
            slot_offsets[slot] = total_values;
            total_values = total_values
                .checked_add((batch_capacity as usize).checked_mul(rows).ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA output arena slot overflows".into())
                })?)
                .ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA output arena values overflow".into())
                })?;
        }
        let transient_bytes = total_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::MemoryBudget("CUDA output arena bytes overflow".into()))?;
        self.make_current()?;
        Ok(PreparedCudaBatchedA8OutputArena {
            context: Rc::clone(&self.inner),
            batch_capacity,
            slot_rows: slot_rows.map(|rows| rows as u32),
            slot_offsets,
            output: DeviceBuffer::allocate(self, transient_bytes)?,
            transient_bytes,
        })
    }

    pub fn prepare_gathered_a8_projection(
        &self,
        projection: &PreparedCudaA8Projection,
        row_ids: &[u32],
    ) -> Result<PreparedCudaGatheredA8Projection> {
        if !Rc::ptr_eq(&self.inner, &projection.context) {
            return Err(EngineError::InvalidState(
                "gathered CUDA projection belongs to another context".into(),
            ));
        }
        let (local_ids, groups) =
            build_gathered_row_plan(&projection.layout, projection.rows, row_ids)?;
        self.make_current()?;
        let row_ids = DeviceBuffer::from_bytes(self, as_bytes(&local_ids))?;
        let output_bytes = local_ids
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("gathered CUDA output size overflows".into()))?;
        let output = DeviceBuffer::allocate(self, output_bytes)?;
        let resident_bytes = row_ids
            .len()
            .checked_add(output.len())
            .ok_or_else(|| EngineError::Shape("gathered CUDA residency overflows".into()))?;
        Ok(PreparedCudaGatheredA8Projection {
            context: Rc::clone(&self.inner),
            rows: u32::try_from(local_ids.len())
                .map_err(|_| EngineError::Shape("gathered CUDA rows exceed u32".into()))?,
            columns: projection.columns,
            groups,
            row_ids,
            output,
            resident_bytes,
        })
    }

    /// Quantizes the corrected activation once into symmetric Q8_B64 blocks.
    /// This legacy prepared object remains matrix-local. Use
    /// [`Self::prepare_shared_a8_activation`] for an actual fan-out.
    pub fn quantize_prepared_a8(&self, prepared: &PreparedCudaA8MatVec) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.base.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA A8 operation belongs to another context".into(),
            ));
        }
        self.launch_a8_quantization(
            prepared.base.input.ptr(),
            prepared.base.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.base.columns,
        )
    }

    pub fn quantize_prepared_mixed_a8(&self, prepared: &PreparedCudaMixedA8MatVec) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared mixed CUDA A8 operation belongs to another context".into(),
            ));
        }
        self.launch_a8_quantization(
            prepared.input.ptr(),
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
        )
    }

    pub fn quantize_shared_a8_activation(&self, prepared: &PreparedCudaA8Activation) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared shared CUDA A8 activation belongs to another context".into(),
            ));
        }
        let input = prepared.input.as_ref().ok_or_else(|| {
            EngineError::InvalidState(
                "mmap-prepared shared CUDA A8 activation requires a producer device view".into(),
            )
        })?;
        self.launch_a8_quantization(
            input.ptr(),
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
        )
    }

    /// Fuse the Qwen FFN `SiLU(gate) * up` edge into the corrected A8 input
    /// quantization for the down projection. Neither producer view is copied
    /// and no intermediate f32 SwiGLU tensor is allocated.
    pub fn quantize_shared_a8_swiglu_device(
        &self,
        prepared: &PreparedCudaA8Activation,
        gate: CudaDeviceF32View<'_>,
        up: CudaDeviceF32View<'_>,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared shared CUDA A8 activation belongs to another context".into(),
            ));
        }
        for (name, view) in [("gate", gate), ("up", up)] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "shared CUDA A8 SwiGLU {name} belongs to another context"
                )));
            }
            if view.values() != prepared.columns as usize {
                return Err(EngineError::Shape(format!(
                    "shared CUDA A8 SwiGLU {name} has {} values, expected {}",
                    view.values(),
                    prepared.columns
                )));
            }
        }
        self.launch_swiglu_a8_quantization(
            gate.ptr()?,
            up.ptr()?,
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
            1,
        )
    }

    /// Complete Qwen FFN middle edge: fused SwiGLU/A8 quantization followed
    /// by one or more identity-bound down projections in the same CUDA stream.
    /// The host synchronizes only after every projection has been enqueued.
    pub fn dispatch_shared_a8_swiglu_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        gate: CudaDeviceF32View<'_>,
        up: CudaDeviceF32View<'_>,
        projections: &[&'a PreparedCudaA8Projection],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        self.validate_shared_a8_fanout(activation, projections, 1)?;
        self.quantize_shared_a8_swiglu_device(activation, gate, up)?;
        for projection in projections {
            self.launch_shared_a8_projection(activation, projection)?;
        }
        self.synchronize_after_launch("shared SwiGLU A8 fan-out context synchronization")?;
        projections
            .iter()
            .map(|projection| (*projection).device_output())
            .collect()
    }

    pub fn quantize_shared_a8_sigmoid_gate_device(
        &self,
        prepared: &PreparedCudaA8Activation,
        attention: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
    ) -> Result<()> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared shared CUDA A8 activation belongs to another context".into(),
            ));
        }
        for (name, view) in [("attention", attention), ("gate", gate)] {
            if !Rc::ptr_eq(&self.inner, view.context) {
                return Err(EngineError::InvalidState(format!(
                    "shared CUDA A8 attention gate {name} belongs to another context"
                )));
            }
            if view.values() != prepared.columns as usize {
                return Err(EngineError::Shape(format!(
                    "shared CUDA A8 attention gate {name} has {} values, expected {}",
                    view.values(),
                    prepared.columns
                )));
            }
        }
        self.launch_sigmoid_gate_a8_quantization(
            attention.ptr()?,
            gate.ptr()?,
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
            1,
        )
    }

    pub fn dispatch_shared_a8_sigmoid_gate_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        attention: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        projections: &[&'a PreparedCudaA8Projection],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        self.validate_shared_a8_fanout(activation, projections, 1)?;
        self.quantize_shared_a8_sigmoid_gate_device(activation, attention, gate)?;
        for projection in projections {
            self.launch_shared_a8_projection(activation, projection)?;
        }
        self.synchronize_after_launch("shared attention-gate A8 fan-out context synchronization")?;
        projections
            .iter()
            .map(|projection| (*projection).device_output())
            .collect()
    }

    /// Quantizes one corrected activation, launches every byte-identity-bound
    /// projection, synchronizes once, and returns outputs in caller order.
    pub fn dispatch_shared_a8_fanout(
        &self,
        activation: &PreparedCudaA8Activation,
        projections: &[&PreparedCudaA8Projection],
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_shared_a8_fanout(activation, projections, 1)?;
        self.quantize_shared_a8_activation(activation)?;
        self.dispatch_prepared_shared_a8_fanout_repeated(activation, projections, 1)
    }

    /// Quantizes a producer-owned corrected activation and dispatches the
    /// complete identity-bound fan-out without an intermediate host or device
    /// copy. Projection outputs remain in their prepared allocations.
    pub fn dispatch_shared_a8_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        input: CudaDeviceF32View<'_>,
        projections: &[&'a PreparedCudaA8Projection],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        self.validate_shared_a8_fanout(activation, projections, 1)?;
        if !Rc::ptr_eq(&self.inner, input.context) {
            return Err(EngineError::InvalidState(
                "shared CUDA A8 device input belongs to another context".into(),
            ));
        }
        if input.values() != activation.columns as usize {
            return Err(EngineError::Shape(format!(
                "shared CUDA A8 device input has {} values, expected {}",
                input.values(),
                activation.columns
            )));
        }
        self.launch_a8_quantization(
            input.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            activation.q8_codes.ptr(),
            activation.q8_scales.ptr(),
            activation.columns,
        )?;
        for projection in projections {
            self.launch_shared_a8_projection(activation, projection)?;
        }
        self.synchronize_after_launch("shared A8 device fan-out context synchronization")?;
        projections
            .iter()
            .map(|projection| (*projection).device_output())
            .collect()
    }

    /// Quantizes one producer-owned row-major prompt chunk and executes an
    /// identity-bound projection fan-out through the SM86 MMQ path. Model
    /// weights stay in the ordinary resident projection owners; only bounded
    /// activation/output workspaces scale with the chunk capacity.
    pub fn dispatch_batched_shared_a8_fanout_device<'a>(
        &self,
        activation: &PreparedCudaBatchedA8Activation,
        input: CudaDeviceF32View<'_>,
        batch_rows: usize,
        projections: &[(
            &'a PreparedCudaA8Projection,
            &'a PreparedCudaBatchedA8Output,
        )],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if projections.is_empty() {
            return Err(EngineError::Shape(
                "batched shared CUDA A8 fan-out requires at least one projection".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &activation.context) || !Rc::ptr_eq(&self.inner, input.context)
        {
            return Err(EngineError::InvalidState(
                "batched shared CUDA A8 input belongs to another context".into(),
            ));
        }
        if batch_rows > activation.batch_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA prefill chunk has {batch_rows} rows, activation capacity is {}",
                activation.batch_capacity
            )));
        }
        let input_values = (batch_rows as usize)
            .checked_mul(activation.columns as usize)
            .ok_or_else(|| EngineError::Shape("batched CUDA A8 input shape overflows".into()))?;
        if input.values() != input_values {
            return Err(EngineError::Shape(format!(
                "batched shared CUDA A8 input has {} values, expected {input_values}",
                input.values()
            )));
        }
        for (projection, output) in projections {
            if !Rc::ptr_eq(&self.inner, &projection.context)
                || !Rc::ptr_eq(&self.inner, &output.context)
            {
                return Err(EngineError::InvalidState(
                    "batched shared CUDA A8 fan-out crosses driver contexts".into(),
                ));
            }
            if projection.columns != activation.columns
                || projection.correction_identity != activation.correction_identity
            {
                return Err(EngineError::InvalidArtifact(
                    "batched shared CUDA A8 projection s_in identity differs".into(),
                ));
            }
            if output.rows != projection.rows || batch_rows > output.batch_capacity {
                return Err(EngineError::MemoryBudget(format!(
                    "CUDA batched output capacity/shape does not admit {batch_rows}x{} projection",
                    projection.rows
                )));
            }
        }

        self.launch_batched_a8_quantization(
            input.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            activation.q8_codes.ptr(),
            activation.q8_scales.ptr(),
            activation.columns,
            batch_rows,
        )?;
        for (projection, output) in projections {
            self.launch_batched_shared_a8_projection(activation, projection, output, batch_rows)?;
        }
        self.synchronize_after_launch("batched shared A8 MMQ fan-out context synchronization")?;
        projections
            .iter()
            .map(|(_, output)| output.device_output(batch_rows as usize))
            .collect()
    }

    /// Production-shaped fan-out over the single maximum-width activation
    /// arena and four fixed output slots. The resident activation contributes
    /// the exact recovery identity and `s_in`; transient codes never become a
    /// second model owner.
    pub fn dispatch_batched_a8_arena_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        workspace: &PreparedCudaBatchedA8Workspace,
        outputs: &'a PreparedCudaBatchedA8OutputArena,
        input: CudaDeviceF32View<'_>,
        batch_rows: usize,
        projections: &[(&PreparedCudaA8Projection, usize)],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if projections.is_empty() {
            return Err(EngineError::Shape(
                "CUDA A8 arena fan-out requires at least one projection".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &activation.context)
            || !Rc::ptr_eq(&self.inner, &workspace.context)
            || !Rc::ptr_eq(&self.inner, &outputs.context)
            || !Rc::ptr_eq(&self.inner, input.context)
        {
            return Err(EngineError::InvalidState(
                "CUDA A8 arena fan-out crosses driver contexts".into(),
            ));
        }
        if batch_rows > workspace.batch_capacity || batch_rows > outputs.batch_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA A8 arena does not admit {batch_rows} prompt rows"
            )));
        }
        if activation.columns > workspace.column_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA activation needs {} columns, arena admits {}",
                activation.columns, workspace.column_capacity
            )));
        }
        let input_values = (batch_rows as usize)
            .checked_mul(activation.columns as usize)
            .ok_or_else(|| EngineError::Shape("CUDA A8 arena input shape overflows".into()))?;
        if input.values() != input_values {
            return Err(EngineError::Shape(format!(
                "CUDA A8 arena input has {} values, expected {input_values}",
                input.values()
            )));
        }
        let mut used_slots = [false; 4];
        for (projection, slot) in projections {
            if *slot >= outputs.slot_rows.len() || used_slots[*slot] {
                return Err(EngineError::InvalidState(format!(
                    "CUDA A8 arena output slot {slot} is invalid or aliased"
                )));
            }
            used_slots[*slot] = true;
            if !Rc::ptr_eq(&self.inner, &projection.context) {
                return Err(EngineError::InvalidState(
                    "CUDA A8 arena projection belongs to another context".into(),
                ));
            }
            if projection.columns != activation.columns
                || projection.correction_identity != activation.correction_identity
            {
                return Err(EngineError::InvalidArtifact(
                    "CUDA A8 arena projection s_in identity differs".into(),
                ));
            }
            if projection.rows > outputs.slot_rows[*slot] {
                return Err(EngineError::MemoryBudget(format!(
                    "CUDA A8 projection needs {} rows, slot {slot} admits {}",
                    projection.rows, outputs.slot_rows[*slot]
                )));
            }
        }

        self.launch_batched_a8_quantization(
            input.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            workspace.q8_codes.ptr(),
            workspace.q8_scales.ptr(),
            activation.columns,
            batch_rows,
        )?;
        for (projection, slot) in projections {
            let output_ptr = device_ptr_offset(
                outputs.output.ptr(),
                outputs.slot_offsets[*slot]
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::Shape("CUDA output slot offset overflows".into())
                    })?,
            )?;
            self.launch_batched_a8_projection_buffers(
                workspace.q8_codes.ptr(),
                workspace.q8_scales.ptr(),
                projection,
                output_ptr,
                batch_rows,
            )?;
        }
        self.synchronize_after_launch("CUDA A8 arena fan-out context synchronization")?;
        projections
            .iter()
            .map(|(projection, slot)| {
                outputs.device_output(*slot, projection.rows(), batch_rows as usize)
            })
            .collect()
    }

    /// Batches the complete Qwen FFN middle edge over one prompt chunk. The
    /// existing verifier kernel uses `grid.y` for token rows, while the
    /// graph-owned A8 and output arenas prevent token-local allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_batched_a8_arena_swiglu_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        workspace: &PreparedCudaBatchedA8Workspace,
        outputs: &'a PreparedCudaBatchedA8OutputArena,
        gate: CudaDeviceF32View<'_>,
        up: CudaDeviceF32View<'_>,
        batch_rows: usize,
        projections: &[(&PreparedCudaA8Projection, usize)],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        let batch_rows = self.validate_batched_a8_arena_fused_fanout(
            activation,
            workspace,
            outputs,
            gate,
            up,
            batch_rows,
            projections,
            "SwiGLU",
        )?;
        self.launch_swiglu_a8_quantization(
            gate.ptr()?,
            up.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            workspace.q8_codes.ptr(),
            workspace.q8_scales.ptr(),
            activation.columns,
            batch_rows,
        )?;
        self.launch_batched_a8_arena_projections(workspace, outputs, batch_rows, projections)?;
        self.synchronize_after_launch("CUDA batched SwiGLU A8 arena context synchronization")?;
        projections
            .iter()
            .map(|(projection, slot)| {
                outputs.device_output(*slot, projection.rows(), batch_rows as usize)
            })
            .collect()
    }

    /// Batches full-attention sigmoid gating, recovery input scaling, A8
    /// quantization, and the output projection without a host token loop.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_batched_a8_arena_sigmoid_gate_fanout_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        workspace: &PreparedCudaBatchedA8Workspace,
        outputs: &'a PreparedCudaBatchedA8OutputArena,
        attention: CudaDeviceF32View<'_>,
        gate: CudaDeviceF32View<'_>,
        batch_rows: usize,
        projections: &[(&PreparedCudaA8Projection, usize)],
    ) -> Result<Vec<CudaDeviceF32View<'a>>> {
        let batch_rows = self.validate_batched_a8_arena_fused_fanout(
            activation,
            workspace,
            outputs,
            attention,
            gate,
            batch_rows,
            projections,
            "attention gate",
        )?;
        self.launch_sigmoid_gate_a8_quantization(
            attention.ptr()?,
            gate.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            workspace.q8_codes.ptr(),
            workspace.q8_scales.ptr(),
            activation.columns,
            batch_rows,
        )?;
        self.launch_batched_a8_arena_projections(workspace, outputs, batch_rows, projections)?;
        self.synchronize_after_launch(
            "CUDA batched attention-gate A8 arena context synchronization",
        )?;
        projections
            .iter()
            .map(|(projection, slot)| {
                outputs.device_output(*slot, projection.rows(), batch_rows as usize)
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_batched_a8_arena_fused_fanout(
        &self,
        activation: &PreparedCudaA8Activation,
        workspace: &PreparedCudaBatchedA8Workspace,
        outputs: &PreparedCudaBatchedA8OutputArena,
        left: CudaDeviceF32View<'_>,
        right: CudaDeviceF32View<'_>,
        batch_rows: usize,
        projections: &[(&PreparedCudaA8Projection, usize)],
        edge: &str,
    ) -> Result<u32> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if projections.is_empty() {
            return Err(EngineError::Shape(format!(
                "CUDA batched {edge} fan-out requires at least one projection"
            )));
        }
        if !Rc::ptr_eq(&self.inner, &activation.context)
            || !Rc::ptr_eq(&self.inner, &workspace.context)
            || !Rc::ptr_eq(&self.inner, &outputs.context)
            || !Rc::ptr_eq(&self.inner, left.context)
            || !Rc::ptr_eq(&self.inner, right.context)
        {
            return Err(EngineError::InvalidState(format!(
                "CUDA batched {edge} fan-out crosses driver contexts"
            )));
        }
        if batch_rows > workspace.batch_capacity || batch_rows > outputs.batch_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched {edge} arenas do not admit {batch_rows} prompt rows"
            )));
        }
        if activation.columns > workspace.column_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched {edge} needs {} columns, arena admits {}",
                activation.columns, workspace.column_capacity
            )));
        }
        let expected_values = (batch_rows as usize)
            .checked_mul(activation.columns as usize)
            .ok_or_else(|| EngineError::Shape(format!("CUDA batched {edge} shape overflows")))?;
        for (name, view) in [("left", left), ("right", right)] {
            if view.values() != expected_values {
                return Err(EngineError::Shape(format!(
                    "CUDA batched {edge} {name} input has {} values, expected {expected_values}",
                    view.values()
                )));
            }
        }
        let mut used_slots = [false; 4];
        for (projection, slot) in projections {
            if *slot >= outputs.slot_rows.len() || used_slots[*slot] {
                return Err(EngineError::InvalidState(format!(
                    "CUDA batched {edge} output slot {slot} is invalid or aliased"
                )));
            }
            used_slots[*slot] = true;
            if !Rc::ptr_eq(&self.inner, &projection.context) {
                return Err(EngineError::InvalidState(format!(
                    "CUDA batched {edge} projection belongs to another context"
                )));
            }
            if projection.columns != activation.columns
                || projection.correction_identity != activation.correction_identity
            {
                return Err(EngineError::InvalidArtifact(format!(
                    "CUDA batched {edge} projection s_in identity differs"
                )));
            }
            if projection.rows > outputs.slot_rows[*slot] {
                return Err(EngineError::MemoryBudget(format!(
                    "CUDA batched {edge} output slot {slot} is too narrow"
                )));
            }
        }
        Ok(batch_rows)
    }

    fn launch_batched_a8_arena_projections(
        &self,
        workspace: &PreparedCudaBatchedA8Workspace,
        outputs: &PreparedCudaBatchedA8OutputArena,
        batch_rows: u32,
        projections: &[(&PreparedCudaA8Projection, usize)],
    ) -> Result<()> {
        for (projection, slot) in projections {
            let output_ptr = device_ptr_offset(
                outputs.output.ptr(),
                outputs.slot_offsets[*slot]
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        EngineError::Shape("CUDA output slot offset overflows".into())
                    })?,
            )?;
            self.launch_batched_a8_projection_buffers(
                workspace.q8_codes.ptr(),
                workspace.q8_scales.ptr(),
                projection,
                output_ptr,
                batch_rows,
            )?;
        }
        Ok(())
    }

    /// Executes only release-bound LM-head rows for an MTP proposal. The
    /// target head remains a separate complete projection and verifies every
    /// proposed token before commit.
    pub fn dispatch_shared_a8_gathered_device<'a>(
        &self,
        activation: &PreparedCudaA8Activation,
        input: CudaDeviceF32View<'_>,
        projection: &PreparedCudaA8Projection,
        gathered: &'a PreparedCudaGatheredA8Projection,
    ) -> Result<CudaDeviceF32View<'a>> {
        self.validate_shared_a8_fanout(activation, &[projection], 1)?;
        if !Rc::ptr_eq(&self.inner, input.context) || !Rc::ptr_eq(&self.inner, &gathered.context) {
            return Err(EngineError::InvalidState(
                "gathered CUDA dispatch crosses driver contexts".into(),
            ));
        }
        if input.values() != activation.columns as usize || gathered.columns != projection.columns {
            return Err(EngineError::Shape(
                "gathered CUDA activation or projection width differs".into(),
            ));
        }
        self.launch_a8_quantization(
            input.ptr()?,
            activation.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            activation.q8_codes.ptr(),
            activation.q8_scales.ptr(),
            activation.columns,
        )?;
        for group in &gathered.groups {
            self.launch_gathered_a8_projection(activation, projection, gathered, *group)?;
        }
        self.synchronize_after_launch("gathered A8 LM-head context synchronization")?;
        gathered.device_output()
    }

    /// Verifier/roofline variant that replays the complete fan-out without
    /// requantizing. Production decode uses one dispatch per projection.
    pub fn dispatch_prepared_shared_a8_fanout_repeated(
        &self,
        activation: &PreparedCudaA8Activation,
        projections: &[&PreparedCudaA8Projection],
        dispatches: usize,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_shared_a8_fanout(activation, projections, dispatches)?;
        self.make_current()?;
        for _ in 0..dispatches {
            for projection in projections {
                self.launch_shared_a8_projection(activation, projection)?;
            }
        }
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "shared A8 fan-out context synchronization",
            )?;
        }
        projections
            .iter()
            .map(|projection| {
                let mut output = vec![0.0_f32; projection.rows as usize];
                projection.output.copy_to(as_bytes_mut(&mut output))?;
                if output.iter().any(|value| !value.is_finite()) {
                    return Err(EngineError::InvalidState(
                        "shared CUDA A8 fan-out produced a non-finite output".into(),
                    ));
                }
                Ok(output)
            })
            .collect()
    }

    fn validate_shared_a8_fanout(
        &self,
        activation: &PreparedCudaA8Activation,
        projections: &[&PreparedCudaA8Projection],
        dispatches: usize,
    ) -> Result<()> {
        if dispatches == 0 || projections.is_empty() {
            return Err(EngineError::Shape(
                "shared CUDA A8 fan-out requires projections and positive dispatches".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &activation.context) {
            return Err(EngineError::InvalidState(
                "prepared shared CUDA A8 activation belongs to another context".into(),
            ));
        }
        for projection in projections {
            if !Rc::ptr_eq(&self.inner, &projection.context) {
                return Err(EngineError::InvalidState(
                    "prepared shared CUDA A8 projection belongs to another context".into(),
                ));
            }
            if projection.columns != activation.columns
                || projection.correction_identity != activation.correction_identity
            {
                return Err(EngineError::InvalidArtifact(
                    "shared CUDA A8 projection s_in identity differs".into(),
                ));
            }
        }
        Ok(())
    }

    fn launch_shared_a8_projection(
        &self,
        activation: &PreparedCudaA8Activation,
        projection: &PreparedCudaA8Projection,
    ) -> Result<()> {
        match &projection.layout {
            CudaA8ProjectionLayout::Pure(dtype) => self.launch_a8_projection(
                *dtype,
                projection.weights.ptr(),
                activation.q8_codes.ptr(),
                activation.q8_scales.ptr(),
                projection.s_out.as_ref().map_or(0, DeviceBuffer::ptr),
                projection.bias.as_ref().map_or(0, DeviceBuffer::ptr),
                projection.output.ptr(),
                projection.rows,
                projection.columns,
                projection.activation,
            ),
            CudaA8ProjectionLayout::Mixed(segments) => {
                for segment in segments {
                    let row_start = segment.row_start as usize;
                    self.launch_a8_projection(
                        segment.descriptor.dtype,
                        device_ptr_offset(projection.weights.ptr(), segment.weight_offset)?,
                        activation.q8_codes.ptr(),
                        activation.q8_scales.ptr(),
                        projection
                            .s_out
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 2))
                            .transpose()?
                            .unwrap_or(0),
                        projection
                            .bias
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 4))
                            .transpose()?
                            .unwrap_or(0),
                        device_ptr_offset(projection.output.ptr(), row_start * 4)?,
                        segment.row_count,
                        projection.columns,
                        projection.activation,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn launch_batched_shared_a8_projection(
        &self,
        activation: &PreparedCudaBatchedA8Activation,
        projection: &PreparedCudaA8Projection,
        output: &PreparedCudaBatchedA8Output,
        batch_rows: u32,
    ) -> Result<()> {
        self.launch_batched_a8_projection_buffers(
            activation.q8_codes.ptr(),
            activation.q8_scales.ptr(),
            projection,
            output.output.ptr(),
            batch_rows,
        )
    }

    fn launch_batched_a8_projection_buffers(
        &self,
        q8_codes: CuDevicePtr,
        q8_scales: CuDevicePtr,
        projection: &PreparedCudaA8Projection,
        output: CuDevicePtr,
        batch_rows: u32,
    ) -> Result<()> {
        match &projection.layout {
            CudaA8ProjectionLayout::Pure(dtype) => self.launch_batched_a8_mmq_projection(
                *dtype,
                projection.weights.ptr(),
                q8_codes,
                q8_scales,
                projection.s_out.as_ref().map_or(0, DeviceBuffer::ptr),
                projection.bias.as_ref().map_or(0, DeviceBuffer::ptr),
                output,
                projection.rows,
                projection.columns,
                batch_rows,
                projection.rows,
                projection.activation,
            ),
            CudaA8ProjectionLayout::Mixed(segments) => {
                for segment in segments {
                    let row_start = segment.row_start as usize;
                    self.launch_batched_a8_mmq_projection(
                        segment.descriptor.dtype,
                        device_ptr_offset(projection.weights.ptr(), segment.weight_offset)?,
                        q8_codes,
                        q8_scales,
                        projection
                            .s_out
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 2))
                            .transpose()?
                            .unwrap_or(0),
                        projection
                            .bias
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 4))
                            .transpose()?
                            .unwrap_or(0),
                        device_ptr_offset(output, row_start * 4)?,
                        segment.row_count,
                        projection.columns,
                        batch_rows,
                        projection.rows,
                        projection.activation,
                    )?;
                }
                Ok(())
            }
        }
    }

    fn launch_a8_quantization(
        &self,
        input_ptr: CuDevicePtr,
        s_in_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        column_count: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut input = input_ptr;
        let mut s_in = s_in_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut columns = column_count;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.a8_quantize_function,
                    column_count.div_ceil(64),
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "A8 quantization launch",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_swiglu_a8_quantization(
        &self,
        gate_ptr: CuDevicePtr,
        up_ptr: CuDevicePtr,
        s_in_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        column_count: u32,
        batch_rows: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut gate = gate_ptr;
        let mut up = up_ptr;
        let mut s_in = s_in_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut columns = column_count;
        let mut params = [
            (&mut gate as *mut CuDevicePtr).cast::<c_void>(),
            (&mut up as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.swiglu_a8_quantize_function,
                    column_count.div_ceil(64),
                    batch_rows,
                    1,
                    64,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "SwiGLU A8 quantization launch",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_sigmoid_gate_a8_quantization(
        &self,
        attention_ptr: CuDevicePtr,
        gate_ptr: CuDevicePtr,
        s_in_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        column_count: u32,
        batch_rows: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut attention = attention_ptr;
        let mut gate = gate_ptr;
        let mut s_in = s_in_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut columns = column_count;
        let mut params = [
            (&mut attention as *mut CuDevicePtr).cast::<c_void>(),
            (&mut gate as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.sigmoid_gate_a8_quantize_function,
                    column_count.div_ceil(64),
                    batch_rows,
                    1,
                    64,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "attention-gate A8 quantization launch",
            )
        }
    }

    pub fn dispatch_a8_fused_matvec(&self, prepared: &PreparedCudaA8MatVec) -> Result<Vec<f32>> {
        self.quantize_prepared_a8(prepared)?;
        self.dispatch_prepared_a8_repeated(prepared, 1)
    }

    /// Executes only the dp4a weight projection. The caller must quantize the
    /// current input first; repeated execution is a verifier/roofline tool.
    pub fn dispatch_prepared_a8_repeated(
        &self,
        prepared: &PreparedCudaA8MatVec,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "CUDA A8 repeated dispatch count must be positive".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &prepared.base.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA A8 operation belongs to another context".into(),
            ));
        }
        self.make_current()?;
        for _ in 0..dispatches {
            self.launch_a8_projection(
                prepared.base.dtype,
                prepared.base.weights.ptr(),
                prepared.q8_codes.ptr(),
                prepared.q8_scales.ptr(),
                prepared.base.s_out.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.base.bias.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.base.output.ptr(),
                prepared.base.rows,
                prepared.base.columns,
                prepared.base.activation,
            )?;
        }
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "A8 context synchronization",
            )?;
        }
        let mut output = vec![0.0_f32; prepared.base.rows as usize];
        prepared.base.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA A8 candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_mixed_a8_fused_matvec(
        &self,
        prepared: &PreparedCudaMixedA8MatVec,
    ) -> Result<Vec<f32>> {
        self.quantize_prepared_mixed_a8(prepared)?;
        self.dispatch_prepared_mixed_a8_repeated(prepared, 1)
    }

    /// Dispatches every canonical mixed row segment without synchronizing or
    /// copying between segments. One output tensor and one transient A8 input
    /// remain resident for the complete mixed projection.
    pub fn dispatch_prepared_mixed_a8_repeated(
        &self,
        prepared: &PreparedCudaMixedA8MatVec,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "mixed CUDA A8 repeated dispatch count must be positive".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared mixed CUDA A8 operation belongs to another context".into(),
            ));
        }
        self.make_current()?;
        for _ in 0..dispatches {
            for segment in &prepared.segments {
                let row_start = segment.row_start as usize;
                let weights = device_ptr_offset(prepared.weights.ptr(), segment.weight_offset)?;
                let s_out = prepared
                    .s_out
                    .as_ref()
                    .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 2))
                    .transpose()?
                    .unwrap_or(0);
                let bias = prepared
                    .bias
                    .as_ref()
                    .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 4))
                    .transpose()?
                    .unwrap_or(0);
                let output = device_ptr_offset(prepared.output.ptr(), row_start * 4)?;
                self.launch_a8_projection(
                    segment.descriptor.dtype,
                    weights,
                    prepared.q8_codes.ptr(),
                    prepared.q8_scales.ptr(),
                    s_out,
                    bias,
                    output,
                    segment.row_count,
                    prepared.columns,
                    prepared.activation,
                )?;
            }
        }
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "mixed A8 context synchronization",
            )?;
        }
        let mut output = vec![0.0_f32; prepared.rows as usize];
        prepared.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "mixed CUDA A8 candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Quantizes and projects a complete row-major prompt batch with one
    /// synchronization. Mixed Q2/Q4 row ranges write into disjoint slices of
    /// the same `[batch_rows, rows]` output allocation.
    pub fn dispatch_batched_a8_matmul(
        &self,
        prepared: &PreparedCudaBatchedA8MatMul,
    ) -> Result<Vec<f32>> {
        self.dispatch_batched_a8_matmul_device(prepared)?;
        let output_values = (prepared.batch_rows as usize)
            .checked_mul(prepared.rows as usize)
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 output read overflows".into()))?;
        let mut output = vec![0.0_f32; output_values];
        prepared.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA batched A8 candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    /// Runs the isolated SM86 tensor-core candidate against the exact same
    /// prepared weights, A8 scratch, recovery scales, and output allocation as
    /// the dp4a baseline. This remains verifier-only until the numerical and
    /// roofline gates promote it.
    pub fn dispatch_batched_a8_mmq(
        &self,
        prepared: &PreparedCudaBatchedA8MatMul,
    ) -> Result<Vec<f32>> {
        self.dispatch_batched_a8_mmq_device(prepared)?;
        let output_values = (prepared.batch_rows as usize)
            .checked_mul(prepared.rows as usize)
            .ok_or_else(|| EngineError::Shape("CUDA batched MMQ output read overflows".into()))?;
        let mut output = vec![0.0_f32; output_values];
        prepared.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA batched MMQ candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    pub fn dispatch_batched_a8_mmq_device<'a>(
        &self,
        prepared: &'a PreparedCudaBatchedA8MatMul,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared batched CUDA MMQ operation belongs to another context".into(),
            ));
        }
        self.launch_batched_a8_quantization(
            prepared.input.ptr(),
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
            prepared.batch_rows,
        )?;
        match &prepared.layout {
            CudaA8ProjectionLayout::Pure(dtype) => self.launch_batched_a8_mmq_projection(
                *dtype,
                prepared.weights.ptr(),
                prepared.q8_codes.ptr(),
                prepared.q8_scales.ptr(),
                prepared.s_out.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.bias.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.output.ptr(),
                prepared.rows,
                prepared.columns,
                prepared.batch_rows,
                prepared.rows,
                prepared.activation,
            )?,
            CudaA8ProjectionLayout::Mixed(segments) => {
                for segment in segments {
                    let row_start = segment.row_start as usize;
                    self.launch_batched_a8_mmq_projection(
                        segment.descriptor.dtype,
                        device_ptr_offset(prepared.weights.ptr(), segment.weight_offset)?,
                        prepared.q8_codes.ptr(),
                        prepared.q8_scales.ptr(),
                        prepared
                            .s_out
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 2))
                            .transpose()?
                            .unwrap_or(0),
                        prepared
                            .bias
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 4))
                            .transpose()?
                            .unwrap_or(0),
                        device_ptr_offset(prepared.output.ptr(), row_start * 4)?,
                        segment.row_count,
                        prepared.columns,
                        prepared.batch_rows,
                        prepared.rows,
                        prepared.activation,
                    )?;
                }
            }
        }
        self.synchronize_after_launch("batched A8 MMQ context synchronization")?;
        prepared.device_output()
    }

    /// Device-resident form used by the future chunked prefill graph. No
    /// activation or projection output crosses host memory.
    pub fn dispatch_batched_a8_matmul_device<'a>(
        &self,
        prepared: &'a PreparedCudaBatchedA8MatMul,
    ) -> Result<CudaDeviceF32View<'a>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared batched CUDA A8 operation belongs to another context".into(),
            ));
        }
        self.launch_batched_a8_quantization(
            prepared.input.ptr(),
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
            prepared.batch_rows,
        )?;
        match &prepared.layout {
            CudaA8ProjectionLayout::Pure(dtype) => self.launch_batched_a8_projection(
                *dtype,
                prepared.weights.ptr(),
                prepared.q8_codes.ptr(),
                prepared.q8_scales.ptr(),
                prepared.s_out.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.bias.as_ref().map_or(0, DeviceBuffer::ptr),
                prepared.output.ptr(),
                prepared.rows,
                prepared.columns,
                prepared.batch_rows,
                prepared.rows,
                prepared.activation,
            )?,
            CudaA8ProjectionLayout::Mixed(segments) => {
                for segment in segments {
                    let row_start = segment.row_start as usize;
                    self.launch_batched_a8_projection(
                        segment.descriptor.dtype,
                        device_ptr_offset(prepared.weights.ptr(), segment.weight_offset)?,
                        prepared.q8_codes.ptr(),
                        prepared.q8_scales.ptr(),
                        prepared
                            .s_out
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 2))
                            .transpose()?
                            .unwrap_or(0),
                        prepared
                            .bias
                            .as_ref()
                            .map(|buffer| device_ptr_offset(buffer.ptr(), row_start * 4))
                            .transpose()?
                            .unwrap_or(0),
                        device_ptr_offset(prepared.output.ptr(), row_start * 4)?,
                        segment.row_count,
                        prepared.columns,
                        prepared.batch_rows,
                        prepared.rows,
                        prepared.activation,
                    )?;
                }
            }
        }
        self.synchronize_after_launch("batched A8 matmul context synchronization")?;
        prepared.device_output()
    }

    fn launch_batched_a8_quantization(
        &self,
        input_ptr: CuDevicePtr,
        s_in_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        column_count: u32,
        batch_rows_count: u32,
    ) -> Result<()> {
        self.make_current()?;
        let mut input = input_ptr;
        let mut s_in = s_in_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut columns = column_count;
        let mut batch_rows = batch_rows_count;
        let mut params = [
            (&mut input as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut batch_rows as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    self.inner.a8_batched_quantize_function,
                    column_count.div_ceil(BLOCK_LEN as u32),
                    batch_rows_count,
                    1,
                    BLOCK_LEN as u32,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched A8 quantization launch",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batched_a8_projection(
        &self,
        dtype: TensorDType,
        weights_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        s_out_ptr: CuDevicePtr,
        bias_ptr: CuDevicePtr,
        output_ptr: CuDevicePtr,
        row_count: u32,
        column_count: u32,
        batch_rows_count: u32,
        output_stride_count: u32,
        activation_code: u32,
    ) -> Result<()> {
        self.make_current()?;
        let function = match dtype {
            TensorDType::Q2B64 => self.inner.q2_a8_batched_function,
            TensorDType::Q4B64 => self.inner.q4_a8_batched_function,
            _ => unreachable!("validated batched CUDA A8 segment is Q2 or Q4"),
        };
        let mut weights = weights_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut s_out = s_out_ptr;
        let mut bias = bias_ptr;
        let mut output = output_ptr;
        let mut rows = row_count;
        let mut columns = column_count;
        let mut batch_rows = batch_rows_count;
        let mut output_stride = output_stride_count;
        let mut activation = activation_code;
        let mut params = [
            (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
            (&mut bias as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut batch_rows as *mut u32).cast::<c_void>(),
            (&mut output_stride as *mut u32).cast::<c_void>(),
            (&mut activation as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    function,
                    row_count.div_ceil(A8_ROWS_PER_BLOCK),
                    batch_rows_count,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched A8 dp4a matmul launch",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batched_a8_mmq_projection(
        &self,
        dtype: TensorDType,
        weights_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        s_out_ptr: CuDevicePtr,
        bias_ptr: CuDevicePtr,
        output_ptr: CuDevicePtr,
        row_count: u32,
        column_count: u32,
        batch_rows_count: u32,
        output_stride_count: u32,
        activation_code: u32,
    ) -> Result<()> {
        self.make_current()?;
        let function = match dtype {
            TensorDType::Q2B64 => self.inner.q2_a8_batched_mmq_function,
            TensorDType::Q4B64 => self.inner.q4_a8_batched_mmq_function,
            _ => unreachable!("validated batched CUDA MMQ segment is Q2 or Q4"),
        };
        let mut weights = weights_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut s_out = s_out_ptr;
        let mut bias = bias_ptr;
        let mut output = output_ptr;
        let mut rows = row_count;
        let mut columns = column_count;
        let mut batch_rows = batch_rows_count;
        let mut output_stride = output_stride_count;
        let mut activation = activation_code;
        let mut params = [
            (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
            (&mut bias as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut batch_rows as *mut u32).cast::<c_void>(),
            (&mut output_stride as *mut u32).cast::<c_void>(),
            (&mut activation as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    function,
                    row_count.div_ceil(MMQ_ROWS_PER_BLOCK),
                    batch_rows_count.div_ceil(MMQ_BATCH_ROWS_PER_BLOCK),
                    1,
                    MMQ_THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "batched A8 tensor-core MMQ launch",
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_a8_projection(
        &self,
        dtype: TensorDType,
        weights_ptr: CuDevicePtr,
        q8_codes_ptr: CuDevicePtr,
        q8_scales_ptr: CuDevicePtr,
        s_out_ptr: CuDevicePtr,
        bias_ptr: CuDevicePtr,
        output_ptr: CuDevicePtr,
        row_count: u32,
        column_count: u32,
        activation_code: u32,
    ) -> Result<()> {
        let function = match dtype {
            TensorDType::Q2B64 => self.inner.q2_a8_function,
            TensorDType::Q4B64 => self.inner.q4_a8_function,
            _ => unreachable!("validated CUDA A8 segment is Q2 or Q4"),
        };
        let mut weights = weights_ptr;
        let mut q8_codes = q8_codes_ptr;
        let mut q8_scales = q8_scales_ptr;
        let mut s_out = s_out_ptr;
        let mut bias = bias_ptr;
        let mut output = output_ptr;
        let mut rows = row_count;
        let mut columns = column_count;
        let mut activation = activation_code;
        let mut params = [
            (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
            (&mut bias as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut activation as *mut u32).cast::<c_void>(),
        ];
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    function,
                    row_count.div_ceil(A8_ROWS_PER_BLOCK),
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "A8 dp4a matvec launch",
            )
        }
    }

    fn launch_gathered_a8_projection(
        &self,
        activation: &PreparedCudaA8Activation,
        projection: &PreparedCudaA8Projection,
        gathered: &PreparedCudaGatheredA8Projection,
        group: CudaGatheredRowGroup,
    ) -> Result<()> {
        let function = match group.dtype {
            TensorDType::Q2B64 => self.inner.q2_a8_gathered_function,
            TensorDType::Q4B64 => self.inner.q4_a8_gathered_function,
            _ => unreachable!("validated gathered CUDA segment is Q2 or Q4"),
        };
        let mut weights = device_ptr_offset(projection.weights.ptr(), group.weight_offset)?;
        let mut q8_codes = activation.q8_codes.ptr();
        let mut q8_scales = activation.q8_scales.ptr();
        let mut s_out = projection
            .s_out
            .as_ref()
            .map(|buffer| device_ptr_offset(buffer.ptr(), group.scale_row_offset * 2))
            .transpose()?
            .unwrap_or(0);
        let mut bias = projection
            .bias
            .as_ref()
            .map(|buffer| device_ptr_offset(buffer.ptr(), group.scale_row_offset * 4))
            .transpose()?
            .unwrap_or(0);
        let mut row_ids = device_ptr_offset(gathered.row_ids.ptr(), group.row_id_offset)?;
        let mut output = device_ptr_offset(gathered.output.ptr(), group.output_offset)?;
        let mut rows = group.row_count;
        let mut columns = projection.columns;
        let mut activation_code = projection.activation;
        let mut params = [
            (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_codes as *mut CuDevicePtr).cast::<c_void>(),
            (&mut q8_scales as *mut CuDevicePtr).cast::<c_void>(),
            (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
            (&mut bias as *mut CuDevicePtr).cast::<c_void>(),
            (&mut row_ids as *mut CuDevicePtr).cast::<c_void>(),
            (&mut output as *mut CuDevicePtr).cast::<c_void>(),
            (&mut rows as *mut u32).cast::<c_void>(),
            (&mut columns as *mut u32).cast::<c_void>(),
            (&mut activation_code as *mut u32).cast::<c_void>(),
        ];
        self.make_current()?;
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.launch_kernel)(
                    function,
                    group.row_count.div_ceil(A8_ROWS_PER_BLOCK),
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "gathered A8 projection launch",
            )
        }
    }

    /// Launches a resident operation repeatedly and synchronizes once. This
    /// amortizes host launch/copy overhead for per-op roofline measurement;
    /// production graph capture remains a separate promotion requirement.
    pub fn dispatch_prepared_repeated(
        &self,
        prepared: &PreparedCudaMatVec,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "CUDA repeated dispatch count must be positive".into(),
            ));
        }
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA operation belongs to another context".into(),
            ));
        }
        self.make_current()?;
        let function = match prepared.dtype {
            TensorDType::Q2B64 => self.inner.q2_function,
            TensorDType::Q4B64 => self.inner.q4_function,
            _ => unreachable!("CUDA validation accepts only Q2/Q4"),
        };
        let warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
        let grid_x = prepared.rows.div_ceil(warps_per_block);
        for _ in 0..dispatches {
            let mut weights = prepared.weights.ptr();
            let mut input = prepared.input.ptr();
            let mut s_in = prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr);
            let mut s_out = prepared.s_out.as_ref().map_or(0, DeviceBuffer::ptr);
            let mut bias = prepared.bias.as_ref().map_or(0, DeviceBuffer::ptr);
            let mut output = prepared.output.ptr();
            let mut rows = prepared.rows;
            let mut columns = prepared.columns;
            let mut activation = prepared.activation;
            let mut params = [
                (&mut weights as *mut CuDevicePtr).cast::<c_void>(),
                (&mut input as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_in as *mut CuDevicePtr).cast::<c_void>(),
                (&mut s_out as *mut CuDevicePtr).cast::<c_void>(),
                (&mut bias as *mut CuDevicePtr).cast::<c_void>(),
                (&mut output as *mut CuDevicePtr).cast::<c_void>(),
                (&mut rows as *mut u32).cast::<c_void>(),
                (&mut columns as *mut u32).cast::<c_void>(),
                (&mut activation as *mut u32).cast::<c_void>(),
            ];
            unsafe {
                self.inner.driver.check(
                    (self.inner.driver.launch_kernel)(
                        function,
                        grid_x,
                        1,
                        1,
                        THREADS_PER_BLOCK,
                        1,
                        1,
                        0,
                        ptr::null_mut(),
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "kernel launch",
                )?;
            }
        }
        unsafe {
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "context synchronization",
            )?;
        }
        let mut output = vec![0.0_f32; prepared.rows as usize];
        prepared.output.copy_to(as_bytes_mut(&mut output))?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }

    fn make_current(&self) -> Result<()> {
        self.inner.make_current()
    }
}

impl CudaContextInner {
    fn make_current(&self) -> Result<()> {
        unsafe {
            self.driver.check(
                (self.driver.ctx_set_current)(self.context),
                "set current context",
            )
        }
    }
}

impl PreparedCudaMatVec {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.rows())
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.columns() {
            return Err(EngineError::Shape(format!(
                "CUDA prepared input has {} values, expected {}",
                input.len(),
                self.columns()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "CUDA prepared input contains a non-finite value".into(),
            ));
        }
        self.input.write(as_bytes(input))
    }
}

impl PreparedCudaA8MatVec {
    pub fn dtype(&self) -> TensorDType {
        self.base.dtype()
    }

    pub fn rows(&self) -> usize {
        self.base.rows()
    }

    pub fn columns(&self) -> usize {
        self.base.columns()
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.base.device_output()
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        self.base.write_input(input)
    }
}

impl PreparedCudaMixedA8MatVec {
    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.rows())
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.columns() {
            return Err(EngineError::Shape(format!(
                "mixed CUDA prepared input has {} values, expected {}",
                input.len(),
                self.columns()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "mixed CUDA prepared input contains a non-finite value".into(),
            ));
        }
        self.input.write(as_bytes(input))
    }
}

impl PreparedCudaBatchedA8MatMul {
    pub fn batch_rows(&self) -> usize {
        self.batch_rows as usize
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        let values = self
            .batch_rows()
            .checked_mul(self.rows())
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 output view overflows".into()))?;
        self.output.f32_view(0, values)
    }

    pub fn write_inputs(&self, inputs: &[f32]) -> Result<()> {
        let expected = self
            .batch_rows()
            .checked_mul(self.columns())
            .ok_or_else(|| EngineError::Shape("CUDA batched A8 input shape overflows".into()))?;
        if inputs.len() != expected {
            return Err(EngineError::Shape(format!(
                "CUDA batched A8 input has {} values, expected {expected}",
                inputs.len()
            )));
        }
        if inputs.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "CUDA batched A8 input contains a non-finite value".into(),
            ));
        }
        self.input.write(as_bytes(inputs))
    }
}

impl PreparedCudaA8Activation {
    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.columns() {
            return Err(EngineError::Shape(format!(
                "shared CUDA A8 input has {} values, expected {}",
                input.len(),
                self.columns()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "shared CUDA A8 input contains a non-finite value".into(),
            ));
        }
        self.input
            .as_ref()
            .ok_or_else(|| {
                EngineError::InvalidState(
                    "mmap-prepared shared CUDA A8 activation has no host staging buffer".into(),
                )
            })?
            .write(as_bytes(input))
    }

    pub fn verifier_read_quantized(&self) -> Result<(Vec<i8>, Vec<f32>)> {
        let mut codes = vec![0_i8; self.columns()];
        let mut scales = vec![0.0_f32; self.columns().div_ceil(BLOCK_LEN)];
        self.q8_codes.copy_to(as_bytes_mut(&mut codes))?;
        self.q8_scales.copy_to(as_bytes_mut(&mut scales))?;
        if scales.iter().any(|scale| !scale.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA verifier read produced a non-finite A8 scale".into(),
            ));
        }
        Ok((codes, scales))
    }
}

impl PreparedCudaBatchedA8Activation {
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

impl PreparedCudaBatchedA8Workspace {
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity as usize
    }

    pub fn column_capacity(&self) -> usize {
        self.column_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    /// Reads the active quantized prefix for a hardware verifier. Production
    /// graph execution keeps both buffers device-resident.
    pub fn verifier_read_quantized(
        &self,
        batch_rows: usize,
        columns: usize,
    ) -> Result<(Vec<i8>, Vec<f32>)> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)? as usize;
        if columns == 0
            || !columns.is_multiple_of(64)
            || columns > self.column_capacity as usize
            || batch_rows > self.batch_capacity as usize
        {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA A8 verifier read does not admit {batch_rows}x{columns}"
            )));
        }
        let mut all_codes = vec![0_i8; self.q8_codes.len()];
        self.q8_codes.copy_to(as_bytes_mut(&mut all_codes))?;
        let active_codes = batch_rows
            .checked_mul(columns)
            .ok_or_else(|| EngineError::Shape("CUDA A8 verifier code count overflows".into()))?;
        all_codes.truncate(active_codes);
        let mut all_scales = vec![0.0_f32; self.q8_scales.len() / std::mem::size_of::<f32>()];
        self.q8_scales.copy_to(as_bytes_mut(&mut all_scales))?;
        all_scales.truncate(active_codes / 64);
        Ok((all_codes, all_scales))
    }
}

impl PreparedCudaA8Projection {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.rows())
    }
}

impl PreparedCudaBatchedA8Output {
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity as usize
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self, batch_rows: usize) -> Result<CudaDeviceF32View<'_>> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if batch_rows > self.batch_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched output request has {batch_rows} rows, capacity is {}",
                self.batch_capacity
            )));
        }
        let values = (batch_rows as usize)
            .checked_mul(self.rows())
            .ok_or_else(|| EngineError::Shape("CUDA batched output view overflows".into()))?;
        self.output.f32_view(0, values)
    }
}

impl PreparedCudaBatchedA8OutputArena {
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity as usize
    }

    pub fn slot_rows(&self) -> [usize; 4] {
        self.slot_rows.map(|rows| rows as usize)
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(
        &self,
        slot: usize,
        rows: usize,
        batch_rows: usize,
    ) -> Result<CudaDeviceF32View<'_>> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if slot >= self.slot_rows.len()
            || rows == 0
            || rows > self.slot_rows[slot] as usize
            || batch_rows > self.batch_capacity
        {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA output arena does not admit slot {slot}, {batch_rows}x{rows}"
            )));
        }
        let values = (batch_rows as usize)
            .checked_mul(rows)
            .ok_or_else(|| EngineError::Shape("CUDA output arena view overflows".into()))?;
        self.output.f32_view(self.slot_offsets[slot], values)
    }
}

impl PreparedCudaGatheredA8Projection {
    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.rows())
    }
}

impl PreparedCudaArgmax {
    pub fn resident_bytes(&self) -> usize {
        self.result.len()
    }
}

impl PreparedCudaTopKTopPSampler {
    pub fn max_values(&self) -> usize {
        self.max_values
    }

    pub fn resident_bytes(&self) -> usize {
        self.scratch.len() + self.result.len()
    }
}

impl PreparedCudaRecoveredRow {
    pub fn dtype(&self) -> TensorDType {
        self.dtype
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.columns())
    }
}

impl PreparedCudaEmbedding {
    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn graph_bytes(&self) -> usize {
        self.graph_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.columns())
    }
}

impl PreparedCudaBatchedEmbeddingWorkspace {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self, tokens: usize) -> Result<CudaDeviceF32View<'_>> {
        if tokens == 0 || tokens > self.token_capacity() {
            return Err(EngineError::Shape(
                "CUDA batched embedding output token count is invalid".into(),
            ));
        }
        self.output.f32_view(0, tokens * self.columns())
    }
}

impl PreparedCudaGatedDelta {
    pub fn config(&self) -> CudaGatedDeltaConfig {
        self.config
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> usize {
        self.checkpoint.len()
    }

    pub fn begin_speculative(&mut self) -> Result<()> {
        if self.poisoned || self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta checkpoint requires a healthy state without an active branch"
                    .into(),
            ));
        }
        self.checkpoint
            .copy_from_buffer(&self.state, "gated-delta checkpoint copy")?;
        self.checkpoint_valid = true;
        Ok(())
    }

    pub fn restore_speculative(&mut self) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta has no speculative checkpoint".into(),
            ));
        }
        self.state
            .copy_from_buffer(&self.checkpoint, "gated-delta checkpoint restore")?;
        self.checkpoint_valid = false;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA gated-delta has no speculative checkpoint".into(),
            ));
        }
        self.checkpoint_valid = false;
        Ok(())
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
        let qk_values = self.config.heads * self.config.key_dim;
        let value_values = self.config.heads * self.config.value_dim;
        for (name, values, expected) in [
            ("query", query, qk_values),
            ("key", key, qk_values),
            ("value", value, value_values),
            ("log_decay", log_decay, self.config.heads),
            ("beta", beta, self.config.heads),
        ] {
            if values.len() != expected || values.iter().any(|item| !item.is_finite()) {
                return Err(EngineError::Shape(format!(
                    "CUDA gated-delta {name} has invalid length or non-finite values"
                )));
            }
        }
        self.query.write(as_bytes(query))?;
        self.key.write(as_bytes(key))?;
        self.value.write(as_bytes(value))?;
        self.log_decay.write(as_bytes(log_decay))?;
        self.beta.write(as_bytes(beta))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state.zero()?;
        self.checkpoint.zero()?;
        self.output.zero()?;
        self.poisoned = false;
        self.checkpoint_valid = false;
        Ok(())
    }

    /// Readback is restricted to verifier evidence. Graph execution never
    /// materializes a host-side state copy.
    pub fn verifier_read_state(&self) -> Result<Vec<half::f16>> {
        let mut state = vec![half::f16::ZERO; self.resident_state_bytes / 2];
        self.state.copy_to(as_bytes_mut(&mut state))?;
        Ok(state)
    }
}

impl PreparedCudaGatedDeltaScanOutput {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self, tokens: usize) -> Result<CudaDeviceF32View<'_>> {
        let tokens = validate_a8_batch_capacity(tokens)?;
        if tokens > self.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA gated-delta scan requests {tokens} tokens, capacity is {}",
                self.token_capacity
            )));
        }
        self.output.f32_view(
            0,
            tokens as usize * self.heads as usize * self.value_dim as usize,
        )
    }
}

impl PreparedCudaGatedDeltaInputs {
    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedCudaGatedDeltaScanInputs {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedCudaCausalConv {
    pub fn config(&self) -> CudaCausalConvConfig {
        self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn resident_state_bytes(&self) -> usize {
        self.resident_state_bytes
    }

    pub fn speculative_checkpoint_bytes(&self) -> usize {
        self.checkpoint.len()
    }

    pub fn begin_speculative(&mut self) -> Result<()> {
        if self.poisoned || self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA convolution checkpoint requires a healthy state without an active branch"
                    .into(),
            ));
        }
        self.checkpoint
            .copy_from_buffer(&self.state, "convolution checkpoint copy")?;
        self.checkpoint_valid = true;
        Ok(())
    }

    pub fn restore_speculative(&mut self) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA convolution has no speculative checkpoint".into(),
            ));
        }
        self.state
            .copy_from_buffer(&self.checkpoint, "convolution checkpoint restore")?;
        self.checkpoint_valid = false;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if !self.checkpoint_valid {
            return Err(EngineError::InvalidState(
                "CUDA convolution has no speculative checkpoint".into(),
            ));
        }
        self.checkpoint_valid = false;
        Ok(())
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        if input.len() != self.config.channels || input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "CUDA convolution input has invalid length or values".into(),
            ));
        }
        self.input.write(as_bytes(input))
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state.zero()?;
        self.checkpoint.zero()?;
        self.output.zero()?;
        self.poisoned = false;
        self.checkpoint_valid = false;
        Ok(())
    }

    pub fn verifier_read_state(&self) -> Result<Vec<half::f16>> {
        let mut state = vec![half::f16::ZERO; self.resident_state_bytes / 2];
        self.state.copy_to(as_bytes_mut(&mut state))?;
        Ok(state)
    }
}

impl PreparedCudaCausalConvScanOutput {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self, tokens: usize) -> Result<CudaDeviceF32View<'_>> {
        let tokens = validate_a8_batch_capacity(tokens)?;
        if tokens > self.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA convolution scan requests {tokens} tokens, capacity is {}",
                self.token_capacity
            )));
        }
        self.output
            .f32_view(0, tokens as usize * self.channels as usize)
    }
}

impl PreparedCudaGatedRmsNorm {
    pub fn config(&self) -> CudaGatedRmsNormConfig {
        self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output
            .f32_view(0, self.config.rows * self.config.columns)
    }

    pub fn write_inputs(&self, input: &[f32], gate: &[f32]) -> Result<()> {
        let expected = self.config.rows * self.config.columns;
        for (name, values) in [("input", input), ("gate", gate)] {
            if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::Shape(format!(
                    "CUDA gated RMSNorm {name} has invalid length or values"
                )));
            }
        }
        self.input.write(as_bytes(input))?;
        self.gate.write(as_bytes(gate))
    }
}

impl PreparedCudaBatchedGatedRmsNormOutput {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self, token_count: usize) -> Result<CudaDeviceF32View<'_>> {
        let token_count = validate_a8_batch_capacity(token_count)?;
        if token_count > self.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched gated RMSNorm requests {token_count} tokens, capacity is {}",
                self.token_capacity
            )));
        }
        self.output.f32_view(
            0,
            token_count as usize * self.heads as usize * self.columns as usize,
        )
    }
}

impl PreparedCudaRmsNorm {
    pub fn config(&self) -> CudaRmsNormConfig {
        self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output
            .f32_view(0, self.config.rows * self.config.columns)
    }

    pub fn write_input(&self, input: &[f32]) -> Result<()> {
        let expected = self.config.rows * self.config.columns;
        if input.len() != expected || input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "CUDA Qwen RMSNorm input has invalid length or values".into(),
            ));
        }
        self.input.write(as_bytes(input))
    }
}

impl PreparedCudaResidualRmsNorm {
    pub fn config(&self) -> CudaRmsNormConfig {
        self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn residual_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.residual_output
            .f32_view(0, self.config.rows * self.config.columns)
    }

    pub fn normalized_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.normalized_output
            .f32_view(0, self.config.rows * self.config.columns)
    }
}

impl PreparedCudaBatchedRmsNormWorkspace {
    pub fn batch_capacity(&self) -> usize {
        self.batch_capacity as usize
    }

    pub fn columns(&self) -> usize {
        self.columns as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn residual_output(&self, batch_rows: usize) -> Result<CudaDeviceF32View<'_>> {
        self.output_view(&self.residual_output, batch_rows)
    }

    pub fn normalized_output(&self, batch_rows: usize) -> Result<CudaDeviceF32View<'_>> {
        self.output_view(&self.normalized_output, batch_rows)
    }

    fn output_view<'a>(
        &'a self,
        buffer: &'a DeviceBuffer,
        batch_rows: usize,
    ) -> Result<CudaDeviceF32View<'a>> {
        let batch_rows = validate_a8_batch_capacity(batch_rows)?;
        if batch_rows > self.batch_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA batched RMSNorm requests {batch_rows} rows, capacity is {}",
                self.batch_capacity
            )));
        }
        buffer.f32_view(0, batch_rows as usize * self.columns())
    }
}

impl PreparedCudaPartialRope {
    pub fn config(&self) -> CudaPartialRopeConfig {
        self.config
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_values(&self) -> Result<CudaDeviceF32View<'_>> {
        self.values
            .f32_view(0, self.config.heads * self.config.head_dim)
    }

    pub fn write_values(&self, values: &[f32]) -> Result<()> {
        let expected = self.config.heads * self.config.head_dim;
        if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(
                "CUDA partial-RoPE values have invalid length or data".into(),
            ));
        }
        self.values.write(as_bytes(values))
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        let half_dim = self.config.rotary_dim / 2;
        let mut cosine = Vec::with_capacity(half_dim);
        let mut sine = Vec::with_capacity(half_dim);
        for index in 0..half_dim {
            let inverse_frequency = self
                .config
                .theta
                .powf(-((2 * index) as f32) / self.config.rotary_dim as f32);
            let angle = position as f32 * inverse_frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
        self.cosine.write(as_bytes(&cosine))?;
        self.sine.write(as_bytes(&sine))
    }
}

impl PreparedCudaBatchedRopeWorkspace {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedCudaQueryGate {
    pub fn config(&self) -> CudaQueryGateConfig {
        self.config
    }

    pub fn model_bytes(&self) -> usize {
        self.model_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn write_position(&self, position: u64) -> Result<()> {
        let half_dim = self.config.rotary_dim / 2;
        let mut cosine = Vec::with_capacity(half_dim);
        let mut sine = Vec::with_capacity(half_dim);
        for index in 0..half_dim {
            let inverse_frequency = self
                .config
                .theta
                .powf(-((2 * index) as f32) / self.config.rotary_dim as f32);
            let angle = position as f32 * inverse_frequency;
            cosine.push(angle.cos());
            sine.push(angle.sin());
        }
        self.cosine.write(as_bytes(&cosine))?;
        self.sine.write(as_bytes(&sine))
    }
}

impl PreparedCudaBatchedQueryGateOutput {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedCudaPagedGqa {
    pub fn config(&self) -> CudaPagedGqaConfig {
        self.config
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn maximum_tokens(&self) -> usize {
        self.config.maximum_tokens
    }

    /// Fixed device allocation for canonical Q2/Q4 pages and descriptors.
    /// Query/output/parameter buffers are reported separately as transient.
    pub fn packed_device_bytes(&self) -> usize {
        self.packed_device_bytes
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn q4_tokens(&self) -> usize {
        self.pages
            .iter()
            .filter(|page| page.precision == KvPrecision::Q4)
            .map(|page| page.tokens)
            .sum()
    }

    pub fn q2_tokens(&self) -> usize {
        self.pages
            .iter()
            .filter(|page| page.precision == KvPrecision::Q2)
            .map(|page| page.tokens)
            .sum()
    }

    pub fn begin_speculative(&mut self) -> Result<()> {
        if self.poisoned || self.speculative_checkpoint.is_some() {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA checkpoint requires a healthy state without an active branch"
                    .into(),
            ));
        }
        if self.free_q4_slots.is_empty() {
            return Err(EngineError::MemoryBudget(
                "CUDA paged GQA has no retained Q4 boundary slot for speculation".into(),
            ));
        }
        self.speculative_checkpoint = Some(CudaPagedGqaCheckpoint {
            tokens: self.tokens,
            pages: self.pages.clone(),
            free_q4_slots: self.free_q4_slots.clone(),
        });
        Ok(())
    }

    pub fn restore_speculative(&mut self) -> Result<()> {
        let checkpoint = self.speculative_checkpoint.take().ok_or_else(|| {
            EngineError::InvalidState("CUDA paged GQA has no speculative checkpoint".into())
        })?;
        self.tokens = checkpoint.tokens;
        self.pages = checkpoint.pages;
        self.free_q4_slots = checkpoint.free_q4_slots;
        self.poisoned = false;
        Ok(())
    }

    pub fn commit_speculative(&mut self) -> Result<()> {
        if self.speculative_checkpoint.take().is_none() {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA has no speculative checkpoint".into(),
            ));
        }
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.tokens = 0;
        self.pages.clear();
        self.free_q4_slots = (0..self.q4_slots).rev().collect();
        self.q2_pages.zero()?;
        self.q4_pages.zero()?;
        self.descriptors.zero()?;
        self.query.zero()?;
        self.key.zero()?;
        self.value.zero()?;
        self.output.zero()?;
        self.params.zero()?;
        self.poisoned = false;
        self.speculative_checkpoint = None;
        Ok(())
    }
}

impl PreparedCudaSplitPagedGqa {
    pub fn maximum_query_tokens(&self) -> usize {
        PAGED_GQA_SPLIT_MAX_QUERY_TOKENS
    }

    pub fn segments(&self) -> usize {
        PAGED_GQA_SPLIT_SEGMENTS
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }
}

impl PreparedCudaPagedGqaPrefillOutput {
    pub fn token_capacity(&self) -> usize {
        self.token_capacity as usize
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
    }

    pub fn device_output(&self, query_tokens: usize) -> Result<CudaDeviceF32View<'_>> {
        let query_tokens = validate_a8_batch_capacity(query_tokens)?;
        if query_tokens > self.token_capacity {
            return Err(EngineError::MemoryBudget(format!(
                "CUDA paged-GQA prefill requests {query_tokens} tokens, capacity is {}",
                self.token_capacity
            )));
        }
        self.output.f32_view(
            0,
            query_tokens as usize * self.query_heads as usize * self.head_dim as usize,
        )
    }
}

impl CudaVerifierF32Tensor {
    pub fn values(&self) -> usize {
        self.values
    }

    pub fn resident_bytes(&self) -> usize {
        self.buffer.len()
    }

    pub fn write(&self, values: &[f32]) -> Result<()> {
        if values.len() != self.values || values.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::Shape(format!(
                "CUDA verifier tensor has {} values, expected finite {} values",
                values.len(),
                self.values
            )));
        }
        self.buffer.write(as_bytes(values))
    }

    pub fn device_view(&self) -> Result<CudaDeviceF32View<'_>> {
        self.buffer.f32_view(0, self.values)
    }
}

impl PreparedCudaF32Concat {
    pub fn values(&self) -> usize {
        self.left_values + self.right_values
    }

    pub fn transient_bytes(&self) -> usize {
        self.output.len()
    }

    pub fn device_output(&self) -> Result<CudaDeviceF32View<'_>> {
        self.output.f32_view(0, self.values())
    }
}

impl PreparedCudaF32Checkpoint {
    pub fn resident_bytes(&self) -> usize {
        self.snapshot.len()
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn commit(&mut self) -> Result<()> {
        if !self.valid {
            return Err(EngineError::InvalidState(
                "CUDA f32 checkpoint is not active".into(),
            ));
        }
        self.valid = false;
        Ok(())
    }

    pub fn clear(&mut self) -> Result<()> {
        self.snapshot.zero()?;
        self.valid = false;
        Ok(())
    }
}

impl Drop for CudaContextInner {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.ctx_set_current)(self.context);
            let _ = (self.driver.module_unload)(self.module);
            let _ = (self.driver.ctx_destroy)(self.context);
        }
    }
}

struct DeviceBuffer {
    context: Rc<CudaContextInner>,
    ptr: CuDevicePtr,
    len: usize,
}

impl DeviceBuffer {
    fn allocate(runtime: &CudaCandidateRuntime, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(EngineError::Shape(
                "CUDA device allocation must be non-empty".into(),
            ));
        }
        runtime.make_current()?;
        let mut ptr = 0;
        unsafe {
            runtime.inner.driver.check(
                (runtime.inner.driver.mem_alloc)(&mut ptr, len),
                "device allocation",
            )?;
        }
        Ok(Self {
            context: Rc::clone(&runtime.inner),
            ptr,
            len,
        })
    }

    fn from_bytes(runtime: &CudaCandidateRuntime, bytes: &[u8]) -> Result<Self> {
        let buffer = Self::allocate(runtime, bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }

    fn write(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != self.len {
            return Err(EngineError::Shape(format!(
                "CUDA host-to-device copy has {} bytes, expected {}",
                bytes.len(),
                self.len
            )));
        }
        self.context.make_current()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_htod)(self.ptr, bytes.as_ptr().cast(), bytes.len()),
                "host-to-device copy",
            )
        }
    }

    fn write_range(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| EngineError::Shape("CUDA ranged copy overflows".into()))?;
        if end > self.len {
            return Err(EngineError::Shape(format!(
                "CUDA ranged copy ends at {end}, buffer has {} bytes",
                self.len
            )));
        }
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.make_current()?;
        let target = device_ptr_offset(self.ptr, offset)?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_htod)(target, bytes.as_ptr().cast(), bytes.len()),
                "ranged host-to-device copy",
            )
        }
    }

    fn copy_to(&self, bytes: &mut [u8]) -> Result<()> {
        if bytes.len() != self.len {
            return Err(EngineError::Shape(format!(
                "CUDA device-to-host copy has {} bytes, expected {}",
                bytes.len(),
                self.len
            )));
        }
        self.context.make_current()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_dtoh)(bytes.as_mut_ptr().cast(), self.ptr, bytes.len()),
                "device-to-host copy",
            )
        }
    }

    fn copy_from_buffer(&self, source: &Self, operation: &'static str) -> Result<()> {
        if !Rc::ptr_eq(&self.context, &source.context) || self.len != source.len {
            return Err(EngineError::InvalidState(
                "CUDA device checkpoint buffers differ in context or length".into(),
            ));
        }
        self.context.make_current()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_dtod)(self.ptr, source.ptr, self.len),
                operation,
            )
        }
    }

    fn copy_from_view(&self, source: CudaDeviceF32View<'_>, operation: &'static str) -> Result<()> {
        let source_bytes = source
            .values()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA device view bytes overflow".into()))?;
        if !Rc::ptr_eq(&self.context, source.context) || self.len != source_bytes {
            return Err(EngineError::InvalidState(
                "CUDA device checkpoint and source differ in context or length".into(),
            ));
        }
        self.context.make_current()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_dtod)(self.ptr, source.ptr()?, self.len),
                operation,
            )
        }
    }

    fn copy_range_to(&self, offset: usize, bytes: &mut [u8]) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| EngineError::Shape("CUDA ranged readback overflows".into()))?;
        if bytes.is_empty() || end > self.len {
            return Err(EngineError::Shape(format!(
                "CUDA ranged readback ends at {end}, buffer has {} bytes",
                self.len
            )));
        }
        self.context.make_current()?;
        let source = device_ptr_offset(self.ptr, offset)?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memcpy_dtoh)(bytes.as_mut_ptr().cast(), source, bytes.len()),
                "ranged device-to-host copy",
            )
        }
    }

    fn zero(&self) -> Result<()> {
        self.context.make_current()?;
        unsafe {
            self.context.driver.check(
                (self.context.driver.memset_d8)(self.ptr, 0, self.len),
                "device-buffer zero",
            )
        }
    }

    fn ptr(&self) -> CuDevicePtr {
        self.ptr
    }

    fn f32_view(&self, offset_values: usize, values: usize) -> Result<CudaDeviceF32View<'_>> {
        let end_values = offset_values
            .checked_add(values)
            .ok_or_else(|| EngineError::Shape("CUDA f32 view shape overflows".into()))?;
        let end_bytes = end_values
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| EngineError::Shape("CUDA f32 view bytes overflow".into()))?;
        if values == 0 || end_bytes > self.len {
            return Err(EngineError::Shape(format!(
                "CUDA f32 view ends at {end_bytes} bytes, buffer has {}",
                self.len
            )));
        }
        Ok(CudaDeviceF32View {
            context: &self.context,
            buffer: self,
            offset_values,
            values,
        })
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = self.context.make_current();
        unsafe {
            let _ = (self.context.driver.mem_free)(self.ptr);
        }
    }
}

fn cuda_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| EngineError::Shape(format!("{label} exceeds u32")))
}

fn optional_scale_buffer(
    runtime: &CudaCandidateRuntime,
    scales: Option<ScaleSlice<'_>>,
) -> Result<Option<DeviceBuffer>> {
    match scales {
        Some(ScaleSlice::F16Le(bytes)) => DeviceBuffer::from_bytes(runtime, bytes).map(Some),
        Some(ScaleSlice::F32(_)) => unreachable!("validated CUDA scales are FP16"),
        None => Ok(None),
    }
}

fn resolve_function(
    driver: &CudaDriver,
    module: CuModule,
    symbol_name: &str,
) -> Result<CuFunction> {
    let name = CString::new(symbol_name).map_err(|_| {
        EngineError::InvalidArtifact(format!("CUDA symbol contains NUL: {symbol_name:?}"))
    })?;
    let mut function = ptr::null_mut();
    unsafe {
        driver.check(
            (driver.module_get_function)(&mut function, module, name.as_ptr()),
            "function lookup",
        )?;
    }
    Ok(function)
}

fn load_driver_library() -> Result<Library> {
    #[cfg(target_os = "windows")]
    const NAMES: &[&str] = &["nvcuda.dll"];
    #[cfg(target_os = "linux")]
    const NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    const NAMES: &[&str] = &[];

    for name in NAMES {
        // SAFETY: CUDA is loaded only for explicit verifier construction and
        // the Library remains owned for the full lifetime of all symbols.
        if let Ok(library) = unsafe { Library::new(name) } {
            return Ok(library);
        }
    }
    Err(EngineError::UnsupportedOperation {
        backend: "cuda",
        operation: "load CUDA Driver API",
        reason: if NAMES.is_empty() {
            "CUDA verifier is supported only on Linux and Windows".into()
        } else {
            format!("none of the CUDA driver libraries could be loaded: {NAMES:?}")
        },
    })
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    library
        .get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|error| {
            EngineError::InvalidState(format!(
                "CUDA Driver API symbol {} is unavailable: {error}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ))
        })
}

unsafe fn nullable_cstr<'a>(value: *const c_char) -> Option<&'a str> {
    (!value.is_null())
        .then(|| CStr::from_ptr(value).to_str().ok())
        .flatten()
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe { slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values)) }
}

fn as_bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), size_of_val(values)) }
}

fn validate_f16_buffer(bytes: &[u8], expected: usize, name: &str) -> Result<()> {
    if bytes.len() != expected || !bytes.len().is_multiple_of(2) {
        return Err(EngineError::Shape(format!(
            "CUDA {name} has {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    if bytes
        .chunks_exact(2)
        .any(|pair| !half::f16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).is_finite())
    {
        return Err(EngineError::InvalidArtifact(format!(
            "CUDA {name} contains a non-finite FP16 value"
        )));
    }
    Ok(())
}

fn validate_f32_buffer(bytes: &[u8], expected: usize, name: &str) -> Result<()> {
    if bytes.len() != expected || !bytes.len().is_multiple_of(4) {
        return Err(EngineError::Shape(format!(
            "CUDA {name} has {} bytes, expected {expected}",
            bytes.len()
        )));
    }
    if bytes
        .chunks_exact(4)
        .any(|word| !f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_finite())
    {
        return Err(EngineError::InvalidArtifact(format!(
            "CUDA {name} contains a non-finite F32 value"
        )));
    }
    Ok(())
}

fn a8_scale_bytes(columns: usize) -> Result<usize> {
    columns
        .checked_div(64)
        .and_then(|blocks| blocks.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| EngineError::Shape("CUDA A8 scale size overflows usize".into()))
}

fn validate_a8_projection_layout(operation: &FusedMatVec<'_>) -> Result<CudaA8ProjectionLayout> {
    match operation.dtype {
        TensorDType::MixedQ2Q4B64 => {
            validate_mixed_operation(operation).map(CudaA8ProjectionLayout::Mixed)
        }
        TensorDType::Q2B64 | TensorDType::Q4B64 => {
            validate_operation(operation)?;
            Ok(CudaA8ProjectionLayout::Pure(operation.dtype))
        }
        _ => Err(EngineError::UnsupportedDType(format!(
            "CUDA A8 projection does not support {:?}",
            operation.dtype
        ))),
    }
}

fn validate_batched_a8_inputs(
    operation: &FusedMatVec<'_>,
    batch_inputs: &[f32],
    batch_rows: usize,
) -> Result<(CudaA8ProjectionLayout, u32, usize)> {
    let layout = validate_a8_projection_layout(operation)?;
    let batch_rows_u32 = validate_a8_batch_capacity(batch_rows)?;
    let input_values = batch_rows
        .checked_mul(operation.columns)
        .ok_or_else(|| EngineError::Shape("CUDA batched A8 input value count overflows".into()))?;
    if batch_inputs.len() != input_values {
        return Err(EngineError::Shape(format!(
            "CUDA batched A8 input has {} values, expected {input_values}",
            batch_inputs.len()
        )));
    }
    if batch_inputs.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::InvalidArtifact(
            "CUDA batched A8 input contains a non-finite value".into(),
        ));
    }
    Ok((layout, batch_rows_u32, input_values))
}

fn validate_a8_batch_capacity(batch_rows: usize) -> Result<u32> {
    if batch_rows == 0 {
        return Err(EngineError::Shape(
            "CUDA batched A8 matmul requires at least one input row".into(),
        ));
    }
    let batch_rows_u32 = u32::try_from(batch_rows)
        .map_err(|_| EngineError::Shape("CUDA A8 batch rows exceed u32".into()))?;
    if batch_rows_u32 > CUDA_GRID_Y_MAX {
        return Err(EngineError::Shape(format!(
            "CUDA A8 batch has {batch_rows} rows, maximum is {CUDA_GRID_Y_MAX}; chunk longer prefills"
        )));
    }
    Ok(batch_rows_u32)
}

fn paged_prefill_segment(
    committed_tokens: usize,
    remaining_tokens: usize,
    pages: &[CudaPagedKvPage],
    page_tokens: usize,
    sink_tokens: usize,
    recent_tokens: usize,
) -> Result<(usize, bool)> {
    if remaining_tokens == 0 || page_tokens == 0 {
        return Err(EngineError::Shape(
            "CUDA prefill segment requires remaining tokens and page capacity".into(),
        ));
    }
    let token_in_page = committed_tokens % page_tokens;
    let page_segment_tokens = remaining_tokens.min(page_tokens - token_in_page);
    let page_segment_end = committed_tokens
        .checked_add(page_segment_tokens)
        .ok_or_else(|| EngineError::Shape("CUDA page segment end overflows".into()))?;
    let future_recent_start = page_segment_end.saturating_sub(recent_tokens);
    let demotion_at_page_segment_end = pages.iter().any(|page| {
        page.precision == KvPrecision::Q4
            && page.first_token >= sink_tokens
            && page.first_token + page.tokens <= future_recent_start
    });
    // Decode demotes immediately before the boundary token's attention. Keep
    // the preceding page prefix batched, then isolate at most that one token.
    if demotion_at_page_segment_end && page_segment_tokens > 1 {
        Ok((page_segment_tokens - 1, false))
    } else {
        Ok((page_segment_tokens, demotion_at_page_segment_end))
    }
}

fn validate_batched_norm_inputs(
    runtime_context: &Rc<CudaContextInner>,
    prepared_context: &Rc<CudaContextInner>,
    workspace: &PreparedCudaBatchedRmsNormWorkspace,
    inputs: &[CudaDeviceF32View<'_>],
    columns: usize,
    batch_rows: usize,
) -> Result<u32> {
    let batch_rows = validate_a8_batch_capacity(batch_rows)?;
    if inputs.is_empty()
        || !Rc::ptr_eq(runtime_context, prepared_context)
        || !Rc::ptr_eq(runtime_context, &workspace.context)
        || inputs
            .iter()
            .any(|input| !Rc::ptr_eq(runtime_context, input.context))
    {
        return Err(EngineError::InvalidState(
            "batched CUDA RMSNorm dispatch crosses driver contexts".into(),
        ));
    }
    if columns != workspace.columns as usize || batch_rows > workspace.batch_capacity {
        return Err(EngineError::MemoryBudget(format!(
            "CUDA batched RMSNorm {batch_rows}x{columns} exceeds workspace {}x{}",
            workspace.batch_capacity, workspace.columns
        )));
    }
    let expected = (batch_rows as usize)
        .checked_mul(columns)
        .ok_or_else(|| EngineError::Shape("CUDA batched RMSNorm shape overflows".into()))?;
    if inputs.iter().any(|input| input.values() != expected) {
        return Err(EngineError::Shape(format!(
            "CUDA batched RMSNorm input does not have expected {expected} values"
        )));
    }
    Ok(batch_rows)
}

fn embedding_row_location(
    layout: &CudaA8ProjectionLayout,
    rows: u32,
    columns: u32,
    row: usize,
) -> Result<(TensorDType, usize)> {
    if row >= rows as usize || columns == 0 || !(columns as usize).is_multiple_of(BLOCK_LEN) {
        return Err(EngineError::Shape(format!(
            "CUDA embedding row {row} or geometry is invalid"
        )));
    }
    let blocks_per_row = columns as usize / BLOCK_LEN;
    match layout {
        CudaA8ProjectionLayout::Pure(dtype) => {
            let block_bytes = match dtype {
                TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                _ => unreachable!("validated pure embedding dtype"),
            };
            let row_bytes = blocks_per_row.checked_mul(block_bytes).ok_or_else(|| {
                EngineError::MemoryBudget("CUDA embedding row bytes overflow".into())
            })?;
            Ok((
                *dtype,
                row.checked_mul(row_bytes).ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA embedding row offset overflows".into())
                })?,
            ))
        }
        CudaA8ProjectionLayout::Mixed(segments) => {
            let segment = segments
                .iter()
                .find(|segment| {
                    let start = segment.row_start as usize;
                    let end = start.saturating_add(segment.row_count as usize);
                    row >= start && row < end
                })
                .ok_or_else(|| {
                    EngineError::InvalidArtifact(format!(
                        "CUDA embedding row {row} has no mixed segment"
                    ))
                })?;
            let row_bytes = blocks_per_row
                .checked_mul(segment.descriptor.block_bytes)
                .ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA mixed embedding row bytes overflow".into())
                })?;
            let local = row - segment.row_start as usize;
            let offset = local
                .checked_mul(row_bytes)
                .and_then(|bytes| bytes.checked_add(segment.weight_offset))
                .ok_or_else(|| {
                    EngineError::MemoryBudget("CUDA mixed embedding offset overflows".into())
                })?;
            Ok((segment.descriptor.dtype, offset))
        }
    }
}

fn validate_recovered_a8_projection_layout(
    recovered: RecoveredMatrixView<'_>,
) -> Result<CudaA8ProjectionLayout> {
    let matrix = recovered.matrix;
    if matrix.rows == 0 || matrix.columns == 0 || !matrix.columns.is_multiple_of(BLOCK_LEN) {
        return Err(EngineError::Shape(
            "mmap CUDA matrix dimensions must be non-zero and columns divisible by 64".into(),
        ));
    }
    u32::try_from(matrix.rows)
        .map_err(|_| EngineError::Shape("mmap CUDA rows exceed u32 launch limit".into()))?;
    u32::try_from(matrix.columns)
        .map_err(|_| EngineError::Shape("mmap CUDA columns exceed u32 launch limit".into()))?;

    for (name, view, values) in [
        ("s_in", recovered.s_in, matrix.columns),
        ("s_out", recovered.s_out, matrix.rows),
    ] {
        let scales = view.as_recovery_scales()?;
        let expected_bytes = values
            .checked_mul(std::mem::size_of::<half::f16>())
            .ok_or_else(|| EngineError::Shape(format!("mmap CUDA {name} size overflows")))?;
        match scales {
            ScaleSlice::F16Le(bytes) => validate_f16_buffer(bytes, expected_bytes, name)?,
            ScaleSlice::F32(_) => unreachable!("recovered scales reject F32"),
        }
    }

    let blocks_per_row = matrix.columns / BLOCK_LEN;
    match matrix.dtype {
        TensorDType::Q2B64 | TensorDType::Q4B64 => {
            if !matrix.segments.is_empty() {
                return Err(EngineError::InvalidArtifact(
                    "pure mmap CUDA Q2/Q4 matrix declares mixed row segments".into(),
                ));
            }
            let block_bytes = match matrix.dtype {
                TensorDType::Q2B64 => Q2_BLOCK_BYTES,
                TensorDType::Q4B64 => Q4_BLOCK_BYTES,
                _ => unreachable!(),
            };
            let expected_weights = matrix
                .rows
                .checked_mul(blocks_per_row)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or_else(|| EngineError::Shape("pure mmap CUDA weight size overflows".into()))?;
            if matrix.weights.len() != expected_weights {
                return Err(EngineError::Shape(format!(
                    "mmap CUDA weight buffer has {} bytes, expected {expected_weights}",
                    matrix.weights.len()
                )));
            }
            Ok(CudaA8ProjectionLayout::Pure(matrix.dtype))
        }
        TensorDType::MixedQ2Q4B64 => {
            if matrix.segments.is_empty() {
                return Err(EngineError::InvalidArtifact(
                    "mixed mmap CUDA Q2/Q4 matrix has no row segments".into(),
                ));
            }
            let mut launches = Vec::with_capacity(matrix.segments.len());
            let mut expected_row = 0_usize;
            let mut expected_offset = 0_usize;
            for (expected_group, segment) in matrix.segments.iter().enumerate() {
                let group_index = usize::try_from(segment.group_index).map_err(|_| {
                    EngineError::InvalidArtifact("mixed mmap CUDA group index overflows".into())
                })?;
                let row_start = usize::try_from(segment.row_start).map_err(|_| {
                    EngineError::InvalidArtifact("mixed mmap CUDA row start overflows".into())
                })?;
                let row_end = usize::try_from(segment.row_end).map_err(|_| {
                    EngineError::InvalidArtifact("mixed mmap CUDA row end overflows".into())
                })?;
                let offset = usize::try_from(segment.offset).map_err(|_| {
                    EngineError::InvalidArtifact("mixed mmap CUDA offset overflows".into())
                })?;
                let length = usize::try_from(segment.length).map_err(|_| {
                    EngineError::InvalidArtifact("mixed mmap CUDA length overflows".into())
                })?;
                if group_index != expected_group
                    || row_start != expected_row
                    || row_end <= row_start
                    || row_end > matrix.rows
                    || offset != expected_offset
                {
                    return Err(EngineError::InvalidArtifact(format!(
                        "mixed mmap CUDA matrix has non-contiguous segment {}",
                        segment.group_index
                    )));
                }
                let (descriptor, block_bytes) = match segment.dtype {
                    TensorDType::Q2B64 => (&Q2_B64_FUSED_MATVEC, Q2_BLOCK_BYTES),
                    TensorDType::Q4B64 => (&Q4_B64_FUSED_MATVEC, Q4_BLOCK_BYTES),
                    other => {
                        return Err(EngineError::InvalidArtifact(format!(
                            "mixed mmap CUDA segment {} has invalid dtype {other:?}",
                            segment.group_index
                        )));
                    }
                };
                let expected_length = row_end
                    .checked_sub(row_start)
                    .and_then(|rows| rows.checked_mul(blocks_per_row))
                    .and_then(|blocks| blocks.checked_mul(block_bytes))
                    .ok_or_else(|| {
                        EngineError::Shape("mixed mmap CUDA segment size overflows".into())
                    })?;
                if length != expected_length {
                    return Err(EngineError::InvalidArtifact(format!(
                        "mixed mmap CUDA segment {} has {length} bytes, expected {expected_length}",
                        segment.group_index
                    )));
                }
                launches.push(CudaMixedRowSegment {
                    descriptor,
                    row_start: u32::try_from(row_start).map_err(|_| {
                        EngineError::Shape("mixed mmap CUDA row start exceeds u32".into())
                    })?,
                    row_count: u32::try_from(row_end - row_start).map_err(|_| {
                        EngineError::Shape("mixed mmap CUDA row count exceeds u32".into())
                    })?,
                    weight_offset: offset,
                });
                expected_row = row_end;
                expected_offset = expected_offset.checked_add(length).ok_or_else(|| {
                    EngineError::Shape("mixed mmap CUDA weight size overflows".into())
                })?;
            }
            if expected_row != matrix.rows || expected_offset != matrix.weights.len() {
                return Err(EngineError::Shape(format!(
                    "mixed mmap CUDA segments cover {expected_row}/{} rows and {expected_offset}/{} bytes",
                    matrix.rows,
                    matrix.weights.len()
                )));
            }
            Ok(CudaA8ProjectionLayout::Mixed(launches))
        }
        other => Err(EngineError::UnsupportedDType(format!(
            "mmap CUDA A8 projection does not support {other:?}"
        ))),
    }
}

fn a8_correction_identity(
    columns: usize,
    s_in: Option<ScaleSlice<'_>>,
) -> Result<A8CorrectionIdentity> {
    let columns_u32 = u32::try_from(columns)
        .map_err(|_| EngineError::Shape("CUDA A8 columns exceed u32".into()))?;
    let mut digest = Sha256::new();
    digest.update(b"ctox.cuda.a8-correction-identity.v1\0");
    digest.update(columns_u32.to_le_bytes());
    match s_in {
        None => digest.update([0]),
        Some(ScaleSlice::F16Le(bytes)) => {
            let expected = columns
                .checked_mul(2)
                .ok_or_else(|| EngineError::Shape("CUDA A8 s_in size overflows".into()))?;
            if bytes.len() != expected {
                return Err(EngineError::Shape(format!(
                    "CUDA A8 s_in has {} bytes, expected {expected}",
                    bytes.len()
                )));
            }
            digest.update([1]);
            digest.update(bytes);
        }
        Some(ScaleSlice::F32(_)) => {
            return Err(EngineError::UnsupportedDType(
                "CUDA A8 correction identity requires packed FP16 s_in".into(),
            ));
        }
    }
    Ok(A8CorrectionIdentity {
        columns: columns_u32,
        s_in_sha256: digest.finalize().into(),
    })
}

fn device_ptr_offset(base: CuDevicePtr, offset: usize) -> Result<CuDevicePtr> {
    let offset = u64::try_from(offset)
        .map_err(|_| EngineError::Shape("CUDA device pointer offset exceeds u64".into()))?;
    base.checked_add(offset)
        .ok_or_else(|| EngineError::Shape("CUDA device pointer offset overflows".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::QuantSegment;
    use crate::loader::{FloatTensorView, QuantizedMatrixView};

    #[test]
    fn driver_2d_descriptor_and_mtp_row_geometry_are_exact() {
        assert_eq!(std::mem::size_of::<CudaMemcpy2DDescriptor>(), 128);
        let geometry = F32Copy2DGeometry {
            source_row_values: 5,
            destination_row_values: 10,
            destination_column: 5,
            rows: 3,
            columns: 5,
        };
        geometry.validate(15, 30).unwrap();
        assert!(geometry.validate(14, 30).is_err());
        assert!(geometry.validate(15, 29).is_err());
        assert!(F32Copy2DGeometry {
            destination_column: 6,
            ..geometry
        }
        .validate(15, 30)
        .is_err());
    }

    #[test]
    fn launch_geometry_covers_partial_last_block() {
        let warps = THREADS_PER_BLOCK / WARP_SIZE;
        assert_eq!(warps, 4);
        assert_eq!(1_u32.div_ceil(warps), 1);
        assert_eq!(4_u32.div_ceil(warps), 1);
        assert_eq!(5_u32.div_ceil(warps), 2);
        assert_eq!(11_u32.div_ceil(warps), 3);
    }

    #[test]
    fn a8_launch_geometry_assigns_two_rows_per_warp() {
        assert_eq!(A8_ROWS_PER_BLOCK, 8);
        assert_eq!(1_u32.div_ceil(A8_ROWS_PER_BLOCK), 1);
        assert_eq!(8_u32.div_ceil(A8_ROWS_PER_BLOCK), 1);
        assert_eq!(9_u32.div_ceil(A8_ROWS_PER_BLOCK), 2);
        assert_eq!(257_u32.div_ceil(A8_ROWS_PER_BLOCK), 33);
    }

    #[test]
    fn mmq_launch_geometry_reuses_each_weight_tile_across_sixty_four_tokens() {
        assert_eq!(MMQ_THREADS_PER_BLOCK, 256);
        assert_eq!(MMQ_ROWS_PER_BLOCK, 128);
        assert_eq!(MMQ_BATCH_ROWS_PER_BLOCK, 64);
        assert_eq!(257_u32.div_ceil(MMQ_ROWS_PER_BLOCK), 3);
        assert_eq!(65_u32.div_ceil(MMQ_BATCH_ROWS_PER_BLOCK), 2);
    }

    #[test]
    fn a8_transient_layout_is_one_code_per_value_and_one_scale_per_block() {
        assert_eq!(a8_scale_bytes(64).unwrap(), 4);
        assert_eq!(a8_scale_bytes(512).unwrap(), 32);
        assert_eq!(512 + a8_scale_bytes(512).unwrap(), 544);
    }

    #[test]
    fn batched_a8_shape_validation_is_fail_closed_before_cuda_allocation() {
        let weights = vec![0_u8; 3 * Q2_BLOCK_BYTES];
        let placeholder = vec![0.0_f32; BLOCK_LEN];
        let operation = FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &weights,
            segments: &[],
            rows: 3,
            columns: BLOCK_LEN,
            input: &placeholder,
            s_in: None,
            s_out: None,
            bias: None,
            activation: Activation::Identity,
        };
        let batch = vec![0.25_f32; 5 * BLOCK_LEN];
        let (layout, rows, values) = validate_batched_a8_inputs(&operation, &batch, 5).unwrap();
        assert!(matches!(
            layout,
            CudaA8ProjectionLayout::Pure(TensorDType::Q2B64)
        ));
        assert_eq!(rows, 5);
        assert_eq!(values, batch.len());
        assert!(validate_batched_a8_inputs(&operation, &batch, 0).is_err());
        assert!(validate_batched_a8_inputs(&operation, &batch[..batch.len() - 1], 5).is_err());
        let mut nonfinite = batch;
        nonfinite[BLOCK_LEN] = f32::NAN;
        assert!(matches!(
            validate_batched_a8_inputs(&operation, &nonfinite, 5),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn shared_a8_identity_requires_exact_packed_s_in_bytes() {
        let one = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let two = half::f16::from_f32(2.0).to_bits().to_le_bytes();
        let mut first = one.repeat(64);
        let second = first.clone();
        let identity = a8_correction_identity(64, Some(ScaleSlice::F16Le(&first))).unwrap();
        assert_eq!(
            identity,
            a8_correction_identity(64, Some(ScaleSlice::F16Le(&second))).unwrap()
        );
        first[..2].copy_from_slice(&two);
        assert_ne!(
            identity,
            a8_correction_identity(64, Some(ScaleSlice::F16Le(&first))).unwrap()
        );
        assert_ne!(identity, a8_correction_identity(64, None).unwrap());
        assert!(a8_correction_identity(64, Some(ScaleSlice::F32(&[1.0; 64]))).is_err());
        assert!(a8_correction_identity(64, Some(ScaleSlice::F16Le(&second[..126]))).is_err());
    }

    #[test]
    fn mmap_a8_layout_accepts_exact_pure_q2_payload() {
        let weights = vec![0_u8; 2 * Q2_BLOCK_BYTES];
        let scales = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let s_in = scales.repeat(BLOCK_LEN);
        let s_out = scales.repeat(2);
        let recovered = RecoveredMatrixView {
            matrix: QuantizedMatrixView {
                dtype: TensorDType::Q2B64,
                weights: &weights,
                segments: &[],
                rows: 2,
                columns: BLOCK_LEN,
            },
            s_in: FloatTensorView::F16Le(&s_in),
            s_out: FloatTensorView::F16Le(&s_out),
        };
        assert!(matches!(
            validate_recovered_a8_projection_layout(recovered).unwrap(),
            CudaA8ProjectionLayout::Pure(TensorDType::Q2B64)
        ));
    }

    #[test]
    fn mmap_a8_layout_preserves_contiguous_mixed_segments() {
        let q2_bytes = 2 * Q2_BLOCK_BYTES;
        let q4_bytes = Q4_BLOCK_BYTES;
        let weights = vec![0_u8; q2_bytes + q4_bytes];
        let segments = vec![
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: 2,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2_bytes as u64,
            },
            QuantSegment {
                group_index: 1,
                row_start: 2,
                row_end: 3,
                dtype: TensorDType::Q4B64,
                offset: q2_bytes as u64,
                length: q4_bytes as u64,
            },
        ];
        let scales = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let s_in = scales.repeat(BLOCK_LEN);
        let s_out = scales.repeat(3);
        let recovered = RecoveredMatrixView {
            matrix: QuantizedMatrixView {
                dtype: TensorDType::MixedQ2Q4B64,
                weights: &weights,
                segments: &segments,
                rows: 3,
                columns: BLOCK_LEN,
            },
            s_in: FloatTensorView::F16Le(&s_in),
            s_out: FloatTensorView::F16Le(&s_out),
        };
        let CudaA8ProjectionLayout::Mixed(launches) =
            validate_recovered_a8_projection_layout(recovered).unwrap()
        else {
            panic!("expected mixed mmap CUDA layout");
        };
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].row_start, 0);
        assert_eq!(launches[0].row_count, 2);
        assert_eq!(launches[1].row_start, 2);
        assert_eq!(launches[1].row_count, 1);
        assert_eq!(launches[1].weight_offset, q2_bytes);
    }

    #[test]
    fn mmap_a8_layout_rejects_nonfinite_recovery_scales() {
        let weights = vec![0_u8; Q2_BLOCK_BYTES];
        let one = half::f16::from_f32(1.0).to_bits().to_le_bytes();
        let mut s_in = one.repeat(BLOCK_LEN);
        s_in[..2].copy_from_slice(&half::f16::INFINITY.to_bits().to_le_bytes());
        let s_out = one.to_vec();
        let recovered = RecoveredMatrixView {
            matrix: QuantizedMatrixView {
                dtype: TensorDType::Q2B64,
                weights: &weights,
                segments: &[],
                rows: 1,
                columns: BLOCK_LEN,
            },
            s_in: FloatTensorView::F16Le(&s_in),
            s_out: FloatTensorView::F16Le(&s_out),
        };
        assert!(matches!(
            validate_recovered_a8_projection_layout(recovered),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn artifact_f32_validation_rejects_wrong_size_and_nonfinite_values() {
        let finite = 1.25_f32.to_le_bytes().repeat(GATED_DELTA_HEADS);
        validate_f32_buffer(&finite, GATED_DELTA_HEADS * 4, "test").unwrap();
        assert!(validate_f32_buffer(&finite[..finite.len() - 4], finite.len(), "test").is_err());
        let mut nonfinite = finite;
        nonfinite[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            validate_f32_buffer(&nonfinite, nonfinite.len(), "test"),
            Err(EngineError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn resident_embedding_rows_resolve_pure_and_mixed_offsets() {
        assert_eq!(
            embedding_row_location(
                &CudaA8ProjectionLayout::Pure(TensorDType::Q2B64),
                3,
                BLOCK_LEN as u32,
                2,
            )
            .unwrap(),
            (TensorDType::Q2B64, 2 * Q2_BLOCK_BYTES)
        );
        let mixed = CudaA8ProjectionLayout::Mixed(vec![
            CudaMixedRowSegment {
                descriptor: &Q2_B64_FUSED_MATVEC,
                row_start: 0,
                row_count: 2,
                weight_offset: 0,
            },
            CudaMixedRowSegment {
                descriptor: &Q4_B64_FUSED_MATVEC,
                row_start: 2,
                row_count: 1,
                weight_offset: 2 * Q2_BLOCK_BYTES,
            },
        ]);
        assert_eq!(
            embedding_row_location(&mixed, 3, BLOCK_LEN as u32, 2).unwrap(),
            (TensorDType::Q4B64, 2 * Q2_BLOCK_BYTES)
        );
        assert!(embedding_row_location(&mixed, 3, BLOCK_LEN as u32, 3).is_err());
    }

    #[test]
    fn gathered_rows_preserve_canonical_order_across_q2_q4_segments() {
        let mixed = CudaA8ProjectionLayout::Mixed(vec![
            CudaMixedRowSegment {
                descriptor: &Q2_B64_FUSED_MATVEC,
                row_start: 0,
                row_count: 3,
                weight_offset: 0,
            },
            CudaMixedRowSegment {
                descriptor: &Q4_B64_FUSED_MATVEC,
                row_start: 3,
                row_count: 2,
                weight_offset: 3 * Q2_BLOCK_BYTES,
            },
            CudaMixedRowSegment {
                descriptor: &Q2_B64_FUSED_MATVEC,
                row_start: 5,
                row_count: 3,
                weight_offset: 3 * Q2_BLOCK_BYTES + 2 * Q4_BLOCK_BYTES,
            },
        ]);
        let (local_ids, groups) = build_gathered_row_plan(&mixed, 8, &[0, 2, 3, 4, 7]).unwrap();
        assert_eq!(local_ids, vec![0, 2, 0, 1, 2]);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].dtype, TensorDType::Q2B64);
        assert_eq!(groups[0].row_count, 2);
        assert_eq!(groups[0].row_id_offset, 0);
        assert_eq!(groups[1].dtype, TensorDType::Q4B64);
        assert_eq!(groups[1].row_count, 2);
        assert_eq!(groups[1].row_id_offset, 2 * std::mem::size_of::<u32>());
        assert_eq!(groups[1].output_offset, 2 * std::mem::size_of::<f32>());
        assert_eq!(groups[1].scale_row_offset, 3);
        assert_eq!(groups[2].dtype, TensorDType::Q2B64);
        assert_eq!(groups[2].row_count, 1);
        assert_eq!(groups[2].scale_row_offset, 5);
        assert!(build_gathered_row_plan(&mixed, 8, &[3, 3]).is_err());
        assert!(build_gathered_row_plan(&mixed, 8, &[8]).is_err());
    }

    #[test]
    fn checked_device_pointer_offsets_fail_closed() {
        assert_eq!(device_ptr_offset(1_024, 256).unwrap(), 1_280);
        assert!(device_ptr_offset(u64::MAX, 1).is_err());
    }

    #[test]
    fn recovered_row_launch_covers_partial_thread_block() {
        assert_eq!(64_u32.div_ceil(THREADS_PER_BLOCK), 1);
        assert_eq!(256_u32.div_ceil(THREADS_PER_BLOCK), 2);
        assert_eq!(5_120_u32.div_ceil(THREADS_PER_BLOCK), 40);
    }

    #[test]
    fn driver_parameter_widths_match_cuda_abi() {
        assert_eq!(std::mem::size_of::<CuDevicePtr>(), 8);
        assert_eq!(std::mem::size_of::<u32>(), 4);
    }

    #[test]
    fn prefill_chunk_capacity_is_bounded_by_cuda_grid_geometry() {
        assert_eq!(validate_a8_batch_capacity(1).unwrap(), 1);
        assert_eq!(
            validate_a8_batch_capacity(CUDA_GRID_Y_MAX as usize).unwrap(),
            CUDA_GRID_Y_MAX
        );
        assert!(validate_a8_batch_capacity(0).is_err());
        assert!(validate_a8_batch_capacity(CUDA_GRID_Y_MAX as usize + 1).is_err());
    }

    #[test]
    fn paged_prefill_isolates_only_the_q2_demotion_boundary_token() {
        let mut pages = (0..3)
            .map(|index| CudaPagedKvPage {
                precision: KvPrecision::Q4,
                physical_slot: index,
                tokens: 128,
                first_token: index * 128,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            paged_prefill_segment(384, 128, &pages, 128, 128, 256).unwrap(),
            (127, false)
        );
        pages.push(CudaPagedKvPage {
            precision: KvPrecision::Q4,
            physical_slot: 3,
            tokens: 127,
            first_token: 384,
        });
        assert_eq!(
            paged_prefill_segment(511, 1, &pages, 128, 128, 256).unwrap(),
            (1, true)
        );
        assert_eq!(
            paged_prefill_segment(0, 512, &[], 128, 128, 256).unwrap(),
            (128, false)
        );
        assert!(paged_prefill_segment(0, 0, &[], 128, 128, 256).is_err());
    }

    #[test]
    fn prepared_graph_objects_own_their_context_without_borrowed_lifetimes() {
        fn assert_owned<T: 'static>() {}
        assert_owned::<PreparedCudaMatVec>();
        assert_owned::<PreparedCudaA8MatVec>();
        assert_owned::<PreparedCudaMixedA8MatVec>();
        assert_owned::<PreparedCudaA8Activation>();
        assert_owned::<PreparedCudaBatchedA8Activation>();
        assert_owned::<PreparedCudaBatchedA8Workspace>();
        assert_owned::<PreparedCudaA8Projection>();
        assert_owned::<PreparedCudaBatchedA8Output>();
        assert_owned::<PreparedCudaBatchedA8OutputArena>();
        assert_owned::<PreparedCudaBatchedRmsNormWorkspace>();
        assert_owned::<PreparedCudaBatchedGatedRmsNormOutput>();
        assert_owned::<PreparedCudaCausalConvScanOutput>();
        assert_owned::<PreparedCudaGatedDeltaScanInputs>();
        assert_owned::<PreparedCudaGatedDeltaScanOutput>();
        assert_owned::<PreparedCudaGatheredA8Projection>();
        assert_owned::<PreparedCudaArgmax>();
        assert_owned::<PreparedCudaRecoveredRow>();
        assert_owned::<PreparedCudaEmbedding>();
        assert_owned::<PreparedCudaBatchedEmbeddingWorkspace>();
        assert_owned::<PreparedCudaPagedGqa>();
        assert_owned::<PreparedCudaPagedGqaPrefillOutput>();
        assert_owned::<PreparedCudaQueryGate>();
        assert_owned::<CudaVerifierF32Tensor>();
    }
}
