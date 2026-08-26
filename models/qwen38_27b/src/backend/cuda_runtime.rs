//! Direct CUDA Driver API execution for the unpromoted SM86 Q2/Q4 candidate.
//!
//! This verifier runtime deliberately does not implement [`super::Backend`].
//! It loads an explicitly supplied cubin, resolves the ABI symbols pinned in
//! [`super::cuda`], owns all device allocations, and fails closed on any
//! driver/shape/profile mismatch. There is no CUDA Runtime API, framework, or
//! CPU fallback in this path.

use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::size_of_val;
use std::ptr;
use std::rc::Rc;
use std::slice;

use libloading::Library;
use sha2::{Digest, Sha256};

use super::cuda::{
    validate_mixed_operation, validate_operation, validate_recovered_row, CudaMixedRowSegment,
    A8_QUANTIZE_SYMBOL, CAUSAL_CONV_F16_SYMBOL, DEMOTE_PAGED_KV_Q4_TO_Q2_SYMBOL,
    GATED_DELTA_F16_SYMBOL, GATED_DELTA_HEADS, GATED_DELTA_KEY_DIM, GATED_DELTA_STATE_BYTES,
    GATED_DELTA_VALUE_DIM, GATED_RMS_NORM_COLUMNS, GATED_RMS_NORM_F16_SYMBOL, GATED_RMS_NORM_ROWS,
    LINEAR_CONV_CHANNELS, LINEAR_CONV_KERNEL_WIDTH, LINEAR_CONV_STATE_BYTES,
    PACK_PAGED_KV_Q4_F32_SYMBOL, PAGED_GQA_DESCRIPTOR_BYTES, PAGED_GQA_PARAMS_BYTES,
    PAGED_Q2Q4_GQA_F32_SYMBOL, PARTIAL_ROPE_F32_SYMBOL, Q2_B64_A8_MATVEC_SYMBOL,
    Q2_B64_FUSED_MATVEC, Q2_B64_RECOVERED_ROW_SYMBOL, Q4_B64_A8_MATVEC_SYMBOL, Q4_B64_FUSED_MATVEC,
    Q4_B64_RECOVERED_ROW_SYMBOL, QWEN_RMS_NORM_F16_SYMBOL,
};
use super::{Activation, FusedMatVec, RecoveredRow, ScaleSlice};
use crate::format::TensorDType;
use crate::kv_cache::KvPrecision;
use crate::quant::{BLOCK_LEN, Q2_BLOCK_BYTES, Q4_BLOCK_BYTES};
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

struct CudaContextInner {
    driver: CudaDriver,
    context: CuContext,
    module: CuModule,
    q2_function: CuFunction,
    q4_function: CuFunction,
    a8_quantize_function: CuFunction,
    q2_a8_function: CuFunction,
    q4_a8_function: CuFunction,
    q2_recovered_row_function: CuFunction,
    q4_recovered_row_function: CuFunction,
    gated_delta_f16_function: CuFunction,
    causal_conv_f16_function: CuFunction,
    gated_rms_norm_f16_function: CuFunction,
    qwen_rms_norm_f16_function: CuFunction,
    partial_rope_f32_function: CuFunction,
    pack_paged_kv_q4_f32_function: CuFunction,
    demote_paged_kv_q4_to_q2_function: CuFunction,
    paged_q2q4_gqa_f32_function: CuFunction,
    device_name: String,
    compute_capability: (u32, u32),
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
    input: DeviceBuffer,
    s_in: Option<DeviceBuffer>,
    q8_codes: DeviceBuffer,
    q8_scales: DeviceBuffer,
    resident_bytes: usize,
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
    output: DeviceBuffer,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
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
    output: DeviceBuffer,
    model_bytes: usize,
    resident_state_bytes: usize,
    transient_bytes: usize,
    poisoned: bool,
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
        Ok(Self {
            inner: Rc::new(CudaContextInner {
                driver,
                context,
                module,
                q2_function,
                q4_function,
                a8_quantize_function,
                q2_a8_function,
                q4_a8_function,
                q2_recovered_row_function,
                q4_recovered_row_function,
                gated_delta_f16_function,
                causal_conv_f16_function,
                gated_rms_norm_f16_function,
                qwen_rms_norm_f16_function,
                partial_rope_f32_function,
                pack_paged_kv_q4_f32_function,
                demote_paged_kv_q4_to_q2_function,
                paged_q2q4_gqa_f32_function,
                device_name,
                compute_capability,
            }),
        })
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
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        state.zero()?;
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
            output,
            resident_state_bytes: GATED_DELTA_STATE_BYTES,
            transient_bytes,
            poisoned: false,
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
        self.make_current()?;
        let mut query = prepared.query.ptr();
        let mut key = prepared.key.ptr();
        let mut value = prepared.value.ptr();
        let mut log_decay = prepared.log_decay.ptr();
        let mut beta = prepared.beta.ptr();
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "gated-delta context synchronization",
            )?;
        }
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
        let output = DeviceBuffer::allocate(self, value_bytes)?;
        input.zero()?;
        state.zero()?;
        output.zero()?;
        Ok(PreparedCudaCausalConv {
            context: Rc::clone(&self.inner),
            config,
            input,
            weight,
            state,
            output,
            model_bytes: weight_f16_le.len(),
            resident_state_bytes: LINEAR_CONV_STATE_BYTES,
            transient_bytes: value_bytes * 2,
            poisoned: false,
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
        self.make_current()?;
        let mut input = prepared.input.ptr();
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "causal-convolution context synchronization",
            )?;
        }
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

    pub fn dispatch_gated_rms_norm_f16(
        &self,
        prepared: &PreparedCudaGatedRmsNorm,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA gated RMSNorm belongs to another context".into(),
            ));
        }
        self.make_current()?;
        let mut input = prepared.input.ptr();
        let mut gate = prepared.gate.ptr();
        let mut weight = prepared.weight.ptr();
        let mut output = prepared.output.ptr();
        let mut rows = prepared.config.rows as u32;
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "gated RMSNorm context synchronization",
            )?;
        }
        let mut result = vec![0.0_f32; prepared.config.rows * prepared.config.columns];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA gated RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(result)
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
        self.make_current()?;
        let mut input = prepared.input.ptr();
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "Qwen RMSNorm context synchronization",
            )?;
        }
        let mut result = vec![0.0_f32; prepared.config.rows * prepared.config.columns];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA Qwen RMSNorm produced a non-finite output".into(),
            ));
        }
        Ok(result)
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

    pub fn dispatch_partial_rope_f32(
        &self,
        prepared: &PreparedCudaPartialRope,
    ) -> Result<Vec<f32>> {
        if !Rc::ptr_eq(&self.inner, &prepared.context) {
            return Err(EngineError::InvalidState(
                "prepared CUDA partial RoPE belongs to another context".into(),
            ));
        }
        self.make_current()?;
        let mut values = prepared.values.ptr();
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "partial-RoPE context synchronization",
            )?;
        }
        let mut result = vec![0.0_f32; prepared.config.heads * prepared.config.head_dim];
        prepared.values.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA partial RoPE produced a non-finite output".into(),
            ));
        }
        Ok(result)
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
        prepared.poisoned = true;
        let result = self.append_and_dispatch_paged_q2q4_gqa_inner(prepared, query, key, value);
        if result.is_ok() {
            prepared.poisoned = false;
        }
        result
    }

    fn append_and_dispatch_paged_q2q4_gqa_inner(
        &self,
        prepared: &mut PreparedCudaPagedGqa,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
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
        prepared.key.write(as_bytes(key))?;
        prepared.value.write(as_bytes(value))?;
        let mut q4_pages_ptr = prepared.q4_pages.ptr();
        let mut key_ptr = prepared.key.ptr();
        let mut value_ptr = prepared.value.ptr();
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

        let recent_start = prepared
            .tokens
            .saturating_sub(prepared.config.recent_tokens);
        let demoted_pages = prepared
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
            .collect::<Vec<_>>();
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
        prepared.query.write(as_bytes(query))?;
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
        prepared.params.write(as_bytes(&params_words))?;

        self.make_current()?;
        let mut query_ptr = prepared.query.ptr();
        let mut q2_pages = prepared.q2_pages.ptr();
        let mut q4_pages = prepared.q4_pages.ptr();
        let mut descriptors = prepared.descriptors.ptr();
        let mut output = prepared.output.ptr();
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
            self.inner.driver.check(
                (self.inner.driver.ctx_synchronize)(),
                "paged Q2/Q4 GQA context synchronization",
            )?;
        }
        let mut result = vec![0.0_f32; prepared.config.query_heads * prepared.config.head_dim];
        prepared.output.copy_to(as_bytes_mut(&mut result))?;
        if result.iter().any(|item| !item.is_finite()) {
            return Err(EngineError::InvalidState(
                "CUDA paged GQA produced a non-finite output".into(),
            ));
        }
        Ok(result)
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
            input,
            s_in,
            q8_codes,
            q8_scales,
            resident_bytes,
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
        self.launch_a8_quantization(
            prepared.input.ptr(),
            prepared.s_in.as_ref().map_or(0, DeviceBuffer::ptr),
            prepared.q8_codes.ptr(),
            prepared.q8_scales.ptr(),
            prepared.columns,
        )
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
        self.input.write(as_bytes(input))
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
}

impl PreparedCudaGatedDelta {
    pub fn config(&self) -> CudaGatedDeltaConfig {
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
        self.output.zero()?;
        self.poisoned = false;
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
        self.output.zero()?;
        self.poisoned = false;
        Ok(())
    }

    pub fn verifier_read_state(&self) -> Result<Vec<half::f16>> {
        let mut state = vec![half::f16::ZERO; self.resident_state_bytes / 2];
        self.state.copy_to(as_bytes_mut(&mut state))?;
        Ok(state)
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

impl PreparedCudaPartialRope {
    pub fn config(&self) -> CudaPartialRopeConfig {
        self.config
    }

    pub fn transient_bytes(&self) -> usize {
        self.transient_bytes
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
    fn a8_transient_layout_is_one_code_per_value_and_one_scale_per_block() {
        assert_eq!(a8_scale_bytes(64).unwrap(), 4);
        assert_eq!(a8_scale_bytes(512).unwrap(), 32);
        assert_eq!(512 + a8_scale_bytes(512).unwrap(), 544);
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
    fn prepared_graph_objects_own_their_context_without_borrowed_lifetimes() {
        fn assert_owned<T: 'static>() {}
        assert_owned::<PreparedCudaMatVec>();
        assert_owned::<PreparedCudaA8MatVec>();
        assert_owned::<PreparedCudaMixedA8MatVec>();
        assert_owned::<PreparedCudaA8Activation>();
        assert_owned::<PreparedCudaA8Projection>();
        assert_owned::<PreparedCudaRecoveredRow>();
    }
}
