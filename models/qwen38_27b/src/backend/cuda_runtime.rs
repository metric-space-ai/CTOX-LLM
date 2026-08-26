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

use super::cuda::{
    validate_mixed_operation, validate_operation, validate_recovered_row, CudaMixedRowSegment,
    A8_QUANTIZE_SYMBOL, Q2_B64_A8_MATVEC_SYMBOL, Q2_B64_FUSED_MATVEC, Q2_B64_RECOVERED_ROW_SYMBOL,
    Q4_B64_A8_MATVEC_SYMBOL, Q4_B64_FUSED_MATVEC, Q4_B64_RECOVERED_ROW_SYMBOL,
};
use super::{Activation, FusedMatVec, RecoveredRow, ScaleSlice};
use crate::format::TensorDType;
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

    /// Quantizes the corrected activation once into symmetric Q8_B64 blocks.
    /// Callers may share this result across projections that consume the same
    /// input, such as Q/K/V or gate/up matrices.
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

fn a8_scale_bytes(columns: usize) -> Result<usize> {
    columns
        .checked_div(64)
        .and_then(|blocks| blocks.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| EngineError::Shape("CUDA A8 scale size overflows usize".into()))
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
        assert_owned::<PreparedCudaRecoveredRow>();
    }
}
