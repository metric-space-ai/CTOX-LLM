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
use std::slice;

use libloading::Library;

use super::cuda::{validate_operation, Q2_B64_FUSED_MATVEC, Q4_B64_FUSED_MATVEC};
use super::{Activation, FusedMatVec, ScaleSlice};
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
    driver: CudaDriver,
    context: CuContext,
    module: CuModule,
    q2_function: CuFunction,
    q4_function: CuFunction,
    device_name: String,
    compute_capability: (u32, u32),
}

/// Device-resident buffers for one pure Q2 or Q4 projection. Immutable model
/// and recovery buffers remain allocated across repeated token dispatches.
pub struct PreparedCudaMatVec<'runtime> {
    runtime: &'runtime CudaCandidateRuntime,
    dtype: TensorDType,
    rows: u32,
    columns: u32,
    activation: u32,
    weights: DeviceBuffer<'runtime>,
    input: DeviceBuffer<'runtime>,
    s_in: Option<DeviceBuffer<'runtime>>,
    s_out: Option<DeviceBuffer<'runtime>>,
    bias: Option<DeviceBuffer<'runtime>>,
    output: DeviceBuffer<'runtime>,
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
        Ok(Self {
            driver,
            context,
            module,
            q2_function,
            q4_function,
            device_name,
            compute_capability,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    pub fn dispatch_fused_matvec(&self, operation: &FusedMatVec<'_>) -> Result<Vec<f32>> {
        let prepared = self.prepare_fused_matvec(operation)?;
        self.dispatch_prepared(&prepared)
    }

    pub fn prepare_fused_matvec<'runtime>(
        &'runtime self,
        operation: &FusedMatVec<'_>,
    ) -> Result<PreparedCudaMatVec<'runtime>> {
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
            runtime: self,
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

    pub fn dispatch_prepared(&self, prepared: &PreparedCudaMatVec<'_>) -> Result<Vec<f32>> {
        self.dispatch_prepared_repeated(prepared, 1)
    }

    /// Launches a resident operation repeatedly and synchronizes once. This
    /// amortizes host launch/copy overhead for per-op roofline measurement;
    /// production graph capture remains a separate promotion requirement.
    pub fn dispatch_prepared_repeated(
        &self,
        prepared: &PreparedCudaMatVec<'_>,
        dispatches: usize,
    ) -> Result<Vec<f32>> {
        if dispatches == 0 {
            return Err(EngineError::Shape(
                "CUDA repeated dispatch count must be positive".into(),
            ));
        }
        if !ptr::eq(self, prepared.runtime) {
            return Err(EngineError::InvalidState(
                "prepared CUDA operation belongs to another context".into(),
            ));
        }
        self.make_current()?;
        let function = match prepared.dtype {
            TensorDType::Q2B64 => self.q2_function,
            TensorDType::Q4B64 => self.q4_function,
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
                self.driver.check(
                    (self.driver.launch_kernel)(
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
            self.driver
                .check((self.driver.ctx_synchronize)(), "context synchronization")?;
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
        unsafe {
            self.driver.check(
                (self.driver.ctx_set_current)(self.context),
                "set current context",
            )
        }
    }
}

impl PreparedCudaMatVec<'_> {
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

impl Drop for CudaCandidateRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.driver.ctx_set_current)(self.context);
            let _ = (self.driver.module_unload)(self.module);
            let _ = (self.driver.ctx_destroy)(self.context);
        }
    }
}

struct DeviceBuffer<'runtime> {
    runtime: &'runtime CudaCandidateRuntime,
    ptr: CuDevicePtr,
    len: usize,
}

impl<'runtime> DeviceBuffer<'runtime> {
    fn allocate(runtime: &'runtime CudaCandidateRuntime, len: usize) -> Result<Self> {
        if len == 0 {
            return Err(EngineError::Shape(
                "CUDA device allocation must be non-empty".into(),
            ));
        }
        runtime.make_current()?;
        let mut ptr = 0;
        unsafe {
            runtime.driver.check(
                (runtime.driver.mem_alloc)(&mut ptr, len),
                "device allocation",
            )?;
        }
        Ok(Self { runtime, ptr, len })
    }

    fn from_bytes(runtime: &'runtime CudaCandidateRuntime, bytes: &[u8]) -> Result<Self> {
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
        self.runtime.make_current()?;
        unsafe {
            self.runtime.driver.check(
                (self.runtime.driver.memcpy_htod)(self.ptr, bytes.as_ptr().cast(), bytes.len()),
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
        self.runtime.make_current()?;
        unsafe {
            self.runtime.driver.check(
                (self.runtime.driver.memcpy_dtoh)(bytes.as_mut_ptr().cast(), self.ptr, bytes.len()),
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

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        let _ = self.runtime.make_current();
        unsafe {
            let _ = (self.runtime.driver.mem_free)(self.ptr);
        }
    }
}

fn optional_scale_buffer<'runtime>(
    runtime: &'runtime CudaCandidateRuntime,
    scales: Option<ScaleSlice<'_>>,
) -> Result<Option<DeviceBuffer<'runtime>>> {
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
    fn driver_parameter_widths_match_cuda_abi() {
        assert_eq!(std::mem::size_of::<CuDevicePtr>(), 8);
        assert_eq!(std::mem::size_of::<u32>(), 4);
    }
}
