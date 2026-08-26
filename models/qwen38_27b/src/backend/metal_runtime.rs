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
    validate_mixed_operation, validate_operation, MetalBufferAbi, MetalFusedMatVecParams,
    MAX_SIMDGROUPS_PER_THREADGROUP, Q2_KERNEL_NAME, Q4_KERNEL_NAME,
};
use super::{FusedMatVec, ScaleSlice};
use crate::format::TensorDType;
use crate::loader::ModelArtifact;
use crate::{EngineError, Result};

const KERNEL_SOURCE: &str = include_str!("../../kernels/metal/q2q4_fused_matvec.metal");
const MAX_THREADS_PER_GROUP: usize = MAX_SIMDGROUPS_PER_THREADGROUP * 32;
const DEFAULT_SIMDGROUPS: usize = 2;
const ROWS_PER_SIMDGROUP: usize = 4;

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
/// offsets into one shared no-copy CTOXQ mapping. Only input, bias, output,
/// and the small parameter block allocate separate Metal storage.
pub struct PreparedMappedMetalMatVec {
    dtype: TensorDType,
    rows: usize,
    columns: usize,
    s_in_offset: u64,
    dispatches: Vec<MappedMetalDispatch>,
    mapping: MappedMetalArtifact,
    input_buffer: Buffer,
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
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            q2_pipeline,
            q4_pipeline,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
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
        self.prepare_mapped_fused_matvec_with_simdgroups(mapping, operation, DEFAULT_SIMDGROUPS)
    }

    pub fn prepare_mapped_fused_matvec_with_simdgroups(
        &self,
        mapping: &MappedMetalArtifact,
        operation: &FusedMatVec<'_>,
        simdgroups: usize,
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
        let input_buffer = buffer_with_data(&self.device, as_bytes(operation.input));
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
        let transient_bytes = size_of_val(operation.input)
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
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-mmap-q2q4-verifier");
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
}
