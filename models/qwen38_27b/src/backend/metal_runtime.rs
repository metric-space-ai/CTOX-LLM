//! Direct Metal execution for the unpromoted Q2/Q4 fused-matvec candidate.
//!
//! This module deliberately does not implement [`super::Backend`]. It exists
//! to generate same-device verifier and benchmark evidence while the public
//! Metal backend remains fail-closed at `PromotionState::Contract`.

use std::ffi::c_void;
use std::mem::size_of_val;
use std::slice;

use metal_driver::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::metal::{
    validate_operation, MetalBufferAbi, MAX_SIMDGROUPS_PER_THREADGROUP, Q2_KERNEL_NAME,
    Q4_KERNEL_NAME,
};
use super::{FusedMatVec, ScaleSlice};
use crate::format::TensorDType;
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
        if input.len() != self.columns {
            return Err(EngineError::Shape(format!(
                "Metal prepared input has {} values, expected {}",
                input.len(),
                self.columns
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidArtifact(
                "Metal prepared input contains a non-finite value".into(),
            ));
        }
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

    /// Dispatches an already resident projection. Command encoding and
    /// completion remain synchronous so verifier and benchmark callers obtain
    /// an unambiguous interval and completed output.
    pub fn dispatch_prepared(&self, prepared: &PreparedMetalMatVec) -> Result<Vec<f32>> {
        self.dispatch_prepared_repeated(prepared, 1)
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
    use crate::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
    use half::f16;

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
}
