//! Direct Metal execution for the unpromoted Q2/Q4 fused-matvec candidate.
//!
//! This module deliberately does not implement [`super::Backend`]. It exists
//! to generate same-device verifier and benchmark evidence while the public
//! Metal backend remains fail-closed at `PromotionState::Contract`.

use std::ffi::c_void;
use std::mem::size_of_val;
use std::slice;

use metal_driver::{
    CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use super::metal::{
    validate_operation, MetalBufferAbi, Q2_KERNEL_NAME, Q4_KERNEL_NAME, REDUCTION_SCRATCH_FLOATS,
};
use super::{FusedMatVec, ScaleSlice};
use crate::format::TensorDType;
use crate::{EngineError, Result};

const KERNEL_SOURCE: &str = include_str!("../../kernels/metal/q2q4_fused_matvec.metal");
const MAX_THREADS_PER_GROUP: usize = REDUCTION_SCRATCH_FLOATS * 32;

/// An explicitly verifier-only Metal runtime owning compiled MSL pipelines.
///
/// Creating this object compiles the in-crate source through the native Metal
/// driver. Dispatches allocate shared buffers for the small verifier inputs;
/// production graph residency and zero-copy weight ownership are separate
/// promotion requirements and are not claimed by this type.
pub struct MetalCandidateRuntime {
    device: Device,
    queue: CommandQueue,
    q2_pipeline: ComputePipelineState,
    q4_pipeline: ComputePipelineState,
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
        let (layout, params) = validate_operation(operation)?;
        let pipeline = match layout.dtype {
            TensorDType::Q2B64 => &self.q2_pipeline,
            TensorDType::Q4B64 => &self.q4_pipeline,
            _ => unreachable!("Metal validation accepts only Q2/Q4"),
        };
        let thread_width = dispatch_width(pipeline)?;
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

        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("ctox-qwen38-q2q4-verifier");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(MetalBufferAbi::WEIGHTS as u64, Some(&weights_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::INPUT as u64, Some(&input_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::S_IN as u64, Some(&s_in_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::S_OUT as u64, Some(&s_out_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::BIAS as u64, Some(&bias_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::OUTPUT as u64, Some(&output_buffer), 0);
        encoder.set_buffer(MetalBufferAbi::PARAMS as u64, Some(&params_buffer), 0);
        encoder.set_threadgroup_memory_length(
            MetalBufferAbi::REDUCTION_SCRATCH as u64,
            (REDUCTION_SCRATCH_FLOATS * std::mem::size_of::<f32>()) as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize {
                width: operation.rows as u64,
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
                "Metal candidate command ended with {:?}",
                command_buffer.status()
            )));
        }

        // StorageModeShared makes the completed result coherently visible to
        // the CPU on Apple Silicon. Copy before the Metal buffer is dropped.
        let output = unsafe {
            slice::from_raw_parts(output_buffer.contents().cast::<f32>(), operation.rows).to_vec()
        };
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidState(
                "Metal candidate produced a non-finite output".into(),
            ));
        }
        Ok(output)
    }
}

fn dispatch_width(pipeline: &ComputePipelineState) -> Result<usize> {
    let execution_width = pipeline.thread_execution_width() as usize;
    let maximum = pipeline.max_total_threads_per_threadgroup() as usize;
    if execution_width == 0 || maximum < execution_width {
        return Err(EngineError::InvalidState(
            "Metal pipeline reports an invalid execution width".into(),
        ));
    }
    let capped = maximum.min(MAX_THREADS_PER_GROUP);
    let width = capped / execution_width * execution_width;
    if width == 0 || width.div_ceil(32) > REDUCTION_SCRATCH_FLOATS {
        return Err(EngineError::InvalidState(
            "Metal pipeline cannot satisfy the reduction scratch ABI".into(),
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
            let actual = runtime
                .dispatch_fused_matvec(&operation)
                .expect("Metal candidate dispatch");
            for (row, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
                let absolute = (expected - actual).abs();
                let tolerance = 2.0e-4_f32.max(expected.abs() * 3.0e-5);
                assert!(
                    absolute <= tolerance,
                    "{dtype:?} row {row}: expected {expected}, got {actual}, absolute error {absolute}, tolerance {tolerance}"
                );
            }
        }
    }
}
