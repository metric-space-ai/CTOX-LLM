use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::backend::{Activation, FusedMatVec, ScaleSlice};
use ctox_qwen38_27b::format::TensorDType;
use ctox_qwen38_27b::quant::{A8Block64, Q2Block64, BLOCK_LEN};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify batched fused Qwen attention-gate and SwiGLU CUDA chunk edges")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 512)]
    tokens: usize,
}

#[derive(Debug, Serialize)]
struct EdgeReport {
    columns: usize,
    values: usize,
    a8_blocks: usize,
    code_sha256: String,
    scale_sha256: String,
    code_mismatches_vs_cpu: usize,
    maximum_scale_absolute_error_vs_cpu: f32,
    selected_sequential_rows: Vec<usize>,
    code_mismatches_vs_sequential_cuda: usize,
    maximum_scale_absolute_delta_vs_sequential_cuda: f32,
    zero_weight_projection_verified: bool,
}

#[derive(Debug, Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    workspace_bytes: usize,
    output_arena_bytes: usize,
    swiglu: EdgeReport,
    attention_gate: EdgeReport,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_peak_bytes: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

struct OperationFixture {
    weights: Vec<u8>,
    input: Vec<f32>,
    s_in: Vec<u8>,
    s_out: Vec<u8>,
    columns: usize,
}

impl OperationFixture {
    fn view(&self) -> FusedMatVec<'_> {
        FusedMatVec {
            dtype: TensorDType::Q2B64,
            weights: &self.weights,
            segments: &[],
            rows: 1,
            columns: self.columns,
            input: &self.input,
            s_in: Some(ScaleSlice::F16Le(&self.s_in)),
            s_out: Some(ScaleSlice::F16Le(&self.s_out)),
            bias: None,
            activation: Activation::Identity,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.tokens > 1, "tokens must exceed one");
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before, _) = runtime.memory_info()?;

    let swiglu_columns = 17_408;
    let attention_columns = 6_144;
    let workspace = runtime.prepare_batched_a8_workspace(args.tokens, swiglu_columns)?;
    let outputs = runtime.prepare_batched_a8_output_arena(args.tokens, [1, 1, 1, 1])?;

    let swiglu_gate = values(args.tokens, swiglu_columns, 0.017, 2.4, 3.0, true);
    let swiglu_up = values(args.tokens, swiglu_columns, 0.013, 1.7, 11.0, false);
    let swiglu_scales = scales(swiglu_columns, 29, 0.82, 0.011);
    let swiglu_fixture = operation(swiglu_columns, &swiglu_scales)?;
    let swiglu_operation = swiglu_fixture.view();
    let swiglu_activation = runtime.prepare_shared_a8_activation(&swiglu_operation)?;
    let swiglu_projection = runtime.prepare_shared_a8_projection(&swiglu_operation)?;
    let swiglu_gate_device = runtime.prepare_verifier_f32_tensor(&swiglu_gate)?;
    let swiglu_up_device = runtime.prepare_verifier_f32_tensor(&swiglu_up)?;
    let swiglu_output = runtime
        .dispatch_batched_a8_arena_swiglu_fanout_device(
            &swiglu_activation,
            &workspace,
            &outputs,
            swiglu_gate_device.device_view()?,
            swiglu_up_device.device_view()?,
            args.tokens,
            &[(&swiglu_projection, 0)],
        )?
        .into_iter()
        .next()
        .context("batched SwiGLU fan-out returned no output")?;
    let swiglu_zero = runtime
        .verifier_read_f32(swiglu_output)?
        .into_iter()
        .all(|value| value == 0.0);
    let (swiglu_codes, swiglu_quant_scales) =
        workspace.verifier_read_quantized(args.tokens, swiglu_columns)?;
    let swiglu_expected = swiglu_gate
        .iter()
        .zip(&swiglu_up)
        .zip(expanded_f16(&swiglu_scales).iter().cycle())
        .map(|((gate, up), scale)| gate / (1.0 + (-gate).exp()) * up * scale)
        .collect::<Vec<_>>();
    let swiglu_sequential = selected_sequential(
        &runtime,
        &swiglu_activation,
        swiglu_gate_device.device_view()?,
        swiglu_up_device.device_view()?,
        args.tokens,
        swiglu_columns,
        true,
    )?;
    let swiglu = edge_report(
        swiglu_columns,
        &swiglu_codes,
        &swiglu_quant_scales,
        &swiglu_expected,
        swiglu_sequential,
        swiglu_zero,
    )?;

    let attention = values(args.tokens, attention_columns, 0.019, 1.8, 5.0, true);
    let attention_gate = values(args.tokens, attention_columns, 0.011, 2.1, 13.0, false);
    let attention_scales = scales(attention_columns, 23, 0.84, 0.012);
    let attention_fixture = operation(attention_columns, &attention_scales)?;
    let attention_operation = attention_fixture.view();
    let attention_activation = runtime.prepare_shared_a8_activation(&attention_operation)?;
    let attention_projection = runtime.prepare_shared_a8_projection(&attention_operation)?;
    let attention_device = runtime.prepare_verifier_f32_tensor(&attention)?;
    let attention_gate_device = runtime.prepare_verifier_f32_tensor(&attention_gate)?;
    let free_after_prepare = runtime.memory_info()?.0;
    let attention_output = runtime
        .dispatch_batched_a8_arena_sigmoid_gate_fanout_device(
            &attention_activation,
            &workspace,
            &outputs,
            attention_device.device_view()?,
            attention_gate_device.device_view()?,
            args.tokens,
            &[(&attention_projection, 0)],
        )?
        .into_iter()
        .next()
        .context("batched attention-gate fan-out returned no output")?;
    let attention_zero = runtime
        .verifier_read_f32(attention_output)?
        .into_iter()
        .all(|value| value == 0.0);
    let (attention_codes, attention_quant_scales) =
        workspace.verifier_read_quantized(args.tokens, attention_columns)?;
    let attention_expected = attention
        .iter()
        .zip(&attention_gate)
        .zip(expanded_f16(&attention_scales).iter().cycle())
        .map(|((attention, gate), scale)| attention / (1.0 + (-gate).exp()) * scale)
        .collect::<Vec<_>>();
    let attention_gate = edge_report(
        attention_columns,
        &attention_codes,
        &attention_quant_scales,
        &attention_expected,
        selected_sequential(
            &runtime,
            &attention_activation,
            attention_device.device_view()?,
            attention_gate_device.device_view()?,
            args.tokens,
            attention_columns,
            false,
        )?,
        attention_zero,
    )?;

    let workspace_bytes = workspace.transient_bytes();
    let output_arena_bytes = outputs.transient_bytes();
    drop(workspace);
    drop(outputs);
    drop(swiglu_activation);
    drop(swiglu_projection);
    drop(swiglu_gate_device);
    drop(swiglu_up_device);
    drop(attention_activation);
    drop(attention_projection);
    drop(attention_device);
    drop(attention_gate_device);
    let free_after = runtime.memory_info()?.0;
    let observed_peak_bytes = free_before.saturating_sub(free_after_prepare);
    let observed_reclaimed_bytes = free_after.saturating_sub(free_after_prepare);

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-batched-fused-a8-verifier.v1",
            status: "pass",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            workspace_bytes,
            output_arena_bytes,
            swiglu,
            attention_gate,
            driver_free_bytes_before_prepare: free_before,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after,
            observed_peak_bytes,
            observed_reclaimed_bytes,
            note: "Verifier-only adaptation of the pinned fused A8 candidates. Both 512-row edges use grid.y and graph-owned arenas; production promotion still requires complete-schedule wiring and roofline evidence.",
        })?
    );
    Ok(())
}

fn operation(columns: usize, scale_values: &[f32]) -> Result<OperationFixture> {
    let weights = Q2Block64::quantize(&vec![0.0; BLOCK_LEN])?
        .encode()
        .repeat(columns / BLOCK_LEN);
    Ok(OperationFixture {
        weights,
        input: vec![0.0; columns],
        s_in: f16_bytes(scale_values),
        s_out: f16_bytes(&[1.0]),
        columns,
    })
}

fn values(
    tokens: usize,
    columns: usize,
    frequency: f32,
    amplitude: f32,
    phase: f32,
    sine: bool,
) -> Vec<f32> {
    (0..tokens * columns)
        .map(|index| {
            let angle = (index as f32 + phase) * frequency;
            (if sine { angle.sin() } else { angle.cos() }) * amplitude
        })
        .collect()
}

fn scales(columns: usize, period: usize, base: f32, step: f32) -> Vec<f32> {
    (0..columns)
        .map(|index| base + (index % period) as f32 * step)
        .collect()
}

fn expanded_f16(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect()
}

type SequentialRows = (Vec<usize>, Vec<(Vec<i8>, Vec<f32>)>);

#[allow(clippy::too_many_arguments)]
fn selected_sequential(
    runtime: &CudaCandidateRuntime,
    activation: &ctox_qwen38_27b::backend::cuda_runtime::PreparedCudaA8Activation,
    left: ctox_qwen38_27b::backend::cuda_runtime::CudaDeviceF32View<'_>,
    right: ctox_qwen38_27b::backend::cuda_runtime::CudaDeviceF32View<'_>,
    tokens: usize,
    columns: usize,
    swiglu: bool,
) -> Result<SequentialRows> {
    let mut rows = vec![0, 1, tokens / 2, tokens - 1];
    rows.sort_unstable();
    rows.dedup();
    let mut results = Vec::with_capacity(rows.len());
    for row in &rows {
        let offset = row * columns;
        if swiglu {
            runtime.quantize_shared_a8_swiglu_device(
                activation,
                left.slice(offset, columns)?,
                right.slice(offset, columns)?,
            )?;
        } else {
            runtime.quantize_shared_a8_sigmoid_gate_device(
                activation,
                left.slice(offset, columns)?,
                right.slice(offset, columns)?,
            )?;
        }
        results.push(activation.verifier_read_quantized()?);
    }
    Ok((rows, results))
}

fn edge_report(
    columns: usize,
    codes: &[i8],
    scales: &[f32],
    expected_values: &[f32],
    sequential: SequentialRows,
    zero_weight_projection_verified: bool,
) -> Result<EdgeReport> {
    let expected = expected_values
        .chunks_exact(BLOCK_LEN)
        .map(A8Block64::quantize)
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(expected.len() == scales.len(), "A8 scale count differs");
    let mut code_mismatches_vs_cpu = 0;
    let mut maximum_scale_absolute_error_vs_cpu = 0.0_f32;
    for (block, (expected, actual_scale)) in expected.iter().zip(scales).enumerate() {
        maximum_scale_absolute_error_vs_cpu =
            maximum_scale_absolute_error_vs_cpu.max((expected.scale - actual_scale).abs());
        code_mismatches_vs_cpu += expected
            .codes
            .iter()
            .zip(&codes[block * BLOCK_LEN..(block + 1) * BLOCK_LEN])
            .filter(|(expected, actual)| expected != actual)
            .count();
    }
    let (selected_sequential_rows, sequential_values) = sequential;
    let mut code_mismatches_vs_sequential_cuda = 0;
    let mut maximum_scale_absolute_delta_vs_sequential_cuda = 0.0_f32;
    for (row, (sequential_codes, sequential_scales)) in
        selected_sequential_rows.iter().zip(&sequential_values)
    {
        let code_start = row * columns;
        code_mismatches_vs_sequential_cuda += sequential_codes
            .iter()
            .zip(&codes[code_start..code_start + columns])
            .filter(|(expected, actual)| expected != actual)
            .count();
        let scale_start = row * columns / BLOCK_LEN;
        for (expected, actual) in sequential_scales
            .iter()
            .zip(&scales[scale_start..scale_start + columns / BLOCK_LEN])
        {
            maximum_scale_absolute_delta_vs_sequential_cuda =
                maximum_scale_absolute_delta_vs_sequential_cuda.max((expected - actual).abs());
        }
    }
    let cpu_code_tolerance = codes.len().div_ceil(10_000);
    anyhow::ensure!(
        code_mismatches_vs_cpu <= cpu_code_tolerance,
        "batched A8 has {code_mismatches_vs_cpu} code mismatches against CPU, tolerance is {cpu_code_tolerance}"
    );
    anyhow::ensure!(
        maximum_scale_absolute_error_vs_cpu <= 2.0e-7,
        "batched A8 scale error exceeds tolerance"
    );
    anyhow::ensure!(
        code_mismatches_vs_sequential_cuda == 0
            && maximum_scale_absolute_delta_vs_sequential_cuda == 0.0,
        "batched A8 output differs from selected sequential CUDA rows"
    );
    anyhow::ensure!(
        zero_weight_projection_verified,
        "zero-weight batched projection produced a non-zero value"
    );
    Ok(EdgeReport {
        columns,
        values: codes.len(),
        a8_blocks: scales.len(),
        code_sha256: hash_i8(codes),
        scale_sha256: format!("{:x}", Sha256::digest(as_bytes(scales))),
        code_mismatches_vs_cpu,
        maximum_scale_absolute_error_vs_cpu,
        selected_sequential_rows,
        code_mismatches_vs_sequential_cuda,
        maximum_scale_absolute_delta_vs_sequential_cuda,
        zero_weight_projection_verified,
    })
}

fn hash_i8(values: &[i8]) -> String {
    format!("{:x}", Sha256::digest(as_bytes(values)))
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    // SAFETY: the caller hashes initialized plain numeric vectors only.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}
