use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use ctox_qwen38_27b::backend::cpu::CpuBackend;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaRmsNormConfig};
use ctox_qwen38_27b::backend::{Activation, Backend, FusedMatVec, ScaleSlice};
use ctox_qwen38_27b::format::{QuantSegment, TensorDType};
use ctox_qwen38_27b::quant::{A8Block64, Q2Block64, Q4Block64, BLOCK_LEN};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DTypeArg {
    Q2,
    Q4,
    Mixed,
}

#[derive(Debug, Parser)]
#[command(about = "Verify one resident batched Q2/Q4 CUDA prefill projection")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, value_enum, default_value_t = DTypeArg::Mixed)]
    dtype: DTypeArg,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 257)]
    rows: usize,
    #[arg(long, default_value_t = 512)]
    columns: usize,
    #[arg(long, default_value_t = 17)]
    batch_rows: usize,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, default_value_t = 2.0e-2)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 1.0e-3)]
    relative_tolerance: f32,
    #[arg(long)]
    skip_cpu_oracle: bool,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    dtype: &'static str,
    rows: usize,
    columns: usize,
    batch_rows: usize,
    cpu_oracle_executed: bool,
    resident_bytes: usize,
    resident_graph_workspace_bytes: usize,
    resident_arena_workspace_bytes: usize,
    resident_norm_workspace_bytes: usize,
    baseline_mean_batch_milliseconds: f64,
    mmq_mean_batch_milliseconds: f64,
    mmq_speedup: f64,
    mmq_prompt_rows_per_second: f64,
    baseline_maximum_absolute_error: Option<f32>,
    baseline_maximum_relative_error: Option<f32>,
    mmq_maximum_absolute_error: Option<f32>,
    mmq_maximum_relative_error: Option<f32>,
    baseline_mmq_maximum_absolute_delta: f32,
    baseline_mmq_maximum_relative_delta: f32,
    mmq_graph_maximum_absolute_delta: f32,
    mmq_graph_maximum_relative_delta: f32,
    mmq_arena_maximum_absolute_delta: f32,
    mmq_arena_maximum_relative_delta: f32,
    norm_workspace_maximum_absolute_delta: f32,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.rows > 1, "rows must exceed one");
    anyhow::ensure!(
        args.columns > 0 && args.columns.is_multiple_of(BLOCK_LEN),
        "columns must be a positive multiple of {BLOCK_LEN}"
    );
    anyhow::ensure!(args.batch_rows > 0, "batch-rows must be positive");
    anyhow::ensure!(args.iterations > 0, "iterations must be positive");
    let dtype = match args.dtype {
        DTypeArg::Q2 => TensorDType::Q2B64,
        DTypeArg::Q4 => TensorDType::Q4B64,
        DTypeArg::Mixed => TensorDType::MixedQ2Q4B64,
    };
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let (weights, segments) = packed_weights_and_segments(dtype, args.rows, args.columns)?;
    let inputs: Vec<f32> = (0..args.batch_rows * args.columns)
        .map(|index| {
            let row = index / args.columns;
            let column = index % args.columns;
            ((column as f32 * 0.031_25) + row as f32 * 0.173).cos()
        })
        .collect();
    let s_in = f16_bytes(
        &(0..args.columns)
            .map(|index| 0.9 + 0.001 * (index % 11) as f32)
            .collect::<Vec<_>>(),
    );
    let s_out = f16_bytes(
        &(0..args.rows)
            .map(|row| 1.1 - 0.002 * (row % 9) as f32)
            .collect::<Vec<_>>(),
    );
    let bias: Vec<f32> = (0..args.rows)
        .map(|row| (row as f32 * 0.77).sin() * 0.05)
        .collect();
    let first_input = &inputs[..args.columns];
    let operation = FusedMatVec {
        dtype,
        weights: &weights,
        segments: &segments,
        rows: args.rows,
        columns: args.columns,
        input: first_input,
        s_in: Some(ScaleSlice::F16Le(&s_in)),
        s_out: Some(ScaleSlice::F16Le(&s_out)),
        bias: Some(&bias),
        activation: Activation::Silu,
    };
    let oracle = if args.skip_cpu_oracle {
        None
    } else {
        let mut values = Vec::with_capacity(args.batch_rows * args.rows);
        for input in inputs.chunks_exact(args.columns) {
            values.extend(a8_execution_oracle(&operation, input)?);
        }
        Some(values)
    };

    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let prepared = runtime.prepare_batched_a8_matmul(&operation, &inputs, args.batch_rows)?;
    let baseline_output = runtime.dispatch_batched_a8_matmul(&prepared)?;
    let (baseline_maximum_absolute_error, baseline_maximum_relative_error) = oracle
        .as_ref()
        .map(|oracle| {
            compare(
                oracle,
                &baseline_output,
                args.absolute_tolerance,
                args.relative_tolerance,
            )
        })
        .transpose()?
        .map_or((None, None), |(absolute, relative)| {
            (Some(absolute), Some(relative))
        });
    let mmq_output = runtime.dispatch_batched_a8_mmq(&prepared)?;
    let (mmq_maximum_absolute_error, mmq_maximum_relative_error) = oracle
        .as_ref()
        .map(|oracle| {
            compare(
                oracle,
                &mmq_output,
                args.absolute_tolerance,
                args.relative_tolerance,
            )
        })
        .transpose()?
        .map_or((None, None), |(absolute, relative)| {
            (Some(absolute), Some(relative))
        });
    let (baseline_mmq_maximum_absolute_delta, baseline_mmq_maximum_relative_delta) = compare(
        &baseline_output,
        &mmq_output,
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    let graph_input = runtime.prepare_verifier_f32_tensor(&inputs)?;
    let graph_activation =
        runtime.prepare_batched_shared_a8_activation(&operation, args.batch_rows)?;
    let graph_projection = runtime.prepare_shared_a8_projection(&operation)?;
    let graph_output = runtime.prepare_batched_a8_output(&graph_projection, args.batch_rows)?;
    let graph_view = runtime
        .dispatch_batched_shared_a8_fanout_device(
            &graph_activation,
            graph_input.device_view()?,
            args.batch_rows,
            &[(&graph_projection, &graph_output)],
        )?
        .into_iter()
        .next()
        .context("batched CUDA graph fan-out returned no output")?;
    let graph_values = runtime.verifier_read_f32(graph_view)?;
    let (mmq_graph_maximum_absolute_delta, mmq_graph_maximum_relative_delta) = compare(
        &mmq_output,
        &graph_values,
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    let resident_graph_workspace_bytes = graph_activation
        .resident_bytes()
        .checked_add(graph_output.resident_bytes())
        .context("CUDA graph workspace residency overflows")?;
    let arena_activation = runtime.prepare_shared_a8_activation(&operation)?;
    let arena_workspace =
        runtime.prepare_batched_a8_workspace(args.batch_rows, args.columns * 2)?;
    let arena_outputs =
        runtime.prepare_batched_a8_output_arena(args.batch_rows, [args.rows, 1, 1, 1])?;
    let arena_view = runtime
        .dispatch_batched_a8_arena_fanout_device(
            &arena_activation,
            &arena_workspace,
            &arena_outputs,
            graph_input.device_view()?,
            args.batch_rows,
            &[(&graph_projection, 0)],
        )?
        .into_iter()
        .next()
        .context("batched CUDA arena fan-out returned no output")?;
    let arena_values = runtime.verifier_read_f32(arena_view)?;
    let (mmq_arena_maximum_absolute_delta, mmq_arena_maximum_relative_delta) = compare(
        &mmq_output,
        &arena_values,
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    let resident_arena_workspace_bytes = arena_workspace
        .transient_bytes()
        .checked_add(arena_outputs.transient_bytes())
        .context("CUDA arena workspace residency overflows")?;
    let norm_weight = f16_bytes(
        &(0..args.columns)
            .map(|column| 0.01 * (column % 7) as f32)
            .collect::<Vec<_>>(),
    );
    let norm = runtime.prepare_qwen_rms_norm_f16(
        CudaRmsNormConfig {
            rows: args.batch_rows,
            columns: args.columns,
            epsilon: 1.0e-6,
        },
        &norm_weight,
    )?;
    let norm_baseline = runtime.verifier_read_f32(
        runtime.dispatch_qwen_rms_norm_f16_device(&norm, graph_input.device_view()?)?,
    )?;
    let norm_workspace =
        runtime.prepare_batched_rms_norm_workspace(args.batch_rows, args.columns)?;
    let norm_graph =
        runtime.verifier_read_f32(runtime.dispatch_batched_qwen_rms_norm_f16_device(
            &norm,
            &norm_workspace,
            graph_input.device_view()?,
            args.batch_rows,
        )?)?;
    let (norm_workspace_maximum_absolute_delta, _) = compare(
        &norm_baseline,
        &norm_graph,
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    for _ in 0..args.warmup {
        std::hint::black_box(runtime.dispatch_batched_a8_matmul(&prepared)?);
        std::hint::black_box(runtime.dispatch_batched_a8_mmq(&prepared)?);
    }
    let baseline_started = Instant::now();
    for _ in 0..args.iterations {
        std::hint::black_box(runtime.dispatch_batched_a8_matmul(&prepared)?);
    }
    let baseline_elapsed = baseline_started.elapsed().as_secs_f64();
    let mmq_started = Instant::now();
    for _ in 0..args.iterations {
        std::hint::black_box(runtime.dispatch_batched_a8_mmq(&prepared)?);
    }
    let mmq_elapsed = mmq_started.elapsed().as_secs_f64();
    let baseline_mean_batch_seconds = baseline_elapsed / args.iterations as f64;
    let mmq_mean_batch_seconds = mmq_elapsed / args.iterations as f64;
    let (major, minor) = runtime.compute_capability();
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-batched-prefill-verifier.v1",
            status: "verifier_only_not_promotion_evidence",
            device: runtime.device_name(),
            compute_capability: format!("{major}.{minor}"),
            module_sha256,
            dtype: match dtype {
                TensorDType::Q2B64 => "q2_b64",
                TensorDType::Q4B64 => "q4_b64",
                TensorDType::MixedQ2Q4B64 => "mixed_q2_q4_b64",
                _ => unreachable!(),
            },
            rows: prepared.rows(),
            columns: prepared.columns(),
            batch_rows: prepared.batch_rows(),
            cpu_oracle_executed: !args.skip_cpu_oracle,
            resident_bytes: prepared.resident_bytes(),
            resident_graph_workspace_bytes,
            resident_arena_workspace_bytes,
            resident_norm_workspace_bytes: norm_workspace.transient_bytes(),
            baseline_mean_batch_milliseconds: baseline_mean_batch_seconds * 1_000.0,
            mmq_mean_batch_milliseconds: mmq_mean_batch_seconds * 1_000.0,
            mmq_speedup: baseline_mean_batch_seconds / mmq_mean_batch_seconds,
            mmq_prompt_rows_per_second: args.batch_rows as f64 / mmq_mean_batch_seconds,
            baseline_maximum_absolute_error,
            baseline_maximum_relative_error,
            mmq_maximum_absolute_error,
            mmq_maximum_relative_error,
            baseline_mmq_maximum_absolute_delta,
            baseline_mmq_maximum_relative_delta,
            mmq_graph_maximum_absolute_delta,
            mmq_graph_maximum_relative_delta,
            mmq_arena_maximum_absolute_delta,
            mmq_arena_maximum_relative_delta,
            norm_workspace_maximum_absolute_delta,
            note: "Verifier-only comparison of the 2-D dp4a baseline, standalone SM86 MMQ tile, and the device-resident graph workspace path. The graph path shares immutable projection weights and keeps activation/output transients separate. Production promotion still requires complete layer-major chunk scheduling, recurrent/attention scans, representative Qwen shapes, and stable roofline evidence.",
        })?
    );
    Ok(())
}

fn compare(
    expected: &[f32],
    actual: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> anyhow::Result<(f32, f32)> {
    anyhow::ensure!(expected.len() == actual.len(), "output lengths differ");
    let mut maximum_absolute_error = 0.0_f32;
    let mut maximum_relative_error = 0.0_f32;
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let absolute = (expected - actual).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        anyhow::ensure!(
            absolute <= absolute_tolerance + relative_tolerance * expected.abs(),
            "batched CUDA result {actual} at {index} differs from A8 oracle {expected} by {absolute}"
        );
    }
    Ok((maximum_absolute_error, maximum_relative_error))
}

fn a8_execution_oracle(operation: &FusedMatVec<'_>, input: &[f32]) -> anyhow::Result<Vec<f32>> {
    let mut corrected = Vec::with_capacity(operation.columns);
    for (index, value) in input.iter().enumerate() {
        corrected.push(
            *value
                * operation
                    .s_in
                    .map(|scales| scales.value(index))
                    .transpose()?
                    .unwrap_or(1.0),
        );
    }
    let mut dequantized = Vec::with_capacity(operation.columns);
    for block in corrected.chunks_exact(BLOCK_LEN) {
        dequantized.extend_from_slice(&A8Block64::quantize(block)?.dequantize());
    }
    Ok(CpuBackend::scalar_verifier().fused_matvec(&FusedMatVec {
        dtype: operation.dtype,
        weights: operation.weights,
        segments: operation.segments,
        rows: operation.rows,
        columns: operation.columns,
        input: &dequantized,
        s_in: None,
        s_out: operation.s_out,
        bias: operation.bias,
        activation: operation.activation,
    })?)
}

fn packed_weights_and_segments(
    dtype: TensorDType,
    rows: usize,
    columns: usize,
) -> anyhow::Result<(Vec<u8>, Vec<QuantSegment>)> {
    if dtype != TensorDType::MixedQ2Q4B64 {
        return Ok((packed_weights(dtype, rows, columns)?, Vec::new()));
    }
    let q2_rows = rows.div_ceil(2);
    let q4_rows = rows - q2_rows;
    let q2 = packed_weights(TensorDType::Q2B64, q2_rows, columns)?;
    let q4 = packed_weights(TensorDType::Q4B64, q4_rows, columns)?;
    let mut weights = Vec::with_capacity(q2.len() + q4.len());
    weights.extend_from_slice(&q2);
    weights.extend_from_slice(&q4);
    Ok((
        weights,
        vec![
            QuantSegment {
                group_index: 0,
                row_start: 0,
                row_end: q2_rows as u64,
                dtype: TensorDType::Q2B64,
                offset: 0,
                length: q2.len() as u64,
            },
            QuantSegment {
                group_index: 1,
                row_start: q2_rows as u64,
                row_end: rows as u64,
                dtype: TensorDType::Q4B64,
                offset: q2.len() as u64,
                length: q4.len() as u64,
            },
        ],
    ))
}

fn packed_weights(dtype: TensorDType, rows: usize, columns: usize) -> anyhow::Result<Vec<u8>> {
    let block_count = rows
        .checked_mul(columns / BLOCK_LEN)
        .ok_or_else(|| anyhow::anyhow!("matrix block count overflows"))?;
    let mut weights = Vec::new();
    for matrix_block in 0..block_count {
        let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
            let phase = (matrix_block * BLOCK_LEN + index) as f32;
            (phase * 0.013_579).sin() * 0.25
        });
        match dtype {
            TensorDType::Q2B64 => {
                weights.extend_from_slice(&Q2Block64::quantize(&values)?.encode())
            }
            TensorDType::Q4B64 => {
                weights.extend_from_slice(&Q4Block64::quantize(&values)?.encode())
            }
            _ => unreachable!(),
        }
    }
    Ok(weights)
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}
