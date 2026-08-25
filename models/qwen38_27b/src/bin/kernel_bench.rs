use std::time::Instant;

use clap::{Parser, ValueEnum};
use ctox_qwen38_27b::backend::cpu::CpuBackend;
use ctox_qwen38_27b::backend::{Activation, Backend, ExecutionPolicy, FusedMatVec};
use ctox_qwen38_27b::format::TensorDType;
use ctox_qwen38_27b::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
use serde::Serialize;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DTypeArg {
    Q2,
    Q4,
}

#[derive(Debug, Parser)]
#[command(about = "Verify and benchmark Q2/Q4 fused CPU matvec")]
struct Args {
    #[arg(long, value_enum, default_value_t = DTypeArg::Q2)]
    dtype: DTypeArg,
    #[arg(long, default_value_t = 512)]
    rows: usize,
    #[arg(long, default_value_t = 512)]
    columns: usize,
    #[arg(long, default_value_t = 20)]
    iterations: usize,
}

#[derive(Serialize)]
struct Report {
    dtype: &'static str,
    rows: usize,
    columns: usize,
    iterations: usize,
    detected_profile: &'static str,
    scalar_milliseconds: f64,
    detected_milliseconds: f64,
    speedup: f64,
    maximum_absolute_error: f32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.rows > 0, "rows must be positive");
    anyhow::ensure!(
        args.columns > 0 && args.columns.is_multiple_of(BLOCK_LEN),
        "columns must be a positive multiple of {BLOCK_LEN}"
    );
    anyhow::ensure!(args.iterations > 0, "iterations must be positive");

    let dtype = match args.dtype {
        DTypeArg::Q2 => TensorDType::Q2B64,
        DTypeArg::Q4 => TensorDType::Q4B64,
    };
    let mut weights = Vec::new();
    for matrix_block in 0..args.rows * (args.columns / BLOCK_LEN) {
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
    let input: Vec<f32> = (0..args.columns)
        .map(|index| (index as f32 * 0.031_25).cos())
        .collect();
    let s_in = vec![1.0_f32; args.columns];
    let s_out = vec![1.0_f32; args.rows];
    let operation = FusedMatVec {
        dtype,
        weights: &weights,
        rows: args.rows,
        columns: args.columns,
        input: &input,
        s_in: Some(&s_in),
        s_out: Some(&s_out),
        bias: None,
        activation: Activation::Silu,
    };
    let scalar = CpuBackend::scalar_verifier();
    let detected = CpuBackend::detect(ExecutionPolicy::Production)?;
    let scalar_output = scalar.fused_matvec(&operation)?;
    let detected_output = detected.fused_matvec(&operation)?;
    let maximum_absolute_error = scalar_output
        .iter()
        .zip(&detected_output)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    anyhow::ensure!(
        maximum_absolute_error <= 1e-4,
        "detected kernel differs from scalar by {maximum_absolute_error}"
    );

    let scalar_start = Instant::now();
    for _ in 0..args.iterations {
        std::hint::black_box(scalar.fused_matvec(std::hint::black_box(&operation))?);
    }
    let scalar_elapsed = scalar_start.elapsed();
    let detected_start = Instant::now();
    for _ in 0..args.iterations {
        std::hint::black_box(detected.fused_matvec(std::hint::black_box(&operation))?);
    }
    let detected_elapsed = detected_start.elapsed();
    let scalar_milliseconds = scalar_elapsed.as_secs_f64() * 1000.0;
    let detected_milliseconds = detected_elapsed.as_secs_f64() * 1000.0;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            dtype: match dtype {
                TensorDType::Q2B64 => "q2_b64",
                TensorDType::Q4B64 => "q4_b64",
                _ => unreachable!(),
            },
            rows: args.rows,
            columns: args.columns,
            iterations: args.iterations,
            detected_profile: detected.profile(),
            scalar_milliseconds,
            detected_milliseconds,
            speedup: scalar_milliseconds / detected_milliseconds,
            maximum_absolute_error,
        })?
    );
    Ok(())
}
