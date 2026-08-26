use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use ctox_qwen38_27b::backend::cpu::CpuBackend;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::backend::{Backend, RecoveredRow, ScaleSlice};
use ctox_qwen38_27b::format::TensorDType;
use ctox_qwen38_27b::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DTypeArg {
    Q2,
    Q4,
}

#[derive(Debug, Parser)]
#[command(about = "Verify and benchmark a recovered Q2/Q4 CUDA embedding row")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, value_enum, default_value_t = DTypeArg::Q2)]
    dtype: DTypeArg,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 5_120)]
    columns: usize,
    #[arg(long, default_value_t = 5)]
    warmup: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 20)]
    dispatches_per_sync: usize,
    #[arg(long, default_value_t = 2.0e-5)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 2.0e-5)]
    relative_tolerance: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    dtype: &'static str,
    columns: usize,
    warmup: usize,
    iterations: usize,
    dispatches_per_sync: usize,
    total_dispatches: usize,
    packed_weight_bytes: usize,
    requested_resident_buffer_bytes: usize,
    elapsed_milliseconds: f64,
    mean_dispatch_milliseconds: f64,
    maximum_absolute_error: f32,
    maximum_relative_error: f32,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.columns > 0 && args.columns.is_multiple_of(BLOCK_LEN),
        "columns must be positive and divisible by 64"
    );
    anyhow::ensure!(args.iterations > 0, "iterations must be positive");
    anyhow::ensure!(
        args.dispatches_per_sync > 0,
        "dispatches-per-sync must be positive"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let dtype = match args.dtype {
        DTypeArg::Q2 => TensorDType::Q2B64,
        DTypeArg::Q4 => TensorDType::Q4B64,
    };
    let weights = packed_row(dtype, args.columns)?;
    let s_in: Vec<u8> = (0..args.columns)
        .flat_map(|index| {
            f16::from_f32(0.91 + (index % 13) as f32 * 0.007)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let operation = RecoveredRow {
        dtype,
        weights: &weights,
        columns: args.columns,
        s_in: ScaleSlice::F16Le(&s_in),
        s_out: 0.873,
    };
    let oracle = CpuBackend::scalar_verifier().recovered_row(&operation)?;
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let prepared = runtime.prepare_recovered_row(&operation)?;
    let device_output = runtime.dispatch_recovered_row(&prepared)?;
    let mut maximum_absolute_error = 0.0_f32;
    let mut maximum_relative_error = 0.0_f32;
    for (expected, actual) in oracle.iter().zip(&device_output) {
        let absolute = (expected - actual).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        maximum_absolute_error = maximum_absolute_error.max(absolute);
        maximum_relative_error = maximum_relative_error.max(relative);
        anyhow::ensure!(
            absolute <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
            "CUDA recovered-row value {actual} differs from oracle {expected} by {absolute}"
        );
    }

    for _ in 0..args.warmup {
        std::hint::black_box(
            runtime
                .dispatch_prepared_recovered_row_repeated(&prepared, args.dispatches_per_sync)?,
        );
    }
    let started = Instant::now();
    for _ in 0..args.iterations {
        std::hint::black_box(
            runtime
                .dispatch_prepared_recovered_row_repeated(&prepared, args.dispatches_per_sync)?,
        );
    }
    let elapsed = started.elapsed().as_secs_f64();
    let total_dispatches = args
        .iterations
        .checked_mul(args.dispatches_per_sync)
        .context("dispatch count overflows")?;
    let mean_seconds = elapsed / total_dispatches as f64;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-recovered-row-benchmark.v1",
            status: "verifier_only_not_promotion_evidence",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            dtype: match dtype {
                TensorDType::Q2B64 => "q2_b64",
                TensorDType::Q4B64 => "q4_b64",
                _ => unreachable!(),
            },
            columns: args.columns,
            warmup: args.warmup,
            iterations: args.iterations,
            dispatches_per_sync: args.dispatches_per_sync,
            total_dispatches,
            packed_weight_bytes: weights.len(),
            requested_resident_buffer_bytes: prepared.resident_bytes(),
            elapsed_milliseconds: elapsed * 1.0e3,
            mean_dispatch_milliseconds: mean_seconds * 1.0e3,
            maximum_absolute_error,
            maximum_relative_error,
            note: "Synchronous verifier interval with one resident packed row. Repetition amortizes synchronization and output-copy overhead; embedding rows are latency-bound and no bandwidth roofline is claimed.",
        })?
    );
    Ok(())
}

fn packed_row(dtype: TensorDType, columns: usize) -> anyhow::Result<Vec<u8>> {
    let mut packed = Vec::new();
    for block in 0..columns / BLOCK_LEN {
        let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
            let linear = block * BLOCK_LEN + index;
            (linear as f32 * 0.013).sin() * 0.7 + (linear as f32 * 0.003).cos() * 0.3
        });
        match dtype {
            TensorDType::Q2B64 => packed.extend_from_slice(&Q2Block64::quantize(&values)?.encode()),
            TensorDType::Q4B64 => packed.extend_from_slice(&Q4Block64::quantize(&values)?.encode()),
            _ => unreachable!(),
        }
    }
    Ok(packed)
}
