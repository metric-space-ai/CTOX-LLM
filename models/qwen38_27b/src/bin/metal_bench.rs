#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use clap::{Parser, ValueEnum};
    use ctox_qwen38_27b::backend::cpu::CpuBackend;
    use ctox_qwen38_27b::backend::metal_runtime::MetalCandidateRuntime;
    use ctox_qwen38_27b::backend::{Activation, Backend, FusedMatVec, ScaleSlice};
    use ctox_qwen38_27b::format::TensorDType;
    use ctox_qwen38_27b::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
    use half::f16;
    use serde::Serialize;

    #[derive(Debug, Clone, Copy, ValueEnum)]
    enum DTypeArg {
        Q2,
        Q4,
    }

    #[derive(Debug, Parser)]
    #[command(about = "Verify and benchmark resident Q2/Q4 Metal matvec")]
    struct Args {
        #[arg(long, value_enum, default_value_t = DTypeArg::Q2)]
        dtype: DTypeArg,
        #[arg(long, default_value_t = 5_120)]
        rows: usize,
        #[arg(long, default_value_t = 5_120)]
        columns: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        #[arg(long, default_value_t = 20)]
        iterations: usize,
        #[arg(long, default_value_t = 2)]
        simdgroups: usize,
        /// Benchmark an already corrected activation vector. Production may
        /// fuse s_in into the preceding norm/scale stage in one command graph.
        #[arg(long)]
        pre_scaled_input: bool,
    }

    #[derive(Serialize)]
    struct Report<'a> {
        format: &'static str,
        status: &'static str,
        device: &'a str,
        dtype: &'static str,
        rows: usize,
        columns: usize,
        warmup: usize,
        iterations: usize,
        simdgroups: usize,
        pre_scaled_input: bool,
        packed_weight_bytes: usize,
        requested_resident_buffer_bytes: usize,
        elapsed_milliseconds: f64,
        mean_dispatch_milliseconds: f64,
        dispatches_per_second: f64,
        packed_weight_gb_per_second: f64,
        maximum_absolute_error: f32,
        note: &'static str,
    }

    pub fn run() -> anyhow::Result<()> {
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
        let weights = packed_weights(dtype, args.rows, args.columns)?;
        let mut input: Vec<f32> = (0..args.columns)
            .map(|index| (index as f32 * 0.031_25).cos())
            .collect();
        let s_in_values: Vec<f32> = (0..args.columns)
            .map(|index| 0.9 + 0.001 * (index % 11) as f32)
            .collect();
        let s_in = f16_bytes(&s_in_values);
        if args.pre_scaled_input {
            for (value, scale) in input.iter_mut().zip(&s_in_values) {
                *value *= f16::from_f32(*scale).to_f32();
            }
        }
        let s_out = f16_bytes(
            &(0..args.rows)
                .map(|row| 1.1 - 0.002 * (row % 9) as f32)
                .collect::<Vec<_>>(),
        );
        let bias: Vec<f32> = (0..args.rows)
            .map(|row| (row as f32 * 0.77).sin() * 0.05)
            .collect();
        let operation = FusedMatVec {
            dtype,
            weights: &weights,
            segments: &[],
            rows: args.rows,
            columns: args.columns,
            input: &input,
            s_in: (!args.pre_scaled_input).then_some(ScaleSlice::F16Le(&s_in)),
            s_out: Some(ScaleSlice::F16Le(&s_out)),
            bias: Some(&bias),
            activation: Activation::Silu,
        };

        let scalar_output = CpuBackend::scalar_verifier().fused_matvec(&operation)?;
        let runtime = MetalCandidateRuntime::new()?;
        let prepared = runtime.prepare_fused_matvec_with_simdgroups(&operation, args.simdgroups)?;
        let metal_output = runtime.dispatch_prepared(&prepared)?;
        let maximum_absolute_error = scalar_output
            .iter()
            .zip(&metal_output)
            .map(|(expected, actual)| (expected - actual).abs())
            .fold(0.0_f32, f32::max);
        anyhow::ensure!(
            maximum_absolute_error <= 2.0e-3,
            "Metal result differs from scalar by {maximum_absolute_error}"
        );
        for _ in 0..args.warmup {
            std::hint::black_box(runtime.dispatch_prepared(&prepared)?);
        }
        let started = Instant::now();
        for _ in 0..args.iterations {
            std::hint::black_box(runtime.dispatch_prepared(&prepared)?);
        }
        let elapsed = started.elapsed().as_secs_f64();
        let mean_seconds = elapsed / args.iterations as f64;
        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                format: "ctox.metal-candidate-benchmark.v1",
                status: "verifier_only_not_promotion_evidence",
                device: runtime.device_name(),
                dtype: match dtype {
                    TensorDType::Q2B64 => "q2_b64",
                    TensorDType::Q4B64 => "q4_b64",
                    _ => unreachable!(),
                },
                rows: args.rows,
                columns: args.columns,
                warmup: args.warmup,
                iterations: args.iterations,
                simdgroups: args.simdgroups,
                pre_scaled_input: args.pre_scaled_input,
                packed_weight_bytes: weights.len(),
                requested_resident_buffer_bytes: prepared.resident_bytes(),
                elapsed_milliseconds: elapsed * 1_000.0,
                mean_dispatch_milliseconds: mean_seconds * 1_000.0,
                dispatches_per_second: mean_seconds.recip(),
                packed_weight_gb_per_second: weights.len() as f64 / mean_seconds / 1.0e9,
                maximum_absolute_error,
                note: "Synchronous device-resident per-op interval; requested buffer bytes exclude allocator page rounding, and packed-weight GB/s is not a hardware-counter roofline measurement.",
            })?
        );
        Ok(())
    }

    fn packed_weights(dtype: TensorDType, rows: usize, columns: usize) -> anyhow::Result<Vec<u8>> {
        let block_count = rows
            .checked_mul(columns / BLOCK_LEN)
            .ok_or_else(|| anyhow::anyhow!("matrix block count overflows"))?;
        let block_bytes = match dtype {
            TensorDType::Q2B64 => 18,
            TensorDType::Q4B64 => 34,
            _ => unreachable!(),
        };
        let mut weights = Vec::with_capacity(
            block_count
                .checked_mul(block_bytes)
                .ok_or_else(|| anyhow::anyhow!("packed matrix size overflows"))?,
        );
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
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("qwen38-metal-bench requires macOS")
}
