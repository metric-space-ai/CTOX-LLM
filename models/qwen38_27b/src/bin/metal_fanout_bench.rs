#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use clap::Parser;
    use ctox_qwen38_27b::backend::cpu::CpuBackend;
    use ctox_qwen38_27b::backend::metal_runtime::MetalCandidateRuntime;
    use ctox_qwen38_27b::backend::{Activation, Backend, FusedMatVec, ScaleSlice};
    use ctox_qwen38_27b::format::TensorDType;
    use ctox_qwen38_27b::quant::{Q2Block64, Q4Block64, BLOCK_LEN};
    use half::f16;
    use serde::Serialize;

    #[derive(Debug, Parser)]
    #[command(about = "Verify and benchmark one shared Metal Q/K/V fan-out")]
    struct Args {
        #[arg(long, default_value_t = 5_120)]
        columns: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 2)]
        simdgroups: usize,
    }

    struct Fixture {
        name: &'static str,
        dtype: TensorDType,
        rows: usize,
        weights: Vec<u8>,
        s_out: Vec<u8>,
        bias: Vec<f32>,
    }

    impl Fixture {
        fn operation<'a>(&'a self, input: &'a [f32], s_in: &'a [u8]) -> FusedMatVec<'a> {
            FusedMatVec {
                dtype: self.dtype,
                weights: &self.weights,
                segments: &[],
                rows: self.rows,
                columns: input.len(),
                input,
                s_in: Some(ScaleSlice::F16Le(s_in)),
                s_out: Some(ScaleSlice::F16Le(&self.s_out)),
                bias: Some(&self.bias),
                activation: Activation::Identity,
            }
        }
    }

    #[derive(Serialize)]
    struct ProjectionReport {
        name: &'static str,
        dtype: &'static str,
        rows: usize,
        packed_weight_bytes: usize,
        maximum_absolute_error: f32,
    }

    #[derive(Serialize)]
    struct Report<'a> {
        format: &'static str,
        status: &'static str,
        device: &'a str,
        columns: usize,
        simdgroups: usize,
        fanout_projections: usize,
        warmup: usize,
        iterations: usize,
        packed_weight_bytes_per_pass: usize,
        isolated_requested_resident_bytes: usize,
        shared_requested_resident_bytes: usize,
        requested_resident_bytes_saved: usize,
        mismatched_s_in_rejected: bool,
        isolated_elapsed_milliseconds: f64,
        shared_elapsed_milliseconds: f64,
        isolated_mean_pass_milliseconds: f64,
        shared_mean_pass_milliseconds: f64,
        command_buffer_fanout_speedup: f64,
        shared_packed_weight_gb_per_second: f64,
        projections: Vec<ProjectionReport>,
        note: &'static str,
    }

    pub fn run() -> anyhow::Result<()> {
        let args = Args::parse();
        anyhow::ensure!(
            args.columns > 0 && args.columns.is_multiple_of(BLOCK_LEN),
            "columns must be a positive multiple of {BLOCK_LEN}"
        );
        anyhow::ensure!(args.iterations > 0, "iterations must be positive");
        let input: Vec<f32> = (0..args.columns)
            .map(|index| (index as f32 * 0.031_25).cos())
            .collect();
        let s_in = f16_bytes(
            &(0..args.columns)
                .map(|index| 0.9 + 0.001 * (index % 11) as f32)
                .collect::<Vec<_>>(),
        );
        let fixtures = [
            fixture("q_proj", TensorDType::Q4B64, 12_288, args.columns, 11)?,
            fixture("k_proj", TensorDType::Q2B64, 1_024, args.columns, 23)?,
            fixture("v_proj", TensorDType::Q2B64, 1_024, args.columns, 37)?,
        ];
        let operations: Vec<_> = fixtures
            .iter()
            .map(|fixture| fixture.operation(&input, &s_in))
            .collect();
        let oracles = operations
            .iter()
            .map(|operation| CpuBackend::scalar_verifier().fused_matvec(operation))
            .collect::<Result<Vec<_>, _>>()?;

        let runtime = MetalCandidateRuntime::new()?;
        let isolated = operations
            .iter()
            .map(|operation| {
                runtime.prepare_fused_matvec_with_simdgroups(operation, args.simdgroups)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let activation = runtime.prepare_shared_activation(&operations[0])?;
        let projections = operations
            .iter()
            .map(|operation| {
                runtime.prepare_shared_projection_with_simdgroups(operation, args.simdgroups)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection_refs: Vec<_> = projections.iter().collect();
        let output = runtime.dispatch_shared_fanout(&activation, &projection_refs)?;
        let projections_report = fixtures
            .iter()
            .zip(oracles.iter().zip(&output))
            .map(|(fixture, (expected, actual))| {
                let maximum_absolute_error = expected
                    .iter()
                    .zip(actual)
                    .map(|(expected, actual)| (expected - actual).abs())
                    .fold(0.0_f32, f32::max);
                anyhow::ensure!(
                    maximum_absolute_error <= 2.0e-3,
                    "Metal {} differs from scalar by {maximum_absolute_error}",
                    fixture.name
                );
                Ok(ProjectionReport {
                    name: fixture.name,
                    dtype: dtype_name(fixture.dtype),
                    rows: fixture.rows,
                    packed_weight_bytes: fixture.weights.len(),
                    maximum_absolute_error,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let mut mismatched_s_in = s_in.clone();
        mismatched_s_in[..2].copy_from_slice(&f16::from_f32(1.25).to_bits().to_le_bytes());
        let mismatched_operation = fixtures[0].operation(&input, &mismatched_s_in);
        let mismatched = runtime
            .prepare_shared_projection_with_simdgroups(&mismatched_operation, args.simdgroups)?;
        let mismatched_s_in_rejected = runtime
            .dispatch_shared_fanout(&activation, &[&mismatched])
            .is_err();
        anyhow::ensure!(
            mismatched_s_in_rejected,
            "Metal fan-out accepted other s_in"
        );

        for _ in 0..args.warmup {
            for prepared in &isolated {
                std::hint::black_box(runtime.dispatch_prepared(prepared)?);
            }
            std::hint::black_box(runtime.dispatch_shared_fanout(&activation, &projection_refs)?);
        }
        let isolated_started = Instant::now();
        for _ in 0..args.iterations {
            for prepared in &isolated {
                std::hint::black_box(runtime.dispatch_prepared(prepared)?);
            }
        }
        let isolated_elapsed = isolated_started.elapsed().as_secs_f64();
        let shared_started = Instant::now();
        for _ in 0..args.iterations {
            std::hint::black_box(runtime.dispatch_shared_fanout(&activation, &projection_refs)?);
        }
        let shared_elapsed = shared_started.elapsed().as_secs_f64();
        let packed_weight_bytes_per_pass = fixtures
            .iter()
            .map(|fixture| fixture.weights.len())
            .sum::<usize>();
        let isolated_requested_resident_bytes = isolated
            .iter()
            .map(|prepared| prepared.resident_bytes())
            .sum();
        let shared_requested_resident_bytes = activation.resident_bytes()
            + projections
                .iter()
                .map(|projection| projection.resident_bytes())
                .sum::<usize>();
        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                format: "ctox.metal-shared-fanout-benchmark.v1",
                status: "verifier_only_not_promotion_evidence",
                device: runtime.device_name(),
                columns: args.columns,
                simdgroups: args.simdgroups,
                fanout_projections: fixtures.len(),
                warmup: args.warmup,
                iterations: args.iterations,
                packed_weight_bytes_per_pass,
                isolated_requested_resident_bytes,
                shared_requested_resident_bytes,
                requested_resident_bytes_saved: isolated_requested_resident_bytes
                    .saturating_sub(shared_requested_resident_bytes),
                mismatched_s_in_rejected,
                isolated_elapsed_milliseconds: isolated_elapsed * 1_000.0,
                shared_elapsed_milliseconds: shared_elapsed * 1_000.0,
                isolated_mean_pass_milliseconds: isolated_elapsed * 1_000.0
                    / args.iterations as f64,
                shared_mean_pass_milliseconds: shared_elapsed * 1_000.0
                    / args.iterations as f64,
                command_buffer_fanout_speedup: isolated_elapsed / shared_elapsed,
                shared_packed_weight_gb_per_second: packed_weight_bytes_per_pass as f64
                    * args.iterations as f64
                    / shared_elapsed
                    / 1.0e9,
                projections: projections_report,
                note: "Exact Qwen Q/K/V shapes with one input/s_in allocation and one command buffer per fan-out. StorageModeShared output copies, host dispatch, uncontrolled clocks, and synthetic weights are included; packed-byte GB/s is not hardware-counter roofline evidence.",
            })?
        );
        Ok(())
    }

    fn fixture(
        name: &'static str,
        dtype: TensorDType,
        rows: usize,
        columns: usize,
        phase_offset: usize,
    ) -> anyhow::Result<Fixture> {
        let weights = packed_weights(dtype, rows, columns, phase_offset)?;
        Ok(Fixture {
            name,
            dtype,
            rows,
            weights,
            s_out: f16_bytes(
                &(0..rows)
                    .map(|row| 1.1 - 0.002 * (row % 9) as f32)
                    .collect::<Vec<_>>(),
            ),
            bias: (0..rows)
                .map(|row| ((row + phase_offset) as f32 * 0.77).sin() * 0.05)
                .collect(),
        })
    }

    fn packed_weights(
        dtype: TensorDType,
        rows: usize,
        columns: usize,
        phase_offset: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let block_count = rows
            .checked_mul(columns / BLOCK_LEN)
            .ok_or_else(|| anyhow::anyhow!("matrix block count overflows"))?;
        let block_bytes = match dtype {
            TensorDType::Q2B64 => 18,
            TensorDType::Q4B64 => 34,
            _ => anyhow::bail!("fan-out fixture supports only Q2/Q4"),
        };
        let mut weights = Vec::with_capacity(block_count * block_bytes);
        for matrix_block in 0..block_count {
            let values: [f32; BLOCK_LEN] = std::array::from_fn(|index| {
                let phase = (phase_offset + matrix_block * BLOCK_LEN + index) as f32;
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

    fn dtype_name(dtype: TensorDType) -> &'static str {
        match dtype {
            TensorDType::Q2B64 => "q2_b64",
            TensorDType::Q4B64 => "q4_b64",
            _ => unreachable!(),
        }
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("qwen38-metal-fanout-bench requires macOS")
}
