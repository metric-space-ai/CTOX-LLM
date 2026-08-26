use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaCausalConvConfig};
use ctox_qwen38_27b::reference::causal_conv_silu_update_f16_state;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify chunked CUDA causal convolution against sequential decode")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 17)]
    tokens: usize,
    #[arg(long, default_value_t = 3.0e-5)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 5.0e-5)]
    relative_tolerance: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    channels: usize,
    kernel_width: usize,
    scan_kernel_registers: usize,
    scan_kernel_spill_bytes: usize,
    sequential_scan_output_exact: bool,
    sequential_scan_state_exact: bool,
    oracle_state_exact: bool,
    maximum_oracle_absolute_error: f32,
    maximum_oracle_relative_error: f32,
    maximum_sequential_scan_absolute_delta: f32,
    verifier_allocated_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.tokens > 0, "tokens must be positive");
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before, _) = runtime.memory_info()?;
    let config = CudaCausalConvConfig::QWEN38_27B;
    let (weights, weight_bytes) = f16_fixture(
        (0..config.channels * config.kernel_width)
            .map(|index| ((index + 1) as f32 * 0.031).cos() * 0.25),
    );
    let inputs: Vec<f32> = (0..args.tokens)
        .flat_map(|token| {
            (0..config.channels)
                .map(move |channel| ((channel + token * 7) as f32 * 0.031).sin() * 0.65)
        })
        .collect();
    let mut oracle_state = vec![f16::ZERO; config.channels * config.kernel_width];
    let mut oracle_outputs = Vec::with_capacity(inputs.len());
    for input in inputs.chunks_exact(config.channels) {
        oracle_outputs.extend(causal_conv_silu_update_f16_state(
            input,
            &mut oracle_state,
            &weights,
            config.channels,
            config.kernel_width,
        )?);
    }

    let mut sequential = runtime.prepare_causal_conv_f16(config, &weight_bytes)?;
    let sequential_input = runtime.prepare_verifier_f32_tensor(&vec![0.0; config.channels])?;
    let mut sequential_outputs = Vec::with_capacity(inputs.len());
    for input in inputs.chunks_exact(config.channels) {
        sequential_input.write(input)?;
        let view = runtime
            .dispatch_causal_conv_f16_device(&mut sequential, sequential_input.device_view()?)?;
        sequential_outputs.extend(runtime.verifier_read_f32(view)?);
    }
    let sequential_state = sequential.verifier_read_state()?;

    let mut scan = runtime.prepare_causal_conv_f16(config, &weight_bytes)?;
    let scan_input = runtime.prepare_verifier_f32_tensor(&inputs)?;
    let scan_output = runtime.prepare_causal_conv_scan_output(config, args.tokens)?;
    let scan_view = runtime.dispatch_causal_conv_f16_scan_device(
        &mut scan,
        &scan_output,
        scan_input.device_view()?,
        args.tokens,
    )?;
    let scan_outputs = runtime.verifier_read_f32(scan_view)?;
    let scan_state = scan.verifier_read_state()?;

    let (maximum_oracle_absolute_error, maximum_oracle_relative_error) =
        compare(&oracle_outputs, &sequential_outputs);
    for (index, (expected, actual)) in oracle_outputs.iter().zip(&sequential_outputs).enumerate() {
        anyhow::ensure!(
            (expected - actual).abs()
                <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
            "sequential CUDA output {index} differs from oracle"
        );
    }
    let (maximum_sequential_scan_absolute_delta, _) = compare(&sequential_outputs, &scan_outputs);
    let sequential_scan_output_exact = sequential_outputs == scan_outputs;
    let sequential_scan_state_exact = sequential_state == scan_state;
    let oracle_state_exact = scan_state == oracle_state;
    anyhow::ensure!(
        sequential_scan_output_exact && sequential_scan_state_exact && oracle_state_exact,
        "CUDA causal scan differs from sequential output or final FP16 state"
    );

    let verifier_allocated_bytes = sequential.model_bytes()
        + sequential.resident_state_bytes()
        + sequential.speculative_checkpoint_bytes()
        + sequential.transient_bytes()
        + sequential_input.resident_bytes()
        + scan.model_bytes()
        + scan.resident_state_bytes()
        + scan.speculative_checkpoint_bytes()
        + scan.transient_bytes()
        + scan_input.resident_bytes()
        + scan_output.transient_bytes();
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(sequential);
    drop(sequential_input);
    drop(scan);
    drop(scan_input);
    drop(scan_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes >= verifier_allocated_bytes,
        "CUDA causal scan verifier did not reclaim all owned allocations"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-causal-conv-scan-verifier.v1",
            status: "pass_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            channels: config.channels,
            kernel_width: config.kernel_width,
            scan_kernel_registers: 22,
            scan_kernel_spill_bytes: 0,
            sequential_scan_output_exact,
            sequential_scan_state_exact,
            oracle_state_exact,
            maximum_oracle_absolute_error,
            maximum_oracle_relative_error,
            maximum_sequential_scan_absolute_delta,
            verifier_allocated_bytes,
            driver_free_bytes_before_prepare: free_before,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "The unpromoted upstream-structured scan advances one token-major prompt chunk in a single launch while preserving the decode FP16 state contract. Promotion still requires representative chunk-size latency and integration with batched GatedDelta preparation/recurrence.",
        })?
    );
    Ok(())
}

fn f16_fixture(values: impl Iterator<Item = f32>) -> (Vec<f32>, Vec<u8>) {
    let packed: Vec<f16> = values.map(f16::from_f32).collect();
    let widened = packed.iter().map(|value| value.to_f32()).collect();
    let bytes = packed
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    (widened, bytes)
}

fn compare(expected: &[f32], actual: &[f32]) -> (f32, f32) {
    expected.iter().zip(actual).fold(
        (0.0_f32, 0.0_f32),
        |(max_absolute, max_relative), (left, right)| {
            let absolute = (left - right).abs();
            (
                max_absolute.max(absolute),
                max_relative.max(absolute / left.abs().max(f32::MIN_POSITIVE)),
            )
        },
    )
}
