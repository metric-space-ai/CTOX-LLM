use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaGatedDeltaConfig};
use ctox_qwen38_27b::reference::recurrent_gated_delta_step_f16_state;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify chunked CUDA GatedDelta against sequential decode")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 11)]
    tokens: usize,
    #[arg(long, default_value_t = 3.0e-4)]
    output_absolute_tolerance: f32,
    #[arg(long, default_value_t = 2.0e-4)]
    output_relative_tolerance: f32,
    #[arg(long, default_value_t = 5.0e-4)]
    state_absolute_tolerance: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    sequential_scan_output_exact: bool,
    sequential_scan_state_exact: bool,
    maximum_oracle_output_absolute_error: f32,
    maximum_oracle_output_relative_error: f32,
    maximum_oracle_state_absolute_error: f32,
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
    let (free_before_prepare, _) = runtime.memory_info()?;
    let config = CudaGatedDeltaConfig::QWEN38_27B;
    let qk_per_token = config.heads * config.key_dim;
    let value_per_token = config.heads * config.value_dim;
    let head_per_token = config.heads;

    let query: Vec<f32> = fixture(args.tokens * qk_per_token, 0.017, 0.41, 3);
    let key: Vec<f32> = fixture(args.tokens * qk_per_token, 0.019, 0.37, 11);
    let value: Vec<f32> = fixture(args.tokens * value_per_token, 0.013, 0.53, 19);
    let log_decay: Vec<f32> = (0..args.tokens * head_per_token)
        .map(|index| -0.015 - (index % 23) as f32 * 0.0013)
        .collect();
    let beta: Vec<f32> = (0..args.tokens * head_per_token)
        .map(|index| 0.18 + (index % 17) as f32 * 0.031)
        .collect();

    let mut oracle_state = vec![f16::ZERO; config.heads * config.key_dim * config.value_dim];
    let mut oracle_outputs = Vec::with_capacity(args.tokens * value_per_token);
    for token in 0..args.tokens {
        let qk_start = token * qk_per_token;
        let value_start = token * value_per_token;
        let head_start = token * head_per_token;
        oracle_outputs.extend(recurrent_gated_delta_step_f16_state(
            &query[qk_start..qk_start + qk_per_token],
            &key[qk_start..qk_start + qk_per_token],
            &value[value_start..value_start + value_per_token],
            &log_decay[head_start..head_start + head_per_token],
            &beta[head_start..head_start + head_per_token],
            &mut oracle_state,
            config.heads,
            config.key_dim,
            config.value_dim,
        )?);
    }

    let mut sequential = runtime.prepare_gated_delta_f16(config)?;
    let mut sequential_outputs = Vec::with_capacity(args.tokens * value_per_token);
    for token in 0..args.tokens {
        let qk_start = token * qk_per_token;
        let value_start = token * value_per_token;
        let head_start = token * head_per_token;
        sequential.write_step(
            &query[qk_start..qk_start + qk_per_token],
            &key[qk_start..qk_start + qk_per_token],
            &value[value_start..value_start + value_per_token],
            &log_decay[head_start..head_start + head_per_token],
            &beta[head_start..head_start + head_per_token],
        )?;
        sequential_outputs.extend(runtime.dispatch_gated_delta_f16(&mut sequential)?);
    }
    let sequential_state = sequential.verifier_read_state()?;

    let mut scan = runtime.prepare_gated_delta_f16(config)?;
    let query_staging = runtime.prepare_verifier_f32_tensor(&query)?;
    let key_staging = runtime.prepare_verifier_f32_tensor(&key)?;
    let value_staging = runtime.prepare_verifier_f32_tensor(&value)?;
    let decay_staging = runtime.prepare_verifier_f32_tensor(&log_decay)?;
    let beta_staging = runtime.prepare_verifier_f32_tensor(&beta)?;
    let scan_output = runtime.prepare_gated_delta_scan_output(config, args.tokens)?;
    let scan_view = runtime.dispatch_gated_delta_f16_scan_device(
        &mut scan,
        &scan_output,
        query_staging.device_view()?,
        key_staging.device_view()?,
        value_staging.device_view()?,
        decay_staging.device_view()?,
        beta_staging.device_view()?,
        args.tokens,
    )?;
    let scan_outputs = runtime.verifier_read_f32(scan_view)?;
    let scan_state = scan.verifier_read_state()?;

    let (maximum_oracle_output_absolute_error, maximum_oracle_output_relative_error) =
        compare(&oracle_outputs, &scan_outputs);
    for (index, (expected, actual)) in oracle_outputs.iter().zip(&scan_outputs).enumerate() {
        anyhow::ensure!(
            (expected - actual).abs()
                <= args.output_absolute_tolerance + args.output_relative_tolerance * expected.abs(),
            "scan CUDA output {index} differs from oracle"
        );
    }
    let maximum_oracle_state_absolute_error = oracle_state
        .iter()
        .zip(&scan_state)
        .map(|(expected, actual)| (expected.to_f32() - actual.to_f32()).abs())
        .fold(0.0_f32, f32::max);
    anyhow::ensure!(
        maximum_oracle_state_absolute_error <= args.state_absolute_tolerance,
        "scan CUDA final state differs from oracle"
    );
    let (maximum_sequential_scan_absolute_delta, _) = compare(&sequential_outputs, &scan_outputs);
    let sequential_scan_output_exact = sequential_outputs == scan_outputs;
    let sequential_scan_state_exact = sequential_state == scan_state;
    anyhow::ensure!(
        sequential_scan_output_exact && sequential_scan_state_exact,
        "CUDA GatedDelta scan differs from sequential output or final FP16 state"
    );

    let verifier_allocated_bytes = sequential.resident_state_bytes()
        + sequential.speculative_checkpoint_bytes()
        + sequential.transient_bytes()
        + scan.resident_state_bytes()
        + scan.speculative_checkpoint_bytes()
        + scan.transient_bytes()
        + query_staging.resident_bytes()
        + key_staging.resident_bytes()
        + value_staging.resident_bytes()
        + decay_staging.resident_bytes()
        + beta_staging.resident_bytes()
        + scan_output.transient_bytes();
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(sequential);
    drop(scan);
    drop(query_staging);
    drop(key_staging);
    drop(value_staging);
    drop(decay_staging);
    drop(beta_staging);
    drop(scan_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes >= verifier_allocated_bytes,
        "CUDA GatedDelta scan verifier did not reclaim all owned allocations"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-gated-delta-scan-verifier.v1",
            status: "pass_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            heads: config.heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            sequential_scan_output_exact,
            sequential_scan_state_exact,
            maximum_oracle_output_absolute_error,
            maximum_oracle_output_relative_error,
            maximum_oracle_state_absolute_error,
            maximum_sequential_scan_absolute_delta,
            verifier_allocated_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "The unpromoted upstream-structured scan advances one token-major prompt chunk in a single launch while preserving the decode FP16 recurrence contract. Promotion still requires representative chunk-size latency and integration with batched Qwen input preparation.",
        })?
    );
    Ok(())
}

fn fixture(values: usize, frequency: f32, scale: f32, offset: usize) -> Vec<f32> {
    (0..values)
        .map(|index| ((index + offset) as f32 * frequency).sin() * scale)
        .collect()
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
