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
#[command(about = "Verify the exact Qwen3.8-27B CUDA FP16 GatedDelta recurrence")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 6)]
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
    heads: usize,
    key_dim: usize,
    value_dim: usize,
    tokens: usize,
    persistent_state_bytes: usize,
    transient_buffer_bytes: usize,
    verifier_device_staging_bytes: usize,
    device_view_path_verified: bool,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_allocation_bytes: usize,
    observed_reclaimed_bytes: usize,
    maximum_output_absolute_error: f32,
    maximum_output_relative_error: f32,
    maximum_state_absolute_error: f32,
    reset_zero_state_verified: bool,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.tokens > 0, "tokens must be positive");
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let config = CudaGatedDeltaConfig::QWEN38_27B;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let mut prepared = runtime.prepare_gated_delta_f16(config)?;
    let query_staging =
        runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads * config.key_dim])?;
    let key_staging =
        runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads * config.key_dim])?;
    let value_staging =
        runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads * config.value_dim])?;
    let log_decay_staging = runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads])?;
    let beta_staging = runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads])?;
    let (free_after_prepare, _) = runtime.memory_info()?;
    let mut oracle_state = vec![f16::ZERO; config.heads * config.key_dim * config.value_dim];
    let mut maximum_output_absolute_error = 0.0_f32;
    let mut maximum_output_relative_error = 0.0_f32;
    let mut maximum_state_absolute_error = 0.0_f32;

    for token in 0..args.tokens {
        let query: Vec<f32> = (0..config.heads * config.key_dim)
            .map(|index| ((index + token * 5) as f32 * 0.023).sin() * 0.4)
            .collect();
        let key: Vec<f32> = (0..config.heads * config.key_dim)
            .map(|index| ((index + token * 7) as f32 * 0.019).cos() * 0.35)
            .collect();
        let value: Vec<f32> = (0..config.heads * config.value_dim)
            .map(|index| ((index + token * 11) as f32 * 0.017).sin() * 0.5)
            .collect();
        let log_decay: Vec<f32> = (0..config.heads)
            .map(|head| -0.012 - head as f32 * 0.0003 - token as f32 * 0.0002)
            .collect();
        let beta: Vec<f32> = (0..config.heads)
            .map(|head| 0.38 + (head % 11) as f32 * 0.025)
            .collect();
        let expected = recurrent_gated_delta_step_f16_state(
            &query,
            &key,
            &value,
            &log_decay,
            &beta,
            &mut oracle_state,
            config.heads,
            config.key_dim,
            config.value_dim,
        )?;
        query_staging.write(&query)?;
        key_staging.write(&key)?;
        value_staging.write(&value)?;
        log_decay_staging.write(&log_decay)?;
        beta_staging.write(&beta)?;
        let actual_view = runtime.dispatch_gated_delta_f16_device(
            &mut prepared,
            query_staging.device_view()?,
            key_staging.device_view()?,
            value_staging.device_view()?,
            log_decay_staging.device_view()?,
            beta_staging.device_view()?,
        )?;
        let actual = runtime.verifier_read_f32(actual_view)?;
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            let absolute = (expected - actual).abs();
            let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
            maximum_output_absolute_error = maximum_output_absolute_error.max(absolute);
            maximum_output_relative_error = maximum_output_relative_error.max(relative);
            anyhow::ensure!(
                absolute
                    <= args.output_absolute_tolerance
                        + args.output_relative_tolerance * expected.abs(),
                "token {token} output {index}: expected {expected}, got {actual}"
            );
        }
        let actual_state = prepared.verifier_read_state()?;
        for (index, (expected, actual)) in oracle_state.iter().zip(&actual_state).enumerate() {
            let absolute = (expected.to_f32() - actual.to_f32()).abs();
            maximum_state_absolute_error = maximum_state_absolute_error.max(absolute);
            anyhow::ensure!(
                absolute <= args.state_absolute_tolerance,
                "token {token} state {index}: expected {expected}, got {actual}"
            );
        }
    }

    prepared.reset()?;
    let reset_zero_state_verified = prepared
        .verifier_read_state()?
        .iter()
        .all(|value| *value == f16::ZERO);
    anyhow::ensure!(
        reset_zero_state_verified,
        "reset left non-zero recurrent state"
    );
    let persistent_state_bytes = prepared.resident_state_bytes();
    let transient_buffer_bytes = prepared.transient_bytes();
    let verifier_device_staging_bytes = query_staging.resident_bytes()
        + key_staging.resident_bytes()
        + value_staging.resident_bytes()
        + log_decay_staging.resident_bytes()
        + beta_staging.resident_bytes();
    drop(prepared);
    drop(query_staging);
    drop(key_staging);
    drop(value_staging);
    drop(log_decay_staging);
    drop(beta_staging);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_allocation_bytes = free_before_prepare.saturating_sub(free_after_prepare);
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes
            >= persistent_state_bytes + transient_buffer_bytes + verifier_device_staging_bytes,
        "dropping the prepared recurrence did not reclaim its requested CUDA buffers"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-gated-delta-f16-verifier.v2",
            status: "pass",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            heads: config.heads,
            key_dim: config.key_dim,
            value_dim: config.value_dim,
            tokens: args.tokens,
            persistent_state_bytes,
            transient_buffer_bytes,
            verifier_device_staging_bytes,
            device_view_path_verified: true,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes,
            observed_reclaimed_bytes,
            maximum_output_absolute_error,
            maximum_output_relative_error,
            maximum_state_absolute_error,
            reset_zero_state_verified,
            note: "Verifier-only candidate; Q/K/V, decay, and beta enter as producer-owned CUDA device views, with readback restricted to the verifier. No CPU fallback and not promoted into the production CUDA ABI.",
        })?
    );
    Ok(())
}
