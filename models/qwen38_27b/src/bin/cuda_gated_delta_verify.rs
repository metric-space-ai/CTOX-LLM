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
    preparation_model_bytes: usize,
    preparation_transient_bytes: usize,
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
    maximum_preparation_absolute_error: f32,
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
    let compact_heads = 16;
    let compact_values = compact_heads * config.key_dim;
    let value_values = config.heads * config.value_dim;
    let a_log: Vec<f32> = (0..config.heads)
        .map(|head| -2.2 + (head % 9) as f32 * 0.035)
        .collect();
    let dt_bias: Vec<f32> = (0..config.heads)
        .map(|head| -0.8 + (head % 7) as f32 * 0.07)
        .collect();
    let (free_before_prepare, _) = runtime.memory_info()?;
    let mut prepared = runtime.prepare_gated_delta_f16(config)?;
    let mut preparation = runtime.prepare_gated_delta_inputs_f32(&a_log, &dt_bias)?;
    let convolved_staging =
        runtime.prepare_verifier_f32_tensor(&vec![0.0; compact_values * 2 + value_values])?;
    let raw_a_staging = runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads])?;
    let raw_b_staging = runtime.prepare_verifier_f32_tensor(&vec![0.0; config.heads])?;
    let (free_after_prepare, _) = runtime.memory_info()?;
    let mut oracle_state = vec![f16::ZERO; config.heads * config.key_dim * config.value_dim];
    let mut maximum_output_absolute_error = 0.0_f32;
    let mut maximum_output_relative_error = 0.0_f32;
    let mut maximum_state_absolute_error = 0.0_f32;
    let mut maximum_preparation_absolute_error = 0.0_f32;

    for token in 0..args.tokens {
        let compact_query: Vec<f32> = (0..compact_values)
            .map(|index| ((index + token * 5) as f32 * 0.023).sin() * 0.4)
            .collect();
        let compact_key: Vec<f32> = (0..compact_values)
            .map(|index| ((index + token * 7) as f32 * 0.019).cos() * 0.35)
            .collect();
        let value: Vec<f32> = (0..value_values)
            .map(|index| ((index + token * 11) as f32 * 0.017).sin() * 0.5)
            .collect();
        let raw_a: Vec<f32> = (0..config.heads)
            .map(|head| -1.1 + head as f32 * 0.013 + token as f32 * 0.004)
            .collect();
        let raw_b: Vec<f32> = (0..config.heads)
            .map(|head| -0.5 + (head % 11) as f32 * 0.09 - token as f32 * 0.003)
            .collect();
        let query = repeat_heads(&compact_query, compact_heads, config.key_dim, 3);
        let key = repeat_heads(&compact_key, compact_heads, config.key_dim, 3);
        let log_decay: Vec<f32> = raw_a
            .iter()
            .zip(&a_log)
            .zip(&dt_bias)
            .map(|((a, a_log), dt_bias)| -a_log.exp() * softplus(a + dt_bias))
            .collect();
        let beta: Vec<f32> = raw_b
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
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
        let mut convolved_qkv = Vec::with_capacity(compact_values * 2 + value_values);
        convolved_qkv.extend_from_slice(&compact_query);
        convolved_qkv.extend_from_slice(&compact_key);
        convolved_qkv.extend_from_slice(&value);
        convolved_staging.write(&convolved_qkv)?;
        raw_a_staging.write(&raw_a)?;
        raw_b_staging.write(&raw_b)?;
        let prepared_views = runtime.dispatch_gated_delta_inputs_device(
            &mut preparation,
            convolved_staging.device_view()?,
            raw_a_staging.device_view()?,
            raw_b_staging.device_view()?,
        )?;
        for (name, expected_values, actual_view) in [
            ("query", query.as_slice(), prepared_views.query),
            ("key", key.as_slice(), prepared_views.key),
            ("log_decay", log_decay.as_slice(), prepared_views.log_decay),
            ("beta", beta.as_slice(), prepared_views.beta),
        ] {
            let actual_values = runtime.verifier_read_f32(actual_view)?;
            for (index, (expected_value, actual_value)) in
                expected_values.iter().zip(&actual_values).enumerate()
            {
                let absolute = (expected_value - actual_value).abs();
                maximum_preparation_absolute_error =
                    maximum_preparation_absolute_error.max(absolute);
                anyhow::ensure!(
                    absolute <= 3.0e-6,
                    "token {token} prepared {name} {index}: expected {expected_value}, got {actual_value}"
                );
            }
        }
        let value_view = convolved_staging
            .device_view()?
            .slice(compact_values * 2, value_values)?;
        let actual_view = runtime.dispatch_gated_delta_f16_device(
            &mut prepared,
            prepared_views.query,
            prepared_views.key,
            value_view,
            prepared_views.log_decay,
            prepared_views.beta,
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
    let preparation_model_bytes = preparation.model_bytes();
    let preparation_transient_bytes = preparation.transient_bytes();
    let verifier_device_staging_bytes = convolved_staging.resident_bytes()
        + raw_a_staging.resident_bytes()
        + raw_b_staging.resident_bytes();
    drop(prepared);
    drop(preparation);
    drop(convolved_staging);
    drop(raw_a_staging);
    drop(raw_b_staging);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_allocation_bytes = free_before_prepare.saturating_sub(free_after_prepare);
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes
            >= persistent_state_bytes
                + transient_buffer_bytes
                + preparation_model_bytes
                + preparation_transient_bytes
                + verifier_device_staging_bytes,
        "dropping the prepared recurrence did not reclaim its requested CUDA buffers"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-gated-delta-f16-verifier.v3",
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
            preparation_model_bytes,
            preparation_transient_bytes,
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
            maximum_preparation_absolute_error,
            reset_zero_state_verified,
            note: "Verifier-only candidate; compact 16-head Q/K, V, and raw A/B enter as producer-owned CUDA device views. Q/K repetition plus A/B-to-decay/beta transforms stay on device and feed the recurrence directly. Readback is restricted to verification; no CPU fallback and not promoted into the production CUDA ABI.",
        })?
    );
    Ok(())
}

fn repeat_heads(values: &[f32], heads: usize, dimension: usize, repeats: usize) -> Vec<f32> {
    assert_eq!(values.len(), heads * dimension);
    let mut output = Vec::with_capacity(values.len() * repeats);
    for head in values.chunks_exact(dimension) {
        for _ in 0..repeats {
            output.extend_from_slice(head);
        }
    }
    output
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}
