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
    maximum_batched_preparation_absolute_error: f32,
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
    let compact_heads = 16;
    let compact_per_token = compact_heads * config.key_dim;
    let compact_query: Vec<f32> = fixture(args.tokens * compact_per_token, 0.017, 0.41, 3);
    let compact_key: Vec<f32> = fixture(args.tokens * compact_per_token, 0.019, 0.37, 11);
    let value: Vec<f32> = fixture(args.tokens * value_per_token, 0.013, 0.53, 19);
    let raw_a: Vec<f32> = (0..args.tokens * head_per_token)
        .map(|index| -1.15 + (index % 29) as f32 * 0.021)
        .collect();
    let raw_b: Vec<f32> = (0..args.tokens * head_per_token)
        .map(|index| -0.7 + (index % 17) as f32 * 0.063)
        .collect();
    let a_log: Vec<f32> = (0..head_per_token)
        .map(|head| -2.15 + (head % 9) as f32 * 0.037)
        .collect();
    let dt_bias: Vec<f32> = (0..head_per_token)
        .map(|head| -0.75 + (head % 7) as f32 * 0.071)
        .collect();
    let mut query = Vec::with_capacity(args.tokens * qk_per_token);
    let mut key = Vec::with_capacity(args.tokens * qk_per_token);
    let mut convolved_qkv = Vec::with_capacity(args.tokens * 10_240);
    for token in 0..args.tokens {
        let compact_start = token * compact_per_token;
        let value_start = token * value_per_token;
        let compact_query_token = &compact_query[compact_start..compact_start + compact_per_token];
        let compact_key_token = &compact_key[compact_start..compact_start + compact_per_token];
        query.extend(repeat_heads(
            compact_query_token,
            compact_heads,
            config.key_dim,
            3,
        ));
        key.extend(repeat_heads(
            compact_key_token,
            compact_heads,
            config.key_dim,
            3,
        ));
        convolved_qkv.extend_from_slice(compact_query_token);
        convolved_qkv.extend_from_slice(compact_key_token);
        convolved_qkv.extend_from_slice(&value[value_start..value_start + value_per_token]);
    }
    let log_decay: Vec<f32> = raw_a
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            let head = index % head_per_token;
            -a_log[head].exp() * softplus(raw + dt_bias[head])
        })
        .collect();
    let beta: Vec<f32> = raw_b.iter().map(|raw| 1.0 / (1.0 + (-raw).exp())).collect();

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
    let mut sequential_preparation = runtime.prepare_gated_delta_inputs_f32(&a_log, &dt_bias)?;
    let sequential_convolved = runtime.prepare_verifier_f32_tensor(&vec![0.0; 10_240])?;
    let sequential_raw_a = runtime.prepare_verifier_f32_tensor(&vec![0.0; head_per_token])?;
    let sequential_raw_b = runtime.prepare_verifier_f32_tensor(&vec![0.0; head_per_token])?;
    let mut sequential_outputs = Vec::with_capacity(args.tokens * value_per_token);
    for token in 0..args.tokens {
        let convolved_start = token * 10_240;
        let head_start = token * head_per_token;
        sequential_convolved.write(&convolved_qkv[convolved_start..convolved_start + 10_240])?;
        sequential_raw_a.write(&raw_a[head_start..head_start + head_per_token])?;
        sequential_raw_b.write(&raw_b[head_start..head_start + head_per_token])?;
        let prepared_views = runtime.dispatch_gated_delta_inputs_device(
            &mut sequential_preparation,
            sequential_convolved.device_view()?,
            sequential_raw_a.device_view()?,
            sequential_raw_b.device_view()?,
        )?;
        let value_view = sequential_convolved
            .device_view()?
            .slice(compact_per_token * 2, value_per_token)?;
        let actual_view = runtime.dispatch_gated_delta_f16_device(
            &mut sequential,
            prepared_views.query,
            prepared_views.key,
            value_view,
            prepared_views.log_decay,
            prepared_views.beta,
        )?;
        sequential_outputs.extend(runtime.verifier_read_f32(actual_view)?);
    }
    let sequential_state = sequential.verifier_read_state()?;

    let mut scan = runtime.prepare_gated_delta_f16(config)?;
    let preparation = runtime.prepare_gated_delta_inputs_f32(&a_log, &dt_bias)?;
    let preparation_workspace = runtime.prepare_gated_delta_scan_inputs(args.tokens)?;
    let convolved_staging = runtime.prepare_verifier_f32_tensor(&convolved_qkv)?;
    let raw_a_staging = runtime.prepare_verifier_f32_tensor(&raw_a)?;
    let raw_b_staging = runtime.prepare_verifier_f32_tensor(&raw_b)?;
    let prepared_views = runtime.dispatch_gated_delta_scan_inputs_device(
        &preparation,
        &preparation_workspace,
        convolved_staging.device_view()?,
        raw_a_staging.device_view()?,
        raw_b_staging.device_view()?,
        args.tokens,
    )?;
    let mut maximum_batched_preparation_absolute_error = 0.0_f32;
    for (name, expected, actual) in [
        ("query", query.as_slice(), prepared_views.query),
        ("key", key.as_slice(), prepared_views.key),
        ("value", value.as_slice(), prepared_views.value),
        ("log_decay", log_decay.as_slice(), prepared_views.log_decay),
        ("beta", beta.as_slice(), prepared_views.beta),
    ] {
        let actual = runtime.verifier_read_f32(actual)?;
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            let absolute = (expected - actual).abs();
            maximum_batched_preparation_absolute_error =
                maximum_batched_preparation_absolute_error.max(absolute);
            anyhow::ensure!(
                absolute <= 3.0e-6,
                "batched preparation {name} {index} differs from oracle"
            );
        }
    }
    let scan_output = runtime.prepare_gated_delta_scan_output(config, args.tokens)?;
    let scan_view = runtime.dispatch_gated_delta_f16_scan_device(
        &mut scan,
        &scan_output,
        prepared_views.query,
        prepared_views.key,
        prepared_views.value,
        prepared_views.log_decay,
        prepared_views.beta,
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
        + sequential_preparation.model_bytes()
        + sequential_preparation.transient_bytes()
        + sequential_convolved.resident_bytes()
        + sequential_raw_a.resident_bytes()
        + sequential_raw_b.resident_bytes()
        + scan.resident_state_bytes()
        + scan.speculative_checkpoint_bytes()
        + scan.transient_bytes()
        + preparation.model_bytes()
        + preparation.transient_bytes()
        + preparation_workspace.transient_bytes()
        + convolved_staging.resident_bytes()
        + raw_a_staging.resident_bytes()
        + raw_b_staging.resident_bytes()
        + scan_output.transient_bytes();
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(sequential);
    drop(sequential_preparation);
    drop(sequential_convolved);
    drop(sequential_raw_a);
    drop(sequential_raw_b);
    drop(scan);
    drop(preparation);
    drop(preparation_workspace);
    drop(convolved_staging);
    drop(raw_a_staging);
    drop(raw_b_staging);
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
            format: "ctox.cuda-sm86-gated-delta-scan-verifier.v2",
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
            maximum_batched_preparation_absolute_error,
            verifier_allocated_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "The unpromoted upstream-structured path prepares compact Q/K, V, and raw A/B for one token-major prompt chunk in one launch, then advances the exact decode FP16 recurrence in one causal scan. Promotion still requires representative chunk-size latency and complete-graph integration.",
        })?
    );
    Ok(())
}

fn fixture(values: usize, frequency: f32, scale: f32, offset: usize) -> Vec<f32> {
    (0..values)
        .map(|index| ((index + offset) as f32 * frequency).sin() * scale)
        .collect()
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
