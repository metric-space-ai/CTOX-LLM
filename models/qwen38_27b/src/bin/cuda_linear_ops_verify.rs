use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaCausalConvConfig, CudaGatedRmsNormConfig,
};
use ctox_qwen38_27b::reference::{causal_conv_silu_update_f16_state, rms_norm_gated};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify Qwen3.8-27B CUDA causal-conv and gated RMSNorm")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 6)]
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
    convolution_channels: usize,
    convolution_kernel_width: usize,
    convolution_model_bytes: usize,
    convolution_state_bytes: usize,
    convolution_transient_bytes: usize,
    gated_norm_rows: usize,
    gated_norm_columns: usize,
    gated_norm_model_bytes: usize,
    gated_norm_transient_bytes: usize,
    maximum_convolution_absolute_error: f32,
    maximum_convolution_relative_error: f32,
    maximum_gated_norm_absolute_error: f32,
    maximum_gated_norm_relative_error: f32,
    state_matches_oracle_exactly: bool,
    reset_zero_state_verified: bool,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
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

fn track_error(expected: &[f32], actual: &[f32], absolute: &mut f32, relative: &mut f32) {
    for (left, right) in expected.iter().zip(actual) {
        let error = (left - right).abs();
        *absolute = absolute.max(error);
        *relative = relative.max(error / left.abs().max(f32::MIN_POSITIVE));
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(args.tokens > 0, "tokens must be positive");
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;

    let conv_config = CudaCausalConvConfig::QWEN38_27B;
    let (conv_weight, conv_weight_bytes) = f16_fixture(
        (0..conv_config.channels * conv_config.kernel_width)
            .map(|index| ((index + 1) as f32 * 0.031).cos() * 0.25),
    );
    let mut conv = runtime.prepare_causal_conv_f16(conv_config, &conv_weight_bytes)?;
    let (free_after_prepare, _) = runtime.memory_info()?;
    let mut oracle_state = vec![f16::ZERO; conv_config.channels * conv_config.kernel_width];
    let mut maximum_convolution_absolute_error = 0.0_f32;
    let mut maximum_convolution_relative_error = 0.0_f32;
    let mut state_matches_oracle_exactly = true;
    for token in 0..args.tokens {
        let input: Vec<f32> = (0..conv_config.channels)
            .map(|channel| ((channel + token * 7) as f32 * 0.031).sin() * 0.65)
            .collect();
        let expected = causal_conv_silu_update_f16_state(
            &input,
            &mut oracle_state,
            &conv_weight,
            conv_config.channels,
            conv_config.kernel_width,
        )?;
        conv.write_input(&input)?;
        let actual = runtime.dispatch_causal_conv_f16(&mut conv)?;
        track_error(
            &expected,
            &actual,
            &mut maximum_convolution_absolute_error,
            &mut maximum_convolution_relative_error,
        );
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            anyhow::ensure!(
                (expected - actual).abs()
                    <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
                "token {token} convolution output {index}: expected {expected}, got {actual}"
            );
        }
        state_matches_oracle_exactly &= conv.verifier_read_state()? == oracle_state;
    }
    anyhow::ensure!(state_matches_oracle_exactly, "convolution state differs");
    conv.reset()?;
    let reset_zero_state_verified = conv
        .verifier_read_state()?
        .iter()
        .all(|value| *value == f16::ZERO);
    anyhow::ensure!(reset_zero_state_verified, "convolution reset failed");

    let norm_config = CudaGatedRmsNormConfig::QWEN38_27B;
    let (norm_weight, norm_weight_bytes) =
        f16_fixture((0..norm_config.columns).map(|index| 0.85 + (index % 17) as f32 * 0.013));
    let norm = runtime.prepare_gated_rms_norm_f16(norm_config, &norm_weight_bytes)?;
    let input: Vec<f32> = (0..norm_config.rows * norm_config.columns)
        .map(|index| ((index + 3) as f32 * 0.017).sin() * 0.75)
        .collect();
    let gate: Vec<f32> = (0..norm_config.rows * norm_config.columns)
        .map(|index| ((index + 7) as f32 * 0.011).cos() * 0.8)
        .collect();
    let expected_norm = rms_norm_gated(
        &input,
        &gate,
        norm_config.rows,
        norm_config.columns,
        &norm_weight,
        norm_config.epsilon,
    )?;
    norm.write_inputs(&input, &gate)?;
    let actual_norm = runtime.dispatch_gated_rms_norm_f16(&norm)?;
    let mut maximum_gated_norm_absolute_error = 0.0_f32;
    let mut maximum_gated_norm_relative_error = 0.0_f32;
    track_error(
        &expected_norm,
        &actual_norm,
        &mut maximum_gated_norm_absolute_error,
        &mut maximum_gated_norm_relative_error,
    );
    for (index, (expected, actual)) in expected_norm.iter().zip(&actual_norm).enumerate() {
        anyhow::ensure!(
            (expected - actual).abs()
                <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
            "gated RMSNorm output {index}: expected {expected}, got {actual}"
        );
    }

    let convolution_model_bytes = conv.model_bytes();
    let convolution_state_bytes = conv.resident_state_bytes();
    let convolution_transient_bytes = conv.transient_bytes();
    let gated_norm_model_bytes = norm.model_bytes();
    let gated_norm_transient_bytes = norm.transient_bytes();
    drop(conv);
    drop(norm);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    let requested_bytes = convolution_model_bytes
        + convolution_state_bytes
        + convolution_transient_bytes
        + gated_norm_model_bytes
        + gated_norm_transient_bytes;
    anyhow::ensure!(
        observed_reclaimed_bytes >= requested_bytes,
        "dropping CUDA linear-op objects did not reclaim requested buffers"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-linear-ops-f16-verifier.v1",
            status: "pass",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            convolution_channels: conv_config.channels,
            convolution_kernel_width: conv_config.kernel_width,
            convolution_model_bytes,
            convolution_state_bytes,
            convolution_transient_bytes,
            gated_norm_rows: norm_config.rows,
            gated_norm_columns: norm_config.columns,
            gated_norm_model_bytes,
            gated_norm_transient_bytes,
            maximum_convolution_absolute_error,
            maximum_convolution_relative_error,
            maximum_gated_norm_absolute_error,
            maximum_gated_norm_relative_error,
            state_matches_oracle_exactly,
            reset_zero_state_verified,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "Verifier-only candidates; no CPU fallback and not promoted into the production CUDA ABI.",
        })?
    );
    Ok(())
}
