use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda::{
    GATED_DELTA_HEADS, GATED_DELTA_VALUE_DIM, GATED_RMS_NORM_COLUMNS, LINEAR_CONV_CHANNELS,
    LINEAR_CONV_KERNEL_WIDTH,
};
use ctox_qwen38_27b::backend::cuda_graph::PreparedCudaLinearMixerLayer;
use ctox_qwen38_27b::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaCausalConvConfig, CudaGatedDeltaConfig, CudaGatedRmsNormConfig,
};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify one complete Qwen linear-attention prefill chunk on CUDA SM86")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 8)]
    tokens: usize,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    output_values: usize,
    output_f32le_sha256: String,
    output_matches_sequential_bit_exact: bool,
    convolution_state_matches_sequential_bit_exact: bool,
    recurrence_state_matches_sequential_bit_exact: bool,
    model_bytes_per_mixer: u64,
    graph_bytes_per_mixer: u64,
    session_bytes_per_mixer: u64,
    speculative_checkpoint_bytes_per_mixer: u64,
    chunk_workspace_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_allocation_bytes: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn packed_f16(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values
        .map(f16::from_f32)
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect()
}

fn packed_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn digest_f32(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (1..=512).contains(&args.tokens),
        "tokens must be in 1..=512"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;

    let convolution_weight = packed_f16(
        (0..LINEAR_CONV_CHANNELS * LINEAR_CONV_KERNEL_WIDTH)
            .map(|index| ((index + 11) as f32 * 0.017).cos() * 0.14),
    );
    let a_log: Vec<f32> = (0..GATED_DELTA_HEADS)
        .map(|head| -2.1 - head as f32 * 0.007)
        .collect();
    let dt_bias: Vec<f32> = (0..GATED_DELTA_HEADS)
        .map(|head| -0.7 + head as f32 * 0.009)
        .collect();
    let a_log = packed_f32(&a_log);
    let dt_bias = packed_f32(&dt_bias);
    let norm_weight =
        packed_f16((0..GATED_RMS_NORM_COLUMNS).map(|column| 0.82 + (column % 19) as f32 * 0.011));

    let mut sequential = PreparedCudaLinearMixerLayer::prepare(
        &runtime,
        0,
        &convolution_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    )?;
    let mut batched = PreparedCudaLinearMixerLayer::prepare(
        &runtime,
        0,
        &convolution_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    )?;
    let model_bytes_per_mixer = batched.model_bytes();
    let graph_bytes_per_mixer = batched.graph_bytes();
    let session_bytes_per_mixer = batched.session_bytes();
    let speculative_checkpoint_bytes_per_mixer = batched.speculative_checkpoint_bytes();

    let mixed_qkv_values = args
        .tokens
        .checked_mul(LINEAR_CONV_CHANNELS)
        .context("mixed-QKV fixture shape overflows")?;
    let gated_values = args
        .tokens
        .checked_mul(GATED_DELTA_HEADS * GATED_DELTA_VALUE_DIM)
        .context("gate fixture shape overflows")?;
    let head_values = args
        .tokens
        .checked_mul(GATED_DELTA_HEADS)
        .context("head fixture shape overflows")?;
    let mixed_qkv: Vec<f32> = (0..mixed_qkv_values)
        .map(|index| ((index + 5) as f32 * 0.0031).sin() * 0.31)
        .collect();
    let gate: Vec<f32> = (0..gated_values)
        .map(|index| ((index + 17) as f32 * 0.0023).cos() * 0.44)
        .collect();
    let raw_a: Vec<f32> = (0..head_values)
        .map(|index| ((index + 23) as f32 * 0.019).sin() * 0.7)
        .collect();
    let raw_b: Vec<f32> = (0..head_values)
        .map(|index| ((index + 31) as f32 * 0.013).cos() * 0.6)
        .collect();
    let mixed_qkv_owner = runtime.prepare_verifier_f32_tensor(&mixed_qkv)?;
    let gate_owner = runtime.prepare_verifier_f32_tensor(&gate)?;
    let raw_a_owner = runtime.prepare_verifier_f32_tensor(&raw_a)?;
    let raw_b_owner = runtime.prepare_verifier_f32_tensor(&raw_b)?;

    let mut expected = Vec::with_capacity(gated_values);
    for token in 0..args.tokens {
        let output = sequential.dispatch_device(
            &runtime,
            mixed_qkv_owner
                .device_view()?
                .slice(token * LINEAR_CONV_CHANNELS, LINEAR_CONV_CHANNELS)?,
            gate_owner.device_view()?.slice(
                token * GATED_DELTA_HEADS * GATED_DELTA_VALUE_DIM,
                GATED_DELTA_HEADS * GATED_DELTA_VALUE_DIM,
            )?,
            raw_a_owner
                .device_view()?
                .slice(token * GATED_DELTA_HEADS, GATED_DELTA_HEADS)?,
            raw_b_owner
                .device_view()?
                .slice(token * GATED_DELTA_HEADS, GATED_DELTA_HEADS)?,
        )?;
        expected.extend(runtime.verifier_read_f32(output)?);
    }
    let sequential_convolution_state = sequential.convolution_mut().verifier_read_state()?;
    let sequential_recurrence_state = sequential.recurrence_mut().verifier_read_state()?;

    let convolution_output =
        runtime.prepare_causal_conv_scan_output(CudaCausalConvConfig::QWEN38_27B, args.tokens)?;
    let input_workspace = runtime.prepare_gated_delta_scan_inputs(args.tokens)?;
    let recurrence_output =
        runtime.prepare_gated_delta_scan_output(CudaGatedDeltaConfig::QWEN38_27B, args.tokens)?;
    let norm_output = runtime
        .prepare_batched_gated_rms_norm_output(CudaGatedRmsNormConfig::QWEN38_27B, args.tokens)?;
    let chunk_workspace_bytes = convolution_output
        .transient_bytes()
        .checked_add(input_workspace.transient_bytes())
        .and_then(|bytes| bytes.checked_add(recurrence_output.transient_bytes()))
        .and_then(|bytes| bytes.checked_add(norm_output.transient_bytes()))
        .context("chunk workspace byte count overflows")?;
    let actual = runtime.verifier_read_f32(batched.dispatch_prefill_device(
        &runtime,
        &convolution_output,
        &input_workspace,
        &recurrence_output,
        &norm_output,
        mixed_qkv_owner.device_view()?,
        gate_owner.device_view()?,
        raw_a_owner.device_view()?,
        raw_b_owner.device_view()?,
        args.tokens,
    )?)?;
    let output_matches_sequential_bit_exact = actual == expected;
    anyhow::ensure!(
        output_matches_sequential_bit_exact,
        "batched linear prefill output differs from sequential device execution"
    );
    let convolution_state_matches_sequential_bit_exact =
        batched.convolution_mut().verifier_read_state()? == sequential_convolution_state;
    anyhow::ensure!(
        convolution_state_matches_sequential_bit_exact,
        "batched linear prefill convolution state differs from sequential execution"
    );
    let recurrence_state_matches_sequential_bit_exact =
        batched.recurrence_mut().verifier_read_state()? == sequential_recurrence_state;
    anyhow::ensure!(
        recurrence_state_matches_sequential_bit_exact,
        "batched linear prefill recurrence state differs from sequential execution"
    );
    let output_f32le_sha256 = digest_f32(&actual);
    let output_values = actual.len();
    let (free_after_prepare, _) = runtime.memory_info()?;

    drop(sequential);
    drop(batched);
    drop(mixed_qkv_owner);
    drop(gate_owner);
    drop(raw_a_owner);
    drop(raw_b_owner);
    drop(convolution_output);
    drop(input_workspace);
    drop(recurrence_output);
    drop(norm_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA linear-prefill verifier retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-linear-prefill.v1",
            status: "bit_exact_sequential_device_match_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            output_values,
            output_f32le_sha256,
            output_matches_sequential_bit_exact,
            convolution_state_matches_sequential_bit_exact,
            recurrence_state_matches_sequential_bit_exact,
            model_bytes_per_mixer,
            graph_bytes_per_mixer,
            session_bytes_per_mixer,
            speculative_checkpoint_bytes_per_mixer,
            chunk_workspace_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes: free_before_prepare.saturating_sub(free_after_prepare),
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            note: "One complete allocation-free linear-attention prompt chunk executes CausalConv scan, fused GatedDelta input preparation, recurrent GatedDelta scan, and batched gated RMSNorm entirely through device views. Output plus both persistent states must match the ordinary token-wise CUDA path bit-for-bit. This is a layer composite gate; projection fanout, graph-wide chunk transaction, all 645 scheduled steps, and roofline promotion remain separate gates.",
        })?
    );
    Ok(())
}
