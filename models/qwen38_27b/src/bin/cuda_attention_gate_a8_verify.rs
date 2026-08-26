use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::backend::{Activation, FusedMatVec, ScaleSlice};
use ctox_qwen38_27b::format::TensorDType;
use ctox_qwen38_27b::quant::{A8Block64, Q2Block64, BLOCK_LEN};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify fused Qwen attention gate plus corrected A8 CUDA quantization")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    columns: usize,
    a8_blocks: usize,
    prepared_bytes: usize,
    verifier_staging_bytes: usize,
    avoided_gated_attention_f32_bytes: usize,
    maximum_scale_absolute_error: f32,
    code_mismatches: usize,
    output_projection_device_chain_verified: bool,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let columns = 6_144;
    let attention: Vec<f32> = (0..columns)
        .map(|index| ((index as f32 + 5.0) * 0.019).sin() * 1.8)
        .collect();
    let gate: Vec<f32> = (0..columns)
        .map(|index| ((index as f32 + 13.0) * 0.011).cos() * 2.1)
        .collect();
    let s_in_values: Vec<f32> = (0..columns)
        .map(|index| 0.84 + (index % 23) as f32 * 0.012)
        .collect();
    let s_in = f16_bytes(&s_in_values);
    let s_in_quantized: Vec<f32> = s_in_values
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect();
    let weights = Q2Block64::quantize(&vec![0.0; BLOCK_LEN])?
        .encode()
        .repeat(columns / BLOCK_LEN);
    let s_out = f16_bytes(&[1.0]);
    let input = vec![0.0; columns];
    let operation = FusedMatVec {
        dtype: TensorDType::Q2B64,
        weights: &weights,
        segments: &[],
        rows: 1,
        columns,
        input: &input,
        s_in: Some(ScaleSlice::F16Le(&s_in)),
        s_out: Some(ScaleSlice::F16Le(&s_out)),
        bias: None,
        activation: Activation::Identity,
    };
    let expected_values: Vec<f32> = attention
        .iter()
        .zip(&gate)
        .zip(&s_in_quantized)
        .map(|((attention, gate), s_in)| attention / (1.0 + (-gate).exp()) * s_in)
        .collect();
    let expected_blocks = expected_values
        .chunks_exact(BLOCK_LEN)
        .map(A8Block64::quantize)
        .collect::<Result<Vec<_>, _>>()?;

    let activation = runtime.prepare_shared_a8_activation(&operation)?;
    let output_projection = runtime.prepare_shared_a8_projection(&operation)?;
    let attention_staging = runtime.prepare_verifier_f32_tensor(&attention)?;
    let gate_staging = runtime.prepare_verifier_f32_tensor(&gate)?;
    let (_, free_total) = runtime.memory_info()?;
    let free_after_prepare = runtime.memory_info()?.0;
    let projection_refs = [&output_projection];
    let output_views = runtime.dispatch_shared_a8_sigmoid_gate_fanout_device(
        &activation,
        attention_staging.device_view()?,
        gate_staging.device_view()?,
        &projection_refs,
    )?;
    let output = runtime.verifier_read_f32(output_views[0])?;
    let output_projection_device_chain_verified = output.iter().all(|value| *value == 0.0);
    anyhow::ensure!(
        output_projection_device_chain_verified,
        "zero-weight attention output projection produced a non-zero value"
    );
    let (actual_codes, actual_scales) = activation.verifier_read_quantized()?;
    let mut maximum_scale_absolute_error = 0.0_f32;
    let mut code_mismatches = 0;
    for (block_index, (expected, actual_scale)) in
        expected_blocks.iter().zip(&actual_scales).enumerate()
    {
        maximum_scale_absolute_error =
            maximum_scale_absolute_error.max((expected.scale - actual_scale).abs());
        let start = block_index * BLOCK_LEN;
        code_mismatches += expected
            .codes
            .iter()
            .zip(&actual_codes[start..start + BLOCK_LEN])
            .filter(|(expected, actual)| expected != actual)
            .count();
    }
    anyhow::ensure!(
        maximum_scale_absolute_error <= 2.0e-7,
        "attention-gate A8 scale error {maximum_scale_absolute_error} exceeds tolerance"
    );
    anyhow::ensure!(code_mismatches == 0, "attention-gate A8 codes differ");

    let prepared_bytes = activation.resident_bytes() + output_projection.resident_bytes();
    let verifier_staging_bytes = attention_staging.resident_bytes() + gate_staging.resident_bytes();
    drop(activation);
    drop(output_projection);
    drop(attention_staging);
    drop(gate_staging);
    let free_after_drop = runtime.memory_info()?.0;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes >= prepared_bytes + verifier_staging_bytes,
        "dropping attention-gate verifier allocations did not reclaim requested CUDA buffers"
    );
    anyhow::ensure!(
        free_after_drop <= free_total,
        "CUDA free memory exceeds total"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-attention-gate-a8-verifier.v1",
            status: "pass",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            columns,
            a8_blocks: columns / BLOCK_LEN,
            prepared_bytes,
            verifier_staging_bytes,
            avoided_gated_attention_f32_bytes: columns * std::mem::size_of::<f32>(),
            maximum_scale_absolute_error,
            code_mismatches,
            output_projection_device_chain_verified,
            observed_reclaimed_bytes,
            note: "Verifier-only candidate; attention*sigmoid(gate), recovery s_in, A8 quantization, and output projection remain one device chain with no f32 gated-attention intermediate or CPU fallback.",
        })?
    );
    Ok(())
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}
