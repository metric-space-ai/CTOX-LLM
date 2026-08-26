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
#[command(about = "Verify fused Qwen SwiGLU plus corrected A8 CUDA quantization")]
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
    prepared_activation_bytes: usize,
    verifier_staging_bytes: usize,
    avoided_swiglu_f32_bytes: usize,
    maximum_scale_absolute_error: f32,
    code_mismatches: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let columns = 17_408;
    let gate: Vec<f32> = (0..columns)
        .map(|index| ((index as f32 + 3.0) * 0.017).sin() * 2.4)
        .collect();
    let up: Vec<f32> = (0..columns)
        .map(|index| ((index as f32 + 11.0) * 0.013).cos() * 1.7)
        .collect();
    let s_in_values: Vec<f32> = (0..columns)
        .map(|index| 0.82 + (index % 29) as f32 * 0.011)
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

    let expected_values: Vec<f32> = gate
        .iter()
        .zip(&up)
        .zip(&s_in_quantized)
        .map(|((gate, up), s_in)| gate / (1.0 + (-gate).exp()) * up * s_in)
        .collect();
    let expected_blocks = expected_values
        .chunks_exact(BLOCK_LEN)
        .map(A8Block64::quantize)
        .collect::<Result<Vec<_>, _>>()?;

    let (free_before_prepare, _) = runtime.memory_info()?;
    let prepared = runtime.prepare_shared_a8_activation(&operation)?;
    let gate_staging = runtime.prepare_verifier_f32_tensor(&gate)?;
    let up_staging = runtime.prepare_verifier_f32_tensor(&up)?;
    let (free_after_prepare, _) = runtime.memory_info()?;
    runtime.quantize_shared_a8_swiglu_device(
        &prepared,
        gate_staging.device_view()?,
        up_staging.device_view()?,
    )?;
    let (actual_codes, actual_scales) = prepared.verifier_read_quantized()?;

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
        "SwiGLU A8 scale error {maximum_scale_absolute_error} exceeds tolerance"
    );
    anyhow::ensure!(code_mismatches == 0, "SwiGLU A8 codes differ");

    let prepared_activation_bytes = prepared.resident_bytes();
    let verifier_staging_bytes = gate_staging.resident_bytes() + up_staging.resident_bytes();
    drop(prepared);
    drop(gate_staging);
    drop(up_staging);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes >= prepared_activation_bytes + verifier_staging_bytes,
        "dropping SwiGLU verifier allocations did not reclaim requested CUDA buffers"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-swiglu-a8-verifier.v1",
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
            prepared_activation_bytes,
            verifier_staging_bytes,
            avoided_swiglu_f32_bytes: columns * std::mem::size_of::<f32>(),
            maximum_scale_absolute_error,
            code_mismatches,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "Verifier-only candidate; SiLU(gate), multiplication by up, recovery s_in, and A8 quantization are one device launch. No f32 SwiGLU intermediate or CPU fallback.",
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
