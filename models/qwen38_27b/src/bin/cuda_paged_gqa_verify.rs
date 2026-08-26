use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaPagedGqaConfig};
use ctox_qwen38_27b::reference::grouped_query_attention;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify Qwen3.8-27B CUDA packed paged Q2/Q4 GQA")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 7)]
    tokens: usize,
    #[arg(long, default_value_t = 5.0e-4)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 2.0e-4)]
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
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    packed_device_bytes: usize,
    transient_bytes: usize,
    verifier_cpu_packed_bytes: usize,
    q4_tokens: usize,
    maximum_absolute_error: f32,
    maximum_relative_error: f32,
    demotion_verified: bool,
    reset_verified: bool,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (7..=8).contains(&args.tokens),
        "tokens must be between 7 and 8 so the verifier exercises Q4-to-Q2 demotion"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let config = CudaPagedGqaConfig {
        query_heads: 24,
        key_value_heads: 4,
        head_dim: 256,
        maximum_tokens: 8,
        page_tokens: 2,
        sink_tokens: 2,
        recent_tokens: 2,
    };

    let mut maximum_absolute_error = 0.0_f32;
    let mut maximum_relative_error = 0.0_f32;
    let packed_device_bytes;
    let transient_bytes;
    let verifier_cpu_packed_bytes;
    let q4_tokens;
    let free_after_prepare;
    let reset_verified;
    {
        let mut prepared = runtime.prepare_paged_q2q4_gqa(config)?;
        (free_after_prepare, _) = runtime.memory_info()?;
        packed_device_bytes = prepared.packed_device_bytes();
        transient_bytes = prepared.transient_bytes();
        for token in 0..args.tokens {
            let query: Vec<f32> = (0..config.query_heads * config.head_dim)
                .map(|index| ((index + token * 7) as f32 * 0.017).sin() * 0.35)
                .collect();
            let key: Vec<f32> = (0..config.key_value_heads * config.head_dim)
                .map(|index| ((index + token * 11) as f32 * 0.021).cos() * 0.45)
                .collect();
            let value: Vec<f32> = (0..config.key_value_heads * config.head_dim)
                .map(|index| ((index + token * 13) as f32 * 0.015).sin() * 0.55)
                .collect();
            let actual =
                runtime.append_and_dispatch_paged_q2q4_gqa(&mut prepared, &query, &key, &value)?;
            let cached_key = prepared.verifier_flattened_key()?;
            let cached_value = prepared.verifier_flattened_value()?;
            let expected = grouped_query_attention(
                &query,
                &cached_key,
                &cached_value,
                config.query_heads,
                config.key_value_heads,
                1,
                token + 1,
                config.head_dim,
                token,
            )?;
            for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
                let absolute = (left - right).abs();
                let relative = absolute / left.abs().max(f32::MIN_POSITIVE);
                maximum_absolute_error = maximum_absolute_error.max(absolute);
                maximum_relative_error = maximum_relative_error.max(relative);
                anyhow::ensure!(
                    absolute <= args.absolute_tolerance + args.relative_tolerance * left.abs(),
                    "token {token} output {index}: expected {left}, got {right}"
                );
            }
        }
        verifier_cpu_packed_bytes = prepared.verifier_cpu_packed_bytes();
        q4_tokens = prepared.q4_tokens();
        anyhow::ensure!(q4_tokens < args.tokens, "no old Q4 page was demoted to Q2");
        prepared.reset()?;
        reset_verified = prepared.tokens() == 0
            && prepared.q4_tokens() == 0
            && prepared.verifier_cpu_packed_bytes() == 0;
        anyhow::ensure!(reset_verified, "packed CUDA GQA reset failed");
    }
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop >= free_after_prepare,
        "dropping the packed GQA buffers did not reclaim device memory"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.qwen38.cuda_paged_q2q4_gqa_verification.v1",
            status: "pass",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            query_heads: config.query_heads,
            key_value_heads: config.key_value_heads,
            head_dim: config.head_dim,
            packed_device_bytes,
            transient_bytes,
            verifier_cpu_packed_bytes,
            q4_tokens,
            maximum_absolute_error,
            maximum_relative_error,
            demotion_verified: true,
            reset_verified,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            note: "Verifier-only CPU quantization mirror remains a promotion blocker; the CUDA cache itself stores only canonical packed Q2/Q4 pages.",
        })?
    );
    Ok(())
}
