use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_graph::PreparedCudaFullAttentionLayer;
use ctox_qwen38_27b::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaPagedGqaConfig, CudaPartialRopeConfig, CudaQueryGateConfig,
};
use ctox_qwen38_27b::kv_cache::{
    DEFAULT_KV_PAGE_TOKENS, DEFAULT_KV_RECENT_TOKENS, DEFAULT_KV_SINK_TOKENS,
};
use ctox_qwen38_27b::Qwen38Config;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify one complete Qwen full-attention prefill chunk on CUDA SM86")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 160)]
    tokens: usize,
    #[arg(long, default_value_t = 4096)]
    maximum_context_tokens: usize,
    #[arg(long, default_value_t = 2.0e-5)]
    absolute_tolerance: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    maximum_context_tokens: usize,
    attention_values: usize,
    gate_values: usize,
    attention_f32le_sha256: String,
    gate_f32le_sha256: String,
    maximum_attention_sequential_absolute_delta: f32,
    maximum_gate_sequential_absolute_delta: f32,
    cached_tokens: usize,
    q2_tokens: usize,
    q4_tokens: usize,
    model_bytes_per_layer: u64,
    graph_bytes_per_layer: u64,
    session_bytes_per_layer: u64,
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

fn digest_f32(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn maximum_absolute_delta(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (1..=512).contains(&args.tokens),
        "tokens must be in 1..=512"
    );
    anyhow::ensure!(
        args.maximum_context_tokens >= args.tokens
            && args.maximum_context_tokens >= DEFAULT_KV_SINK_TOKENS + DEFAULT_KV_RECENT_TOKENS,
        "maximum context does not admit the prompt"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let config = Qwen38Config::default();
    let q_norm_weight =
        packed_f16((0..config.head_dim).map(|column| -0.08 + (column % 23) as f32 * 0.008));
    let k_norm_weight =
        packed_f16((0..config.head_dim).map(|column| 0.04 - (column % 29) as f32 * 0.006));
    let mut sequential = PreparedCudaFullAttentionLayer::prepare(
        &runtime,
        &config,
        "fixture:sequential",
        &q_norm_weight,
        &k_norm_weight,
        args.maximum_context_tokens,
    )?;
    let mut batched = PreparedCudaFullAttentionLayer::prepare(
        &runtime,
        &config,
        "fixture:batched",
        &q_norm_weight,
        &k_norm_weight,
        args.maximum_context_tokens,
    )?;
    let model_bytes_per_layer = batched.model_bytes();
    let graph_bytes_per_layer = batched.graph_bytes();
    let session_bytes_per_layer = batched.session_bytes();

    let query_values = config
        .num_attention_heads
        .checked_mul(config.head_dim)
        .context("query width overflows")?;
    let query_gate_values = args
        .tokens
        .checked_mul(query_values * 2)
        .context("query/gate fixture overflows")?;
    let key_value_width = config
        .num_key_value_heads
        .checked_mul(config.head_dim)
        .context("key/value width overflows")?;
    let key_values = args
        .tokens
        .checked_mul(key_value_width)
        .context("key fixture overflows")?;
    let query_gate: Vec<f32> = (0..query_gate_values)
        .map(|index| ((index + 7) as f32 * 0.0017).sin() * 0.42)
        .collect();
    let key: Vec<f32> = (0..key_values)
        .map(|index| ((index + 19) as f32 * 0.0031).cos() * 0.38)
        .collect();
    let value: Vec<f32> = (0..key_values)
        .map(|index| ((index + 31) as f32 * 0.0023).sin() * 0.36)
        .collect();
    let query_gate_owner = runtime.prepare_verifier_f32_tensor(&query_gate)?;
    let key_owner = runtime.prepare_verifier_f32_tensor(&key)?;
    let value_owner = runtime.prepare_verifier_f32_tensor(&value)?;

    let mut expected_attention = Vec::with_capacity(args.tokens * query_values);
    let mut expected_gate = Vec::with_capacity(args.tokens * query_values);
    for token in 0..args.tokens {
        let (attention, gate) = sequential.dispatch_device(
            &runtime,
            query_gate_owner
                .device_view()?
                .slice(token * query_values * 2, query_values * 2)?,
            key_owner
                .device_view()?
                .slice(token * key_value_width, key_value_width)?,
            value_owner
                .device_view()?
                .slice(token * key_value_width, key_value_width)?,
            token as u64,
        )?;
        expected_attention.extend(runtime.verifier_read_f32(attention)?);
        expected_gate.extend(runtime.verifier_read_f32(gate)?);
    }

    let key_norm_workspace = runtime.prepare_batched_rms_norm_workspace(
        args.tokens * config.num_key_value_heads,
        config.head_dim,
    )?;
    let rope_workspace = runtime.prepare_batched_rope_workspace(
        CudaPartialRopeConfig {
            heads: config.num_attention_heads,
            head_dim: config.head_dim,
            rotary_dim: config.rotary_dim,
            theta: config.rope_theta,
        },
        args.tokens,
    )?;
    let query_gate_output =
        runtime.prepare_batched_query_gate_output(CudaQueryGateConfig::QWEN38_27B, args.tokens)?;
    let attention_output = runtime.prepare_paged_q2q4_gqa_prefill_output(
        CudaPagedGqaConfig {
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            maximum_tokens: args.maximum_context_tokens,
            page_tokens: DEFAULT_KV_PAGE_TOKENS,
            sink_tokens: DEFAULT_KV_SINK_TOKENS,
            recent_tokens: DEFAULT_KV_RECENT_TOKENS,
        },
        args.tokens,
    )?;
    let chunk_workspace_bytes = key_norm_workspace
        .transient_bytes()
        .checked_add(rope_workspace.transient_bytes())
        .and_then(|bytes| bytes.checked_add(query_gate_output.transient_bytes()))
        .and_then(|bytes| bytes.checked_add(attention_output.transient_bytes()))
        .context("attention workspace byte count overflows")?;
    let (attention, gate) = batched.dispatch_prefill_device(
        &runtime,
        &key_norm_workspace,
        &rope_workspace,
        &query_gate_output,
        &attention_output,
        query_gate_owner.device_view()?,
        key_owner.device_view()?,
        value_owner.device_view()?,
        0,
        args.tokens,
    )?;
    let actual_attention = runtime.verifier_read_f32(attention)?;
    let actual_gate = runtime.verifier_read_f32(gate)?;
    let maximum_attention_sequential_absolute_delta =
        maximum_absolute_delta(&expected_attention, &actual_attention);
    let maximum_gate_sequential_absolute_delta =
        maximum_absolute_delta(&expected_gate, &actual_gate);
    anyhow::ensure!(
        maximum_attention_sequential_absolute_delta <= args.absolute_tolerance,
        "batched full-attention output delta {maximum_attention_sequential_absolute_delta} exceeds {}",
        args.absolute_tolerance
    );
    anyhow::ensure!(
        maximum_gate_sequential_absolute_delta <= args.absolute_tolerance,
        "batched full-attention gate delta {maximum_gate_sequential_absolute_delta} exceeds {}",
        args.absolute_tolerance
    );
    let cached_tokens = batched.kv_mut().tokens();
    let q2_tokens = batched.kv_mut().q2_tokens();
    let q4_tokens = batched.kv_mut().q4_tokens();
    anyhow::ensure!(
        cached_tokens == args.tokens && q2_tokens + q4_tokens == args.tokens,
        "batched full-attention cache metadata is incomplete"
    );
    anyhow::ensure!(
        sequential.kv_mut().tokens() == cached_tokens
            && sequential.kv_mut().q2_tokens() == q2_tokens
            && sequential.kv_mut().q4_tokens() == q4_tokens,
        "batched and sequential cache precision maps differ"
    );
    let attention_values = actual_attention.len();
    let gate_values = actual_gate.len();
    let attention_f32le_sha256 = digest_f32(&actual_attention);
    let gate_f32le_sha256 = digest_f32(&actual_gate);
    let (free_after_prepare, _) = runtime.memory_info()?;

    drop(sequential);
    drop(batched);
    drop(query_gate_owner);
    drop(key_owner);
    drop(value_owner);
    drop(key_norm_workspace);
    drop(rope_workspace);
    drop(query_gate_output);
    drop(attention_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA attention-prefill verifier retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-attention-prefill.v1",
            status: "sequential_device_match_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            maximum_context_tokens: args.maximum_context_tokens,
            attention_values,
            gate_values,
            attention_f32le_sha256,
            gate_f32le_sha256,
            maximum_attention_sequential_absolute_delta,
            maximum_gate_sequential_absolute_delta,
            cached_tokens,
            q2_tokens,
            q4_tokens,
            model_bytes_per_layer,
            graph_bytes_per_layer,
            session_bytes_per_layer,
            chunk_workspace_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes: free_before_prepare.saturating_sub(free_after_prepare),
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            note: "One complete full-attention prompt chunk builds one shared device RoPE table, fuses query deinterleave/Q-norm/RoPE, applies batched K-norm/RoPE, packs canonical paged Q2/Q4 KV, and executes causal paged GQA without a host token loop. Sequential CUDA is the numerical and cache-precision oracle. This is a layer-composite gate; projection fanout, attention gate/output projection, graph-wide commit, all 645 scheduled steps, and roofline promotion remain separate gates.",
        })?
    );
    Ok(())
}
