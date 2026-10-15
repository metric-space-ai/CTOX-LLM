use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{
    CudaCandidateRuntime, CudaPartialRopeConfig, CudaQueryGateConfig,
};
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify shared-table batched CUDA RoPE against sequential CUDA")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 40)]
    tokens: usize,
    #[arg(long, default_value_t = 131_071)]
    start_position: u64,
    #[arg(long, default_value_t = 5.0e-4)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 1_000)]
    graph_replays: usize,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    tokens: usize,
    start_position: u64,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    shared_position_table: bool,
    maximum_query_sequential_absolute_delta: f32,
    maximum_key_sequential_absolute_delta: f32,
    query_tail_exact: bool,
    key_tail_exact: bool,
    maximum_query_gate_query_sequential_absolute_delta: f32,
    maximum_query_gate_gate_sequential_absolute_delta: f32,
    query_gate_output_bytes: usize,
    workspace_bytes: usize,
    staging_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    graph_replays: usize,
    graph_microseconds_per_replay: f64,
    graph_maximum_absolute_delta: f32,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (1..=512).contains(&args.tokens),
        "tokens must be within 1..=512"
    );
    args.start_position
        .checked_add(args.tokens as u64)
        .context("position range overflows")?;
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let graph_verification =
        runtime.verifier_capture_rope_graph(args.start_position, args.graph_replays)?;
    let (free_before_prepare, _) = runtime.memory_info()?;

    let query_config = CudaPartialRopeConfig {
        heads: 24,
        head_dim: 256,
        rotary_dim: 64,
        theta: 10_000_000.0,
    };
    let key_config = CudaPartialRopeConfig {
        heads: 4,
        ..query_config
    };
    let query_row_values = query_config.heads * query_config.head_dim;
    let key_row_values = key_config.heads * key_config.head_dim;
    let query_input: Vec<f32> = (0..args.tokens * query_row_values)
        .map(|index| ((index + 31) as f32 * 0.009).sin() * 0.55)
        .collect();
    let key_input: Vec<f32> = (0..args.tokens * key_row_values)
        .map(|index| ((index + 37) as f32 * 0.017).cos() * 0.45)
        .collect();

    let query_single = runtime.prepare_partial_rope_f32(query_config)?;
    let key_single = runtime.prepare_partial_rope_f32(key_config)?;
    let query_single_staging =
        runtime.prepare_verifier_f32_tensor(&query_input[..query_row_values])?;
    let key_single_staging = runtime.prepare_verifier_f32_tensor(&key_input[..key_row_values])?;
    let mut sequential_query = Vec::with_capacity(query_input.len());
    let mut sequential_key = Vec::with_capacity(key_input.len());
    for token in 0..args.tokens {
        let query_start = token * query_row_values;
        let key_start = token * key_row_values;
        query_single_staging.write(&query_input[query_start..query_start + query_row_values])?;
        key_single_staging.write(&key_input[key_start..key_start + key_row_values])?;
        let position = args.start_position + token as u64;
        query_single.write_position(position)?;
        key_single.write_position(position)?;
        sequential_query.extend(runtime.verifier_read_f32(
            runtime.dispatch_partial_rope_f32_device(
                &query_single,
                query_single_staging.device_view()?,
            )?,
        )?);
        sequential_key.extend(
            runtime.verifier_read_f32(runtime.dispatch_partial_rope_f32_device(
                &key_single,
                key_single_staging.device_view()?,
            )?)?,
        );
    }

    let workspace = runtime.prepare_batched_rope_workspace(query_config, args.tokens)?;
    let query_batch = runtime.prepare_verifier_f32_tensor(&query_input)?;
    let key_batch = runtime.prepare_verifier_f32_tensor(&key_input)?;
    runtime.write_batched_rope_positions(&workspace, args.start_position, args.tokens)?;
    let query_view = runtime.dispatch_batched_partial_rope_with_table_f32_device(
        &workspace,
        query_config,
        query_batch.device_view()?,
        args.tokens,
    )?;
    let key_view = runtime.dispatch_batched_partial_rope_with_table_f32_device(
        &workspace,
        key_config,
        key_batch.device_view()?,
        args.tokens,
    )?;
    let actual_query = runtime.verifier_read_f32(query_view)?;
    let actual_key = runtime.verifier_read_f32(key_view)?;
    let maximum_query_sequential_absolute_delta = maximum_delta(&sequential_query, &actual_query);
    let maximum_key_sequential_absolute_delta = maximum_delta(&sequential_key, &actual_key);
    anyhow::ensure!(
        maximum_query_sequential_absolute_delta <= args.absolute_tolerance
            && maximum_key_sequential_absolute_delta <= args.absolute_tolerance,
        "batched RoPE differs from sequential CUDA: query {}, key {}",
        maximum_query_sequential_absolute_delta,
        maximum_key_sequential_absolute_delta
    );
    let query_tail_exact = tails_equal(&query_input, &actual_query, query_config);
    let key_tail_exact = tails_equal(&key_input, &actual_key, key_config);
    anyhow::ensure!(
        query_tail_exact && key_tail_exact,
        "batched RoPE modified its tail"
    );

    let query_gate_config = CudaQueryGateConfig::QWEN38_27B;
    let query_gate_row_values = query_gate_config.heads * query_gate_config.head_dim * 2;
    let query_gate_input: Vec<f32> = (0..args.tokens * query_gate_row_values)
        .map(|index| ((index + 41) as f32 * 0.011).sin() * 0.65)
        .collect();
    let q_norm_weight: Vec<u8> = (0..query_gate_config.head_dim)
        .map(|index| ((index % 31) as f32 - 15.0) * 0.002)
        .map(f16::from_f32)
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let query_gate = runtime.prepare_query_gate_norm_rope_f32(query_gate_config, &q_norm_weight)?;
    let query_gate_single_staging =
        runtime.prepare_verifier_f32_tensor(&query_gate_input[..query_gate_row_values])?;
    let query_gate_output_values = query_gate_config.heads * query_gate_config.head_dim;
    let mut sequential_query_gate_query =
        Vec::with_capacity(args.tokens * query_gate_output_values);
    let mut sequential_query_gate_gate = Vec::with_capacity(args.tokens * query_gate_output_values);
    for token in 0..args.tokens {
        let start = token * query_gate_row_values;
        query_gate_single_staging.write(&query_gate_input[start..start + query_gate_row_values])?;
        query_gate.write_position(args.start_position + token as u64)?;
        let (query, gate) = runtime.dispatch_query_gate_norm_rope_device(
            &query_gate,
            query_gate_single_staging.device_view()?,
        )?;
        sequential_query_gate_query.extend(runtime.verifier_read_f32(query)?);
        sequential_query_gate_gate.extend(runtime.verifier_read_f32(gate)?);
    }
    let query_gate_batch_staging = runtime.prepare_verifier_f32_tensor(&query_gate_input)?;
    let query_gate_batch_output =
        runtime.prepare_batched_query_gate_output(query_gate_config, args.tokens)?;
    let (query_gate_query_view, query_gate_gate_view) = runtime
        .dispatch_batched_query_gate_norm_rope_with_table_device(
            &query_gate,
            &workspace,
            &query_gate_batch_output,
            query_gate_batch_staging.device_view()?,
            args.tokens,
        )?;
    let actual_query_gate_query = runtime.verifier_read_f32(query_gate_query_view)?;
    let actual_query_gate_gate = runtime.verifier_read_f32(query_gate_gate_view)?;
    let maximum_query_gate_query_sequential_absolute_delta =
        maximum_delta(&sequential_query_gate_query, &actual_query_gate_query);
    let maximum_query_gate_gate_sequential_absolute_delta =
        maximum_delta(&sequential_query_gate_gate, &actual_query_gate_gate);
    anyhow::ensure!(
        maximum_query_gate_query_sequential_absolute_delta <= args.absolute_tolerance
            && maximum_query_gate_gate_sequential_absolute_delta == 0.0,
        "batched query/gate fusion differs from sequential CUDA: query {}, gate {}",
        maximum_query_gate_query_sequential_absolute_delta,
        maximum_query_gate_gate_sequential_absolute_delta
    );

    let workspace_bytes = workspace.transient_bytes();
    let query_gate_output_bytes = query_gate_batch_output.transient_bytes();
    let staging_bytes = query_single_staging.resident_bytes()
        + key_single_staging.resident_bytes()
        + query_batch.resident_bytes()
        + key_batch.resident_bytes()
        + query_gate_single_staging.resident_bytes()
        + query_gate_batch_staging.resident_bytes();
    let single_operator_bytes = query_single.transient_bytes() + key_single.transient_bytes();
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(query_single);
    drop(key_single);
    drop(query_single_staging);
    drop(key_single_staging);
    drop(workspace);
    drop(query_batch);
    drop(key_batch);
    drop(query_gate);
    drop(query_gate_single_staging);
    drop(query_gate_batch_staging);
    drop(query_gate_batch_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes
            >= workspace_bytes + query_gate_output_bytes + staging_bytes + single_operator_bytes,
        "batched RoPE verifier did not reclaim all owned allocations"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-batched-rope-verifier.v1",
            status: "pass_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            start_position: args.start_position,
            query_heads: query_config.heads,
            key_value_heads: key_config.heads,
            head_dim: query_config.head_dim,
            rotary_dim: query_config.rotary_dim,
            shared_position_table: true,
            maximum_query_sequential_absolute_delta,
            maximum_key_sequential_absolute_delta,
            query_tail_exact,
            key_tail_exact,
            maximum_query_gate_query_sequential_absolute_delta,
            maximum_query_gate_gate_sequential_absolute_delta,
            query_gate_output_bytes,
            workspace_bytes,
            staging_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            graph_replays: graph_verification.iterations,
            graph_microseconds_per_replay: graph_verification.microseconds_per_replay,
            graph_maximum_absolute_delta: graph_verification.maximum_absolute_delta,
            note: "One device-built [token, rotary-pair] table is shared by token-major query and key in-place kernels. The same table drives batched Q/Gate deinterleave, resident Q RMSNorm, and query RoPE. Sequential CUDA is the oracle; no host token loop exists in the batched path and no model operation falls back to CPU. Dedicated-stream CUDA graph capture/replay is hardware-verified, but its one-kernel timing is not a model throughput benchmark.",
        })?
    );
    Ok(())
}

fn maximum_delta(expected: &[f32], actual: &[f32]) -> f32 {
    expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}

fn tails_equal(before: &[f32], after: &[f32], config: CudaPartialRopeConfig) -> bool {
    before
        .chunks_exact(config.head_dim)
        .zip(after.chunks_exact(config.head_dim))
        .all(|(left, right)| left[config.rotary_dim..] == right[config.rotary_dim..])
}
