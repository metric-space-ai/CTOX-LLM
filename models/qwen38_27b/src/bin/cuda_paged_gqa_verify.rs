use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda::{PAGED_GQA_SPLIT_MAX_QUERY_TOKENS, PAGED_GQA_SPLIT_SEGMENTS};
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaPagedGqaConfig};
use ctox_qwen38_27b::kv_cache::PagedKvCache;
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
    #[arg(long, default_value_t = 40)]
    tokens: usize,
    #[arg(long, default_value_t = 5.0e-4)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 2.0e-4)]
    relative_tolerance: f32,
    #[arg(long, default_value_t = 20)]
    benchmark_iterations: usize,
    #[arg(long, default_value_t = 1_000)]
    graph_replays: usize,
    #[arg(long, default_value_t = 131_072)]
    graph_maximum_tokens: usize,
    /// Number of newest causal queries measured by the split-KV path. One is
    /// the ordinary long-context decode shape; five is the MTP4 verification
    /// block shape.
    #[arg(long, default_value_t = PAGED_GQA_SPLIT_MAX_QUERY_TOKENS)]
    split_query_tokens: usize,
    #[arg(long, value_delimiter = ',', default_value = "1536,16384")]
    latency_contexts: Vec<usize>,
}

#[derive(Serialize)]
struct LatencyPoint {
    context_tokens: usize,
    packed_device_bytes: usize,
    split_transient_bytes: usize,
    q2_tokens: usize,
    q4_tokens: usize,
    iterations: usize,
    sequential_full_context_microseconds: f64,
    split_causal_microseconds: f64,
    split_speedup: f64,
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
    verifier_device_staging_bytes: usize,
    verifier_cpu_packed_bytes: usize,
    q4_tokens: usize,
    graph_tokenwise_packed_device_bytes: usize,
    graph_packed_device_bytes: usize,
    graph_target_16_layer_packed_device_bytes: usize,
    graph_target_mtp_17_layer_packed_device_bytes: usize,
    graph_maximum_absolute_delta: f32,
    graph_device_position_verified: bool,
    graph_maximum_tokens: usize,
    graph_replays: usize,
    graph_microseconds_per_replay: f64,
    graph_replay_maximum_absolute_error: f32,
    maximum_absolute_error: f32,
    maximum_relative_error: f32,
    split_query_tokens: usize,
    split_segments: usize,
    split_transient_bytes: usize,
    split_maximum_absolute_error: f32,
    split_maximum_relative_error: f32,
    split_device_view_path_verified: bool,
    split_tail_causality_verified: bool,
    benchmark_iterations: usize,
    sequential_full_context_microseconds: f64,
    split_causal_microseconds: f64,
    split_speedup: f64,
    latency_sweep: Vec<LatencyPoint>,
    device_view_path_verified: bool,
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
        (24..=128).contains(&args.tokens),
        "tokens must be between 24 and 128 so the verifier exercises Q4-to-Q2 demotion and non-empty split-KV segments"
    );
    anyhow::ensure!(
        (1..=PAGED_GQA_SPLIT_MAX_QUERY_TOKENS).contains(&args.split_query_tokens)
            && args.split_query_tokens <= args.tokens,
        "split-query-tokens must be within 1..={PAGED_GQA_SPLIT_MAX_QUERY_TOKENS} and no greater than tokens"
    );
    anyhow::ensure!(
        !args.latency_contexts.is_empty()
            && args
                .latency_contexts
                .iter()
                .all(|tokens| (256..=131_072).contains(tokens)),
        "latency contexts must be non-empty and within 256..=131072"
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
        maximum_tokens: args.tokens,
        page_tokens: 8,
        sink_tokens: 8,
        recent_tokens: 8,
    };

    let mut maximum_absolute_error = 0.0_f32;
    let mut maximum_relative_error = 0.0_f32;
    let mut split_maximum_absolute_error = 0.0_f32;
    let mut split_maximum_relative_error = 0.0_f32;
    let mut graph_maximum_absolute_delta = 0.0_f32;
    let packed_device_bytes;
    let transient_bytes;
    let verifier_device_staging_bytes;
    let verifier_cpu_packed_bytes;
    let q4_tokens;
    let graph_tokenwise_packed_device_bytes;
    let graph_device_position_verified;
    let split_transient_bytes;
    let free_after_prepare;
    let reset_verified;
    let benchmark;
    let mut latency_sweep = Vec::with_capacity(args.latency_contexts.len());
    let graph_replay =
        runtime.verifier_capture_graph_paged_gqa(args.graph_maximum_tokens, args.graph_replays)?;
    let graph_query: Vec<f32> = (0..config.query_heads * config.head_dim)
        .map(|index| (index as f32 * 0.017).sin() * 0.35)
        .collect();
    let graph_key: Vec<f32> = (0..config.key_value_heads * config.head_dim)
        .map(|index| (index as f32 * 0.021).cos() * 0.45)
        .collect();
    let graph_value: Vec<f32> = (0..config.key_value_heads * config.head_dim)
        .map(|index| (index as f32 * 0.015).sin() * 0.55)
        .collect();
    let mut graph_oracle = PagedKvCache::new(
        args.graph_replays,
        config.key_value_heads * config.head_dim,
        128,
        128,
        256,
    )?;
    for _ in 0..args.graph_replays {
        graph_oracle.push(&graph_key, &graph_value)?;
    }
    let graph_expected = grouped_query_attention(
        &graph_query,
        &graph_oracle.flattened_key(config.key_value_heads, config.head_dim)?,
        &graph_oracle.flattened_value(config.key_value_heads, config.head_dim)?,
        config.query_heads,
        config.key_value_heads,
        1,
        args.graph_replays,
        config.head_dim,
        args.graph_replays - 1,
    )?;
    let mut graph_replay_maximum_absolute_error = 0.0_f32;
    for (index, (expected, actual)) in graph_expected
        .iter()
        .zip(&graph_replay.final_output)
        .enumerate()
    {
        let absolute = (expected - actual).abs();
        graph_replay_maximum_absolute_error = graph_replay_maximum_absolute_error.max(absolute);
        anyhow::ensure!(
            absolute <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
            "graph replay output {index}: expected {expected}, got {actual}"
        );
    }
    {
        let mut prepared = runtime.prepare_paged_q2q4_gqa(config)?;
        let mut split = runtime.prepare_paged_q2q4_gqa_split(&prepared)?;
        let graph = runtime.prepare_graph_paged_q2q4_gqa(config)?;
        let graph_split = runtime.prepare_graph_paged_q2q4_gqa_split(&graph)?;
        let graph_state = runtime.prepare_decode_state(0, 0)?;
        let query_staging =
            runtime
                .prepare_verifier_f32_tensor(&vec![0.0; config.query_heads * config.head_dim])?;
        let key_staging = runtime.prepare_verifier_f32_tensor(&vec![
            0.0;
            config.key_value_heads
                * config.head_dim
        ])?;
        let value_staging = runtime.prepare_verifier_f32_tensor(&vec![
            0.0;
            config.key_value_heads
                * config.head_dim
        ])?;
        let split_staging = runtime.prepare_verifier_f32_tensor(&vec![
            0.0;
            args.split_query_tokens
                * config.query_heads
                * config.head_dim
        ])?;
        verifier_device_staging_bytes = query_staging.resident_bytes()
            + key_staging.resident_bytes()
            + value_staging.resident_bytes()
            + split_staging.resident_bytes();
        let mut oracle = PagedKvCache::new(
            config.maximum_tokens,
            config.key_value_heads * config.head_dim,
            config.page_tokens,
            config.sink_tokens,
            config.recent_tokens,
        )?;
        (free_after_prepare, _) = runtime.memory_info()?;
        packed_device_bytes = prepared.packed_device_bytes();
        graph_tokenwise_packed_device_bytes = graph.packed_device_bytes();
        transient_bytes = prepared.transient_bytes();
        split_transient_bytes = split.transient_bytes();
        let mut query_history =
            Vec::with_capacity(args.tokens * config.query_heads * config.head_dim);
        for token in 0..args.tokens {
            let query: Vec<f32> = (0..config.query_heads * config.head_dim)
                .map(|index| ((index + token * 7) as f32 * 0.017).sin() * 0.35)
                .collect();
            query_history.extend_from_slice(&query);
            let key: Vec<f32> = (0..config.key_value_heads * config.head_dim)
                .map(|index| ((index + token * 11) as f32 * 0.021).cos() * 0.45)
                .collect();
            let value: Vec<f32> = (0..config.key_value_heads * config.head_dim)
                .map(|index| ((index + token * 13) as f32 * 0.015).sin() * 0.55)
                .collect();
            query_staging.write(&query)?;
            key_staging.write(&key)?;
            value_staging.write(&value)?;
            let output = runtime.append_and_dispatch_paged_q2q4_gqa_device(
                &mut prepared,
                query_staging.device_view()?,
                key_staging.device_view()?,
                value_staging.device_view()?,
            )?;
            let actual = runtime.verifier_read_f32(output)?;
            let graph_output = runtime.append_and_dispatch_graph_paged_q2q4_gqa_split_device(
                &graph,
                &graph_split,
                &graph_state,
                query_staging.device_view()?,
                key_staging.device_view()?,
                value_staging.device_view()?,
            )?;
            let graph_actual = runtime.verifier_read_f32(graph_output)?;
            for (index, (descriptor, graph_value)) in actual.iter().zip(&graph_actual).enumerate() {
                let absolute = (descriptor - graph_value).abs();
                graph_maximum_absolute_delta = graph_maximum_absolute_delta.max(absolute);
                anyhow::ensure!(
                    absolute
                        <= args.absolute_tolerance
                            + args.relative_tolerance * descriptor.abs(),
                    "token {token} graph output {index}: descriptor {descriptor}, graph {graph_value}"
                );
            }
            runtime.advance_decode_position_device(&graph_state)?;
            oracle.push(&key, &value)?;
            let cached_key = oracle.flattened_key(config.key_value_heads, config.head_dim)?;
            let cached_value = oracle.flattened_value(config.key_value_heads, config.head_dim)?;
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
        let split_query_values = args.split_query_tokens * config.query_heads * config.head_dim;
        let split_query_start = query_history.len() - split_query_values;
        let split_queries = &query_history[split_query_start..];
        let split_actual = runtime.dispatch_paged_q2q4_gqa_split(
            &prepared,
            &mut split,
            split_queries,
            args.split_query_tokens,
        )?;
        split_staging.write(split_queries)?;
        let split_device = runtime.dispatch_paged_q2q4_gqa_split_device(
            &prepared,
            &mut split,
            split_staging.device_view()?,
            args.split_query_tokens,
        )?;
        let split_device_actual = runtime.verifier_read_f32(split_device)?;
        anyhow::ensure!(
            split_actual
                .iter()
                .zip(&split_device_actual)
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "split-KV host-staging and borrowed-device paths differ"
        );

        let cached_key = oracle.flattened_key(config.key_value_heads, config.head_dim)?;
        let cached_value = oracle.flattened_value(config.key_value_heads, config.head_dim)?;
        let mut reference_queries = vec![0.0_f32; split_query_values];
        for head in 0..config.query_heads {
            for query_token in 0..args.split_query_tokens {
                let source = (query_token * config.query_heads + head) * config.head_dim;
                let target = (head * args.split_query_tokens + query_token) * config.head_dim;
                reference_queries[target..target + config.head_dim]
                    .copy_from_slice(&split_queries[source..source + config.head_dim]);
            }
        }
        let split_expected = grouped_query_attention(
            &reference_queries,
            &cached_key,
            &cached_value,
            config.query_heads,
            config.key_value_heads,
            args.split_query_tokens,
            args.tokens,
            config.head_dim,
            args.tokens - args.split_query_tokens,
        )?;
        for (index, (left, right)) in split_expected.iter().zip(&split_actual).enumerate() {
            let absolute = (left - right).abs();
            let relative = absolute / left.abs().max(f32::MIN_POSITIVE);
            split_maximum_absolute_error = split_maximum_absolute_error.max(absolute);
            split_maximum_relative_error = split_maximum_relative_error.max(relative);
            anyhow::ensure!(
                absolute <= args.absolute_tolerance + args.relative_tolerance * left.abs(),
                "split-KV output {index}: expected {left}, got {right}"
            );
        }
        benchmark = runtime.benchmark_paged_q2q4_gqa_split(
            &prepared,
            &mut split,
            split_staging.device_view()?,
            args.split_query_tokens,
            args.benchmark_iterations,
        )?;

        for &context_tokens in &args.latency_contexts {
            let latency_config = CudaPagedGqaConfig {
                query_heads: config.query_heads,
                key_value_heads: config.key_value_heads,
                head_dim: config.head_dim,
                maximum_tokens: context_tokens,
                page_tokens: 128,
                sink_tokens: 128,
                recent_tokens: 256.min(context_tokens),
            };
            let mut latency_cache = runtime.prepare_paged_q2q4_gqa(latency_config)?;
            let mut latency_split = runtime.prepare_paged_q2q4_gqa_split(&latency_cache)?;
            let component_values = latency_config.key_value_heads * latency_config.head_dim;
            let fixture_values = context_tokens
                .checked_mul(component_values)
                .context("latency fixture shape overflows")?;
            let latency_keys: Vec<f32> = (0..fixture_values)
                .map(|index| ((index.wrapping_mul(13) % 257) as f32 - 128.0) * 0.0015)
                .collect();
            let latency_values: Vec<f32> = (0..fixture_values)
                .map(|index| ((index.wrapping_mul(29) % 263) as f32 - 131.0) * 0.00125)
                .collect();
            runtime.seed_paged_q2q4_gqa_verifier(
                &mut latency_cache,
                &latency_keys,
                &latency_values,
            )?;
            drop(latency_keys);
            drop(latency_values);

            let latency_query_values =
                args.split_query_tokens * latency_config.query_heads * latency_config.head_dim;
            let latency_queries: Vec<f32> = (0..latency_query_values)
                .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) * 0.00175)
                .collect();
            let latency_staging = runtime.prepare_verifier_f32_tensor(&latency_queries)?;
            let latency_benchmark = runtime.benchmark_paged_q2q4_gqa_split(
                &latency_cache,
                &mut latency_split,
                latency_staging.device_view()?,
                args.split_query_tokens,
                args.benchmark_iterations,
            )?;
            latency_sweep.push(LatencyPoint {
                context_tokens,
                packed_device_bytes: latency_cache.packed_device_bytes(),
                split_transient_bytes: latency_split.transient_bytes(),
                q2_tokens: latency_cache.q2_tokens(),
                q4_tokens: latency_cache.q4_tokens(),
                iterations: latency_benchmark.iterations,
                sequential_full_context_microseconds: latency_benchmark
                    .sequential_full_context_microseconds,
                split_causal_microseconds: latency_benchmark.split_causal_microseconds,
                split_speedup: latency_benchmark.speedup,
            });
        }
        verifier_cpu_packed_bytes = oracle.packed_bytes();
        q4_tokens = prepared.q4_tokens();
        let (graph_token, graph_position) = runtime.verifier_read_decode_state(&graph_state)?;
        graph_device_position_verified = graph_token == 0 && graph_position == args.tokens as u64;
        anyhow::ensure!(
            graph_device_position_verified,
            "graph CUDA decode state ended at token {graph_token}, position {graph_position}, expected token 0 and position {}",
            args.tokens
        );
        anyhow::ensure!(
            prepared.q2_tokens() > 0 && q4_tokens == oracle.q4_tokens(),
            "device page demotion differs from the canonical CPU policy"
        );
        prepared.reset()?;
        oracle.reset();
        reset_verified = prepared.tokens() == 0
            && prepared.q4_tokens() == 0
            && prepared.q2_tokens() == 0
            && oracle.packed_bytes() == 0;
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
            format: "ctox.qwen38.cuda_paged_q2q4_gqa_verification.v4",
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
            verifier_device_staging_bytes,
            verifier_cpu_packed_bytes,
            q4_tokens,
            graph_tokenwise_packed_device_bytes,
            graph_packed_device_bytes: graph_replay.packed_device_bytes,
            graph_target_16_layer_packed_device_bytes: graph_replay
                .packed_device_bytes
                .checked_mul(16)
                .context("16-layer graph KV bytes overflow")?,
            graph_target_mtp_17_layer_packed_device_bytes: graph_replay
                .packed_device_bytes
                .checked_mul(17)
                .context("17-layer graph KV bytes overflow")?,
            graph_maximum_absolute_delta,
            graph_device_position_verified,
            graph_maximum_tokens: args.graph_maximum_tokens,
            graph_replays: graph_replay.iterations,
            graph_microseconds_per_replay: graph_replay.microseconds_per_replay,
            graph_replay_maximum_absolute_error,
            maximum_absolute_error,
            maximum_relative_error,
            split_query_tokens: args.split_query_tokens,
            split_segments: PAGED_GQA_SPLIT_SEGMENTS,
            split_transient_bytes,
            split_maximum_absolute_error,
            split_maximum_relative_error,
            split_device_view_path_verified: true,
            split_tail_causality_verified: true,
            benchmark_iterations: benchmark.iterations,
            sequential_full_context_microseconds: benchmark
                .sequential_full_context_microseconds,
            split_causal_microseconds: benchmark.split_causal_microseconds,
            split_speedup: benchmark.speedup,
            latency_sweep,
            device_view_path_verified: true,
            demotion_verified: true,
            reset_verified,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            note: "Q/K/V enter the GQA dispatcher as context-bound device views and its output is read back only by the explicit verifier API. The CPU cache exists solely as the external numerical oracle; CUDA packs Q4 and demotes Q4-to-Q2 entirely on device with no host packed-cache mirror. The graph candidate stores logical Q2 pages plus a fixed sink/recent Q4 ring, derives both precision and physical Q4 slot from a device-resident position, performs append/demotion/attention without host metadata mutation or synchronization, and is compared token-by-token with the canonical descriptor path. The isolated one-to-five-query path splits canonical mixed Q2/Q4 KV across 16 partial blocks per head, combines online-softmax state on device, and verifies tail causality against the scalar oracle before scheduler integration. The latency comparison gives both paths one final context barrier; its sequential baseline conservatively exposes the full cache to every query, while the split path applies exact per-tail-query causality.",
        })?
    );
    Ok(())
}
