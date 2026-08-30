use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaPagedGqaConfig};
use ctox_qwen38_27b::kv_cache::PagedKvCache;
use ctox_qwen38_27b::reference::grouped_query_attention;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify causal CUDA paged-GQA prefill against decode and scalar oracles")]
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
    all_q4_prefill_decode_exact: bool,
    maximum_all_q4_prefill_decode_absolute_delta: f32,
    all_q4_batch_page_launches: usize,
    mixed_q2_tokens: usize,
    mixed_q4_tokens: usize,
    mixed_batch_page_launches: usize,
    graph_prefill_matches_sequential_decode: bool,
    maximum_graph_prefill_decode_absolute_delta: f32,
    graph_prefill_committed_tokens: usize,
    maximum_mixed_oracle_absolute_error: f32,
    maximum_mixed_oracle_relative_error: f32,
    verifier_allocated_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (24..=512).contains(&args.tokens),
        "tokens must be within 24..=512"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let query_heads = 24;
    let key_value_heads = 4;
    let head_dim = 256;
    let query_values = query_heads * head_dim;
    let component_values = key_value_heads * head_dim;
    let queries: Vec<f32> = (0..args.tokens * query_values)
        .map(|index| ((index.wrapping_mul(7) % 521) as f32 - 260.0) * 0.0017)
        .collect();
    let keys: Vec<f32> = (0..args.tokens * component_values)
        .map(|index| ((index.wrapping_mul(11) % 509) as f32 - 254.0) * 0.0019)
        .collect();
    let values: Vec<f32> = (0..args.tokens * component_values)
        .map(|index| ((index.wrapping_mul(13) % 503) as f32 - 251.0) * 0.0021)
        .collect();
    let query_staging = runtime.prepare_verifier_f32_tensor(&queries)?;
    let key_staging = runtime.prepare_verifier_f32_tensor(&keys)?;
    let value_staging = runtime.prepare_verifier_f32_tensor(&values)?;

    let all_q4_config = CudaPagedGqaConfig {
        query_heads,
        key_value_heads,
        head_dim,
        maximum_tokens: args.tokens,
        page_tokens: 8,
        sink_tokens: 8,
        recent_tokens: args.tokens,
    };
    let mut sequential = runtime.prepare_paged_q2q4_gqa(all_q4_config)?;
    let mut sequential_outputs = Vec::with_capacity(args.tokens * query_values);
    for token in 0..args.tokens {
        let query_start = token * query_values;
        let component_start = token * component_values;
        sequential_outputs.extend(runtime.append_and_dispatch_paged_q2q4_gqa(
            &mut sequential,
            &queries[query_start..query_start + query_values],
            &keys[component_start..component_start + component_values],
            &values[component_start..component_start + component_values],
        )?);
    }
    let mut all_q4_prefill = runtime.prepare_paged_q2q4_gqa(all_q4_config)?;
    let all_q4_batch_page_launches =
        runtime.seed_paged_q2q4_gqa_batch_verifier(&mut all_q4_prefill, &keys, &values)?;
    anyhow::ensure!(
        all_q4_batch_page_launches == args.tokens.div_ceil(all_q4_config.page_tokens),
        "all-Q4 batch packer did not submit exactly one launch per page"
    );
    let all_q4_output =
        runtime.prepare_paged_q2q4_gqa_prefill_output(all_q4_config, args.tokens)?;
    let all_q4_view = runtime.dispatch_paged_q2q4_gqa_prefill_device(
        &all_q4_prefill,
        &all_q4_output,
        query_staging.device_view()?,
        args.tokens,
    )?;
    let all_q4_actual = runtime.verifier_read_f32(all_q4_view)?;
    let (maximum_all_q4_prefill_decode_absolute_delta, _) =
        compare(&sequential_outputs, &all_q4_actual);
    let all_q4_prefill_decode_exact = sequential_outputs == all_q4_actual;
    anyhow::ensure!(
        all_q4_prefill_decode_exact,
        "all-Q4 causal prefill differs from sequential decode"
    );

    let mixed_config = CudaPagedGqaConfig {
        query_heads,
        key_value_heads,
        head_dim,
        maximum_tokens: args.tokens,
        page_tokens: 8,
        sink_tokens: 8,
        recent_tokens: 8,
    };
    let mut mixed = runtime.prepare_paged_q2q4_gqa(mixed_config)?;
    let mut mixed_sequential = runtime.prepare_paged_q2q4_gqa(mixed_config)?;
    let mut mixed_sequential_outputs = Vec::with_capacity(args.tokens * query_values);
    for token in 0..args.tokens {
        let query_start = token * query_values;
        let component_start = token * component_values;
        mixed_sequential_outputs.extend(runtime.append_and_dispatch_paged_q2q4_gqa(
            &mut mixed_sequential,
            &queries[query_start..query_start + query_values],
            &keys[component_start..component_start + component_values],
            &values[component_start..component_start + component_values],
        )?);
    }
    let mixed_batch_page_launches =
        runtime.seed_paged_q2q4_gqa_batch_verifier(&mut mixed, &keys, &values)?;
    anyhow::ensure!(
        mixed_batch_page_launches == args.tokens.div_ceil(mixed_config.page_tokens),
        "mixed batch packer did not submit exactly one launch per page"
    );
    anyhow::ensure!(mixed.q2_tokens() > 0, "mixed verifier produced no Q2 pages");
    let mixed_output = runtime.prepare_paged_q2q4_gqa_prefill_output(mixed_config, args.tokens)?;
    let mixed_view = runtime.dispatch_paged_q2q4_gqa_prefill_device(
        &mixed,
        &mixed_output,
        query_staging.device_view()?,
        args.tokens,
    )?;
    let mixed_actual = runtime.verifier_read_f32(mixed_view)?;

    let mut graph_mixed = runtime.prepare_graph_paged_q2q4_gqa(mixed_config)?;
    let graph_output = runtime.prepare_paged_q2q4_gqa_prefill_output(mixed_config, args.tokens)?;
    let graph_view = runtime.append_and_dispatch_graph_paged_q2q4_gqa_prefill_device(
        &mut graph_mixed,
        &graph_output,
        query_staging.device_view()?,
        key_staging.device_view()?,
        value_staging.device_view()?,
        args.tokens,
    )?;
    let graph_actual = runtime.verifier_read_f32(graph_view)?;
    let (maximum_graph_prefill_decode_absolute_delta, _) =
        compare(&mixed_sequential_outputs, &graph_actual);
    let graph_prefill_matches_sequential_decode = mixed_sequential_outputs == graph_actual;
    anyhow::ensure!(
        graph_prefill_matches_sequential_decode,
        "graph-ring causal prefill differs from sequential descriptor decode (max abs {maximum_graph_prefill_decode_absolute_delta})"
    );

    let mut oracle = PagedKvCache::new(
        mixed_config.maximum_tokens,
        component_values,
        mixed_config.page_tokens,
        mixed_config.sink_tokens,
        mixed_config.recent_tokens,
    )?;
    for token in 0..args.tokens {
        let start = token * component_values;
        oracle.push(
            &keys[start..start + component_values],
            &values[start..start + component_values],
        )?;
    }
    let cached_key = oracle.flattened_key(key_value_heads, head_dim)?;
    let cached_value = oracle.flattened_value(key_value_heads, head_dim)?;
    let mut head_major_queries = vec![0.0_f32; queries.len()];
    for token in 0..args.tokens {
        for head in 0..query_heads {
            let source = (token * query_heads + head) * head_dim;
            let target = (head * args.tokens + token) * head_dim;
            head_major_queries[target..target + head_dim]
                .copy_from_slice(&queries[source..source + head_dim]);
        }
    }
    let mixed_expected = grouped_query_attention(
        &head_major_queries,
        &cached_key,
        &cached_value,
        query_heads,
        key_value_heads,
        args.tokens,
        args.tokens,
        head_dim,
        0,
    )?;
    let (maximum_mixed_oracle_absolute_error, maximum_mixed_oracle_relative_error) =
        compare(&mixed_expected, &mixed_actual);
    for (index, (expected, actual)) in mixed_expected.iter().zip(&mixed_actual).enumerate() {
        anyhow::ensure!(
            (expected - actual).abs()
                <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
            "mixed prefill output {index} differs from scalar oracle"
        );
    }

    let verifier_allocated_bytes = query_staging.resident_bytes()
        + key_staging.resident_bytes()
        + value_staging.resident_bytes()
        + sequential.packed_device_bytes()
        + sequential.transient_bytes()
        + all_q4_prefill.packed_device_bytes()
        + all_q4_prefill.transient_bytes()
        + all_q4_output.transient_bytes()
        + mixed.packed_device_bytes()
        + mixed.transient_bytes()
        + mixed_output.transient_bytes()
        + mixed_sequential.packed_device_bytes()
        + mixed_sequential.transient_bytes()
        + graph_mixed.packed_device_bytes()
        + graph_output.transient_bytes();
    let mixed_q2_tokens = mixed.q2_tokens();
    let mixed_q4_tokens = mixed.q4_tokens();
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(query_staging);
    drop(key_staging);
    drop(value_staging);
    drop(sequential);
    drop(all_q4_prefill);
    drop(all_q4_output);
    drop(mixed);
    drop(mixed_output);
    drop(mixed_sequential);
    let graph_prefill_committed_tokens = graph_mixed.tokens();
    drop(graph_mixed);
    drop(graph_output);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        observed_reclaimed_bytes >= verifier_allocated_bytes,
        "CUDA paged-GQA prefill verifier did not reclaim all owned allocations"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-paged-gqa-prefill-verifier.v1",
            status: "pass_verifier_only",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            tokens: args.tokens,
            query_heads,
            key_value_heads,
            head_dim,
            all_q4_prefill_decode_exact,
            maximum_all_q4_prefill_decode_absolute_delta,
            all_q4_batch_page_launches,
            mixed_q2_tokens,
            mixed_q4_tokens,
            mixed_batch_page_launches,
            graph_prefill_matches_sequential_decode,
            maximum_graph_prefill_decode_absolute_delta,
            graph_prefill_committed_tokens,
            maximum_mixed_oracle_absolute_error,
            maximum_mixed_oracle_relative_error,
            verifier_allocated_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "The graph-ring causal prefill writes directly into decode's logical-Q2/fixed-Q4-ring cache and is compared against sequential descriptor decode. The production-shaped KV packer consumes device K/V views and submits page-bounded two-dimensional launches, never one host dispatch per token. Controlled full-model latency remains required.",
        })?
    );
    Ok(())
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
