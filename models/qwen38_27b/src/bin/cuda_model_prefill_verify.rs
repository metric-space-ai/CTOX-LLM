use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_graph::PreparedCudaProjectionGraph;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaSubmissionStats};
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use ctox_qwen38_27b::{config::LayerKind, Qwen38Config};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Compare complete sequential and batched Qwen target prefill on CUDA SM86")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    /// Comma-delimited tokenizer IDs. One complete prompt chunk is verified.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    token_ids: Vec<u32>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 4_096)]
    maximum_context_tokens: usize,
    #[arg(long, default_value_t = 2.0e-3)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 1.0e-3)]
    relative_tolerance: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SubmissionDelta {
    token_submission_attempts: u64,
    token_submission_commits: u64,
    deferred_operator_synchronizations: u64,
    context_synchronizations: u64,
}

#[derive(Debug, Eq, PartialEq)]
struct StateEvidence {
    linear_f16le_sha256: String,
    attention_metadata_sha256: String,
    attention_layers: usize,
    cached_tokens_per_attention_layer: usize,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    artifact_manifest_sha256: String,
    token_ids: Vec<u32>,
    tokens: usize,
    maximum_context_tokens: usize,
    sequential_logits_f32le_sha256: String,
    batched_logits_f32le_sha256: String,
    logits_bit_exact: bool,
    maximum_absolute_logit_error: f32,
    maximum_relative_logit_error: f32,
    sequential_greedy_token: u32,
    batched_greedy_token: u32,
    linear_state_f16le_sha256: String,
    linear_state_matches_bit_exact: bool,
    attention_metadata_sha256: String,
    attention_metadata_matches_bit_exact: bool,
    attention_layers: usize,
    cached_tokens_per_attention_layer: usize,
    sequential_submissions: SubmissionDelta,
    batched_submissions: SubmissionDelta,
    synchronization_reduction: u64,
    graph_prepare_milliseconds: f64,
    sequential_milliseconds: f64,
    batched_milliseconds: f64,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_allocation_bytes: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (1..=512).contains(&args.token_ids.len()),
        "token_ids must contain one complete chunk in 1..=512"
    );
    anyhow::ensure!(
        args.maximum_context_tokens >= args.token_ids.len(),
        "maximum context does not admit the prompt"
    );
    anyhow::ensure!(
        args.token_ids
            .iter()
            .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE),
        "prompt contains a token outside the tokenizer vocabulary"
    );
    anyhow::ensure!(
        args.absolute_tolerance.is_finite()
            && args.absolute_tolerance >= 0.0
            && args.relative_tolerance.is_finite()
            && args.relative_tolerance >= 0.0,
        "logit tolerances must be finite and non-negative"
    );

    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let config = Qwen38Config::default();
    let prepare_started = Instant::now();
    let mut graph = PreparedCudaProjectionGraph::prepare(
        &runtime,
        &artifact,
        &config,
        args.maximum_context_tokens,
        None,
    )?;
    let graph_prepare_milliseconds = prepare_started.elapsed().as_secs_f64() * 1.0e3;
    let (free_after_prepare, _) = runtime.memory_info()?;

    let sequential_stats_before = runtime.submission_stats();
    let sequential_started = Instant::now();
    let mut sequential_logits = Vec::new();
    for (position, token) in args.token_ids.iter().copied().enumerate() {
        let logits =
            graph.dispatch_target_token_device(&runtime, &config, token as usize, position)?;
        if position + 1 == args.token_ids.len() {
            sequential_logits = runtime.verifier_read_f32_device(logits)?;
        }
    }
    let sequential_milliseconds = sequential_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_logits(&sequential_logits, "sequential")?;
    let sequential_state = capture_state(&mut graph, &config)?;
    let sequential_stats = submission_delta(sequential_stats_before, runtime.submission_stats())?;
    anyhow::ensure!(
        graph.target_tokens() == args.token_ids.len()
            && sequential_stats.token_submission_attempts == args.token_ids.len() as u64
            && sequential_stats.token_submission_commits == args.token_ids.len() as u64,
        "sequential path did not commit exactly one transaction per token"
    );

    graph.reset_session()?;
    anyhow::ensure!(
        graph.target_tokens() == 0 && graph.mtp_tokens() == 0,
        "session reset retained token state"
    );

    let batched_stats_before = runtime.submission_stats();
    let batched_started = Instant::now();
    let batched_view = graph.dispatch_target_prefill_chunk_without_mtp_device(
        &runtime,
        &config,
        &args.token_ids,
        0,
    )?;
    let batched_logits = runtime.verifier_read_f32_device(batched_view)?;
    let batched_milliseconds = batched_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_logits(&batched_logits, "batched")?;
    let batched_state = capture_state(&mut graph, &config)?;
    let batched_stats = submission_delta(batched_stats_before, runtime.submission_stats())?;
    anyhow::ensure!(
        graph.target_tokens() == args.token_ids.len()
            && batched_stats.token_submission_attempts == 1
            && batched_stats.token_submission_commits == 1,
        "batched path did not commit the chunk as one transaction"
    );

    let (maximum_absolute_logit_error, maximum_relative_logit_error) = compare_logits(
        &sequential_logits,
        &batched_logits,
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    let sequential_greedy_token = greedy_token(&sequential_logits)?;
    let batched_greedy_token = greedy_token(&batched_logits)?;
    anyhow::ensure!(
        sequential_greedy_token == batched_greedy_token,
        "batched greedy token {batched_greedy_token} differs from sequential token {sequential_greedy_token}"
    );
    let linear_state_matches_bit_exact =
        sequential_state.linear_f16le_sha256 == batched_state.linear_f16le_sha256;
    anyhow::ensure!(
        linear_state_matches_bit_exact,
        "batched linear recurrence/convolution state differs from sequential execution"
    );
    let attention_metadata_matches_bit_exact =
        sequential_state.attention_metadata_sha256 == batched_state.attention_metadata_sha256;
    anyhow::ensure!(
        attention_metadata_matches_bit_exact,
        "batched full-attention cache metadata differs from sequential execution"
    );
    anyhow::ensure!(
        batched_stats.context_synchronizations < sequential_stats.context_synchronizations,
        "batched prefill did not reduce CUDA context barriers"
    );

    let sequential_logits_f32le_sha256 = digest_f32(&sequential_logits);
    let batched_logits_f32le_sha256 = digest_f32(&batched_logits);
    let logits_bit_exact = sequential_logits == batched_logits;
    let synchronization_reduction = sequential_stats
        .context_synchronizations
        .saturating_sub(batched_stats.context_synchronizations);
    graph.reset_session()?;
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA model prefill graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-model-prefill.v1",
            status: "full_model_verifier_only_not_promoted",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            token_ids: args.token_ids,
            tokens: sequential_state.cached_tokens_per_attention_layer,
            maximum_context_tokens: args.maximum_context_tokens,
            sequential_logits_f32le_sha256,
            batched_logits_f32le_sha256,
            logits_bit_exact,
            maximum_absolute_logit_error,
            maximum_relative_logit_error,
            sequential_greedy_token,
            batched_greedy_token,
            linear_state_f16le_sha256: sequential_state.linear_f16le_sha256,
            linear_state_matches_bit_exact,
            attention_metadata_sha256: sequential_state.attention_metadata_sha256,
            attention_metadata_matches_bit_exact,
            attention_layers: sequential_state.attention_layers,
            cached_tokens_per_attention_layer: sequential_state.cached_tokens_per_attention_layer,
            sequential_submissions: sequential_stats,
            batched_submissions: batched_stats,
            synchronization_reduction,
            graph_prepare_milliseconds,
            sequential_milliseconds,
            batched_milliseconds,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes: free_before_prepare.saturating_sub(free_after_prepare),
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            note: "Runs the same complete 64-layer target model first through sequential token transactions and then through one 645-step target-only prefill chunk after an explicit reset. It compares final logits under declared tolerances, the greedy decision, bit-exact linear recurrence/convolution state, every full-attention layer's cache precision metadata, synchronization counts, and allocator reclamation. MTP batching, BF16 quality, throughput promotion, and multi-chunk boundary equivalence remain separate gates.",
        })?
    );
    Ok(())
}

fn capture_state(
    graph: &mut PreparedCudaProjectionGraph,
    config: &Qwen38Config,
) -> anyhow::Result<StateEvidence> {
    let mut linear = Sha256::new();
    let mut attention = Sha256::new();
    let mut attention_layers = 0_usize;
    let mut cached_tokens_per_attention_layer = None;
    for layer in 0..config.num_hidden_layers {
        match config
            .layer_kind(layer)
            .context("frozen layer kind is missing")?
        {
            LayerKind::LinearAttention => {
                linear.update((layer as u64).to_le_bytes());
                let mixer = graph.linear_mixer_mut(layer)?;
                for value in mixer.convolution_mut().verifier_read_state()? {
                    linear.update(value.to_bits().to_le_bytes());
                }
                for value in mixer.recurrence_mut().verifier_read_state()? {
                    linear.update(value.to_bits().to_le_bytes());
                }
            }
            LayerKind::FullAttention => {
                attention_layers += 1;
                let cached = {
                    let kv = graph
                        .full_attention_mut(&format!("target:{layer}"))?
                        .kv_mut();
                    let cached = kv.tokens();
                    attention.update((layer as u64).to_le_bytes());
                    attention.update((cached as u64).to_le_bytes());
                    attention.update((kv.q2_tokens() as u64).to_le_bytes());
                    attention.update((kv.q4_tokens() as u64).to_le_bytes());
                    cached
                };
                match cached_tokens_per_attention_layer {
                    Some(expected) => anyhow::ensure!(
                        cached == expected,
                        "full-attention layers committed different token counts"
                    ),
                    None => cached_tokens_per_attention_layer = Some(cached),
                }
            }
        }
    }
    anyhow::ensure!(
        graph.target_tokens() == cached_tokens_per_attention_layer.unwrap_or(0),
        "target token counter differs from full-attention cache metadata"
    );
    Ok(StateEvidence {
        linear_f16le_sha256: format!("{:x}", linear.finalize()),
        attention_metadata_sha256: format!("{:x}", attention.finalize()),
        attention_layers,
        cached_tokens_per_attention_layer: cached_tokens_per_attention_layer.unwrap_or(0),
    })
}

fn submission_delta(
    before: CudaSubmissionStats,
    after: CudaSubmissionStats,
) -> anyhow::Result<SubmissionDelta> {
    Ok(SubmissionDelta {
        token_submission_attempts: checked_counter_delta(
            before.token_submission_attempts,
            after.token_submission_attempts,
            "submission attempts",
        )?,
        token_submission_commits: checked_counter_delta(
            before.token_submission_commits,
            after.token_submission_commits,
            "submission commits",
        )?,
        deferred_operator_synchronizations: checked_counter_delta(
            before.deferred_operator_synchronizations,
            after.deferred_operator_synchronizations,
            "deferred synchronizations",
        )?,
        context_synchronizations: checked_counter_delta(
            before.context_synchronizations,
            after.context_synchronizations,
            "context synchronizations",
        )?,
    })
}

fn checked_counter_delta(before: u64, after: u64, label: &str) -> anyhow::Result<u64> {
    after
        .checked_sub(before)
        .with_context(|| format!("CUDA {label} counter regressed"))
}

fn ensure_logits(logits: &[f32], label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        logits.len() == Qwen38Config::default().vocab_size
            && logits.iter().all(|value| value.is_finite()),
        "{label} logits are not a finite model-vocabulary distribution"
    );
    Ok(())
}

fn compare_logits(
    expected: &[f32],
    actual: &[f32],
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> anyhow::Result<(f32, f32)> {
    anyhow::ensure!(expected.len() == actual.len(), "logit lengths differ");
    let mut maximum_absolute = 0.0_f32;
    let mut maximum_relative = 0.0_f32;
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let absolute = (expected - actual).abs();
        let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
        maximum_absolute = maximum_absolute.max(absolute);
        maximum_relative = maximum_relative.max(relative);
        anyhow::ensure!(
            absolute <= absolute_tolerance + relative_tolerance * expected.abs(),
            "logit {index} differs: expected={expected}, actual={actual}, absolute={absolute}"
        );
    }
    Ok((maximum_absolute, maximum_relative))
}

fn greedy_token(logits: &[f32]) -> anyhow::Result<u32> {
    logits[..TOKENIZER_VOCAB_SIZE]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(token, _)| token as u32)
        .context("logits are empty")
}

fn digest_f32(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_logit_tolerance_scales_with_reference() {
        let (absolute, relative) = compare_logits(&[100.0], &[100.05], 0.0, 0.001).unwrap();
        assert!((absolute - 0.05).abs() < 1.0e-5);
        assert!((relative - 0.0005).abs() < 1.0e-6);
    }

    #[test]
    fn counter_delta_rejects_regression() {
        assert!(checked_counter_delta(2, 1, "fixture").is_err());
    }
}
