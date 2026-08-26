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
#[command(about = "Compare complete sequential and chunked Qwen prefill on CUDA SM86")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    /// Comma-delimited tokenizer IDs. Prompts longer than the prepared chunk
    /// capacity exercise the cross-chunk target/MTP state boundary.
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
    #[arg(long)]
    mtp_enabled: bool,
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
    mtp_cached_tokens: usize,
}

#[derive(Default)]
struct ContinuationEvidence {
    input_token: Option<u32>,
    mtp_logits: Option<Vec<f32>>,
    target_logits: Option<Vec<f32>>,
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
    chunks: usize,
    maximum_chunk_tokens: usize,
    maximum_context_tokens: usize,
    mtp_enabled: bool,
    sequential_logits_f32le_sha256: String,
    batched_logits_f32le_sha256: String,
    logits_bit_exact: bool,
    maximum_absolute_logit_error: f32,
    maximum_relative_logit_error: f32,
    sequential_greedy_token: u32,
    batched_greedy_token: u32,
    continuation_input_token: Option<u32>,
    sequential_mtp_continuation_f32le_sha256: Option<String>,
    batched_mtp_continuation_f32le_sha256: Option<String>,
    maximum_absolute_mtp_continuation_error: Option<f32>,
    sequential_target_continuation_f32le_sha256: Option<String>,
    batched_target_continuation_f32le_sha256: Option<String>,
    maximum_absolute_target_continuation_error: Option<f32>,
    linear_state_f16le_sha256: String,
    linear_state_matches_bit_exact: bool,
    attention_metadata_sha256: String,
    attention_metadata_matches_bit_exact: bool,
    attention_layers: usize,
    cached_tokens_per_attention_layer: usize,
    mtp_cached_tokens: usize,
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
        (2..=args.maximum_context_tokens).contains(&args.token_ids.len()),
        "token_ids must contain 2..=maximum_context_tokens tokens"
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
    let maximum_chunk_tokens = graph.prefill_workspaces().max_chunk_tokens();
    let chunks = args.token_ids.len().div_ceil(maximum_chunk_tokens);

    let sequential_stats_before = runtime.submission_stats();
    let sequential_started = Instant::now();
    let mut sequential_logits = Vec::new();
    for (position, token) in args.token_ids.iter().copied().enumerate() {
        if args.mtp_enabled && position > 0 {
            let _ = graph.dispatch_mtp_draft_device(&runtime, &config, token as usize, position)?;
        }
        let logits =
            graph.dispatch_target_token_device(&runtime, &config, token as usize, position)?;
        if position + 1 == args.token_ids.len() {
            sequential_logits = runtime.verifier_read_f32_device(logits)?;
        }
    }
    let sequential_milliseconds = sequential_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_logits(&sequential_logits, "sequential")?;
    let sequential_continuation = dispatch_continuation(
        &runtime,
        &mut graph,
        &config,
        &sequential_logits,
        args.mtp_enabled,
    )?;
    let sequential_state = capture_state(&mut graph, &config)?;
    let sequential_stats = submission_delta(sequential_stats_before, runtime.submission_stats())?;
    let expected_sequential_submissions = args.token_ids.len()
        + usize::from(args.mtp_enabled) * (args.token_ids.len().saturating_sub(1) + 2);
    let expected_target_tokens = args.token_ids.len() + usize::from(args.mtp_enabled);
    let expected_mtp_tokens = usize::from(args.mtp_enabled) * args.token_ids.len();
    anyhow::ensure!(
        graph.target_tokens() == expected_target_tokens
            && graph.mtp_tokens() == expected_mtp_tokens
            && sequential_stats.token_submission_attempts == expected_sequential_submissions as u64
            && sequential_stats.token_submission_commits == expected_sequential_submissions as u64,
        "sequential path did not commit the exact target/MTP transactions"
    );

    graph.reset_session()?;
    anyhow::ensure!(
        graph.target_tokens() == 0 && graph.mtp_tokens() == 0,
        "session reset retained token state"
    );

    let batched_stats_before = runtime.submission_stats();
    let batched_started = Instant::now();
    let mut batched_logits = Vec::new();
    for chunk in args.token_ids.chunks(maximum_chunk_tokens) {
        let start_position = graph.target_tokens();
        let batched_view = if args.mtp_enabled {
            graph.dispatch_target_prefill_chunk_with_mtp_device(
                &runtime,
                &config,
                chunk,
                start_position,
            )?
        } else {
            graph.dispatch_target_prefill_chunk_without_mtp_device(
                &runtime,
                &config,
                chunk,
                start_position,
            )?
        };
        batched_logits = runtime.verifier_read_f32_device(batched_view)?;
    }
    let batched_milliseconds = batched_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_logits(&batched_logits, "batched")?;
    let batched_continuation = dispatch_continuation(
        &runtime,
        &mut graph,
        &config,
        &batched_logits,
        args.mtp_enabled,
    )?;
    let batched_state = capture_state(&mut graph, &config)?;
    let batched_stats = submission_delta(batched_stats_before, runtime.submission_stats())?;
    anyhow::ensure!(
        graph.target_tokens() == expected_target_tokens
            && graph.mtp_tokens() == expected_mtp_tokens
            && batched_stats.token_submission_attempts
                == chunks as u64 + 2 * u64::from(args.mtp_enabled)
            && batched_stats.token_submission_commits
                == chunks as u64 + 2 * u64::from(args.mtp_enabled),
        "batched path did not commit each chunk as one transaction"
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
    anyhow::ensure!(
        sequential_continuation.input_token == batched_continuation.input_token,
        "batched continuation selected another input token"
    );
    let (
        sequential_mtp_continuation_f32le_sha256,
        batched_mtp_continuation_f32le_sha256,
        maximum_absolute_mtp_continuation_error,
    ) = compare_optional_logits(
        "MTP continuation",
        sequential_continuation.mtp_logits.as_deref(),
        batched_continuation.mtp_logits.as_deref(),
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
    let (
        sequential_target_continuation_f32le_sha256,
        batched_target_continuation_f32le_sha256,
        maximum_absolute_target_continuation_error,
    ) = compare_optional_logits(
        "target continuation",
        sequential_continuation.target_logits.as_deref(),
        batched_continuation.target_logits.as_deref(),
        args.absolute_tolerance,
        args.relative_tolerance,
    )?;
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
    let prompt_tokens = args.token_ids.len();

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
            tokens: prompt_tokens,
            chunks,
            maximum_chunk_tokens,
            maximum_context_tokens: args.maximum_context_tokens,
            mtp_enabled: args.mtp_enabled,
            sequential_logits_f32le_sha256,
            batched_logits_f32le_sha256,
            logits_bit_exact,
            maximum_absolute_logit_error,
            maximum_relative_logit_error,
            sequential_greedy_token,
            batched_greedy_token,
            continuation_input_token: sequential_continuation.input_token,
            sequential_mtp_continuation_f32le_sha256,
            batched_mtp_continuation_f32le_sha256,
            maximum_absolute_mtp_continuation_error,
            sequential_target_continuation_f32le_sha256,
            batched_target_continuation_f32le_sha256,
            maximum_absolute_target_continuation_error,
            linear_state_f16le_sha256: sequential_state.linear_f16le_sha256,
            linear_state_matches_bit_exact,
            attention_metadata_sha256: sequential_state.attention_metadata_sha256,
            attention_metadata_matches_bit_exact,
            attention_layers: sequential_state.attention_layers,
            cached_tokens_per_attention_layer: sequential_state.cached_tokens_per_attention_layer,
            mtp_cached_tokens: sequential_state.mtp_cached_tokens,
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
            note: "Runs the same complete 64-layer target and optional causally shifted MTP prompt state first through sequential token transactions and then through one or more bounded 645-step layer-major chunks after an explicit reset. Prompts above the prepared chunk capacity exercise the retained target-hidden MTP boundary. The verifier compares final target logits, the greedy decision, the next MTP and target continuation logits, bit-exact target linear recurrence/convolution state, target and MTP cache precision metadata, synchronization counts, and allocator reclamation. BF16 quality and throughput promotion remain separate gates.",
        })?
    );
    Ok(())
}

fn dispatch_continuation(
    runtime: &CudaCandidateRuntime,
    graph: &mut PreparedCudaProjectionGraph,
    config: &Qwen38Config,
    prompt_logits: &[f32],
    enabled: bool,
) -> anyhow::Result<ContinuationEvidence> {
    if !enabled {
        return Ok(ContinuationEvidence::default());
    }
    let input_token = greedy_token(prompt_logits)?;
    let position = graph.target_tokens();
    let mtp_view =
        graph.dispatch_mtp_draft_device(runtime, config, input_token as usize, position)?;
    let mtp_logits = runtime.verifier_read_f32_device(mtp_view)?;
    ensure_logits(&mtp_logits, "MTP continuation")?;
    let target_view =
        graph.dispatch_target_token_device(runtime, config, input_token as usize, position)?;
    let target_logits = runtime.verifier_read_f32_device(target_view)?;
    ensure_logits(&target_logits, "target continuation")?;
    Ok(ContinuationEvidence {
        input_token: Some(input_token),
        mtp_logits: Some(mtp_logits),
        target_logits: Some(target_logits),
    })
}

fn compare_optional_logits(
    label: &str,
    expected: Option<&[f32]>,
    actual: Option<&[f32]>,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) -> anyhow::Result<(Option<String>, Option<String>, Option<f32>)> {
    match (expected, actual) {
        (None, None) => Ok((None, None, None)),
        (Some(expected), Some(actual)) => {
            let (maximum_absolute, _) =
                compare_logits(expected, actual, absolute_tolerance, relative_tolerance)?;
            let expected_greedy = greedy_token(expected)?;
            let actual_greedy = greedy_token(actual)?;
            anyhow::ensure!(
                expected_greedy == actual_greedy,
                "{label} greedy token differs: {expected_greedy}/{actual_greedy}"
            );
            Ok((
                Some(digest_f32(expected)),
                Some(digest_f32(actual)),
                Some(maximum_absolute),
            ))
        }
        _ => anyhow::bail!("{label} exists on only one execution path"),
    }
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
    let mtp_cached_tokens = {
        let kv = graph.full_attention_mut("mtp:0")?.kv_mut();
        attention.update(u64::MAX.to_le_bytes());
        attention.update((kv.tokens() as u64).to_le_bytes());
        attention.update((kv.q2_tokens() as u64).to_le_bytes());
        attention.update((kv.q4_tokens() as u64).to_le_bytes());
        kv.tokens()
    };
    anyhow::ensure!(
        graph.mtp_tokens() == mtp_cached_tokens,
        "MTP token counter differs from its full-attention cache metadata"
    );
    Ok(StateEvidence {
        linear_f16le_sha256: format!("{:x}", linear.finalize()),
        attention_metadata_sha256: format!("{:x}", attention.finalize()),
        attention_layers: attention_layers + 1,
        cached_tokens_per_attention_layer: cached_tokens_per_attention_layer.unwrap_or(0),
        mtp_cached_tokens,
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
