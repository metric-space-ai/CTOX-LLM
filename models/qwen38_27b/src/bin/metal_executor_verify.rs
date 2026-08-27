use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda::{GATED_DELTA_STATE_BYTES, LINEAR_CONV_STATE_BYTES};
use ctox_qwen38_27b::backend::metal_executor::ThreadedMetalModelExecutor;
use ctox_qwen38_27b::engine::{CancellationToken, ExecutorStep, ModelExecutor};
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::memory::{LinearStateDType, SpeculativeStateStrategy};
use ctox_qwen38_27b::release::{KvMemoryFormula, MemoryProfile};
use ctox_qwen38_27b::sampler::SamplerConfig;
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use ctox_qwen38_27b::{EngineError, Qwen38Config};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify the complete mmap-backed Metal executor lifecycle and MTP4 transaction")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    /// Canonical strictly-increasing u32-LE release draft vocabulary.
    #[arg(long)]
    mtp_draft_vocabulary: PathBuf,
    /// Exact signed release-pack hardware profile.
    #[arg(long)]
    hardware_profile: String,
    #[arg(long, default_value_t = 16)]
    maximum_context_tokens: u64,
    /// Comma-delimited prompt token IDs. Tokenization is intentionally kept
    /// outside this verifier so the exact sequence remains evidence-bound.
    #[arg(long, value_delimiter = ',', default_value = "1")]
    prefill_tokens: Vec<u32>,
    /// Absolute mapped-model + graph + session residency ceiling.
    #[arg(long)]
    hard_limit_bytes: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    status: &'static str,
    hardware_profile: String,
    artifact_manifest_sha256: String,
    artifact_file_bytes: u64,
    mtp_draft_vocabulary_sha256: String,
    mtp_draft_vocabulary_tokens: usize,
    maximum_context_tokens: u64,
    prefill_tokens: Vec<u32>,
    decode_input_token: u32,
    draft_tokens: Vec<u32>,
    target_tokens: Vec<u32>,
    bonus_token: u32,
    accepted_drafts: u32,
    target_tokens_after_commit: usize,
    mtp_tokens_after_commit: usize,
    requested_model_bytes: u64,
    requested_graph_bytes: u64,
    requested_session_bytes: u64,
    requested_total_bytes: u64,
    hard_limit_bytes: u64,
    allocator_process_baseline_bytes: u64,
    allocator_peak_bytes: u64,
    allocator_after_unload_bytes: u64,
    allocator_reclaimed_bytes: u64,
    allocator_residual_bytes: u64,
    load_milliseconds: f64,
    prefill_milliseconds: f64,
    decode_milliseconds: f64,
    commit_milliseconds: f64,
    reset_milliseconds: f64,
    unload_milliseconds: f64,
    cancellation_failed_closed: bool,
    allocations_zero_after_unload: bool,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        !args.hardware_profile.trim().is_empty(),
        "hardware profile must not be empty"
    );
    anyhow::ensure!(
        !args.prefill_tokens.is_empty()
            && args
                .prefill_tokens
                .iter()
                .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE),
        "prefill tokens must be non-empty and inside the tokenizer vocabulary"
    );
    anyhow::ensure!(
        args.prefill_tokens.len() as u64 + 5 <= args.maximum_context_tokens,
        "verifier context must reserve input plus the complete MTP4 span"
    );

    let draft_vocabulary_bytes = fs::read(&args.mtp_draft_vocabulary).with_context(|| {
        format!(
            "failed to read MTP draft vocabulary {}",
            args.mtp_draft_vocabulary.display()
        )
    })?;
    let mtp_draft_vocabulary_sha256 = digest_bytes(&draft_vocabulary_bytes);
    let mtp_draft_token_ids = parse_canonical_u32le(&draft_vocabulary_bytes)?;
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let artifact_file_bytes = artifact.file_bytes();
    let profile = verifier_profile(
        &artifact,
        args.maximum_context_tokens,
        args.hard_limit_bytes,
    )?;

    let mut executor = ThreadedMetalModelExecutor::new_for_profile(args.hardware_profile.clone())?;
    let capabilities = executor.capabilities();
    anyhow::ensure!(
        capabilities.compact_greedy_mtp_verification
            && capabilities.resident_target_selection
            && capabilities.no_hidden_fallbacks,
        "Metal executor does not expose the complete compact verifier contract"
    );
    let load_started = Instant::now();
    executor.load(&artifact, &profile, &mtp_draft_token_ids)?;
    executor.warmup()?;
    let load_milliseconds = load_started.elapsed().as_secs_f64() * 1.0e3;
    let allocations = executor.allocations();
    let requested_total_bytes = allocations.total_bytes()?;
    anyhow::ensure!(
        allocations.model_bytes == artifact_file_bytes
            && allocations.graph_bytes > 0
            && allocations.session_bytes > 0
            && allocations.global_cache_bytes == 0
            && requested_total_bytes <= args.hard_limit_bytes,
        "Metal executor residency is incomplete or exceeds the hard limit: {allocations:?}"
    );

    let cancellation = CancellationToken::default();
    let prefill_started = Instant::now();
    let prefill = executor.prefill(&args.prefill_tokens, true, &cancellation)?;
    let prefill_milliseconds = prefill_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_resident_step(&prefill, false)?;
    let decode_input_token = executor
        .select_target_token(greedy_sampler(), 0.0)?
        .context("Metal prefill delegated greedy target selection to the host")?;

    let decode_started = Instant::now();
    let decoded = executor.decode(decode_input_token, true, &cancellation)?;
    let decode_milliseconds = decode_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_resident_step(&decoded, true)?;
    let verification = decoded
        .compact_greedy_mtp
        .context("Metal MTP4 decode omitted compact target verification")?;
    anyhow::ensure!(
        verification.draft_tokens.len() == 4
            && verification.target_tokens.len() == 4
            && verification
                .draft_tokens
                .iter()
                .all(|token| mtp_draft_token_ids.binary_search(token).is_ok()),
        "Metal MTP4 output is incomplete or escaped the release draft vocabulary"
    );
    let accepted_drafts = verification
        .draft_tokens
        .iter()
        .zip(&verification.target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count() as u32;

    let commit_started = Instant::now();
    executor.commit_speculative(accepted_drafts, &cancellation)?;
    let commit_milliseconds = commit_started.elapsed().as_secs_f64() * 1.0e3;
    let (target_tokens_after_commit, mtp_tokens_after_commit) =
        executor.session_token_counters()?;
    let expected_target = args.prefill_tokens.len() + 1 + accepted_drafts as usize;
    let expected_mtp = args.prefill_tokens.len() + accepted_drafts as usize;
    anyhow::ensure!(
        (target_tokens_after_commit, mtp_tokens_after_commit) == (expected_target, expected_mtp),
        "Metal MTP commit produced target/MTP counters {target_tokens_after_commit}/{mtp_tokens_after_commit}, expected {expected_target}/{expected_mtp}"
    );

    let reset_started = Instant::now();
    executor.reset_session()?;
    let reset_milliseconds = reset_started.elapsed().as_secs_f64() * 1.0e3;
    anyhow::ensure!(
        executor.session_token_counters()? == (0, 0),
        "Metal reset retained target or MTP state"
    );
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancellation_failed_closed = matches!(
        executor.prefill(&args.prefill_tokens, true, &cancelled),
        Err(EngineError::Cancelled)
    );
    anyhow::ensure!(
        cancellation_failed_closed && executor.session_token_counters()? == (0, 0),
        "Metal cancelled prefill mutated session state or returned the wrong error"
    );

    let unload_started = Instant::now();
    executor.unload()?;
    let unload_milliseconds = unload_started.elapsed().as_secs_f64() * 1.0e3;
    let allocator_stats = executor.allocator_stats()?;
    let allocations_zero_after_unload = executor.allocations().is_zero();
    anyhow::ensure!(
        allocations_zero_after_unload && allocator_stats.residual_bytes == 0,
        "Metal executor retained accounted or process-visible allocations after unload"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.metal-executor-full-artifact-verifier.v1",
            status: "lifecycle_mtp4_verifier_only_not_promoted",
            hardware_profile: args.hardware_profile,
            artifact_manifest_sha256,
            artifact_file_bytes,
            mtp_draft_vocabulary_sha256,
            mtp_draft_vocabulary_tokens: mtp_draft_token_ids.len(),
            maximum_context_tokens: args.maximum_context_tokens,
            prefill_tokens: args.prefill_tokens,
            decode_input_token,
            draft_tokens: verification.draft_tokens,
            target_tokens: verification.target_tokens,
            bonus_token: verification.bonus_token,
            accepted_drafts,
            target_tokens_after_commit,
            mtp_tokens_after_commit,
            requested_model_bytes: allocations.model_bytes,
            requested_graph_bytes: allocations.graph_bytes,
            requested_session_bytes: allocations.session_bytes,
            requested_total_bytes,
            hard_limit_bytes: args.hard_limit_bytes,
            allocator_process_baseline_bytes: allocator_stats.process_baseline_bytes,
            allocator_peak_bytes: allocator_stats.peak_bytes,
            allocator_after_unload_bytes: allocator_stats.after_unload_bytes,
            allocator_reclaimed_bytes: allocator_stats.reclaimed_bytes,
            allocator_residual_bytes: allocator_stats.residual_bytes,
            load_milliseconds,
            prefill_milliseconds,
            decode_milliseconds,
            commit_milliseconds,
            reset_milliseconds,
            unload_milliseconds,
            cancellation_failed_closed,
            allocations_zero_after_unload,
            note: "Checksummed full-artifact load, exact packed draft vocabulary, serial prefill, resident target selection, fused MTP4 target verification, device-resolved commit acknowledgement, reset, cancellation, and accounted unload through the thread-affine executor. Promotion still requires BF16/logit golden comparison, allocator high-watermark measured outside this process, chunked prefill, stochastic sampling, and controlled roofline evidence.",
        })?
    );
    Ok(())
}

fn ensure_resident_step(step: &ExecutorStep, expect_mtp: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        step.target_logits.is_empty()
            && step.draft_logits.is_empty()
            && step.target_verification_logits.is_empty()
            && step.bonus_logits.is_none()
            && (step.compact_greedy_mtp.is_some() == expect_mtp),
        "Metal executor returned host logits or the wrong compact verifier shape"
    );
    Ok(())
}

fn verifier_profile(
    artifact: &ModelArtifact,
    maximum_context_tokens: u64,
    hard_limit_bytes: u64,
) -> anyhow::Result<MemoryProfile> {
    let config = Qwen38Config::default();
    anyhow::ensure!(
        maximum_context_tokens > 0
            && maximum_context_tokens <= config.max_position_embeddings as u64,
        "verifier context exceeds the frozen model capacity"
    );
    let linear_state_bytes = u64::try_from(
        config
            .linear_attention_layers()
            .checked_mul(GATED_DELTA_STATE_BYTES + LINEAR_CONV_STATE_BYTES)
            .context("linear-state bytes overflow")?,
    )?;
    Ok(MemoryProfile {
        profile_id: "metal-executor-full-artifact-verifier".into(),
        pack_id: "direct-ctoxq-verifier".into(),
        context_tokens: maximum_context_tokens,
        sessions: 1,
        resident_model_bytes: artifact.file_bytes(),
        persistent_backend_graph_bytes: 0,
        persistent_runtime_bytes: 0,
        linear_state_dtype: LinearStateDType::F16,
        linear_state_bytes_per_session: linear_state_bytes,
        mtp_draft_tokens: 4,
        speculative_state_strategy: SpeculativeStateStrategy::ReplayOnReject,
        speculative_linear_state_bytes_per_session: linear_state_bytes,
        kv: zero_kv_formula(),
        mtp_kv: zero_kv_formula(),
        prefill_scratch_peak_bytes: 0,
        decode_scratch_peak_bytes: 0,
        loader_transient_peak_bytes: 0,
        accelerator_unattributed_reserve_bytes: 0,
        hard_limit_bytes,
    })
}

fn zero_kv_formula() -> KvMemoryFormula {
    KvMemoryFormula {
        fixed_bytes_per_session: 0,
        bytes_per_token_per_session: 0,
        retained_q4_tokens_per_session: 0,
        q4_delta_bytes_per_token: 0,
    }
}

fn greedy_sampler() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        seed: 0,
    }
}

fn parse_canonical_u32le(bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "MTP draft vocabulary must contain canonical u32-LE values"
    );
    let token_ids = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        token_ids
            .iter()
            .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE)
            && token_ids.windows(2).all(|pair| pair[0] < pair[1]),
        "MTP draft vocabulary must be unique, strictly increasing, and in range"
    );
    Ok(token_ids)
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
