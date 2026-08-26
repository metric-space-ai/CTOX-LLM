use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda::{GATED_DELTA_STATE_BYTES, LINEAR_CONV_STATE_BYTES};
use ctox_qwen38_27b::backend::cuda_executor::ThreadedCudaModelExecutor;
use ctox_qwen38_27b::engine::{CancellationToken, DraftDistribution, ModelExecutor};
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::memory::{LinearStateDType, SpeculativeStateStrategy};
use ctox_qwen38_27b::release::{KvMemoryFormula, MemoryProfile};
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use ctox_qwen38_27b::Qwen38Config;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify the embeddable Qwen CUDA executor lifecycle and MTP4 replay")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    /// Canonical strictly-increasing u32-LE release draft vocabulary.
    #[arg(long)]
    mtp_draft_vocabulary: PathBuf,
    #[arg(long)]
    token_id: u32,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 4_096)]
    maximum_context_tokens: u64,
    /// Cap the causally verified prefix committed by this run.
    #[arg(long, default_value_t = 4)]
    maximum_accepted_drafts: u32,
}

#[derive(Serialize)]
struct Report {
    format: &'static str,
    status: &'static str,
    thread_affine_worker: bool,
    module_sha256: String,
    artifact_manifest_sha256: String,
    mtp_draft_vocabulary_sha256: String,
    mtp_draft_vocabulary_tokens: usize,
    maximum_context_tokens: u64,
    prefill_input_token: u32,
    decode_input_token: u32,
    draft_tokens: Vec<u32>,
    verified_draft_prefix: u32,
    accepted_drafts: u32,
    bonus_token: u32,
    target_logits_f32le_sha256: String,
    draft_logits_f32le_sha256: Vec<String>,
    target_verification_logits_f32le_sha256: Vec<String>,
    bonus_logits_f32le_sha256: String,
    target_tokens_after_replay: usize,
    mtp_tokens_after_replay: usize,
    token_submission_attempts: u64,
    token_submission_commits: u64,
    deferred_operator_synchronizations: u64,
    context_synchronizations: u64,
    requested_model_bytes: u64,
    requested_graph_bytes: u64,
    requested_session_bytes: u64,
    load_milliseconds: f64,
    prefill_milliseconds: f64,
    decode_milliseconds: f64,
    replay_milliseconds: f64,
    unload_milliseconds: f64,
    allocations_zero_after_unload: bool,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        (args.token_id as usize) < TOKENIZER_VOCAB_SIZE,
        "prefill token exceeds tokenizer vocabulary"
    );
    anyhow::ensure!(
        args.maximum_accepted_drafts <= 4,
        "maximum accepted draft prefix exceeds MTP4"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = digest_bytes(&module);
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
    let profile = verifier_profile(&artifact, args.maximum_context_tokens)?;

    let mut executor = ThreadedCudaModelExecutor::new_sm86(&module, args.device)?;
    let load_started = Instant::now();
    executor.load(&artifact, &profile, &mtp_draft_token_ids)?;
    executor.warmup()?;
    let load_milliseconds = load_started.elapsed().as_secs_f64() * 1.0e3;
    let allocations = executor.allocations();
    anyhow::ensure!(
        allocations.model_bytes > 0
            && allocations.graph_bytes > 0
            && allocations.session_bytes > 0
            && allocations.global_cache_bytes == 0,
        "CUDA executor did not report bounded graph allocations"
    );

    let cancellation = CancellationToken::default();
    let prefill_started = Instant::now();
    let prefill = executor.prefill(&[args.token_id], true, &cancellation)?;
    let prefill_milliseconds = prefill_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_full_logits(&prefill.target_logits, "prefill target")?;
    anyhow::ensure!(
        prefill.draft_logits.is_empty()
            && prefill.target_verification_logits.is_empty()
            && prefill.bonus_logits.is_none(),
        "CUDA prefill unexpectedly returned speculative outputs"
    );
    let decode_input_token = greedy_token(&prefill.target_logits)?;

    let decode_started = Instant::now();
    let decoded = executor.decode(decode_input_token, true, &cancellation)?;
    let decode_milliseconds = decode_started.elapsed().as_secs_f64() * 1.0e3;
    ensure_full_logits(&decoded.target_logits, "decode target")?;
    anyhow::ensure!(
        decoded.draft_logits.len() == 4
            && decoded.target_verification_logits.len() == 4
            && decoded.target_verification_logits[0] == decoded.target_logits,
        "CUDA executor did not return a verifiable MTP4 block"
    );
    let bonus_logits = decoded
        .bonus_logits
        .as_ref()
        .context("CUDA MTP4 block omitted target bonus logits")?;
    ensure_full_logits(bonus_logits, "MTP4 bonus")?;
    let draft_tokens = decoded
        .draft_logits
        .iter()
        .map(greedy_draft_token)
        .collect::<anyhow::Result<Vec<_>>>()?;
    let target_tokens = decoded
        .target_verification_logits
        .iter()
        .map(|logits| greedy_token(logits))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let verified_draft_prefix = draft_tokens
        .iter()
        .zip(&target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count() as u32;
    let accepted_drafts = verified_draft_prefix.min(args.maximum_accepted_drafts);
    let bonus_token = greedy_token(bonus_logits)?;
    let target_logits_f32le_sha256 = digest_logits(&decoded.target_logits);
    let draft_logits_f32le_sha256 = decoded
        .draft_logits
        .iter()
        .map(digest_draft_logits)
        .collect();
    let target_verification_logits_f32le_sha256 = decoded
        .target_verification_logits
        .iter()
        .map(|logits| digest_logits(logits))
        .collect();
    let bonus_logits_f32le_sha256 = digest_logits(bonus_logits);

    let replay_started = Instant::now();
    executor.commit_speculative(accepted_drafts, &cancellation)?;
    let replay_milliseconds = replay_started.elapsed().as_secs_f64() * 1.0e3;
    let (target_tokens_after_replay, mtp_tokens_after_replay) =
        executor.session_token_counters()?;
    anyhow::ensure!(
        target_tokens_after_replay == 2 + accepted_drafts as usize
            && mtp_tokens_after_replay + 1 == target_tokens_after_replay,
        "CUDA partial-prefix replay produced target/MTP counters {target_tokens_after_replay}/{mtp_tokens_after_replay}"
    );
    let submission_stats = executor.submission_stats()?;
    let expected_submissions = 10 + u64::from(accepted_drafts) * 2;
    anyhow::ensure!(
        submission_stats.token_submission_attempts == expected_submissions
            && submission_stats.token_submission_commits == expected_submissions,
        "CUDA executor committed {}/{} submissions, expected {expected_submissions}/{expected_submissions}",
        submission_stats.token_submission_commits,
        submission_stats.token_submission_attempts,
    );
    anyhow::ensure!(
        submission_stats.context_synchronizations == expected_submissions + 10,
        "CUDA executor used {} context barriers, expected {} commits plus ten verifier readbacks",
        submission_stats.context_synchronizations,
        expected_submissions,
    );

    executor.reset_session()?;
    anyhow::ensure!(
        executor.session_token_counters()? == (0, 0),
        "CUDA executor reset retained token state"
    );
    let unload_started = Instant::now();
    executor.unload()?;
    let unload_milliseconds = unload_started.elapsed().as_secs_f64() * 1.0e3;
    let allocations_zero_after_unload = executor.allocations().is_zero();
    anyhow::ensure!(
        allocations_zero_after_unload,
        "CUDA executor retained accounted allocations after unload"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-executor.v1",
            status: "lifecycle_mtp4_partial_replay_verifier_only_not_promoted",
            thread_affine_worker: true,
            module_sha256,
            artifact_manifest_sha256,
            mtp_draft_vocabulary_sha256,
            mtp_draft_vocabulary_tokens: mtp_draft_token_ids.len(),
            maximum_context_tokens: args.maximum_context_tokens,
            prefill_input_token: args.token_id,
            decode_input_token,
            draft_tokens,
            verified_draft_prefix,
            accepted_drafts,
            bonus_token,
            target_logits_f32le_sha256,
            draft_logits_f32le_sha256,
            target_verification_logits_f32le_sha256,
            bonus_logits_f32le_sha256,
            target_tokens_after_replay,
            mtp_tokens_after_replay,
            token_submission_attempts: submission_stats.token_submission_attempts,
            token_submission_commits: submission_stats.token_submission_commits,
            deferred_operator_synchronizations: submission_stats
                .deferred_operator_synchronizations,
            context_synchronizations: submission_stats.context_synchronizations,
            requested_model_bytes: allocations.model_bytes,
            requested_graph_bytes: allocations.graph_bytes,
            requested_session_bytes: allocations.session_bytes,
            load_milliseconds,
            prefill_milliseconds,
            decode_milliseconds,
            replay_milliseconds,
            unload_milliseconds,
            allocations_zero_after_unload,
            note: "Exercises the sendable Rust ModelExecutor ABI through the dedicated thread that owns every CUDA object, without CPU model-operation fallback. Full logits cross the host at verifier token boundaries. The supplied release-bound draft vocabulary is identity-checked but gathered-row MTP and device sampling remain performance work. Promotion still requires BF16/logit, quality, long-context, unload, and roofline evidence on the release checkpoint.",
        })?
    );
    Ok(())
}

fn verifier_profile(
    artifact: &ModelArtifact,
    maximum_context_tokens: u64,
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
        profile_id: "cuda-sm86-executor-verifier".into(),
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
        hard_limit_bytes: u64::MAX,
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

fn parse_canonical_u32le(bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "MTP draft vocabulary is not a non-empty u32-LE array"
    );
    let ids = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        ids.windows(2).all(|pair| pair[0] < pair[1])
            && ids
                .iter()
                .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE),
        "MTP draft vocabulary is not canonical"
    );
    Ok(ids)
}

fn ensure_full_logits(logits: &[f32], label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        logits.len() == TOKENIZER_VOCAB_SIZE && logits.iter().all(|value| value.is_finite()),
        "{label} logits are not a finite tokenizer-vocabulary distribution"
    );
    Ok(())
}

fn greedy_token(logits: &[f32]) -> anyhow::Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token, _)| token as u32)
        .context("logit vector is empty")
}

fn greedy_draft_token(distribution: &DraftDistribution) -> anyhow::Result<u32> {
    match distribution {
        DraftDistribution::Full(logits) => greedy_token(logits),
        DraftDistribution::Restricted { token_ids, logits } => {
            let index = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .context("restricted draft distribution is empty")?;
            token_ids
                .get(index)
                .copied()
                .context("restricted draft IDs and logits differ in length")
        }
    }
}

fn digest_draft_logits(distribution: &DraftDistribution) -> String {
    let mut hasher = Sha256::new();
    match distribution {
        DraftDistribution::Full(logits) => {
            hasher.update(b"full\0");
            update_logits_digest(&mut hasher, logits);
        }
        DraftDistribution::Restricted { token_ids, logits } => {
            hasher.update(b"restricted\0");
            for token in token_ids {
                hasher.update(token.to_le_bytes());
            }
            update_logits_digest(&mut hasher, logits);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn digest_logits(logits: &[f32]) -> String {
    let mut hasher = Sha256::new();
    update_logits_digest(&mut hasher, logits);
    format!("{:x}", hasher.finalize())
}

fn update_logits_digest(hasher: &mut Sha256, logits: &[f32]) {
    for value in logits {
        hasher.update(value.to_le_bytes());
    }
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
