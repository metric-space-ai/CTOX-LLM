use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda::{GATED_DELTA_STATE_BYTES, LINEAR_CONV_STATE_BYTES};
use ctox_qwen38_27b::backend::cuda_executor::ThreadedCudaModelExecutor;
use ctox_qwen38_27b::engine::{CancellationToken, ModelExecutor};
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::memory::{LinearStateDType, SpeculativeStateStrategy};
use ctox_qwen38_27b::release::{KvMemoryFormula, MemoryProfile};
use ctox_qwen38_27b::sampler::SamplerConfig;
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use ctox_qwen38_27b::Qwen38Config;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Benchmark real compact Qwen CUDA prefill and MTP decode")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    mtp_draft_vocabulary: PathBuf,
    #[arg(long)]
    token_id: u32,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long)]
    prefill_tokens: usize,
    #[arg(long, default_value_t = 8)]
    decode_blocks: usize,
    #[arg(long, default_value_t = 4)]
    maximum_accepted_drafts: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    status: &'static str,
    artifact_manifest_sha256: String,
    module_sha256: String,
    mtp_draft_vocabulary_sha256: String,
    mtp_draft_vocabulary_tokens: usize,
    device: i32,
    prefill_tokens: usize,
    prefill_milliseconds: f64,
    prefill_tokens_per_second: f64,
    decode_blocks: usize,
    accepted_drafts: usize,
    proposed_drafts: usize,
    acceptance_rate: f64,
    emitted_decode_tokens: usize,
    decode_milliseconds: f64,
    decode_emitted_tokens_per_second: f64,
    mean_decode_block_milliseconds: f64,
    load_and_warmup_milliseconds: f64,
    unload_milliseconds: f64,
    target_tokens_before_reset: usize,
    mtp_tokens_before_reset: usize,
    requested_model_bytes: u64,
    requested_graph_bytes: u64,
    requested_session_bytes: u64,
    allocator_free_before_load_bytes: u64,
    allocator_minimum_free_bytes: u64,
    allocator_sampled_peak_bytes: u64,
    allocator_free_after_unload_bytes: u64,
    allocator_residual_bytes: u64,
    allocations_zero_after_unload: bool,
    compact_host_logit_values: usize,
    no_cpu_model_operator_fallback: bool,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.prefill_tokens > 0 && args.decode_blocks > 0,
        "prefill-tokens and decode-blocks must be positive"
    );
    anyhow::ensure!(
        args.maximum_accepted_drafts <= 4,
        "maximum accepted drafts exceeds MTP4"
    );
    anyhow::ensure!(
        (args.token_id as usize) < TOKENIZER_VOCAB_SIZE,
        "token-id exceeds tokenizer vocabulary"
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
    let admitted_context = args
        .prefill_tokens
        .checked_add(args.decode_blocks.saturating_mul(5))
        .context("admitted context overflow")?;
    let profile = benchmark_profile(&artifact, admitted_context as u64)?;

    let mut executor = ThreadedCudaModelExecutor::new_sm86(&module, args.device)?;
    anyhow::ensure!(
        executor.capabilities().compact_greedy_mtp_verification
            && executor.capabilities().resident_target_selection
            && executor.capabilities().no_hidden_fallbacks,
        "CUDA executor lacks compact resident no-fallback capabilities"
    );
    let load_started = Instant::now();
    executor.load(&artifact, &profile, &mtp_draft_token_ids)?;
    executor.warmup()?;
    let load_and_warmup_milliseconds = load_started.elapsed().as_secs_f64() * 1.0e3;
    let allocations = executor.allocations();

    let cancellation = CancellationToken::default();
    let tokens = vec![args.token_id; args.prefill_tokens];
    let prefill_started = Instant::now();
    let prefill = executor.prefill(&tokens, true, &cancellation)?;
    let prefill_milliseconds = prefill_started.elapsed().as_secs_f64() * 1.0e3;
    let mut compact_host_logit_values = executor_step_host_values(&prefill);
    anyhow::ensure!(
        compact_host_logit_values == 0 && prefill.compact_greedy_mtp.is_none(),
        "compact CUDA prefill returned host logits or speculative output"
    );
    let mut next_token = executor
        .select_target_token(
            SamplerConfig {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                seed: 0,
            },
            0.0,
        )?
        .context("compact CUDA prefill delegated target selection to the host")?;

    let decode_started = Instant::now();
    let mut accepted_drafts = 0_usize;
    let mut proposed_drafts = 0_usize;
    let mut emitted_decode_tokens = 0_usize;
    for _ in 0..args.decode_blocks {
        let decoded = executor.decode(next_token, true, &cancellation)?;
        compact_host_logit_values = compact_host_logit_values
            .checked_add(executor_step_host_values(&decoded))
            .context("host-logit counter overflow")?;
        let verification = decoded
            .compact_greedy_mtp
            .as_ref()
            .context("compact CUDA decode omitted target-verified MTP decisions")?;
        anyhow::ensure!(
            verification.draft_tokens.len() == 4
                && verification.target_tokens.len() == 4
                && compact_host_logit_values == 0,
            "compact CUDA decode returned an invalid MTP4 block"
        );
        let verified_prefix = verification
            .draft_tokens
            .iter()
            .zip(&verification.target_tokens)
            .take_while(|(draft, target)| draft == target)
            .count();
        let accepted = verified_prefix.min(args.maximum_accepted_drafts);
        next_token = if accepted < verification.target_tokens.len() {
            verification.target_tokens[accepted]
        } else {
            verification.bonus_token
        };
        executor.commit_speculative(accepted as u32, &cancellation)?;
        accepted_drafts += accepted;
        proposed_drafts += verification.draft_tokens.len();
        emitted_decode_tokens += accepted + 1;
    }
    let decode_milliseconds = decode_started.elapsed().as_secs_f64() * 1.0e3;
    let (target_tokens_before_reset, mtp_tokens_before_reset) =
        executor.session_token_counters()?;
    anyhow::ensure!(
        target_tokens_before_reset == args.prefill_tokens + args.decode_blocks + accepted_drafts
            && mtp_tokens_before_reset + 1 == target_tokens_before_reset,
        "CUDA benchmark session counters differ from committed token work"
    );

    executor.reset_session()?;
    anyhow::ensure!(
        executor.session_token_counters()? == (0, 0),
        "CUDA benchmark reset retained session state"
    );
    let _ = executor.allocator_stats()?;
    let unload_started = Instant::now();
    executor.unload()?;
    let unload_milliseconds = unload_started.elapsed().as_secs_f64() * 1.0e3;
    let allocator = executor.allocator_stats()?;
    let allocations_zero_after_unload = executor.allocations().is_zero();
    anyhow::ensure!(
        allocations_zero_after_unload && allocator.residual_bytes == 0,
        "CUDA benchmark retained allocations after unload"
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-executor-benchmark.v1",
            status: "measured_candidate_not_promoted",
            artifact_manifest_sha256,
            module_sha256,
            mtp_draft_vocabulary_sha256,
            mtp_draft_vocabulary_tokens: mtp_draft_token_ids.len(),
            device: args.device,
            prefill_tokens: args.prefill_tokens,
            prefill_milliseconds,
            prefill_tokens_per_second: args.prefill_tokens as f64 * 1.0e3
                / prefill_milliseconds,
            decode_blocks: args.decode_blocks,
            accepted_drafts,
            proposed_drafts,
            acceptance_rate: accepted_drafts as f64 / proposed_drafts as f64,
            emitted_decode_tokens,
            decode_milliseconds,
            decode_emitted_tokens_per_second: emitted_decode_tokens as f64 * 1.0e3
                / decode_milliseconds,
            mean_decode_block_milliseconds: decode_milliseconds / args.decode_blocks as f64,
            load_and_warmup_milliseconds,
            unload_milliseconds,
            target_tokens_before_reset,
            mtp_tokens_before_reset,
            requested_model_bytes: allocations.model_bytes,
            requested_graph_bytes: allocations.graph_bytes,
            requested_session_bytes: allocations.session_bytes,
            allocator_free_before_load_bytes: allocator.free_before_load_bytes,
            allocator_minimum_free_bytes: allocator.minimum_free_bytes,
            allocator_sampled_peak_bytes: allocator.sampled_peak_bytes,
            allocator_free_after_unload_bytes: allocator.free_after_unload_bytes,
            allocator_residual_bytes: allocator.residual_bytes,
            allocations_zero_after_unload,
            compact_host_logit_values,
            no_cpu_model_operator_fallback: true,
            note: "Measures the compact CUDA model executor with real model weights, batched prefill, resident target selection, target-verified MTP4 decode, commit, reset, and unload. It is candidate evidence, not a final-checkpoint promotion result.",
        })?
    );
    Ok(())
}

fn executor_step_host_values(step: &ctox_qwen38_27b::engine::ExecutorStep) -> usize {
    step.target_logits
        .len()
        .saturating_add(
            step.draft_logits
                .iter()
                .map(ctox_qwen38_27b::engine::DraftDistribution::len)
                .sum::<usize>(),
        )
        .saturating_add(
            step.target_verification_logits
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
        )
        .saturating_add(step.bonus_logits.as_ref().map_or(0, Vec::len))
}

fn benchmark_profile(
    artifact: &ModelArtifact,
    maximum_context_tokens: u64,
) -> anyhow::Result<MemoryProfile> {
    let config = Qwen38Config::default();
    anyhow::ensure!(
        maximum_context_tokens > 0
            && maximum_context_tokens <= config.max_position_embeddings as u64,
        "benchmark context exceeds the frozen model capacity"
    );
    let linear_state_bytes = u64::try_from(
        config
            .linear_attention_layers()
            .checked_mul(GATED_DELTA_STATE_BYTES + LINEAR_CONV_STATE_BYTES)
            .context("linear-state bytes overflow")?,
    )?;
    Ok(MemoryProfile {
        profile_id: "cuda-sm86-executor-benchmark".into(),
        pack_id: "direct-ctoxq-benchmark".into(),
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

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
