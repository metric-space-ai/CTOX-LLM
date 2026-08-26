use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_graph::PreparedCudaProjectionGraph;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use ctox_qwen38_27b::Qwen38Config;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Execute one complete Qwen target token on CUDA SM86")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    token_id: usize,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 4_096)]
    maximum_context_tokens: usize,
}

#[derive(Debug, Serialize)]
struct RankedLogit {
    token_id: usize,
    value: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    artifact_manifest_sha256: String,
    token_id: usize,
    position: usize,
    maximum_context_tokens: usize,
    target_logits: usize,
    target_logits_f32le_sha256: String,
    target_top_logits: Vec<RankedLogit>,
    selected_target_token: usize,
    mtp_logits: usize,
    mtp_logits_f32le_sha256: String,
    mtp_top_logits: Vec<RankedLogit>,
    mtp_draft_token: usize,
    verification_logits_f32le_sha256: String,
    verification_top_logits: Vec<RankedLogit>,
    target_verified_token: usize,
    mtp_draft_accepted: bool,
    checkpoint_mtp_logits_f32le_sha256: String,
    checkpoint_draft_token: usize,
    speculative_target_logits_f32le_sha256: String,
    replayed_target_logits_f32le_sha256: String,
    speculative_restore_exact: bool,
    graph_prepare_milliseconds: f64,
    target_dispatch_milliseconds: f64,
    mtp_dispatch_milliseconds: f64,
    target_verify_milliseconds: f64,
    checkpoint_replay_milliseconds: f64,
    token_submission_attempts: u64,
    token_submission_commits: u64,
    deferred_operator_synchronizations: u64,
    context_synchronizations: u64,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let prepare_started = Instant::now();
    let mut graph = PreparedCudaProjectionGraph::prepare(
        &runtime,
        &artifact,
        &Qwen38Config::default(),
        args.maximum_context_tokens,
    )?;
    let graph_prepare_milliseconds = prepare_started.elapsed().as_secs_f64() * 1.0e3;
    let (free_after_prepare, _) = runtime.memory_info()?;

    let dispatch_started = Instant::now();
    let logits_view =
        graph.dispatch_target_token_device(&runtime, &Qwen38Config::default(), args.token_id, 0)?;
    let target_logits = runtime.verifier_read_f32_device(logits_view)?;
    let target_dispatch_milliseconds = dispatch_started.elapsed().as_secs_f64() * 1.0e3;
    anyhow::ensure!(
        target_logits.len() == Qwen38Config::default().vocab_size,
        "CUDA target logits have {} values, expected {}",
        target_logits.len(),
        Qwen38Config::default().vocab_size
    );
    let target_top_logits = rank_valid_logits(&target_logits)?;
    let selected_target_token = target_top_logits[0].token_id;
    let target_logits_f32le_sha256 = digest_logits(&target_logits);
    anyhow::ensure!(
        graph.target_tokens() == 1,
        "CUDA target token did not commit"
    );

    let mtp_started = Instant::now();
    let mtp_view = graph.dispatch_mtp_draft_device(
        &runtime,
        &Qwen38Config::default(),
        selected_target_token,
        1,
    )?;
    let mtp_logits = runtime.verifier_read_f32_device(mtp_view)?;
    let mtp_dispatch_milliseconds = mtp_started.elapsed().as_secs_f64() * 1.0e3;
    anyhow::ensure!(
        mtp_logits.len() == Qwen38Config::default().vocab_size,
        "CUDA MTP logits have {} values, expected {}",
        mtp_logits.len(),
        Qwen38Config::default().vocab_size
    );
    let mtp_top_logits = rank_valid_logits(&mtp_logits)?;
    let mtp_draft_token = mtp_top_logits[0].token_id;
    let mtp_logits_f32le_sha256 = digest_logits(&mtp_logits);
    anyhow::ensure!(graph.mtp_tokens() == 1, "CUDA MTP token did not commit");

    let verify_started = Instant::now();
    let verification_view = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        selected_target_token,
        1,
    )?;
    let verification_logits = runtime.verifier_read_f32_device(verification_view)?;
    let target_verify_milliseconds = verify_started.elapsed().as_secs_f64() * 1.0e3;
    let verification_top_logits = rank_valid_logits(&verification_logits)?;
    let target_verified_token = verification_top_logits[0].token_id;
    let mtp_draft_accepted = mtp_draft_token == target_verified_token;
    let verification_logits_f32le_sha256 = digest_logits(&verification_logits);
    anyhow::ensure!(
        graph.target_tokens() == 2,
        "CUDA target verifier token did not commit"
    );

    let checkpoint_started = Instant::now();
    anyhow::ensure!(
        graph.target_tokens() == 2 && graph.mtp_tokens() == 1,
        "CUDA checkpoint base does not keep target exactly one token ahead of MTP"
    );
    graph.begin_speculative_branch(&runtime)?;
    anyhow::ensure!(
        graph.speculative_branch_active(),
        "CUDA speculative checkpoint did not become active"
    );
    let checkpoint_mtp_view = graph.dispatch_mtp_draft_device(
        &runtime,
        &Qwen38Config::default(),
        target_verified_token,
        2,
    )?;
    let checkpoint_mtp_logits = runtime.verifier_read_f32_device(checkpoint_mtp_view)?;
    let checkpoint_mtp_logits_f32le_sha256 = digest_logits(&checkpoint_mtp_logits);
    let checkpoint_draft_token = rank_valid_logits(&checkpoint_mtp_logits)?[0].token_id;
    let speculative_view = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        target_verified_token,
        2,
    )?;
    let speculative_logits = runtime.verifier_read_f32_device(speculative_view)?;
    let speculative_target_logits_f32le_sha256 = digest_logits(&speculative_logits);
    graph.restore_speculative_branch(&runtime)?;
    anyhow::ensure!(
        !graph.speculative_branch_active() && graph.target_tokens() == 2 && graph.mtp_tokens() == 1,
        "CUDA speculative restore did not return to the checkpoint counters"
    );
    let _ = graph.dispatch_mtp_draft_device(
        &runtime,
        &Qwen38Config::default(),
        target_verified_token,
        2,
    )?;
    let replayed_view = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        target_verified_token,
        2,
    )?;
    let replayed_logits = runtime.verifier_read_f32_device(replayed_view)?;
    let replayed_target_logits_f32le_sha256 = digest_logits(&replayed_logits);
    let speculative_restore_exact = speculative_logits == replayed_logits;
    anyhow::ensure!(
        speculative_restore_exact,
        "CUDA replay after speculative restore changed target logits"
    );
    anyhow::ensure!(
        graph.target_tokens() == 3 && graph.mtp_tokens() == 2,
        "CUDA accepted-prefix replay did not preserve target-one-ahead state"
    );
    let checkpoint_replay_milliseconds = checkpoint_started.elapsed().as_secs_f64() * 1.0e3;
    graph.reset_session()?;
    anyhow::ensure!(
        graph.target_tokens() == 0,
        "CUDA session reset did not commit"
    );
    anyhow::ensure!(graph.mtp_tokens() == 0, "CUDA MTP reset did not commit");
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA target graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );
    let submission_stats = runtime.submission_stats();
    anyhow::ensure!(
        submission_stats.token_submission_attempts == 7
            && submission_stats.token_submission_commits == 7,
        "CUDA target/MTP chain committed {}/{} token submissions, expected 7/7",
        submission_stats.token_submission_commits,
        submission_stats.token_submission_attempts,
    );
    anyhow::ensure!(
        submission_stats.context_synchronizations == 13,
        "CUDA target/MTP chain used {} context barriers, expected seven commits and six verifier readbacks",
        submission_stats.context_synchronizations,
    );
    anyhow::ensure!(
        submission_stats.deferred_operator_synchronizations > 200,
        "CUDA target/MTP chain deferred only {} operator barriers",
        submission_stats.deferred_operator_synchronizations,
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-target-token.v4",
            status: "finite_logits_verifier_only_not_promoted",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            token_id: args.token_id,
            position: 0,
            maximum_context_tokens: args.maximum_context_tokens,
            target_logits: target_logits.len(),
            target_logits_f32le_sha256,
            target_top_logits,
            selected_target_token,
            mtp_logits: mtp_logits.len(),
            mtp_logits_f32le_sha256,
            mtp_top_logits,
            mtp_draft_token,
            verification_logits_f32le_sha256,
            verification_top_logits,
            target_verified_token,
            mtp_draft_accepted,
            checkpoint_mtp_logits_f32le_sha256,
            checkpoint_draft_token,
            speculative_target_logits_f32le_sha256,
            replayed_target_logits_f32le_sha256,
            speculative_restore_exact,
            graph_prepare_milliseconds,
            target_dispatch_milliseconds,
            mtp_dispatch_milliseconds,
            target_verify_milliseconds,
            checkpoint_replay_milliseconds,
            token_submission_attempts: submission_stats.token_submission_attempts,
            token_submission_commits: submission_stats.token_submission_commits,
            deferred_operator_synchronizations: submission_stats
                .deferred_operator_synchronizations,
            context_synchronizations: submission_stats.context_synchronizations,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            note: "Executes embedding, all 64 target layers, final norm, LM head, target-selected-token MTP drafts, target verification, one device-side speculative checkpoint, restore, bit-exact accepted-prefix replay through both MTP and target state, reset, and unload. The checkpoint starts with target exactly one token ahead of MTP, matching the executor contract. Each target/MTP step has one commit barrier; logits cross the host only at explicit verifier boundaries. Draft rejection is a valid result and is reported, not hidden. This proves one bounded replay transition, not the complete MTP4 executor: chained draft assembly, multi-token partial-prefix replay, production sampling, prefill, BF16/CPU comparison, and roofline promotion remain open.",
        })?
    );
    Ok(())
}

fn rank_valid_logits(logits: &[f32]) -> anyhow::Result<Vec<RankedLogit>> {
    anyhow::ensure!(
        logits.len() >= TOKENIZER_VOCAB_SIZE,
        "logit vector is narrower than the tokenizer vocabulary"
    );
    let mut ranking: Vec<_> = logits[..TOKENIZER_VOCAB_SIZE]
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, value)| RankedLogit { token_id, value })
        .collect();
    ranking.sort_unstable_by(|left, right| right.value.total_cmp(&left.value));
    ranking.truncate(16);
    Ok(ranking)
}

fn digest_logits(logits: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in logits {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
