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
#[command(about = "Verify and benchmark the four-draft Qwen target/MTP CUDA graphs")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    mtp_draft_vocabulary: PathBuf,
    #[arg(long)]
    seed_token_id: usize,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 4_096)]
    maximum_context_tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmup_blocks: usize,
    #[arg(long, default_value_t = 4)]
    measured_blocks: usize,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    artifact_manifest_sha256: String,
    artifact_file_bytes: u64,
    mtp_draft_vocabulary_sha256: String,
    mtp_draft_vocabulary_rows: usize,
    maximum_context_tokens: usize,
    selected_seed_token: usize,
    warmup_blocks: usize,
    warmup_verified_emitted_tokens: usize,
    measured_blocks: usize,
    measured_milliseconds: f64,
    speculative_blocks_per_second: f64,
    verified_emitted_tokens: usize,
    verified_emitted_tokens_per_second: f64,
    accepted_drafts: usize,
    mean_accepted_drafts_per_block: f64,
    final_target_tokens: usize,
    final_mtp_tokens: usize,
    compact_device_to_host_readbacks: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.measured_blocks > 0,
        "measured block count must be positive"
    );
    anyhow::ensure!(
        1 + 5 * (args.warmup_blocks + args.measured_blocks) <= args.maximum_context_tokens,
        "MTP4 benchmark's conservative five-token blocks exceed context capacity"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let artifact_file_bytes = fs::metadata(&args.artifact)
        .with_context(|| format!("failed to stat CTOXQ artifact {}", args.artifact.display()))?
        .len();
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let draft_vocabulary = fs::read(&args.mtp_draft_vocabulary).with_context(|| {
        format!(
            "failed to read MTP draft vocabulary {}",
            args.mtp_draft_vocabulary.display()
        )
    })?;
    let mtp_draft_vocabulary_sha256 = format!("{:x}", Sha256::digest(&draft_vocabulary));
    let draft_token_ids = parse_canonical_u32le(&draft_vocabulary)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let mut graph = PreparedCudaProjectionGraph::prepare(
        &runtime,
        &artifact,
        &Qwen38Config::default(),
        args.maximum_context_tokens,
        Some(&draft_token_ids),
    )?;
    let (free_after_prepare, _) = runtime.memory_info()?;

    let seed_logits = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    let selected_seed_token = argmax_token(&runtime.verifier_read_f32_device(seed_logits)?)?;
    // The ordinary target capture also seeds the persistent device token used
    // by the first MTP draft. It is not replayed by this benchmark.
    graph.capture_target_decode_graph(&runtime, &Qwen38Config::default(), selected_seed_token)?;
    graph.capture_mtp_decode_graph(&runtime, &Qwen38Config::default())?;
    graph.capture_mtp4_draft_graph(&runtime, &Qwen38Config::default())?;
    graph.capture_mtp4_target_verify_graph(&runtime, &Qwen38Config::default())?;

    let mut warmup_verified_emitted_tokens = 0_usize;
    for _ in 0..args.warmup_blocks {
        graph.begin_speculative_branch(&runtime)?;
        graph.launch_captured_mtp4_drafts()?;
        graph.launch_captured_mtp4_target_verify()?;
        let accepted = graph.verifier_read_captured_mtp4_acceptance(&runtime)?;
        warmup_verified_emitted_tokens +=
            graph.resolve_captured_mtp4_acceptance(&runtime, accepted)?;
    }

    let started = Instant::now();
    let mut accepted_drafts = 0_usize;
    let mut verified_emitted_tokens = 0_usize;
    for _ in 0..args.measured_blocks {
        graph.begin_speculative_branch(&runtime)?;
        graph.launch_captured_mtp4_drafts()?;
        graph.launch_captured_mtp4_target_verify()?;
        let accepted = graph.verifier_read_captured_mtp4_acceptance(&runtime)?;
        accepted_drafts += accepted.accepted_count as usize;
        verified_emitted_tokens += graph.resolve_captured_mtp4_acceptance(&runtime, accepted)?;
    }
    let measured_milliseconds = started.elapsed().as_secs_f64() * 1.0e3;
    let seconds = measured_milliseconds / 1.0e3;
    anyhow::ensure!(seconds > 0.0, "measured MTP4 duration is not positive");
    let final_target_tokens = graph.target_tokens();
    let final_mtp_tokens = graph.mtp_tokens();
    anyhow::ensure!(
        final_target_tokens == final_mtp_tokens.saturating_add(1),
        "MTP4 benchmark ended with invalid target/MTP state {final_target_tokens}/{final_mtp_tokens}"
    );
    anyhow::ensure!(
        final_target_tokens == 1 + warmup_verified_emitted_tokens + verified_emitted_tokens,
        "MTP4 benchmark target counter does not match committed token count"
    );

    graph.reset_session()?;
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA MTP4 graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-target-mtp4-graph-benchmark.v1",
            status: "measured_mtp4_committed_generation",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            artifact_file_bytes,
            mtp_draft_vocabulary_sha256,
            mtp_draft_vocabulary_rows: draft_token_ids.len(),
            maximum_context_tokens: args.maximum_context_tokens,
            selected_seed_token,
            warmup_blocks: args.warmup_blocks,
            warmup_verified_emitted_tokens,
            measured_blocks: args.measured_blocks,
            measured_milliseconds,
            speculative_blocks_per_second: args.measured_blocks as f64 / seconds,
            verified_emitted_tokens,
            verified_emitted_tokens_per_second: verified_emitted_tokens as f64 / seconds,
            accepted_drafts,
            mean_accepted_drafts_per_block: accepted_drafts as f64
                / args.measured_blocks as f64,
            final_target_tokens,
            final_mtp_tokens,
            compact_device_to_host_readbacks: args.warmup_blocks + args.measured_blocks,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            note: "Real committed generation: four device-chained MTP drafts, one five-row target verification graph, one compact D2H acceptance readback, direct full-block commit, and device-only target/MTP graph replay of exactly the accepted prefix plus target correction after partial rejection. Model load, seed prefill, and graph capture are excluded from measured time.",
        })?
    );
    Ok(())
}

fn argmax_token(logits: &[f32]) -> anyhow::Result<usize> {
    anyhow::ensure!(
        logits.len() >= TOKENIZER_VOCAB_SIZE,
        "logit vector is narrower than tokenizer vocabulary"
    );
    logits[..TOKENIZER_VOCAB_SIZE]
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(token, _)| token)
        .context("logit vector is empty")
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
