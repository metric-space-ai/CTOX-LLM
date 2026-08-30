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
#[command(about = "Verify and benchmark the chained Qwen target/MTP CUDA graphs")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    seed_token_id: usize,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 4_096)]
    maximum_context_tokens: usize,
    #[arg(long, default_value_t = 1)]
    warmup_pairs: usize,
    #[arg(long, default_value_t = 8)]
    measured_pairs: usize,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    artifact_manifest_sha256: String,
    maximum_context_tokens: usize,
    selected_seed_token: usize,
    reference_mtp_draft_token: usize,
    reference_target_verified_token: usize,
    captured_mtp_maximum_absolute_error: f32,
    captured_target_maximum_absolute_error: f32,
    captured_mtp_exact: bool,
    captured_target_exact: bool,
    captured_draft_accepted: bool,
    warmup_pairs: usize,
    measured_pairs: usize,
    measured_milliseconds: f64,
    verified_target_tokens_per_second: f64,
    target_plus_mtp_graph_steps_per_second: f64,
    final_target_tokens: usize,
    final_mtp_tokens: usize,
    final_draft_accepted: bool,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.measured_pairs > 0,
        "measured pair count must be positive"
    );
    anyhow::ensure!(
        1 + args.warmup_pairs + args.measured_pairs <= args.maximum_context_tokens,
        "warmup plus measured graph pairs exceed context capacity"
    );
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let mut graph = PreparedCudaProjectionGraph::prepare(
        &runtime,
        &artifact,
        &Qwen38Config::default(),
        args.maximum_context_tokens,
        None,
    )?;
    let (free_after_prepare, _) = runtime.memory_info()?;

    let seed = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    let seed_logits = runtime.verifier_read_f32_device(seed)?;
    let selected_seed_token = argmax_token(&seed_logits)?;
    let reference_mtp = graph.dispatch_mtp_draft_device(
        &runtime,
        &Qwen38Config::default(),
        selected_seed_token,
        1,
    )?;
    let reference_mtp = runtime.verifier_read_f32_device(reference_mtp)?;
    let reference_mtp_draft_token = argmax_token(&reference_mtp)?;
    let reference_target = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        selected_seed_token,
        1,
    )?;
    let reference_target = runtime.verifier_read_f32_device(reference_target)?;
    let reference_target_verified_token = argmax_token(&reference_target)?;

    graph.reset_session()?;
    let _ = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    graph.capture_target_decode_graph(&runtime, &Qwen38Config::default(), selected_seed_token)?;
    graph.capture_mtp_decode_graph(&runtime, &Qwen38Config::default())?;
    graph.launch_captured_mtp_decode()?;
    let captured_mtp = runtime.verifier_read_f32_device(graph.target_logits_device()?)?;
    graph.launch_captured_target_decode()?;
    let captured_target = runtime.verifier_read_f32_device(graph.target_logits_device()?)?;
    graph.enqueue_captured_mtp_acceptance(&runtime)?;
    let captured_acceptance = graph.verifier_read_captured_mtp_acceptance(&runtime)?;
    let captured_mtp_maximum_absolute_error = maximum_absolute_error(&reference_mtp, &captured_mtp);
    let captured_target_maximum_absolute_error =
        maximum_absolute_error(&reference_target, &captured_target);
    anyhow::ensure!(
        captured_mtp_maximum_absolute_error <= 1.0e-5
            && captured_target_maximum_absolute_error <= 1.0e-5,
        "captured target/MTP graphs differ from sequential paths by {captured_target_maximum_absolute_error}/{captured_mtp_maximum_absolute_error}"
    );
    anyhow::ensure!(
        captured_acceptance.accepted_count
            == u32::from(reference_mtp_draft_token == reference_target_verified_token),
        "device MTP acceptance disagrees with host oracle"
    );

    graph.reset_session()?;
    let _ = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    graph.capture_target_decode_graph(&runtime, &Qwen38Config::default(), selected_seed_token)?;
    graph.capture_mtp_decode_graph(&runtime, &Qwen38Config::default())?;
    for _ in 0..args.warmup_pairs {
        graph.launch_captured_mtp_decode()?;
        graph.launch_captured_target_decode()?;
        graph.enqueue_captured_mtp_acceptance(&runtime)?;
    }
    let started = Instant::now();
    for _ in 0..args.measured_pairs {
        graph.launch_captured_mtp_decode()?;
        graph.launch_captured_target_decode()?;
        graph.enqueue_captured_mtp_acceptance(&runtime)?;
    }
    let final_acceptance = graph.verifier_read_captured_mtp_acceptance(&runtime)?;
    let measured_milliseconds = started.elapsed().as_secs_f64() * 1.0e3;
    let verified_target_tokens_per_second =
        args.measured_pairs as f64 / (measured_milliseconds / 1.0e3);
    let target_plus_mtp_graph_steps_per_second =
        (2 * args.measured_pairs) as f64 / (measured_milliseconds / 1.0e3);
    let final_target_tokens = graph.target_tokens();
    let final_mtp_tokens = graph.mtp_tokens();

    graph.reset_session()?;
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA target/MTP graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-target-mtp-graph-benchmark.v1",
            status: "measured_sequential_target_mtp_graph_pair_not_mtp4",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            maximum_context_tokens: args.maximum_context_tokens,
            selected_seed_token,
            reference_mtp_draft_token,
            reference_target_verified_token,
            captured_mtp_maximum_absolute_error,
            captured_target_maximum_absolute_error,
            captured_mtp_exact: reference_mtp == captured_mtp,
            captured_target_exact: reference_target == captured_target,
            captured_draft_accepted: captured_acceptance.accepted_count == 1,
            warmup_pairs: args.warmup_pairs,
            measured_pairs: args.measured_pairs,
            measured_milliseconds,
            verified_target_tokens_per_second,
            target_plus_mtp_graph_steps_per_second,
            final_target_tokens,
            final_mtp_tokens,
            final_draft_accepted: final_acceptance.accepted_count == 1,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            note: "Real complete target plus one-layer MTP graph chain with device-side greedy selection and TensorRT-LLM one-draft acceptance. It verifies captured outputs against sequential complete paths. This is a sequential one-draft correctness/overhead benchmark, not MTP4 acceleration: every emitted token still executes one complete target graph. Prefill and model load are excluded; only one final compact acceptance readback is included in measured time.",
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

fn maximum_absolute_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max)
}
