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
#[command(about = "Benchmark the complete device-resident Qwen target CUDA graph")]
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
    #[arg(long, default_value_t = 2)]
    warmup_tokens: usize,
    #[arg(long, default_value_t = 16)]
    measured_tokens: usize,
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
    warmup_tokens: usize,
    measured_tokens: usize,
    selected_seed_token: usize,
    captured_graph_correctness_maximum_absolute_error: f32,
    captured_graph_correctness_exact: bool,
    measured_milliseconds: f64,
    target_decode_tokens_per_second: f64,
    final_target_tokens: usize,
    final_logits_f32le_sha256: String,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.measured_tokens > 0,
        "measured token count must be positive"
    );
    anyhow::ensure!(
        1 + args.warmup_tokens + args.measured_tokens <= args.maximum_context_tokens,
        "warmup plus measured decode exceeds context capacity"
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
    let reference = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        selected_seed_token,
        1,
    )?;
    let reference = runtime.verifier_read_f32_device(reference)?;

    graph.reset_session()?;
    let _ = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    graph.capture_target_decode_graph(&runtime, &Qwen38Config::default(), selected_seed_token)?;
    anyhow::ensure!(
        graph.has_captured_target_decode_graph(),
        "target graph capture did not commit"
    );
    graph.launch_captured_target_decode()?;
    let captured = runtime.verifier_read_f32_device(graph.target_logits_device()?)?;
    let correctness_maximum_absolute_error = reference
        .iter()
        .zip(&captured)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    anyhow::ensure!(
        correctness_maximum_absolute_error <= 1.0e-5,
        "captured target graph differs from sequential decode by {correctness_maximum_absolute_error}"
    );

    graph.reset_session()?;
    let _ = graph.dispatch_target_token_device(
        &runtime,
        &Qwen38Config::default(),
        args.seed_token_id,
        0,
    )?;
    graph.capture_target_decode_graph(&runtime, &Qwen38Config::default(), selected_seed_token)?;
    for _ in 0..args.warmup_tokens {
        graph.launch_captured_target_decode()?;
    }
    let started = Instant::now();
    for _ in 0..args.measured_tokens {
        graph.launch_captured_target_decode()?;
    }
    let final_logits = runtime.verifier_read_f32_device(graph.target_logits_device()?)?;
    let measured_milliseconds = started.elapsed().as_secs_f64() * 1.0e3;
    let target_decode_tokens_per_second =
        args.measured_tokens as f64 / (measured_milliseconds / 1.0e3);
    let final_target_tokens = graph.target_tokens();
    let final_logits_f32le_sha256 = digest_logits(&final_logits);

    graph.reset_session()?;
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA target graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );

    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-target-graph-benchmark.v1",
            status: "measured_complete_target_graph_without_mtp",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            maximum_context_tokens: args.maximum_context_tokens,
            warmup_tokens: args.warmup_tokens,
            measured_tokens: args.measured_tokens,
            selected_seed_token,
            captured_graph_correctness_maximum_absolute_error:
                correctness_maximum_absolute_error,
            captured_graph_correctness_exact: reference == captured,
            measured_milliseconds,
            target_decode_tokens_per_second,
            final_target_tokens,
            final_logits_f32le_sha256,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            note: "Real end-to-end target-only decode benchmark: embedding, all 64 layers, final norm, LM head, device argmax and device position advance are one captured CUDA graph. Timing includes all graph launches and one final synchronization/readback, but excludes model load, prompt prefill, sampling beyond greedy argmax, and MTP. No result from this verifier may be reported as MTP throughput.",
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

fn digest_logits(logits: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in logits {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
