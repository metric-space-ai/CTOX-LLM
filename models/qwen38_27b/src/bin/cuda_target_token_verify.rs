use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_graph::PreparedCudaProjectionGraph;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
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
    logits: usize,
    logits_f32le_sha256: String,
    top_logits: Vec<RankedLogit>,
    graph_prepare_milliseconds: f64,
    target_dispatch_milliseconds: f64,
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
    let logits = runtime.verifier_read_f32_device(logits_view)?;
    let target_dispatch_milliseconds = dispatch_started.elapsed().as_secs_f64() * 1.0e3;
    anyhow::ensure!(
        logits.len() == Qwen38Config::default().vocab_size,
        "CUDA target logits have {} values, expected {}",
        logits.len(),
        Qwen38Config::default().vocab_size
    );
    let mut ranking: Vec<_> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token_id, value)| RankedLogit { token_id, value })
        .collect();
    ranking.sort_unstable_by(|left, right| right.value.total_cmp(&left.value));
    ranking.truncate(16);
    let mut logits_digest = Sha256::new();
    for value in &logits {
        logits_digest.update(value.to_le_bytes());
    }
    let logits_f32le_sha256 = format!("{:x}", logits_digest.finalize());
    anyhow::ensure!(
        graph.target_tokens() == 1,
        "CUDA target token did not commit"
    );
    graph.reset_session()?;
    anyhow::ensure!(
        graph.target_tokens() == 0,
        "CUDA session reset did not commit"
    );
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
            format: "ctox.cuda-sm86-target-token.v1",
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
            logits: logits.len(),
            logits_f32le_sha256,
            top_logits: ranking,
            graph_prepare_milliseconds,
            target_dispatch_milliseconds,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            note: "Executes embedding, all 64 target layers, final norm, and LM head through device views with no tensor readback before the token boundary. Finite logits and exact unload are necessary but not sufficient: BF16/CPU logit comparison, removal of per-op synchronizations, MTP, sampling, and roofline promotion remain open.",
        })?
    );
    Ok(())
}
