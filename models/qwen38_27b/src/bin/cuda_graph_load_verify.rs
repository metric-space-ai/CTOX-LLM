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
#[command(about = "Load and release the complete Qwen CUDA projection graph")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 131_072)]
    maximum_context_tokens: usize,
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
    activation_groups: usize,
    projections: usize,
    linear_mixer_layers: usize,
    full_attention_states: usize,
    maximum_context_tokens: usize,
    requested_model_bytes: u64,
    requested_graph_bytes: u64,
    requested_session_bytes: u64,
    requested_resident_bytes: u64,
    driver_total_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_allocation_bytes: usize,
    observed_reclaimed_bytes: usize,
    checksum_and_prepare_milliseconds: f64,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let started = Instant::now();
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_file_bytes = artifact.file_bytes();
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, total) = runtime.memory_info()?;
    let graph = PreparedCudaProjectionGraph::prepare(
        &runtime,
        &artifact,
        &Qwen38Config::default(),
        args.maximum_context_tokens,
    )?;
    let activation_groups = graph.plan().group_count();
    let projections = graph.plan().projection_count();
    let linear_mixer_layers = graph.linear_mixer_count();
    let full_attention_states = graph.full_attention_count();
    let requested_model_bytes = graph.model_bytes();
    let requested_graph_bytes = graph.graph_bytes();
    let requested_session_bytes = graph.session_bytes();
    let requested_resident_bytes = graph.resident_bytes()?;
    let (free_after_prepare, _) = runtime.memory_info()?;
    anyhow::ensure!(
        graph.artifact_manifest_sha256() == artifact_manifest_sha256,
        "prepared CUDA graph changed artifact identity"
    );
    drop(graph);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA projection graph retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );
    let elapsed_milliseconds = started.elapsed().as_secs_f64() * 1.0e3;
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-projection-graph-load.v1",
            status: "verifier_only_not_production_executor",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            module_sha256,
            artifact_manifest_sha256,
            artifact_file_bytes,
            activation_groups,
            projections,
            linear_mixer_layers,
            full_attention_states,
            maximum_context_tokens: args.maximum_context_tokens,
            requested_model_bytes,
            requested_graph_bytes,
            requested_session_bytes,
            requested_resident_bytes,
            driver_total_bytes: total,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes: free_before_prepare.saturating_sub(free_after_prepare),
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            checksum_and_prepare_milliseconds: elapsed_milliseconds,
            note: "Full checksum, all 505 non-embedding target/MTP projections, 48 linear-attention state groups, and 16 target plus one MTP packed Q2/Q4 KV state. This proves artifact binding/residency/unload only; embedding, decoder execution, logits, and roofline promotion remain separate gates.",
        })?
    );
    Ok(())
}
