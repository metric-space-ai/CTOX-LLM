use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use ctox_qwen38_27b::format::RecoveryProvenance;
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use ctox_qwen38_27b::memory::FoldMemoryPlan;
use ctox_qwen38_27b::tensor_contract::validate_tensor_contract;
use ctox_qwen38_27b::Qwen38Config;
use serde::Serialize;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Checksums {
    Manifest,
    All,
}

#[derive(Debug, Parser)]
#[command(about = "Validate a CTOX Q2/Q4 model artifact")]
struct Args {
    artifact: PathBuf,
    #[arg(long, value_enum, default_value_t = Checksums::All)]
    checksums: Checksums,
    #[arg(long, default_value_t = 131_072)]
    context: u64,
}

#[derive(Serialize)]
struct Report<'a> {
    valid: bool,
    model: &'a str,
    revision: &'a str,
    target: &'a str,
    recovery: Option<&'a RecoveryProvenance>,
    tensors: usize,
    artifact_bytes: u64,
    resident_weights_bytes: u64,
    fold_plan: FoldMemoryPlan,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let artifact = ModelArtifact::open(
        &args.artifact,
        match args.checksums {
            Checksums::Manifest => ChecksumPolicy::ManifestOnly,
            Checksums::All => ChecksumPolicy::AllTensors,
        },
    )?;
    let artifact_bytes = std::fs::metadata(&args.artifact)?.len();
    validate_tensor_contract(artifact.manifest(), &Qwen38Config::default())?;
    let resident_weights_bytes = artifact
        .manifest()
        .tensors
        .iter()
        .map(|tensor| tensor.offset + tensor.length)
        .max()
        .unwrap_or(0);
    let fold_plan = FoldMemoryPlan::for_context(
        &Qwen38Config::default(),
        args.context,
        resident_weights_bytes,
    )?;
    fold_plan.verify()?;
    let manifest = artifact.manifest();
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            valid: true,
            model: &manifest.model,
            revision: &manifest.revision,
            target: &manifest.target,
            recovery: manifest.recovery.as_ref(),
            tensors: manifest.tensors.len(),
            artifact_bytes,
            resident_weights_bytes,
            fold_plan,
        })?
    );
    Ok(())
}
