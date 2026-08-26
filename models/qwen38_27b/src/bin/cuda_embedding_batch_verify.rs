use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cpu::CpuBackend;
use ctox_qwen38_27b::backend::cuda_runtime::CudaCandidateRuntime;
use ctox_qwen38_27b::backend::{Backend, RecoveredRow};
use ctox_qwen38_27b::format::TensorDType;
use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
use serde::Serialize;
use sha2::{Digest, Sha256};

const EMBEDDING_MATRIX: &str = "model.language_model.embed_tokens.weight";

#[derive(Debug, Parser)]
#[command(about = "Verify mixed Q2/Q4 batched CUDA embedding lookup")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    module: PathBuf,
    #[arg(long)]
    kernel_source: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 2.0e-5)]
    absolute_tolerance: f32,
    #[arg(long, default_value_t = 2.0e-5)]
    relative_tolerance: f32,
}

#[derive(Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    kernel_source_sha256: String,
    module_sha256: String,
    artifact_manifest_sha256: String,
    artifact_file_bytes: u64,
    tensor: &'static str,
    rows: usize,
    columns: usize,
    segment_count: usize,
    q2_segment_count: usize,
    q4_segment_count: usize,
    requested_row_ids: Vec<u32>,
    embedding_model_bytes: usize,
    embedding_graph_bytes: usize,
    batched_workspace_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_allocation_bytes: usize,
    observed_reclaimed_bytes: usize,
    batch_vs_single_bit_mismatches: usize,
    maximum_batch_vs_single_absolute_error: f32,
    maximum_batch_vs_single_relative_error: f32,
    maximum_batch_vs_cpu_absolute_error: f32,
    maximum_batch_vs_cpu_relative_error: f32,
    note: &'static str,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let kernel_source = fs::read(&args.kernel_source).with_context(|| {
        format!(
            "failed to read CUDA kernel source {}",
            args.kernel_source.display()
        )
    })?;
    let kernel_source_sha256 = format!("{:x}", Sha256::digest(&kernel_source));
    let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
        .with_context(|| format!("failed to open CTOXQ artifact {}", args.artifact.display()))?;
    let artifact_file_bytes = artifact.file_bytes();
    let artifact_manifest_sha256 = artifact.manifest_sha256().to_owned();
    let recovered = artifact.recovered_matrix(EMBEDDING_MATRIX)?;
    anyhow::ensure!(
        recovered.matrix.dtype == TensorDType::MixedQ2Q4B64,
        "release embedding must be mixed Q2/Q4 for this verifier"
    );
    let q2_segments: Vec<_> = recovered
        .matrix
        .segments
        .iter()
        .filter(|segment| segment.dtype == TensorDType::Q2B64)
        .collect();
    let q4_segments: Vec<_> = recovered
        .matrix
        .segments
        .iter()
        .filter(|segment| segment.dtype == TensorDType::Q4B64)
        .collect();
    let first_q2 = q2_segments.first().context("embedding has no Q2 segment")?;
    let last_q2 = q2_segments.last().context("embedding has no Q2 segment")?;
    let first_q4 = q4_segments.first().context("embedding has no Q4 segment")?;
    let last_q4 = q4_segments.last().context("embedding has no Q4 segment")?;
    let row_ids = vec![
        row_start(first_q2)?,
        row_end_minus_one(last_q4)?,
        row_end_minus_one(first_q2)?,
        row_start(first_q4)?,
        row_start(last_q2)?,
        row_end_minus_one(first_q4)?,
        row_start(first_q2)?,
        row_start(last_q4)?,
    ];

    let cpu = CpuBackend::scalar_verifier();
    let mut cpu_outputs = Vec::with_capacity(row_ids.len());
    for row_id in &row_ids {
        let row = usize::try_from(*row_id).context("row ID exceeds usize")?;
        let packed = recovered.matrix.row(row)?;
        cpu_outputs.push(cpu.recovered_row(&RecoveredRow {
            dtype: packed.dtype,
            weights: packed.weights,
            columns: packed.columns,
            s_in: recovered.s_in.as_recovery_scales()?,
            s_out: recovered.s_out.value(row)?,
        })?);
    }

    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let prepared = runtime.prepare_embedding_recovered(recovered)?;
    let workspace = runtime.prepare_batched_embedding_workspace(&prepared, row_ids.len())?;
    let embedding_model_bytes = prepared.model_bytes();
    let embedding_graph_bytes = prepared.graph_bytes();
    let batched_workspace_bytes = workspace.transient_bytes();
    let (free_after_prepare, _) = runtime.memory_info()?;

    let mut single_outputs = Vec::with_capacity(row_ids.len());
    for row_id in &row_ids {
        let row = usize::try_from(*row_id).context("row ID exceeds usize")?;
        let view =
            runtime.dispatch_embedding_row_device(&prepared, row, recovered.s_out.value(row)?)?;
        single_outputs.push(runtime.verifier_read_f32_device(view)?);
    }
    let batch_view = runtime.dispatch_embedding_rows_device(&prepared, &workspace, &row_ids)?;
    let batch_output = runtime.verifier_read_f32_device(batch_view)?;
    anyhow::ensure!(
        batch_output.len() == row_ids.len() * recovered.matrix.columns,
        "batched embedding returned the wrong number of values"
    );

    let mut batch_vs_single_bit_mismatches = 0_usize;
    let mut maximum_batch_vs_single_absolute_error = 0.0_f32;
    let mut maximum_batch_vs_single_relative_error = 0.0_f32;
    let mut maximum_batch_vs_cpu_absolute_error = 0.0_f32;
    let mut maximum_batch_vs_cpu_relative_error = 0.0_f32;
    for token in 0..row_ids.len() {
        let start = token * recovered.matrix.columns;
        let actual = &batch_output[start..start + recovered.matrix.columns];
        for ((batch, single), expected) in actual
            .iter()
            .zip(&single_outputs[token])
            .zip(&cpu_outputs[token])
        {
            let single_absolute = (batch - single).abs();
            let single_relative = single_absolute / single.abs().max(f32::MIN_POSITIVE);
            let cpu_absolute = (batch - expected).abs();
            let cpu_relative = cpu_absolute / expected.abs().max(f32::MIN_POSITIVE);
            maximum_batch_vs_single_absolute_error =
                maximum_batch_vs_single_absolute_error.max(single_absolute);
            maximum_batch_vs_single_relative_error =
                maximum_batch_vs_single_relative_error.max(single_relative);
            maximum_batch_vs_cpu_absolute_error =
                maximum_batch_vs_cpu_absolute_error.max(cpu_absolute);
            maximum_batch_vs_cpu_relative_error =
                maximum_batch_vs_cpu_relative_error.max(cpu_relative);
            if batch.to_bits() != single.to_bits() {
                batch_vs_single_bit_mismatches += 1;
            }
            anyhow::ensure!(
                cpu_absolute <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
                "CUDA batched embedding value {batch} differs from CPU oracle {expected} by {cpu_absolute}"
            );
        }
    }
    anyhow::ensure!(
        batch_vs_single_bit_mismatches == 0,
        "batched embedding differs from sequential CUDA lookup in {batch_vs_single_bit_mismatches} values"
    );

    drop(workspace);
    drop(prepared);
    let (free_after_drop, _) = runtime.memory_info()?;
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "CUDA embedding verifier retained {} bytes after drop",
        free_before_prepare.saturating_sub(free_after_drop)
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.cuda-sm86-mixed-embedding-batch-verifier.v1",
            status: "verifier_only_not_promotion_evidence",
            device: runtime.device_name(),
            compute_capability: format!(
                "{}.{}",
                runtime.compute_capability().0,
                runtime.compute_capability().1
            ),
            kernel_source_sha256,
            module_sha256,
            artifact_manifest_sha256,
            artifact_file_bytes,
            tensor: EMBEDDING_MATRIX,
            rows: recovered.matrix.rows,
            columns: recovered.matrix.columns,
            segment_count: recovered.matrix.segments.len(),
            q2_segment_count: q2_segments.len(),
            q4_segment_count: q4_segments.len(),
            requested_row_ids: row_ids,
            embedding_model_bytes,
            embedding_graph_bytes,
            batched_workspace_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_allocation_bytes: free_before_prepare.saturating_sub(free_after_prepare),
            observed_reclaimed_bytes: free_after_drop.saturating_sub(free_after_prepare),
            batch_vs_single_bit_mismatches,
            maximum_batch_vs_single_absolute_error,
            maximum_batch_vs_single_relative_error,
            maximum_batch_vs_cpu_absolute_error,
            maximum_batch_vs_cpu_relative_error,
            note: "One resident mixed Q2/Q4 embedding table serves arbitrary and repeated row IDs through at most one launch per canonical segment. The batched token-major result must be bit-identical to sequential CUDA lookup, stay within the CPU recovery oracle tolerance, and reclaim every verifier-owned allocation. This is per-operation correctness/lifecycle evidence, not a promoted executor.",
        })?
    );
    Ok(())
}

fn row_start(segment: &ctox_qwen38_27b::format::QuantSegment) -> anyhow::Result<u32> {
    u32::try_from(segment.row_start).context("segment row start exceeds u32")
}

fn row_end_minus_one(segment: &ctox_qwen38_27b::format::QuantSegment) -> anyhow::Result<u32> {
    let row = segment
        .row_end
        .checked_sub(1)
        .context("embedding segment is empty")?;
    u32::try_from(row).context("segment row end exceeds u32")
}
