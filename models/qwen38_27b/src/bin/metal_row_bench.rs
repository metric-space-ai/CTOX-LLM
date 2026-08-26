#[cfg(target_os = "macos")]
mod macos {
    use std::path::PathBuf;
    use std::time::Instant;

    use anyhow::Context;
    use clap::Parser;
    use ctox_qwen38_27b::backend::cpu::CpuBackend;
    use ctox_qwen38_27b::backend::metal_runtime::MetalCandidateRuntime;
    use ctox_qwen38_27b::backend::Backend;
    use ctox_qwen38_27b::format::TensorDType;
    use ctox_qwen38_27b::loader::{ChecksumPolicy, ModelArtifact};
    use serde::Serialize;

    const DEFAULT_EMBEDDING: &str = "model.language_model.embed_tokens.weight";

    #[derive(Debug, Parser)]
    #[command(about = "Verify and benchmark a no-copy recovered Metal embedding row")]
    struct Args {
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long, default_value = DEFAULT_EMBEDDING)]
        tensor: String,
        #[arg(long, default_value_t = 0)]
        row: usize,
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        #[arg(long, default_value_t = 100)]
        iterations: usize,
        #[arg(long, default_value_t = 20)]
        dispatches_per_command: usize,
        #[arg(long, default_value_t = 2.0e-5)]
        absolute_tolerance: f32,
        #[arg(long, default_value_t = 3.0e-5)]
        relative_tolerance: f32,
    }

    #[derive(Serialize)]
    struct Report<'a> {
        format: &'static str,
        status: &'static str,
        device: &'a str,
        artifact: String,
        manifest_sha256: String,
        mapped_file_bytes: u64,
        copied_model_bytes: u64,
        tensor: &'a str,
        row: usize,
        dtype: &'static str,
        columns: usize,
        packed_row_bytes: usize,
        transient_buffer_bytes: usize,
        warmup: usize,
        iterations: usize,
        dispatches_per_command: usize,
        total_dispatches: usize,
        elapsed_milliseconds: f64,
        mean_dispatch_microseconds: f64,
        maximum_absolute_error: f32,
        maximum_relative_error: f32,
        note: &'static str,
    }

    pub fn run() -> anyhow::Result<()> {
        let args = Args::parse();
        anyhow::ensure!(args.iterations > 0, "iterations must be positive");
        anyhow::ensure!(
            args.dispatches_per_command > 0,
            "dispatches-per-command must be positive"
        );
        let artifact = ModelArtifact::open(&args.artifact, ChecksumPolicy::AllTensors)
            .with_context(|| format!("failed to open {}", args.artifact.display()))?;
        let manifest_sha256 = artifact.manifest_sha256().to_owned();
        let matrix = artifact
            .recovered_matrix(&args.tensor)
            .with_context(|| format!("failed to resolve recovered matrix {}", args.tensor))?;
        let operation = matrix
            .row_operation(args.row)
            .with_context(|| format!("failed to resolve row {}", args.row))?;
        let dtype = operation.dtype;
        let packed_row_bytes = operation.weights.len();
        let oracle = CpuBackend::scalar_verifier().recovered_row(&operation)?;
        let runtime = MetalCandidateRuntime::new()?;
        let mapping = runtime.map_artifact_no_copy(&artifact)?;
        let prepared = runtime.prepare_mapped_recovered_row(&mapping, matrix, args.row)?;
        let mapped_file_bytes = mapping.mapped_file_bytes();
        let copied_model_bytes = prepared.copied_model_bytes();
        let transient_buffer_bytes = prepared.transient_bytes();
        let columns = prepared.columns();
        drop(mapping);
        drop(artifact);

        let device_output = runtime.dispatch_mapped_recovered_row(&prepared)?;
        let mut maximum_absolute_error = 0.0_f32;
        let mut maximum_relative_error = 0.0_f32;
        for (expected, actual) in oracle.iter().zip(&device_output) {
            let absolute = (expected - actual).abs();
            let relative = absolute / expected.abs().max(f32::MIN_POSITIVE);
            maximum_absolute_error = maximum_absolute_error.max(absolute);
            maximum_relative_error = maximum_relative_error.max(relative);
            anyhow::ensure!(
                absolute <= args.absolute_tolerance + args.relative_tolerance * expected.abs(),
                "Metal recovered-row value {actual} differs from oracle {expected} by {absolute}"
            );
        }
        for _ in 0..args.warmup {
            std::hint::black_box(
                runtime.dispatch_mapped_recovered_row_repeated(
                    &prepared,
                    args.dispatches_per_command,
                )?,
            );
        }
        let started = Instant::now();
        for _ in 0..args.iterations {
            std::hint::black_box(
                runtime.dispatch_mapped_recovered_row_repeated(
                    &prepared,
                    args.dispatches_per_command,
                )?,
            );
        }
        let elapsed = started.elapsed().as_secs_f64();
        let total_dispatches = args
            .iterations
            .checked_mul(args.dispatches_per_command)
            .context("total dispatch count overflows")?;
        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                format: "ctox.metal-recovered-row-benchmark.v1",
                status: "verifier_only_not_promotion_evidence",
                device: runtime.device_name(),
                artifact: args.artifact.display().to_string(),
                manifest_sha256,
                mapped_file_bytes,
                copied_model_bytes,
                tensor: &args.tensor,
                row: args.row,
                dtype: match dtype {
                    TensorDType::Q2B64 => "q2_b64",
                    TensorDType::Q4B64 => "q4_b64",
                    _ => unreachable!("loader resolves one pure Q2/Q4 row"),
                },
                columns,
                packed_row_bytes,
                transient_buffer_bytes,
                warmup: args.warmup,
                iterations: args.iterations,
                dispatches_per_command: args.dispatches_per_command,
                total_dispatches,
                elapsed_milliseconds: elapsed * 1.0e3,
                mean_dispatch_microseconds: elapsed / total_dispatches as f64 * 1.0e6,
                maximum_absolute_error,
                maximum_relative_error,
                note: "One mapped recovered row remains resident; repeated encoding amortizes command synchronization. Output readback remains in each command interval, and embedding lookup is latency-bound rather than a bandwidth-roofline claim.",
            })?
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("qwen38-metal-row-bench requires macOS")
}
