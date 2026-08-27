//! End-to-end CUDA measurement through the same `Engine` and executor ABI used
//! by the local server. This binary deliberately accepts only release-bound
//! artifacts and real token IDs; it cannot substitute a kernel microbenchmark
//! or a synthetic weight fixture for model throughput.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_executor::ThreadedCudaModelExecutor;
use ctox_qwen38_27b::backend::ExecutionPolicy;
use ctox_qwen38_27b::engine::{
    CancellationToken, Engine, GeneratedStep, LoadProgress, SessionOptions,
};
use ctox_qwen38_27b::release::ReleaseManifest;
use ctox_qwen38_27b::sampler::SamplerConfig;
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Measure real Qwen3.8 CUDA prefill/decode through the Engine ABI")]
struct Args {
    /// Directory containing every release-relative file.
    #[arg(long)]
    release_root: PathBuf,
    /// Preverified release manifest. Production admission still requires the
    /// separately verified signed-release path.
    #[arg(long)]
    release_manifest: PathBuf,
    #[arg(long)]
    pack_id: String,
    #[arg(long)]
    memory_profile_id: String,
    /// Exact SM86 module used by the executor.
    #[arg(long)]
    module: PathBuf,
    /// Non-empty little-endian u32 tokenizer IDs for one real prompt.
    #[arg(long)]
    prompt_token_ids: PathBuf,
    /// Atomic JSON evidence output. Existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    /// Number of output tokens timed after prefill. MTP may cross this bound by
    /// at most one accepted block; the actual count is reported.
    #[arg(long, default_value_t = 128)]
    decode_output_tokens: usize,
    #[arg(long, default_value_t = 0)]
    warmup_repetitions: usize,
    #[arg(long, default_value_t = 3)]
    measured_repetitions: usize,
    #[arg(long)]
    mtp_enabled: bool,
}

#[derive(Debug, Serialize)]
struct RunMeasurement {
    repetition: usize,
    prompt_tokens: usize,
    prefill_seconds: f64,
    prompt_tokens_per_second: f64,
    time_to_first_token_seconds: f64,
    decode_calls: usize,
    decode_output_tokens_requested: usize,
    decode_output_tokens_actual: usize,
    decode_seconds: f64,
    decode_output_tokens_per_second: f64,
    draft_tokens_proposed: u64,
    draft_tokens_verified: u64,
    accepted_draft_tokens: u64,
    draft_acceptance_ratio: f64,
    final_context_tokens: u64,
    generated_sequence_u32le_sha256: String,
}

#[derive(Debug, Serialize)]
struct Summary {
    measured_repetitions: usize,
    median_prefill_seconds: f64,
    median_prompt_tokens_per_second: f64,
    median_decode_seconds: f64,
    median_decode_output_tokens_per_second: f64,
    total_decode_output_tokens: usize,
    total_draft_tokens_proposed: u64,
    total_accepted_draft_tokens: u64,
    aggregate_draft_acceptance_ratio: f64,
    deterministic_generated_sequence: bool,
    promotion_sample_count_gate: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    format: &'static str,
    status: &'static str,
    measurement_scope: &'static str,
    release_id: String,
    release_manifest_path: String,
    release_manifest_bytes: u64,
    release_manifest_sha256: String,
    release_root: String,
    pack_id: String,
    memory_profile_id: String,
    hardware_profile: String,
    cuda_device_ordinal: i32,
    module_path: String,
    module_bytes: u64,
    module_sha256: String,
    prompt_token_ids_path: String,
    prompt_token_ids_bytes: u64,
    prompt_token_ids_sha256: String,
    prompt_tokens: usize,
    admitted_context_tokens: u64,
    mtp_enabled: bool,
    no_hidden_fallbacks: bool,
    resident_target_selection: bool,
    warmup_repetitions: usize,
    load_progress: Vec<&'static str>,
    load_seconds: f64,
    executor_warmup_seconds: f64,
    resident_model_bytes: u64,
    resident_graph_bytes: u64,
    resident_session_bytes: u64,
    engine_prefill_calls: u64,
    engine_decode_calls: u64,
    engine_last_prefill_micros: u64,
    engine_last_decode_micros: u64,
    runs: Vec<RunMeasurement>,
    summary: Summary,
    unload_seconds: f64,
    allocations_zero_after_unload: bool,
    limitations: Vec<&'static str>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.decode_output_tokens > 0,
        "decode-output-tokens must be positive"
    );
    anyhow::ensure!(
        args.measured_repetitions > 0,
        "measured-repetitions must be positive"
    );
    anyhow::ensure!(
        !args.output.exists(),
        "refusing to overwrite {}",
        args.output.display()
    );

    let release_root = args.release_root.canonicalize().with_context(|| {
        format!(
            "failed to resolve release root {}",
            args.release_root.display()
        )
    })?;
    let release_manifest_path = args.release_manifest.canonicalize().with_context(|| {
        format!(
            "failed to resolve release manifest {}",
            args.release_manifest.display()
        )
    })?;
    let module_path = args
        .module
        .canonicalize()
        .with_context(|| format!("failed to resolve CUDA module {}", args.module.display()))?;
    let prompt_path = args.prompt_token_ids.canonicalize().with_context(|| {
        format!(
            "failed to resolve prompt token file {}",
            args.prompt_token_ids.display()
        )
    })?;

    let release_bytes = fs::read(&release_manifest_path)?;
    let release: ReleaseManifest = serde_json::from_slice(&release_bytes)?;
    release.validate()?;
    let profile = release.memory_profile(&args.memory_profile_id)?.clone();
    anyhow::ensure!(
        profile.pack_id == args.pack_id,
        "memory profile belongs to another backend pack"
    );
    let module = fs::read(&module_path)?;
    let prompt_bytes = fs::read(&prompt_path)?;
    let prompt_tokens = parse_token_ids(&prompt_bytes)?;
    let reserved_tail = usize::from(args.mtp_enabled) * 4;
    let required_context = prompt_tokens
        .len()
        .checked_add(args.decode_output_tokens)
        .and_then(|tokens| tokens.checked_add(reserved_tail))
        .context("benchmark context requirement overflows")?;
    anyhow::ensure!(
        required_context as u64 <= profile.context_tokens,
        "prompt, requested decode, and MTP tail require {required_context} tokens but profile admits {}",
        profile.context_tokens
    );

    let executor = ThreadedCudaModelExecutor::new_sm86(&module, args.device)?;
    let mut load_progress = Vec::new();
    let load_started = Instant::now();
    let mut engine = Engine::load_preverified_release(
        &release_root,
        &release,
        &args.pack_id,
        &args.memory_profile_id,
        ExecutionPolicy::Verifier,
        executor,
        |progress| load_progress.push(load_progress_name(progress)),
    )?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let capabilities = engine.capabilities().clone();
    anyhow::ensure!(
        capabilities.no_hidden_fallbacks && capabilities.resident_target_selection,
        "CUDA benchmark requires no hidden fallbacks and resident target selection"
    );
    let warmup_started = Instant::now();
    engine.warmup()?;
    let executor_warmup_seconds = warmup_started.elapsed().as_secs_f64();
    let resident = engine.health().allocations;
    anyhow::ensure!(
        resident.model_bytes > 0
            && resident.graph_bytes > 0
            && resident.session_bytes > 0
            && resident.global_cache_bytes == 0,
        "CUDA executor did not report bounded resident allocations"
    );

    let mut warmup_decode_calls = 0_u64;
    for repetition in 0..args.warmup_repetitions {
        let warmup = run_once(
            &mut engine,
            &prompt_tokens,
            args.decode_output_tokens,
            args.mtp_enabled,
            u64::MAX - repetition as u64,
            repetition,
        )?;
        warmup_decode_calls = warmup_decode_calls
            .checked_add(warmup.decode_calls as u64)
            .context("warmup decode count overflows")?;
        engine.reset_session()?;
    }

    let mut runs = Vec::with_capacity(args.measured_repetitions);
    for repetition in 0..args.measured_repetitions {
        runs.push(run_once(
            &mut engine,
            &prompt_tokens,
            args.decode_output_tokens,
            args.mtp_enabled,
            repetition as u64 + 1,
            repetition,
        )?);
        engine.reset_session()?;
    }
    let deterministic_generated_sequence = runs.windows(2).all(|pair| {
        pair[0].generated_sequence_u32le_sha256 == pair[1].generated_sequence_u32le_sha256
    });
    anyhow::ensure!(
        deterministic_generated_sequence,
        "identical greedy benchmark repetitions produced different token sequences"
    );
    let summary = summarize(&runs, deterministic_generated_sequence)?;
    if args.mtp_enabled {
        anyhow::ensure!(
            summary.total_draft_tokens_proposed > 0,
            "MTP benchmark completed without a draft proposal"
        );
    }
    let engine_metrics = engine.metrics().clone();
    let expected_prefill_calls = (args.warmup_repetitions + args.measured_repetitions) as u64;
    let expected_decode_calls = warmup_decode_calls
        .checked_add(runs.iter().map(|run| run.decode_calls as u64).sum())
        .context("total decode count overflows")?;
    anyhow::ensure!(
        engine_metrics.prefill_calls == expected_prefill_calls
            && engine_metrics.decode_calls == expected_decode_calls,
        "engine operation counters differ from the benchmark record"
    );

    let unload_started = Instant::now();
    engine.unload()?;
    let unload_seconds = unload_started.elapsed().as_secs_f64();
    let allocations_zero_after_unload = engine.health().allocations.is_zero();
    anyhow::ensure!(
        allocations_zero_after_unload,
        "CUDA engine retained allocations after unload"
    );

    let report = Report {
        format: "ctox.cuda-sm86-engine-e2e-benchmark.v1",
        status: "measured_verifier_only_not_promoted",
        measurement_scope: "complete_release_bound_engine_prefill_and_incremental_decode",
        release_id: release.release_id.clone(),
        release_manifest_path: release_manifest_path.display().to_string(),
        release_manifest_bytes: release_bytes.len() as u64,
        release_manifest_sha256: digest(&release_bytes),
        release_root: release_root.display().to_string(),
        pack_id: args.pack_id,
        memory_profile_id: args.memory_profile_id,
        hardware_profile: engine.health().hardware_profile,
        cuda_device_ordinal: args.device,
        module_path: module_path.display().to_string(),
        module_bytes: module.len() as u64,
        module_sha256: digest(&module),
        prompt_token_ids_path: prompt_path.display().to_string(),
        prompt_token_ids_bytes: prompt_bytes.len() as u64,
        prompt_token_ids_sha256: digest(&prompt_bytes),
        prompt_tokens: prompt_tokens.len(),
        admitted_context_tokens: profile.context_tokens,
        mtp_enabled: args.mtp_enabled,
        no_hidden_fallbacks: capabilities.no_hidden_fallbacks,
        resident_target_selection: capabilities.resident_target_selection,
        warmup_repetitions: args.warmup_repetitions,
        load_progress,
        load_seconds,
        executor_warmup_seconds,
        resident_model_bytes: resident.model_bytes,
        resident_graph_bytes: resident.graph_bytes,
        resident_session_bytes: resident.session_bytes,
        engine_prefill_calls: engine_metrics.prefill_calls,
        engine_decode_calls: engine_metrics.decode_calls,
        engine_last_prefill_micros: engine_metrics.last_prefill_micros,
        engine_last_decode_micros: engine_metrics.last_decode_micros,
        runs,
        summary,
        unload_seconds,
        allocations_zero_after_unload,
        limitations: vec![
            "Verifier policy is used until CUDA passes the full promotion gate.",
            "This command reports only observations from the supplied release, prompt, module, device, and run; it performs no extrapolation.",
            "Promotion additionally requires controlled clocks/thermals, a same-device pinned reference, BF16 golden logits, and the complete context sweep.",
        ],
    };
    write_atomic_json(&args.output, &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn run_once(
    engine: &mut Engine<ThreadedCudaModelExecutor>,
    prompt_tokens: &[u32],
    decode_output_tokens_requested: usize,
    mtp_enabled: bool,
    session_id: u64,
    repetition: usize,
) -> anyhow::Result<RunMeasurement> {
    let cancellation = CancellationToken::default();
    let options = SessionOptions {
        id: session_id,
        mtp_enabled,
        sampling: SamplerConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            seed: 0,
        },
    };
    let prefill_started = Instant::now();
    let first = engine.prefill(options, prompt_tokens, &cancellation)?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    anyhow::ensure!(prefill_seconds > 0.0, "prefill timer did not advance");

    let mut generated = Sha256::new();
    update_token_digest(&mut generated, first.token_id);
    let mut next_token = first.token_id;
    let mut decode_calls = 0_usize;
    let mut decode_output_tokens_actual = 0_usize;
    let mut draft_tokens_proposed = 0_u64;
    let mut draft_tokens_verified = 0_u64;
    let mut accepted_draft_tokens = 0_u64;
    let decode_started = Instant::now();
    while decode_output_tokens_actual < decode_output_tokens_requested {
        let step = engine.decode(session_id, next_token, &cancellation)?;
        account_step(
            &step,
            &mut generated,
            &mut draft_tokens_proposed,
            &mut draft_tokens_verified,
            &mut accepted_draft_tokens,
        );
        decode_calls += 1;
        decode_output_tokens_actual = decode_output_tokens_actual
            .checked_add(1 + step.accepted_draft_tokens.len())
            .context("decoded token count overflows")?;
        next_token = step.token_id;
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    anyhow::ensure!(decode_seconds > 0.0, "decode timer did not advance");
    let final_context_tokens = engine
        .health()
        .session
        .context("benchmark session disappeared before measurement")?
        .context_tokens;
    anyhow::ensure!(
        final_context_tokens == (prompt_tokens.len() + decode_output_tokens_actual) as u64,
        "engine context counter differs from emitted decode tokens"
    );
    let draft_acceptance_ratio = ratio(accepted_draft_tokens, draft_tokens_proposed);
    Ok(RunMeasurement {
        repetition,
        prompt_tokens: prompt_tokens.len(),
        prefill_seconds,
        prompt_tokens_per_second: prompt_tokens.len() as f64 / prefill_seconds,
        time_to_first_token_seconds: prefill_seconds,
        decode_calls,
        decode_output_tokens_requested,
        decode_output_tokens_actual,
        decode_seconds,
        decode_output_tokens_per_second: decode_output_tokens_actual as f64 / decode_seconds,
        draft_tokens_proposed,
        draft_tokens_verified,
        accepted_draft_tokens,
        draft_acceptance_ratio,
        final_context_tokens,
        generated_sequence_u32le_sha256: format!("{:x}", generated.finalize()),
    })
}

fn account_step(
    step: &GeneratedStep,
    digest: &mut Sha256,
    proposed: &mut u64,
    verified: &mut u64,
    accepted: &mut u64,
) {
    *proposed = proposed.saturating_add(u64::from(step.draft_tokens_proposed));
    *verified = verified.saturating_add(u64::from(step.draft_tokens_verified));
    *accepted = accepted.saturating_add(step.accepted_draft_tokens.len() as u64);
    for token in &step.accepted_draft_tokens {
        update_token_digest(digest, *token);
    }
    update_token_digest(digest, step.token_id);
}

fn summarize(runs: &[RunMeasurement], deterministic: bool) -> anyhow::Result<Summary> {
    anyhow::ensure!(!runs.is_empty(), "benchmark has no measured repetitions");
    let total_decode_output_tokens = runs.iter().map(|run| run.decode_output_tokens_actual).sum();
    let total_draft_tokens_proposed = runs.iter().map(|run| run.draft_tokens_proposed).sum();
    let total_accepted_draft_tokens = runs.iter().map(|run| run.accepted_draft_tokens).sum();
    Ok(Summary {
        measured_repetitions: runs.len(),
        median_prefill_seconds: median(runs.iter().map(|run| run.prefill_seconds))?,
        median_prompt_tokens_per_second: median(
            runs.iter().map(|run| run.prompt_tokens_per_second),
        )?,
        median_decode_seconds: median(runs.iter().map(|run| run.decode_seconds))?,
        median_decode_output_tokens_per_second: median(
            runs.iter().map(|run| run.decode_output_tokens_per_second),
        )?,
        total_decode_output_tokens,
        total_draft_tokens_proposed,
        total_accepted_draft_tokens,
        aggregate_draft_acceptance_ratio: ratio(
            total_accepted_draft_tokens,
            total_draft_tokens_proposed,
        ),
        deterministic_generated_sequence: deterministic,
        promotion_sample_count_gate: runs.len() >= 3,
    })
}

fn median(values: impl IntoIterator<Item = f64>) -> anyhow::Result<f64> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    anyhow::ensure!(
        !values.is_empty() && values.iter().all(|value| value.is_finite()),
        "median requires finite observations"
    );
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Ok(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn parse_token_ids(bytes: &[u8]) -> anyhow::Result<Vec<u32>> {
    anyhow::ensure!(
        !bytes.is_empty() && bytes.len().is_multiple_of(4),
        "prompt token file must be a non-empty u32-LE array"
    );
    let tokens = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        tokens
            .iter()
            .all(|token| (*token as usize) < TOKENIZER_VOCAB_SIZE),
        "prompt contains a token outside the tokenizer vocabulary"
    );
    Ok(tokens)
}

fn load_progress_name(progress: LoadProgress) -> &'static str {
    match progress {
        LoadProgress::SignatureVerified => "signature_verified",
        LoadProgress::TokenizerVerified => "tokenizer_verified",
        LoadProgress::DraftVocabularyVerified => "draft_vocabulary_verified",
        LoadProgress::ArtifactOpened => "artifact_opened",
        LoadProgress::ArtifactAdmitted => "artifact_admitted",
        LoadProgress::BackendLoaded => "backend_loaded",
    }
}

fn update_token_digest(digest: &mut Sha256, token: u32) {
    digest.update(token.to_le_bytes());
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_atomic_json(path: &Path, report: &Report) -> anyhow::Result<()> {
    let parent = path.parent().context("output has no parent directory")?;
    fs::create_dir_all(parent)?;
    anyhow::ensure!(!path.exists(), "refusing to overwrite {}", path.display());
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .context("output filename is not UTF-8")?,
        std::process::id()
    ));
    anyhow::ensure!(
        !temporary.exists(),
        "temporary output already exists: {}",
        temporary.display()
    );
    let payload = serde_json::to_vec_pretty(report)?;
    let result = (|| -> anyhow::Result<()> {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(&payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_file_is_strict_u32le_and_range_checked() {
        let bytes = [0_u32, 7, (TOKENIZER_VOCAB_SIZE - 1) as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            parse_token_ids(&bytes).unwrap(),
            [0, 7, (TOKENIZER_VOCAB_SIZE - 1) as u32]
        );
        assert!(parse_token_ids(&[]).is_err());
        assert!(parse_token_ids(&[0, 1, 2]).is_err());
        assert!(parse_token_ids(&(TOKENIZER_VOCAB_SIZE as u32).to_le_bytes()).is_err());
    }

    #[test]
    fn median_handles_even_and_odd_measurement_counts() {
        assert_eq!(median([3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median([4.0, 1.0, 3.0, 2.0]).unwrap(), 2.5);
        assert!(median([]).is_err());
        assert!(median([f64::NAN]).is_err());
    }

    #[test]
    fn accepted_tokens_precede_the_verified_fallback_in_sequence_hash() {
        let step = GeneratedStep {
            token_id: 11,
            draft_tokens_proposed: 4,
            draft_tokens_verified: 3,
            accepted_draft_tokens: vec![7, 9],
        };
        let mut observed = Sha256::new();
        let mut proposed = 0;
        let mut verified = 0;
        let mut accepted = 0;
        account_step(
            &step,
            &mut observed,
            &mut proposed,
            &mut verified,
            &mut accepted,
        );
        let mut expected = Sha256::new();
        for token in [7_u32, 9, 11] {
            update_token_digest(&mut expected, token);
        }
        assert_eq!(observed.finalize(), expected.finalize());
        assert_eq!((proposed, verified, accepted), (4, 3, 2));
    }
}
