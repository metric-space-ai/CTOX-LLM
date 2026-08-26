use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ctox_qwen38_27b::backend::cuda_runtime::{CudaCandidateRuntime, CudaSampledToken};
use ctox_qwen38_27b::sampler::{Sampler, SamplerConfig};
use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Verify deterministic CUDA top-k/top-p sampling parity")]
struct Args {
    #[arg(long)]
    module: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
}

#[derive(Debug, Serialize)]
struct CaseReport {
    name: &'static str,
    values: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    draws: Vec<f32>,
    expected_tokens: Vec<u32>,
    actual_tokens: Vec<u32>,
    nucleus_lengths: Vec<u32>,
    nucleus_totals: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct Report<'a> {
    format: &'static str,
    status: &'static str,
    device: &'a str,
    compute_capability: String,
    module_sha256: String,
    cases: Vec<CaseReport>,
    device_sampling_launches: u64,
    sampler_resident_bytes: usize,
    input_resident_bytes: usize,
    driver_free_bytes_before_prepare: usize,
    driver_free_bytes_after_prepare: usize,
    driver_free_bytes_after_drop: usize,
    observed_reclaimed_bytes: usize,
    note: &'static str,
}

fn verify_case(
    runtime: &CudaCandidateRuntime,
    sampler: &ctox_qwen38_27b::backend::cuda_runtime::PreparedCudaTopKTopPSampler,
    name: &'static str,
    logits: &[f32],
    config: SamplerConfig,
    draws: &[f32],
) -> anyhow::Result<(CaseReport, usize)> {
    let input = runtime.prepare_verifier_f32_tensor(logits)?;
    let oracle = Sampler::new(config)?;
    let mut expected_tokens = Vec::with_capacity(draws.len());
    let mut actual_tokens = Vec::with_capacity(draws.len());
    let mut nucleus_lengths = Vec::with_capacity(draws.len());
    let mut nucleus_totals = Vec::with_capacity(draws.len());
    for draw in draws {
        let expected = u32::try_from(oracle.sample_with_draw(logits, *draw)?)?;
        let CudaSampledToken {
            token,
            nucleus_len,
            nucleus_total,
        } = runtime.dispatch_topk_topp_sample_f32_device(
            sampler,
            input.device_view()?,
            config,
            *draw,
        )?;
        anyhow::ensure!(
            token == expected,
            "{name} draw {draw}: host selected {expected}, CUDA selected {token}"
        );
        expected_tokens.push(expected);
        actual_tokens.push(token);
        nucleus_lengths.push(nucleus_len);
        nucleus_totals.push(nucleus_total);
    }
    let input_bytes = input.resident_bytes();
    Ok((
        CaseReport {
            name,
            values: logits.len(),
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            draws: draws.to_vec(),
            expected_tokens,
            actual_tokens,
            nucleus_lengths,
            nucleus_totals,
        },
        input_bytes,
    ))
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let module = fs::read(&args.module)
        .with_context(|| format!("failed to read CUDA module {}", args.module.display()))?;
    let module_sha256 = format!("{:x}", Sha256::digest(&module));
    let runtime = CudaCandidateRuntime::new(&module, args.device)?;
    let (free_before_prepare, _) = runtime.memory_info()?;
    let sampler = runtime.prepare_topk_topp_sampler(TOKENIZER_VOCAB_SIZE)?;
    let sampler_resident_bytes = sampler.resident_bytes();

    let mut vocabulary_logits: Vec<f32> = (0..TOKENIZER_VOCAB_SIZE)
        .map(|token| -20.0 - (token % 997) as f32 * 0.000_031)
        .collect();
    for rank in 0..96_usize {
        let token = (rank * 7_919 + 31) % TOKENIZER_VOCAB_SIZE;
        vocabulary_logits[token] = 9.0 - rank as f32 * 0.137;
    }
    let standard_draws = [0.0, 0.007_812_5, 0.062_5, 0.25, 0.5, 0.75, 0.968_75];
    let (full_report, input_resident_bytes) = verify_case(
        &runtime,
        &sampler,
        "full-vocabulary-top40-p092",
        &vocabulary_logits,
        SamplerConfig {
            temperature: 0.73,
            top_k: 40,
            top_p: 0.92,
            seed: 0,
        },
        &standard_draws,
    )?;

    let (top_one_report, _) = verify_case(
        &runtime,
        &sampler,
        "later-index-tie-top1",
        &[-3.0, 2.0, 2.0, 1.0, -0.0, 0.0],
        SamplerConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: 0,
        },
        &[0.0, 0.5, 0.999],
    )?;

    let (wide_nucleus_report, _) = verify_case(
        &runtime,
        &sampler,
        "compact-top64-p1",
        &vocabulary_logits[..4_096],
        SamplerConfig {
            temperature: 1.25,
            top_k: 64,
            top_p: 1.0,
            seed: 0,
        },
        &standard_draws,
    )?;

    let cases = vec![full_report, top_one_report, wide_nucleus_report];
    let expected_launches: u64 = cases.iter().map(|case| case.draws.len() as u64).sum();
    let stats = runtime.submission_stats();
    anyhow::ensure!(
        stats.device_sampling_launches == expected_launches,
        "observed {} sampling launches, expected {expected_launches}",
        stats.device_sampling_launches
    );
    let (free_after_prepare, _) = runtime.memory_info()?;
    drop(sampler);
    let (free_after_drop, _) = runtime.memory_info()?;
    let observed_reclaimed_bytes = free_after_drop.saturating_sub(free_after_prepare);
    anyhow::ensure!(
        free_after_drop == free_before_prepare,
        "sampler drop restored {free_after_drop} free bytes, expected baseline {free_before_prepare}"
    );

    let device = runtime.device_name().to_owned();
    let compute_capability = format!(
        "{}.{}",
        runtime.compute_capability().0,
        runtime.compute_capability().1
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            format: "ctox.qwen38.cuda-sampling-verification.v1",
            status: "passed",
            device: &device,
            compute_capability,
            module_sha256,
            cases,
            device_sampling_launches: stats.device_sampling_launches,
            sampler_resident_bytes,
            input_resident_bytes,
            driver_free_bytes_before_prepare: free_before_prepare,
            driver_free_bytes_after_prepare: free_after_prepare,
            driver_free_bytes_after_drop: free_after_drop,
            observed_reclaimed_bytes,
            note: "Verifier-only single-request top-k/top-p candidate derived from pinned TensorRT-LLM selection stages. Exact token parity is checked for canonical caller-supplied draws over full-vocabulary, tie, and wide-nucleus fixtures. Production integration, on-device RNG state, unrestricted top-p, stochastic MTP rejection sampling, distribution tests, and roofline evidence remain open.",
        })?
    );
    Ok(())
}
