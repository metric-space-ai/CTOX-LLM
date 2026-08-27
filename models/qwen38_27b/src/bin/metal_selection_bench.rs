#[cfg(target_os = "macos")]
mod macos {
    use std::time::Instant;

    use clap::Parser;
    use ctox_qwen38_27b::backend::metal_runtime::MetalCandidateRuntime;
    use ctox_qwen38_27b::sampler::{Sampler, SamplerConfig};
    use ctox_qwen38_27b::tokenizer::TOKENIZER_VOCAB_SIZE;
    use serde::Serialize;

    #[derive(Debug, Parser)]
    #[command(about = "Verify and benchmark full-vocabulary Metal argmax")]
    struct Args {
        #[arg(long, default_value_t = 20)]
        warmup: usize,
        #[arg(long, default_value_t = 200)]
        iterations: usize,
        #[arg(long, default_value_t = 32)]
        dispatches_per_command: usize,
        #[arg(long, default_value_t = 32)]
        groups: usize,
        #[arg(long, default_value_t = 2)]
        sampling_warmup: usize,
        #[arg(long, default_value_t = 20)]
        sampling_iterations: usize,
    }

    #[derive(Serialize)]
    struct Report<'a> {
        format: &'static str,
        status: &'static str,
        device: &'a str,
        vocabulary_values: usize,
        logical_input_bytes: usize,
        returned_bytes: usize,
        requested_resident_buffer_bytes: usize,
        warmup: usize,
        iterations: usize,
        dispatches_per_command: usize,
        total_dispatches: usize,
        groups: usize,
        selected_token: u32,
        metal_elapsed_milliseconds: f64,
        metal_mean_microseconds: f64,
        metal_logical_gb_per_second: f64,
        host_elapsed_milliseconds: f64,
        host_mean_microseconds: f64,
        sampling_temperature: f32,
        sampling_top_k: usize,
        sampling_top_p: f32,
        sampling_draw: f32,
        sampled_token: u32,
        sampling_nucleus_len: u32,
        sampling_metal_elapsed_milliseconds: f64,
        sampling_metal_mean_microseconds: f64,
        sampling_host_elapsed_milliseconds: f64,
        sampling_host_mean_microseconds: f64,
        note: &'static str,
    }

    pub fn run() -> anyhow::Result<()> {
        let args = Args::parse();
        anyhow::ensure!(args.iterations > 0, "iterations must be positive");
        anyhow::ensure!(
            args.sampling_iterations > 0,
            "sampling-iterations must be positive"
        );
        anyhow::ensure!(
            args.dispatches_per_command > 0,
            "dispatches-per-command must be positive"
        );
        let mut logits: Vec<f32> = (0..TOKENIZER_VOCAB_SIZE)
            .map(|index| ((index as f32 * 0.001_953_125).sin() * 4.0) - 5.0)
            .collect();
        logits[17] = 12.0;
        logits[TOKENIZER_VOCAB_SIZE - 1] = 12.0;
        let expected = host_argmax(&logits)?;
        let runtime = MetalCandidateRuntime::new()?;
        let prepared = runtime.prepare_argmax_f32_with_groups(&logits, args.groups)?;
        let selected = runtime.dispatch_argmax_f32(&prepared)?;
        anyhow::ensure!(
            selected == expected,
            "Metal selected {selected}, host oracle selected {expected}"
        );
        for _ in 0..args.warmup {
            std::hint::black_box(
                runtime.dispatch_argmax_f32_repeated(&prepared, args.dispatches_per_command)?,
            );
        }
        let metal_started = Instant::now();
        for _ in 0..args.iterations {
            std::hint::black_box(
                runtime.dispatch_argmax_f32_repeated(&prepared, args.dispatches_per_command)?,
            );
        }
        let metal_elapsed = metal_started.elapsed().as_secs_f64();
        let total_dispatches = args
            .iterations
            .checked_mul(args.dispatches_per_command)
            .ok_or_else(|| anyhow::anyhow!("total dispatch count overflows"))?;
        let host_started = Instant::now();
        for _ in 0..total_dispatches {
            std::hint::black_box(host_argmax(std::hint::black_box(&logits))?);
        }
        let host_elapsed = host_started.elapsed().as_secs_f64();
        let sampling = SamplerConfig::default();
        let sampling_draw = 0.625_f32;
        let expected_sample = Sampler::new(sampling)?.sample_with_draw(&logits, sampling_draw)?;
        let prepared_sampler = runtime.prepare_topk_topp_f32(&logits)?;
        let sampled =
            runtime.dispatch_topk_topp_sample_f32(&prepared_sampler, sampling, sampling_draw)?;
        anyhow::ensure!(
            sampled.token as usize == expected_sample,
            "Metal sampled {}, host oracle sampled {expected_sample}",
            sampled.token
        );
        for _ in 0..args.sampling_warmup {
            std::hint::black_box(runtime.dispatch_topk_topp_sample_f32(
                &prepared_sampler,
                sampling,
                sampling_draw,
            )?);
        }
        let sampling_metal_started = Instant::now();
        for _ in 0..args.sampling_iterations {
            std::hint::black_box(runtime.dispatch_topk_topp_sample_f32(
                &prepared_sampler,
                sampling,
                sampling_draw,
            )?);
        }
        let sampling_metal_elapsed = sampling_metal_started.elapsed().as_secs_f64();
        let host_sampler = Sampler::new(sampling)?;
        let sampling_host_started = Instant::now();
        for _ in 0..args.sampling_iterations {
            std::hint::black_box(
                host_sampler.sample_with_draw(std::hint::black_box(&logits), sampling_draw)?,
            );
        }
        let sampling_host_elapsed = sampling_host_started.elapsed().as_secs_f64();
        let logical_input_bytes = logits.len() * std::mem::size_of::<f32>();
        let metal_mean_seconds = metal_elapsed / total_dispatches as f64;
        let host_mean_seconds = host_elapsed / total_dispatches as f64;
        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                format: "ctox.metal-argmax-candidate-benchmark.v1",
                status: "verifier_only_not_promotion_evidence",
                device: runtime.device_name(),
                vocabulary_values: logits.len(),
                logical_input_bytes,
                returned_bytes: 2 * std::mem::size_of::<u32>(),
                requested_resident_buffer_bytes: prepared.resident_bytes(),
                warmup: args.warmup,
                iterations: args.iterations,
                dispatches_per_command: args.dispatches_per_command,
                total_dispatches,
                groups: prepared.groups(),
                selected_token: selected,
                metal_elapsed_milliseconds: metal_elapsed * 1.0e3,
                metal_mean_microseconds: metal_mean_seconds * 1.0e6,
                metal_logical_gb_per_second: logical_input_bytes as f64
                    / metal_mean_seconds
                    / 1.0e9,
                host_elapsed_milliseconds: host_elapsed * 1.0e3,
                host_mean_microseconds: host_mean_seconds * 1.0e6,
                sampling_temperature: sampling.temperature,
                sampling_top_k: sampling.top_k,
                sampling_top_p: sampling.top_p,
                sampling_draw,
                sampled_token: sampled.token,
                sampling_nucleus_len: sampled.nucleus_len,
                sampling_metal_elapsed_milliseconds: sampling_metal_elapsed * 1.0e3,
                sampling_metal_mean_microseconds: sampling_metal_elapsed
                    / args.sampling_iterations as f64
                    * 1.0e6,
                sampling_host_elapsed_milliseconds: sampling_host_elapsed * 1.0e3,
                sampling_host_mean_microseconds: sampling_host_elapsed
                    / args.sampling_iterations as f64
                    * 1.0e6,
                note: "Argmax intervals encode repeated resident two-stage selections in one command buffer and amortize command overhead. Sampling intervals execute the bounded top-k/top-p kernel against the same resident full-vocabulary logits and verify every selected token against the canonical Rust sampler. Both remain candidate evidence rather than a hardware-counter roofline measurement.",
            })?
        );
        Ok(())
    }

    fn host_argmax(logits: &[f32]) -> anyhow::Result<u32> {
        anyhow::ensure!(
            !logits.is_empty() && logits.iter().all(|value| value.is_finite()),
            "host argmax requires finite non-empty logits"
        );
        logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32)
            .ok_or_else(|| anyhow::anyhow!("host argmax input is empty"))
    }
}

#[cfg(target_os = "macos")]
fn main() -> anyhow::Result<()> {
    macos::run()
}

#[cfg(not(target_os = "macos"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("qwen38-metal-selection-bench requires macOS")
}
