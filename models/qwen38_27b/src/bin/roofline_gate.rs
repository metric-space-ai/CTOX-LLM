use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use ctox_qwen38_27b::roofline::RooflineMeasurement;

#[derive(Debug, Parser)]
#[command(about = "Evaluate one measured Qwen3.8 hardware roofline interval")]
struct Args {
    #[arg(long)]
    input: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let encoded = fs::read(&args.input)
        .with_context(|| format!("failed to read {}", args.input.display()))?;
    let measurement: RooflineMeasurement = serde_json::from_slice(&encoded)
        .with_context(|| format!("invalid roofline input {}", args.input.display()))?;
    let report = measurement.evaluate()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    anyhow::ensure!(
        report.optimized_gate_passed,
        "hardware profile misses the practical roofline gate"
    );
    Ok(())
}
