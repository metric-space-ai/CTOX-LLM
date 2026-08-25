use clap::Parser;
use ctox_qwen38_27b::memory::{FoldMemoryPlan, FOLD_WEIGHT_LIMIT_BYTES};
use ctox_qwen38_27b::Qwen38Config;

#[derive(Debug, Parser)]
#[command(about = "Calculate and verify the Qwen3.8 Fold memory plan")]
struct Args {
    #[arg(long, default_value_t = 131_072)]
    context: u64,
    #[arg(long, default_value_t = FOLD_WEIGHT_LIMIT_BYTES)]
    weights_bytes: u64,
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let plan =
        FoldMemoryPlan::for_context(&Qwen38Config::default(), args.context, args.weights_bytes)?;
    plan.verify()?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
    } else {
        println!("context:       {} tokens", plan.context_tokens);
        println!(
            "weights:       {:.3} GiB",
            FoldMemoryPlan::gib(plan.weights_bytes)
        );
        println!(
            "KV Q2 + scales: {:.3} GiB",
            FoldMemoryPlan::gib(plan.kv_raw_q2_bytes + plan.kv_scale_bytes)
        );
        println!(
            "KV Q4 delta:    {:.3} GiB",
            FoldMemoryPlan::gib(plan.kv_q4_recent_and_sink_bytes)
        );
        println!(
            "linear state:   {:.3} GiB",
            FoldMemoryPlan::gib(plan.linear_state_bytes)
        );
        println!(
            "runtime:        {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_bytes)
        );
        println!(
            "total:          {:.3} GiB",
            FoldMemoryPlan::gib(plan.total_bytes)
        );
        println!(
            "target:         {:.3} GiB",
            FoldMemoryPlan::gib(plan.target_bytes)
        );
    }
    Ok(())
}
