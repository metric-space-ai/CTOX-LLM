use clap::{Parser, ValueEnum};
use ctox_qwen38_27b::memory::{
    FoldMemoryPlan, LinearStateDType, SpeculativeStateStrategy, FOLD_WEIGHT_LIMIT_BYTES,
};
use ctox_qwen38_27b::Qwen38Config;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum LinearStateArg {
    F16,
    F32,
}

impl From<LinearStateArg> for LinearStateDType {
    fn from(value: LinearStateArg) -> Self {
        match value {
            LinearStateArg::F16 => Self::F16,
            LinearStateArg::F32 => Self::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SpeculativeStateArg {
    Disabled,
    ReplayOnReject,
    AlignedPages,
}

impl From<SpeculativeStateArg> for SpeculativeStateStrategy {
    fn from(value: SpeculativeStateArg) -> Self {
        match value {
            SpeculativeStateArg::Disabled => Self::Disabled,
            SpeculativeStateArg::ReplayOnReject => Self::ReplayOnReject,
            SpeculativeStateArg::AlignedPages => Self::AlignedPages,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Calculate and verify the Qwen3.8 Fold memory plan")]
struct Args {
    #[arg(long, default_value_t = 131_072)]
    context: u64,
    #[arg(long, default_value_t = FOLD_WEIGHT_LIMIT_BYTES)]
    weights_bytes: u64,
    #[arg(long, value_enum, default_value_t = LinearStateArg::F32)]
    linear_state: LinearStateArg,
    #[arg(long, default_value_t = 0)]
    mtp_draft_tokens: u32,
    #[arg(long, value_enum, default_value_t = SpeculativeStateArg::Disabled)]
    speculative_state: SpeculativeStateArg,
    #[arg(long)]
    json: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let plan = FoldMemoryPlan::for_execution(
        &Qwen38Config::default(),
        args.context,
        args.weights_bytes,
        args.linear_state.into(),
        args.mtp_draft_tokens,
        args.speculative_state.into(),
    )?;
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
            "MTP KV:         {:.3} GiB",
            FoldMemoryPlan::gib(plan.mtp_kv_bytes)
        );
        println!(
            "linear state:   {:.3} GiB (recurrent {:.3} + conv {:.3})",
            FoldMemoryPlan::gib(plan.linear_state_bytes),
            FoldMemoryPlan::gib(plan.linear_recurrent_state_bytes),
            FoldMemoryPlan::gib(plan.linear_convolution_state_bytes)
        );
        println!(
            "spec state:     {:.3} GiB ({:?}, {} drafts)",
            FoldMemoryPlan::gib(plan.speculative_extra_linear_state_bytes),
            plan.speculative_state_strategy,
            plan.speculative_draft_tokens
        );
        println!(
            "runtime:        {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_bytes)
        );
        println!(
            "  code/rodata:  {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.executable_code_and_rodata_bytes)
        );
        println!(
            "  Java/JNI/UI:  {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.java_jni_ui_bytes)
        );
        println!(
            "  graph/control:{:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.tokenizer_sampler_graph_bytes)
        );
        println!(
            "  native heap:  {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.native_heap_stacks_allocator_bytes)
        );
        println!(
            "  accel control:{:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.accelerator_commands_descriptors_bytes)
        );
        println!(
            "  workspaces:   {:.3} GiB",
            FoldMemoryPlan::gib(plan.runtime_budget.kernel_workspaces_bytes)
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
