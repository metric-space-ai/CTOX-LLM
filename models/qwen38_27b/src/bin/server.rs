use std::path::PathBuf;

use clap::Parser;
use ctox_qwen38_27b::server::{run_unix, ServerState};

#[derive(Debug, Parser)]
#[command(about = "Run the Qwen3.8 local Responses transport")]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    socket: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let state = ServerState::load(args.artifact)?;
    run_unix(args.socket, &state)?;
    Ok(())
}
