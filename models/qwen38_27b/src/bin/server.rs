use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use ctox_qwen38_27b::backend::ExecutionPolicy;
use ctox_qwen38_27b::decoder::CpuCorrectnessExecutor;
use ctox_qwen38_27b::engine::LoadProgress;
use ctox_qwen38_27b::release::ReleaseManifest;
use ctox_qwen38_27b::server::{run_unix, EngineServer, ServerState};
use ctox_qwen38_27b::{EngineError, Qwen38Config};

#[derive(Debug, Parser)]
#[command(about = "Run the Qwen3.8 local Responses transport")]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    /// Start the artifact-inspection bring-up service. It cannot generate.
    #[arg(long, conflicts_with = "verification_cpu")]
    artifact: Option<PathBuf>,
    /// Start the scalar/SIMD CPU correctness executor under verifier policy.
    /// This mode is never admitted as a production backend.
    #[arg(long, conflicts_with = "artifact")]
    verification_cpu: bool,
    #[arg(long, requires = "verification_cpu")]
    release_root: Option<PathBuf>,
    #[arg(long, requires = "verification_cpu")]
    release_manifest: Option<PathBuf>,
    #[arg(long, requires = "verification_cpu")]
    pack_id: Option<String>,
    #[arg(long, requires = "verification_cpu")]
    memory_profile_id: Option<String>,
    #[arg(long, requires = "verification_cpu")]
    expected_key_id: Option<String>,
    /// File containing a raw 32-byte Ed25519 public key or 64 lowercase hex
    /// characters. The key is an operator trust root, not a release asset.
    #[arg(long, requires = "verification_cpu")]
    trusted_public_key: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match (&args.artifact, args.verification_cpu) {
        (Some(artifact), false) => {
            let state = ServerState::load(artifact)?;
            run_unix(args.socket, &state)?;
        }
        (None, true) => run_cpu_verifier(args)?,
        _ => {
            return Err(EngineError::InvalidArtifact(
                "select exactly one of --artifact or --verification-cpu".into(),
            )
            .into())
        }
    }
    Ok(())
}

fn run_cpu_verifier(args: Args) -> anyhow::Result<()> {
    let release_root = required(args.release_root, "--release-root")?;
    let manifest_path = required(args.release_manifest, "--release-manifest")?;
    let pack_id = required(args.pack_id, "--pack-id")?;
    let memory_profile_id = required(args.memory_profile_id, "--memory-profile-id")?;
    let expected_key_id = required(args.expected_key_id, "--expected-key-id")?;
    let trusted_key_path = required(args.trusted_public_key, "--trusted-public-key")?;

    let release: ReleaseManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let trusted_public_key = read_public_key(&trusted_key_path)?;
    let executor = CpuCorrectnessExecutor::detected(Qwen38Config::default())?;
    let state = EngineServer::load_signed(
        release_root,
        &release,
        &pack_id,
        &memory_profile_id,
        &expected_key_id,
        &trusted_public_key,
        ExecutionPolicy::Verifier,
        executor,
        report_load_progress,
    )?;
    run_unix(args.socket, &state)?;
    Ok(())
}

fn required<T>(value: Option<T>, name: &'static str) -> anyhow::Result<T> {
    value.ok_or_else(|| EngineError::InvalidArtifact(format!("{name} is required")).into())
}

fn read_public_key(path: &Path) -> anyhow::Result<[u8; 32]> {
    let bytes = fs::read(path)?;
    if bytes.len() == 32 {
        return Ok(bytes.try_into().expect("length checked above"));
    }
    let encoded = std::str::from_utf8(&bytes)?.trim();
    if encoded.len() != 64
        || encoded
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(EngineError::InvalidArtifact(
            "trusted public key must be 32 raw bytes or 64 lowercase hex characters".into(),
        )
        .into());
    }
    let mut key = [0_u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&encoded[index * 2..index * 2 + 2], 16).map_err(|_| {
            EngineError::InvalidArtifact("trusted public key contains invalid hex".into())
        })?;
    }
    Ok(key)
}

fn report_load_progress(progress: LoadProgress) {
    eprintln!("load_progress={progress:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_key_accepts_raw_and_lowercase_hex() {
        let directory = tempfile::tempdir().unwrap();
        let raw_path = directory.path().join("raw.key");
        let hex_path = directory.path().join("hex.key");
        fs::write(&raw_path, [0xabu8; 32]).unwrap();
        fs::write(&hex_path, "ab".repeat(32)).unwrap();
        assert_eq!(read_public_key(&raw_path).unwrap(), [0xab; 32]);
        assert_eq!(read_public_key(&hex_path).unwrap(), [0xab; 32]);
    }

    #[test]
    fn public_key_rejects_ambiguous_encoding() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.key");
        fs::write(&path, "AB".repeat(32)).unwrap();
        assert!(read_public_key(&path).is_err());
    }
}
