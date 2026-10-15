#!/usr/bin/env bash
set -euo pipefail
student_source=$(realpath "$1")
out=$(realpath -m "$2")
: "${CARGO_TARGET_DIR:?Use the admitted build lane}"
mkdir -p "$out/crate/chat_calibration" "$out/crate/sampling_tv"
cp "$(dirname "$0")/student.rs" "$(dirname "$0")/head.rs" "$out/crate/chat_calibration/"
cp "$(dirname "$0")/../sampling_tv/student.rs" "$out/crate/sampling_tv/"
cat > "$out/crate/Cargo.toml" <<TOML
[package]
name = "chat-calibration-student"
version = "0.1.0"
edition = "2021"
[workspace]
[[bin]]
name = "chat-calibration-student"
path = "chat_calibration/student.rs"
[[bin]]
name = "chat-calibration-head"
path = "chat_calibration/head.rs"
[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ctox-qwen38-27b = { path = "$student_source", features = ["cuda"] }
TOML
cargo build --release -j 2 --manifest-path "$out/crate/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/chat-calibration-student" "$CARGO_TARGET_DIR/release/chat-calibration-head" "$out/"
