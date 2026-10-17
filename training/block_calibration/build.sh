#!/usr/bin/env bash
set -euo pipefail
student_source=$(realpath "$1"); headers=$(realpath "$2"); out=$(realpath -m "$3")
: "${CARGO_TARGET_DIR:?Use admitted build lane}"
mkdir -p "$out/crate/block_calibration" "$out/crate/sampling_tv"
cp "$(dirname "$0")/student.rs" "$out/crate/block_calibration/"
cp "$(dirname "$0")/../sampling_tv/student.rs" "$out/crate/sampling_tv/"
cat > "$out/crate/Cargo.toml" <<TOML
[package]
name="block-calibration-student"
version="0.1.0"
edition="2021"
[workspace]
[[bin]]
name="block-calibration-student"
path="block_calibration/student.rs"
[dependencies]
anyhow="1"
serde={version="1",features=["derive"]}
serde_json="1"
ctox-qwen38-27b={path="$student_source",features=["cuda"]}
TOML
c++ -std=c++17 -O2 -fPIC -shared -I"$headers" "$(dirname "$0")/bridge.cpp" -o "$out/block_bridge.so"
cargo build --release -j 2 --manifest-path "$out/crate/Cargo.toml"
cp "$CARGO_TARGET_DIR/release/block-calibration-student" "$out/"
