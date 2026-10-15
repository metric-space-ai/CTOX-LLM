#!/usr/bin/env bash
set -euo pipefail
# Build against the explicitly supplied, provenance-recorded engine snapshot.
# Never alter the historical snapshot or assume current main is equivalent.
student_source=$(realpath "$1")
headers=$(realpath "$2")
out=$(realpath -m "$3")
: "${CARGO_TARGET_DIR:?Use the admitted build lane target directory}"
mkdir -p "$out/crate"
cp "$(dirname "$0")/student.rs" "$out/crate/main.rs"
cat > "$out/crate/Cargo.toml" <<TOML
[package]
name = "sampling-tv-student"
version = "0.1.0"
edition = "2021"
[workspace]
[[bin]]
name = "sampling-tv-student"
path = "main.rs"
[dependencies]
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ctox-qwen38-27b = { path = "$student_source", features = ["cuda"] }
TOML
c++ -std=c++17 -O2 -fPIC -shared -I"$headers" "$(dirname "$0")/q8_bridge.cpp" -o "$out/q8_bridge.so"
cargo build --release -j 2 --manifest-path "$out/crate/Cargo.toml"
cargo test --release -j 2 --manifest-path "$out/crate/Cargo.toml" -- --test-threads=1
cp "$CARGO_TARGET_DIR/release/sampling-tv-student" "$out/"
