# CTOX-LLM

Model-specific, bare-metal inference engines for CTOX. Each model integration
owns its Rust driver, model graph, loader, memory planner, wire protocol, and
hardware kernels. Runtime dependencies on general inference frameworks are not
accepted.

The first integration is `Qwen3.8-27B`, targeting:

- x86-64 and AArch64 CPUs;
- NVIDIA CUDA;
- Apple Metal;
- Qualcomm Hexagon HTP plus Adreno Vulkan on Android.

## Status

The repository is under active bring-up. The model-independent Q2/Q4 format,
Qwen3.8 topology, memory-budget verifier, scalar correctness oracle, and CPU
feature dispatcher are implemented. Accelerator backends are promoted only
after their per-op verifier and same-hardware benchmark gates pass.

No backend is called production-ready merely because it compiles. See
[`docs/PROMOTION_GATES.md`](docs/PROMOTION_GATES.md).

## Build

```sh
cargo test --manifest-path models/qwen38_27b/Cargo.toml
cargo run --manifest-path models/qwen38_27b/Cargo.toml \
  --bin qwen38-memory-plan -- --context 131072
```

Large weights are published separately. The source tree must not contain model
checkpoints, datasets, SDK installations, generated caches, or build output.

## Design rules

- Every model crate is runtime-self-contained. A later model must not depend on
  the Qwen3.8 crate.
- Large weight matrices use Q2 or Q4. Q3 is rejected by the loader.
- Scalar kernels are correctness oracles, never silent production fallbacks.
- Proprietary Qualcomm SDK content is supplied by the developer and is never
  committed.
- Vendored kernels require an immutable upstream pin, license record, and
  source-location anchors in their Rust dispatcher.

## License

Original source code is licensed under Apache-2.0. Model weights, datasets,
vendor sources, and SDKs retain their own licenses; see [`NOTICE`](NOTICE).
