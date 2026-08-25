# ctox-qwen38-27b

Self-contained Qwen3.8-27B Q2/Q4 inference integration. This crate owns all
runtime code for this model and must not depend on another model crate.

The frozen text configuration is 64 layers with a repeating three
linear-attention plus one full-attention pattern. Full attention uses four KV
heads of dimension 256. Text supports 262,144 positions; the Fold acceptance
profile is 131,072 positions.

## Backend status

| Backend | Current state | Production fallback |
|---|---|---|
| CPU scalar | verifier | forbidden |
| CPU AVX2/NEON | experimental dot kernels | none |
| CUDA | contract | none |
| Metal | contract | none |
| Snapdragon HTP/Vulkan | contract | none |

The status table is intentionally conservative. Update it only with verifier
and benchmark artifacts.

## First native artifact

The first direct BF16-to-native pack is independently verified at 7.7695 GiB
including its manifest. It contains text plus resident MTP and produces a
9.5748 GiB calculated 128K Fold plan. Its recovery scales are still identity
values; it is a format/memory baseline, not the final quality checkpoint. Exact
hashes and counts are recorded in [`docs/NATIVE_ARTIFACT_V1.json`](docs/NATIVE_ARTIFACT_V1.json).
