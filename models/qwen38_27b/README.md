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

The first activation-weighted recovery smoke is recorded in
[`docs/RECOVERY_SMOKE_V1.json`](docs/RECOVERY_SMOKE_V1.json). It proves the
BF16 teacher cache, real-graph activation statistics, exact Q2/Q4 sensitivity
simulation, and byte-exact assignment path. The 10-sample/128-token assignment
is pipeline evidence only and is deliberately not promoted as a model artifact.

The expanded provisional calibration in
[`docs/CALIBRATION_160_V1.json`](docs/CALIBRATION_160_V1.json) combines 80
Nemotron and 80 German samples (162,176 observed tokens). It keeps exactly the
same 154-Q4/352-Q2 assignment as the Nemotron-only run. This establishes useful
assignment stability, but the head/MTP and genuine long-context coverage gaps
still prevent recovery training from being called final.

The v2 coverage smoke in
[`docs/ACTIVATION_COVERAGE_SMOKE_V2.json`](docs/ACTIVATION_COVERAGE_SMOKE_V2.json)
closes the tooling gap for all 506 planned matrices: embedding and LM-head
statistics use their correct weighting modes, and the resident MTP checkpoint
is loaded and executed fail-closed. It also verifies mixed 256-row Q2/Q4
segments through the packer and Rust loader. Its ten short samples and identity
recovery scales make it pipeline evidence only, not a model-quality candidate.

The recovery corpus v2 evidence in
[`docs/RECOVERY_CORPUS_V2.json`](docs/RECOVERY_CORPUS_V2.json) records the first
tool-schema-complete NVIDIA Agentic samples plus disjoint bilingual 32K, 64K,
and 128K calibration/evaluation cohorts. These samples are generated and hash
verified. [`docs/TEACHER_CACHE_SMOKE_V1.json`](docs/TEACHER_CACHE_SMOKE_V1.json)
adds real stateful BF16 teacher passes at 32K, 64K, and 128K and complete 32K
activation coverage for all 506 quantized matrices. The same evidence now
includes count-weighted 32K/64K/128K activation bands, a 506-matrix
sensitivity pass, and an exact 7.797994-GiB budget candidate. Those are
single-sample-per-band smoke results; release-size multilingual teacher and
activation cohorts plus recovery training are still pending.
