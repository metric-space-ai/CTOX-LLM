# Metal Q2/Q4 fused matvec candidate notes

Status: **candidate, not promoted**. The backend reports
`PromotionState::Contract` and its production `fused_matvec` entry point fails
closed with `EngineError::UnsupportedOperation`. A separately named
`MetalCandidateRuntime` now performs direct same-device verifier dispatch, but
it cannot be selected as a production `Backend` until complete-graph residency,
quality, unload, and controlled benchmark evidence exists. No MLX/MPSGraph
usage, no scalar fallback.

## Files

- Kernel source: `kernels/metal/q2q4_fused_matvec.metal`
- Rust ABI/dispatch contract: `src/backend/metal.rs`
- Direct verifier runtime: `src/backend/metal_runtime.rs`

## Entry points

| Kernel | DType | Block layout |
|---|---|---|
| `q2_b64_fused_matvec` | Q2_B64 | 18 bytes: fp16-LE scale + 16 code bytes (4 x 2-bit values per byte) |
| `q4_b64_fused_matvec` | Q4_B64 | 34 bytes: fp16-LE scale + 32 code bytes (2 x 4-bit values per byte) |
| `q2_b64_recovered_row` | Q2_B64 embedding row | one packed byte/four corrected outputs per thread |
| `q4_b64_recovered_row` | Q4_B64 embedding row | one packed byte/two corrected outputs per thread |

64 values per block, row-major block order, codebook matching
`src/quant.rs` (Q2: {-1, -1/3, 1/3, 1}; Q4: (code - 7.5) / 7.5). Q3 does not
exist in this format.

## ABI

- Buffers: 0 weights (`uchar`), 1 input (`float`), 2 s_in (`half`), 3 s_out
  (`half`), 4 bias (`float`), 5 output (`float`), 6 params
  (`FusedMatVecParams`, eight LE u32 words, 32 bytes: rows, columns,
  blocks_per_row, has_s_in, has_s_out, has_bias, activation, reserved).
- No threadgroup scratch allocation: every output row is reduced wholly inside
  one simdgroup.
- Fused semantics identical to the CPU oracle:
  `y[r] = act(s_out[r] * (sum_c w[r,c] * x[c] * s_in[c] + bias[r]))`,
  activation 0 = identity, 1 = SiLU (`x / (1 + exp(-x))`).
- `s_in` and `s_out` stay byte-identical to the FP16 CTOXQ recovery tensors;
  kernels widen individual values in registers. No startup expansion to f32
  or duplicate scale allocation is permitted.

## Organization

The current candidate assigns four output rows to each 32-wide simdgroup. A
lane processes two positions from every 64-value block and reuses its corrected
input values across the four rows. Each row ends in an independent `simd_sum`;
lane zero applies bias, `s_out`, and activation. The host can sweep one, two,
four, or eight simdgroups per threadgroup and will eventually pin the winner
per hardware profile and matrix shape.

`PreparedMetalMatVec` allocates immutable weights, packed recovery scales,
bias, parameters, input, and output once. Repeated dispatches reuse those
buffers, and `write_input` changes only the decode activation vector. This
proves resident per-operation ownership but is not yet the final zero-copy
CTOXQ import or full-graph arena. `dispatch_prepared_repeated` additionally
records multiple resident dispatches in one command encoder, so the benchmark
can separate kernel work from a commit/wait round trip per operation. The
production graph must generalize this to distinct dependent operations rather
than repeating one projection.

`PreparedMetalActivation` plus `PreparedMetalProjection` now provide that
distinct-projection primitive for one fan-out: input and packed FP16 `s_in`
are allocated once, Q2/Q4 matrices keep only projection-local state, and every
projection is encoded into one command buffer with one completion wait. The
dispatcher binds the column count and SHA-256 of the exact `s_in` bytes and
rejects a mismatched projection before encoding. The existing MSL kernels
still multiply `input * s_in` inside each projection; this change proves shared
ownership and scheduling, not a pre-corrected or A8 Metal compute path.

`MappedMetalArtifact` now imports the complete immutable CTOXQ file mapping as
one `newBufferWithBytesNoCopy` shared-memory buffer. Prepared projections bind
their weight, `s_in`, and `s_out` tensors as validated byte offsets into that
single mapping; slices with identical contents but a different address are
rejected. The owner retains an `Arc<Mmap>` through a cloned `ModelArtifact`
until after the Metal buffer is released, so dropping the loader's original
artifact cannot leave a dangling GPU mapping. Per-projection allocations are
limited to input, optional bias, output, and a 32-byte parameter block. The
same-device test dispatches after dropping both original mapping handles,
matches the scalar recovered-Q4 oracle, reports zero copied model bytes, and
updates only the existing input buffer for a second dispatch. This proves the
no-copy ownership primitive; the complete 506-matrix graph still needs a
single shared mapping, measured allocator high-watermark, and unload evidence.
Canonical `MixedQ2Q4B64` matrices now use the same ownership path. The host
validates contiguous manifest group indices, rows, payload offsets, lengths,
and per-row block sizes, then encodes one existing Q2/Q4 kernel dispatch per
homogeneous segment in a single command encoder. Weight, row-scale, bias, and
output bindings advance by checked offsets; the logical codes remain in their
original tensor and the input correction is shared across all segments.

The restricted MTP LM head now has dedicated gathered Q2/Q4 candidates.
Canonical token IDs remain strictly increasing and are grouped by their
manifest row segment without changing output order. Each segment receives
local row IDs while its original weight and `s_out` ranges remain offsets into
the shared CTOXQ mapping; `s_in` and the final hidden vector are shared across
the complete batch. Only the ID list and requested scalar logits are transient.
Every draft is still verified by the complete target distribution in the Rust
engine, so gathered evaluation cannot change target semantics.

Embedding lookup uses two separate recovered-row entry points. The loader
resolves the requested global row to its exact Q2 or Q4 manifest segment, and
the runtime binds that packed row plus the complete `s_in` vector and one
packed `s_out` half as offsets into the shared mapping. Q2 threads decode four
adjacent columns and Q4 threads decode two; the only new allocation is the
f32 hidden vector plus the fixed parameter block.

Q2 decoding uses the exact affine identity `normalized = code * 2/3 - 1`
instead of a four-way select. Sixteen lanes each load one unique packed byte
and decode its four adjacent weights, avoiding redundant packed-byte reads.
This changes neither the logical Q2 codes nor the CTOXQ artifact layout.

## Validation evidence (this worktree, Apple M5)

- `xcrun -sdk macosx metal -c kernels/metal/q2q4_fused_matvec.metal -o target/fleet-metal.air` — compiles clean.
- The `metal` Cargo feature links only the native `metal-rs` driver binding and
  compiles the in-crate MSL source with fast math disabled. It does not link an
  inference framework.
- `cargo test --features metal q2_and_q4_device_results_match_scalar_oracle`
  dispatches Q2 and Q4 on the 10-core Apple M5 GPU. Eleven 192-column rows use
  non-identity packed-FP16 `s_in`/`s_out`, bias, and SiLU, and pass the scalar
  CPU oracle tolerance for every output row.
- `prepared_projection_reuses_resident_buffers_and_updates_only_input` proves
  that a second dispatch changes output after updating only the existing input
  buffer and rejects a mismatched input shape.
- `shared_fanout_matches_oracles_reuses_residency_and_rejects_other_s_in`
  executes mixed Q2/Q4 projections in one command buffer, matches both scalar
  oracles, saves exactly one duplicated input/scale allocation for two
  projections, accepts in-place input updates, and rejects another `s_in`.
- `mmap_artifact_is_shared_without_copy_and_outlives_original_owner` imports a
  deliberately non-page-multiple CTOXQ fixture as one no-copy Metal buffer,
  binds quant codes and both recovery scales by offset, survives the original
  mmap owners being dropped, matches the scalar oracle, and rejects copied
  same-valued tensor slices.
- `mixed_mmap_segments_dispatch_without_repacking_or_duplicate_weights` opens
  a checksummed CTOXQ-v2 container with Q2 and Q4 row groups, validates its
  exact manifest coverage, dispatches both groups from one no-copy buffer, and
  matches every row against the mixed CPU oracle.
- `mixed_gathered_lm_head_batches_canonical_rows_from_one_mapping` evaluates
  sparse non-contiguous rows across both Q2 and Q4 segments, rejects unsorted
  or out-of-range IDs, survives loader ownership being dropped, supports an
  in-place input update, and matches the full mixed CPU projection.
- `mixed_embedding_rows_decode_from_one_mapping_without_model_copies` selects
  one Q2 and one Q4 row from the same mixed container, drops all loader
  handles, and matches the recovered CPU embedding oracle while reporting zero
  copied model bytes.
- `qwen38-metal-bench` performs synchronous warmups and repeated dispatches on
  those resident buffers, reports the exact requested buffer bytes, and keeps
  its output marked `verifier_only_not_promotion_evidence`.
- `qwen38-metal-fanout-bench` uses the exact Qwen Q/K/V shapes: Q4
  12,288x5,120 plus two Q2 1,024x5,120 matrices. Five uncontrolled 50-pass
  runs on Apple M5 had median 2.685 ms per shared fan-out versus 4.327 ms for
  three isolated command buffers (1.415x), while saving 61,440 requested
  buffer bytes. Maximum scalar-oracle error was `3.51e-5`; variance was high,
  so this remains verifier evidence, not a roofline or promotion claim. The
  raw run summary is in
  `benchmarks/metal/apple-m5-shared-fanout-20260826.json`.
- `qwen38-metal-row-bench --artifact <pack.ctoxq>` opens a checksummed release
  container, resolves an arbitrary embedding token row, verifies it against
  the recovered CPU oracle, and reports repeated-command latency, the bound
  manifest hash, mapped file bytes, transient bytes, and zero copied model
  bytes. It is ready for the final trained pack; no synthetic result is
  presented as release evidence.
- The complete suite also covers ABI constants against `src/quant.rs`, invalid
  shape/buffer rejection, dispatch-name checks, and an in-test `xcrun metal`
  compilation of the source.

Independent review corrected the generated candidate's positive Q2 code order
to `{+1/3, +1}` for codes `{2, 3}` and removed its per-thread 64-float
dequantization array before this source was accepted.

## Not yet done (promotion blockers)

- The no-copy CTOXQ import is verified on a small real container, but complete
  resident model tensor ownership, allocator high-watermark, and complete
  unload have not yet been measured on the 7.8-GiB artifact.
- Exploratory 17408x5120 FFN measurements with eight dispatches per command
  reached roughly 26.55 GB/s for Q2 (four simdgroups/threadgroup) and
  43.95 GB/s for Q4 (two simdgroups/threadgroup). The earlier CTOX M5 hardware
  probe measured approximately 60.6 GB/s sustained read bandwidth at a large
  working set, putting those observations near 44% and 73% respectively. Q4
  is therefore approaching a useful candidate range, while Q2 still has a
  large decode-efficiency gap. Repetition of the same 25/47 MB projection can
  benefit from GPU caches, desktop load and thermal state are uncontrolled,
  and these figures deliberately remain exploratory rather than promotion
  evidence.
- No controlled size/residue/thermal sweep or hardware-counter roofline
  evidence exists yet.
- Recovered embedding rows exist as verifier candidates; no full attention,
  GatedDeltaNet, MTP block, sampling, or model-graph Metal execution exists yet.
- Per `docs/PROMOTION_GATES.md`, all promotion evidence is required before any state change;
  the backend therefore remains fail-closed.
