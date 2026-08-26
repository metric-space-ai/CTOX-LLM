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
- Frozen complete-decode schedule: `src/backend/metal_schedule.rs`
- Complete artifact-resource binding: `src/backend/metal_graph.rs`
- Direct verifier runtime: `src/backend/metal_runtime.rs`

The model-specific schedule contains exactly 645 ordered steps: embedding and
initial norm, ten operations for each of 64 frozen layers, target LM head,
native MTP verification, and one command-buffer commit/wait. Validation rejects
an altered topology, an unavailable producer slot, a missing causal state
mutation, or any intermediate host wait. This is an assembly contract, not yet
a complete Metal executor or promotion result.

`MetalDecodeBindingPlan` resolves those 645 steps against all 505
non-embedding projections, 262 recovery activation groups, 48 linear mixers,
16 target plus one MTP full-attention state, four regular norms, 130 residual
norms, and the final token barrier. The names and activation-group policy are
derived from the same tensor contract and fan-out policy as CUDA; Metal may
change physical layout but may not requantize or alter logical Q2/Q4 codes.
This remains a binding plan rather than executable graph state.
Its execution cursor rejects stale/out-of-context token positions and any
skipped, duplicated, reordered, or wrong-layer dispatch. An incomplete cursor
cannot advance the committed token count; accelerator-state rollback remains
an explicit prerequisite for the complete executor.

`MetalDecodeWorkspacePlan` derives produced-value live intervals directly
from the validated schedule and assigns all 21 named f32 activation slots to
one 256-byte-aligned arena. For the frozen 40,000-row restricted MTP draft
vocabulary the arena is exactly 1,173,760 bytes versus 1,633,280 bytes for
independent buffers, saving 459,520 bytes. Byte aliasing is rejected whenever
any live interval overlaps; target logits and MTP draft logits consequently
remain distinct through the final commit. Persistent weights, packed KV,
causal-convolution/recurrent state, and kernel parameter blocks are explicitly
outside this transient plan. `prepare_decode_workspace` materializes it as
exactly one zeroed shared Metal buffer for the plan and exposes the same buffer
with validated per-slot offsets. The Apple-device verifier writes and reads an
exact slot, rejects a wrong-sized write, proves distinct slot views share the
one buffer, then drops and recreates the arena. This is allocation/lifetime
evidence for the decode arena, not yet a complete allocator high-watermark or
zero-residue unload measurement; per-step encoder binding remains executor
work.

## Entry points

| Kernel | DType | Block layout |
|---|---|---|
| `q2_b64_fused_matvec` | Q2_B64 | 18 bytes: fp16-LE scale + 16 code bytes (4 x 2-bit values per byte) |
| `q4_b64_fused_matvec` | Q4_B64 | 34 bytes: fp16-LE scale + 32 code bytes (2 x 4-bit values per byte) |
| `q2_b64_recovered_row` | Q2_B64 embedding row | one packed byte/four corrected outputs per thread |
| `q4_b64_recovered_row` | Q4_B64 embedding row | one packed byte/two corrected outputs per thread |
| `qwen_rms_norm_1p_f32` | FP16 weight, f32 activation | one 32-wide simdgroup per row |
| `qwen_rms_norm_gated_f32` | FP16 weight, f32 core/gate | one 32-wide simdgroup per value head |
| `qwen_partial_rope_f32` | f32 Q/K heads in place | one thread per non-interleaved rotary pair |
| `qwen_paged_q2q4_gqa_decode_f32` | f32 query/output, packed Q2/Q4 K/V | one 32-wide simdgroup per query head |
| `qwen_gated_delta_recurrent_f16` | f32 step inputs/output, FP16 recurrent state | one threadgroup per value head |
| `qwen_causal_conv_silu_f16` | f32 input/output, mmap FP16 weight/state | one thread per channel |
| `qwen_argmax_f32_partial` / `qwen_argmax_f32_final` | f32 logits, bounded partials, two-u32 result | tuned 32 parallel groups plus one final group |

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

The complete embedding owner now retains every pure or mixed Q2/Q4 row through
the same artifact mapping and selects a token by changing only the packed row
and scalar `s_out` offsets. It allocates one reusable hidden vector and one
32-byte parameter block regardless of vocabulary size. The first resident
decode chain encodes embedding lookup, Qwen RMSNorm, and an external-input
mixed projection in one command encoder; neither the embedding nor normalized
hidden vector returns to the host between operations. The same prepared table
selects Q2 and Q4 tokens repeatedly and remains valid after the original loader
handles are dropped. A fan-out variant encodes RMSNorm once and binds its exact
resident output plus one byte-identical packed `s_in` to every Q/K/V or Gate/Up
projection branch; a mismatched mapping or correction offset fails before a
command buffer is submitted.

The Qwen RMSNorm candidate implements the model-specific `(1 + weight)`
convention rather than Llama's direct-weight convention. One simdgroup owns a
complete row, reduces the f32 sum of squares without threadgroup scratch, and
writes corrected columns using an mmap-backed FP16 weight. Reusable f32
input/output buffers cover single-token decode and multi-row prefill.
An external-input projection mode omits its own activation allocation. The
composed verifier encodes decode RMSNorm first and then binds that exact output
buffer as the mixed Q2/Q4 projection input in the same command encoder. Direct
dispatch of such a projection fails closed because only an explicit upstream
graph operation can supply its input.

Partial RoPE rotates Qwen's non-interleaved halves in place and never touches
dimensions at or above `rotary_dim`. Query and key buffers are encoded in one
command encoder. An initial candidate computed `pow/cos/sin` independently in
MSL and differed from the f32 reference by roughly `1.65e-4` at position
12,345, so it was rejected. The accepted path creates the 32 f32 cosine/sine
pairs with the pinned reference equation, retains those 256 bytes in reusable
Metal buffers, and performs all head rotations on the GPU. Updating position
rewrites only these tables and the 32-byte parameter block.

The decode-only grouped-query-attention candidate keeps persistent K/V pages
packed on the Metal device. Logical pages have deterministic Q2 arena slots;
a bounded Q4 arena holds only sink pages, recent pages, and one
precision-boundary page. One 16-byte descriptor per logical page selects the
arena and physical slot. Stable three-pass softmax decodes Q2/Q4 blocks
directly into registers and never creates an f32 K/V device cache. An append
uploads only its changed page plus any page crossing the Q4-to-Q2 boundary.

For the frozen 24-query-head/4-KV-head/256-wide topology, a token contains
2,048 combined K/V values: 576 Q2 bytes or 1,088 Q4 bytes. A 128K layer with
128-token pages, 128 sink tokens, and 256 recent tokens reserves 75,497,472
Q2-arena bytes, 557,056 Q4-arena bytes, and 16,384 descriptor bytes:
76,070,912 bytes (72.546875 MiB) per GQA layer and 1,217,134,592 bytes
(1.133545 GiB) across all 16 attention layers. The verifier currently retains
a CPU packed mirror for deterministic page transitions; that duplicate is not
included in these device figures and must be eliminated by GPU-side packing
and demotion before promotion.

The recurrent GatedDeltaNet candidate keeps its matrix state exclusively in
FP16. One threadgroup owns one value head and one thread owns one value column;
each thread walks the key dimension, applies decay with an immediate FP16
rounding, computes the delta, writes the updated value with the second FP16
rounding, and accumulates the output in f32. Q/K normalization is calculated
once per head in threadgroup memory. The frozen 48-head, 128x128 state occupies
1,572,864 bytes (1.5 MiB) per linear-attention layer and 75,497,472 bytes
(72 MiB) across all 48 layers. Reusable f32 inputs/output and the parameter
block add 98,704 transient bytes per independently prepared layer; complete
graph scheduling must share those transient buffers instead of multiplying
them by 48.

The depthwise causal-convolution candidate completes the persistent
linear-attention state pair. Its FP16 weight remains an offset into the shared
CTOXQ mmap, its four-token history remains FP16, and input insertion rounds to
FP16 before the convolution. SiLU is fused into the f32 output write. At the
frozen 10,240-channel, width-4 geometry, history is 81,920 bytes per layer and
3,932,160 bytes (3.75 MiB) across 48 layers. Together with the recurrent state,
all linear-attention state is therefore 75.75 MiB. Independently prepared
layers use 81,936 transient bytes each; the complete graph must reuse these
input/output buffers.

The GatedDeltaNet output normalization is a separate direct-weight candidate:
it computes RMSNorm over each 128-wide value head and fuses `SiLU(z)` into the
same output write. The FP16 norm weight remains an offset into the CTOXQ mmap;
core, gate, and output use reusable f32 graph buffers. At the frozen
48-head-by-128 geometry, an independently prepared operation owns 73,744
transient bytes and zero copied model bytes. Complete graph scheduling must
bind the recurrence output and `in_proj_z` output directly instead of retaining
those two standalone input buffers.

The finite-checking argmax candidate scans all 248,077 valid tokenizer logits
with 32 parallel threadgroups, reduces their 512-byte partial array in a second
kernel, and returns only `{token_id, invalid_count}`. The target matrix has
248,320 padded rows, but padded rows are never selectable. Equal scores select
the larger valid token ID, matching the pinned Rust sampler. Its reusable
standalone verifier owns one logit buffer. The composed final RMSNorm,
recovered Q2/Q4 LM-head, and argmax verifier instead binds the same kernels
directly to the mapped LM-head output in one command encoder and retains only
536 bytes of partial/result/parameter state. Any NaN or infinity fails closed.

On Apple M5, an interleaved five-run comparison measured the selected 32-group
profile at a median 14.580 microseconds per resident selection (68.060 logical
GB/s) versus 17.425 microseconds (56.947 logical GB/s) for 256 groups. These
figures amortize command-buffer overhead with 64 selections per command and do
not constitute hardware-counter roofline evidence. The raw sweep, alternating
run order, hashes, and limitations are recorded in
`benchmarks/metal/apple-m5-vocabulary-selection-20260826.json`.

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
- `mapped_qwen_rms_norm_matches_oracle_and_reuses_input_buffer` dispatches
  multi-row Qwen RMSNorm after dropping loader ownership, rejects copied
  same-valued weights and invalid epsilon, supports an in-place input update,
  and matches the exact scalar `(1 + weight)` equation.
- `mapped_rms_norm_feeds_mixed_projection_without_host_intermediate` verifies
  the two-operation device chain against scalar norm plus mixed-matrix
  oracles, proves the external-input projection saves exactly one activation
  vector, rejects incorrect standalone use, and updates only the upstream norm
  input for a second zero-output dispatch.
- `partial_rope_pair_matches_qwen_oracle_and_preserves_tail` uses the exact
  24-query/4-key-head, 256-wide, 64-rotary-dimension topology at position
  12,345, encodes Q and K in one command, checks every value against the
  scalar oracle, proves every non-rotary tail value is bit-identical, rejects
  invalid contracts, and reuses buffers at position zero.
- `paged_q2q4_gqa_decode_matches_quantized_oracle_and_demotes_pages` forces a
  Q4-to-Q2 page transition, compares every decode step with the scalar GQA
  oracle using the identical quantized cache, verifies bounded arena byte
  counts, and proves reset/reuse without an f32 device cache.
- `gated_delta_f16_matches_recurrent_oracle_and_reuses_state` executes six
  dependent recurrent steps, compares outputs and persistent state with the
  FP16 scalar oracle after every token, rejects invalid geometry, and proves
  reset/reuse without an f32 state shadow.
- `mapped_causal_conv_f16_matches_oracle_and_reuses_state` executes six
  dependent steps with a loader-owned FP16 weight, drops the loader before
  dispatch, compares output and history with the FP16 scalar oracle, proves
  zero copied model bytes, and rejects copied same-valued weights.
- `mapped_gated_rms_norm_matches_oracle_and_reuses_both_inputs` uses the exact
  48x128 Qwen geometry, survives loader teardown, matches the direct-weight
  gated scalar oracle, updates both graph inputs in place, reports zero copied
  model bytes, and rejects copied weights and malformed contracts.
- `device_argmax_matches_full_vocab_oracle_reuses_buffers_and_rejects_nonfinite`
  dispatches the complete 248,077-token vocabulary, proves the larger-token
  tie rule, reuses the resident buffers for a changed winner, returns only two
  u32 values, and rejects a device-observed NaN.
- `mapped_rms_norm_lm_head_argmax_chain_stays_on_device` encodes final RMSNorm,
  a mixed Q2/Q4 LM head, and deterministic selection into one command encoder,
  proves the selected token against the scalar chain, rejects a selector wider
  than the physical output, and never copies the logit vector to the host.
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
- Recovered embedding rows, both Qwen RMSNorm variants, partial RoPE, packed
  decode GQA, causal convolution, and FP16 recurrent GatedDeltaNet exist as
  verifier candidates. GQA still
  duplicates its packed pages in a CPU correctness mirror; neither attention
  path has controlled performance evidence. The MTP block and sampling do not
  exist yet. A deterministic shared decode arena and its single Metal buffer
  exist, but per-step encoder binding, the prefill arena, transactional state,
  and complete model-graph execution remain unfinished.
- Per `docs/PROMOTION_GATES.md`, all promotion evidence is required before any state change;
  the backend therefore remains fail-closed.
