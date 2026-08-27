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
one buffer, then drops and recreates the arena. `bind_decode_program` now
resolves every logical read and write of all 645 resource-bound steps to a
typed view containing the same real Metal buffer, exact offset, width, and byte
range. It rejects non-contiguous step indices and duplicate per-step access
slots. Its Apple-device verifier checks every resulting view and confirms the
final barrier reads both target and MTP logits. This is allocation/lifetime and
buffer-view evidence for the decode arena, not yet a complete allocator
high-watermark, zero-residue unload measurement, or kernel encoder dispatch.

`PreparedMetalF32Checkpoint` adds a bounded device-only snapshot for one arena
slot. Snapshot and restore use a native Metal blit between the shared arena and
one same-sized checkpoint buffer; no f32 state is copied through Rust memory.
The checkpoint becomes active only after a completed blit, is consumed by
restore or explicit commit, and rejects width mismatches and repeated lifecycle
operations. The verifier proves restore and commit semantics on the target
hidden-width slot. This covers the target-hidden replay primitive only: paged
KV and the graph-level transaction were still required at that point.
Individual FP16 causal-convolution and GatedDelta owners now also allocate one
same-sized device checkpoint and expose fail-closed begin/restore/commit
operations. Their Apple-device verifiers prove that a speculative state
advance restores bit-exactly, while commit preserves the advance; no state
payload crosses the host.

`PreparedMappedMetalFullAttentionFanout` is the first executable operation of
the frozen full-attention slice. Its loader binds the layer-specific Q/K/V
recovered matrices to one mmap and admits separately stored recovery `s_in`
tensors only when their packed FP16 bytes are identical. The dispatch consumes
`Normalized` and writes `QueryGate`, `Key`, and `Value` at their shared-arena
offsets in one encoder; it retains no input or output activation buffers.

`PreparedMappedMetalQueryGate` retains the layer-specific mmap-backed FP16
query norm plus only the reusable cosine/sine tables and a 32-byte parameter
block. The Metal kernel ports the verified CUDA equation: it deinterleaves each
256-value Query/Gate head, reduces query variance in one simdgroup, applies
`(1 + weight)`, rotates the first 64 dimensions, and writes Query plus Gate to
their exact arena views without a temporary normalized vector.

`PreparedMappedMetalAttentionOutput` closes the full-attention mixer without a
gated activation allocation. Dedicated Q2/Q4 entry points read
`AttentionOutput` and `AttentionGate` from their arena views, evaluate the
sigmoid gate and recovery `s_in` in registers, and write the recovered output
projection directly to `MixerOutput`.

`PreparedMappedMetalFullAttentionLayer` is the corresponding closed ten-step
resource owner. It binds the exact layer Q/K/V, query norm, `o_proj`, both
residual norms, three FFN matrices, and one packed KV owner to one canonical
mmap identity. Production KV metadata and device descriptors are prepared
before encoding; Q/K/V fan-out, Query/Gate norm+RoPE, per-head Key
RMSNorm+RoPE, packed append,
paged GQA, fused gated output projection, and the complete residual/FFN tail
then execute in one command encoder and one wait. Test builds snapshot K/V on
device before later legal arena aliases and update their independent CPU oracle
only after successful completion. That verifier path is absent from releases.
The all-layer entry point constructs exactly 16 such owners in frozen topology
order and returns no partial vector if any tensor, cache geometry, or mapping
identity fails validation.
`PreparedMappedMetalTargetLayers` is the next ownership boundary: it constructs
all 64 target layers in model order as an exact 48-linear/16-full enum, rejects
any identity or topology drift, aggregates persistent state with checked
arithmetic, and updates all Full-Attention position tables without exposing a
partially admitted graph. The weights in every variant remain views of the same
canonical no-copy artifact mapping.
The complete linear- and full-attention paths now each expose an internal
encoder-only stage. Admission and tensor-identity checks happen before that
stage; the encoder-only functions append the frozen ten operations to a
caller-owned compute encoder and never commit or wait themselves. Standalone
Golden wrappers still commit once per layer, while the model executor can reuse
the same code to encode all target layers into one command buffer.
`validate_prepared_mapped_target_layers` performs the model-wide admission pass
over all 64 schedule slices. It invokes the exact per-kind validator, checks
canonical layer indices and the 48/16 topology, uses checked aggregate state
accounting, and rejects cross-artifact owners. Device checkpoint readiness is
kept as an execution concern, so this preflight itself performs no mutation.
`dispatch_prepared_mapped_target_layers` is the corresponding graph core. It
opens one owner-level transaction over all KV, convolution, and recurrent
states, encodes all 64 ten-operation slices into one compute encoder, and
performs one commit/wait. Encoding errors, command-buffer failures, and
test-oracle mismatches restore every checkpoint; success commits the transaction
only after all layers complete. The current checkpoint creation uses the
existing synchronous state-copy primitives and remains a tuning target even
though the target-layer compute itself has no per-layer command boundary.
`PreparedMappedMetalTargetCore` atomically loads the recovered embedding,
Layer-0 input RMSNorm, all 64 target layers, and `lm_head.weight` from one
canonical mapping. It rejects any vocabulary/hidden-size mismatch, owned graph
I/O buffer, tensor-identity drift, or cross-artifact resource and never returns
a partial core.
The 64-layer implementation is split once more at the graph boundary:
`encode_prepared_mapped_target_layers` appends all layer kernels to a supplied
encoder and returns the 16 append plans without committing or waiting. The
public layer-graph dispatcher now wraps that primitive with transaction,
completion, verifier, and rollback handling.

`dispatch_prepared_mapped_target_core` is the first complete target-side graph
execution boundary. It validates the schedule frontend and target head, opens
one transaction across every target-layer state owner, and encodes recovered
embedding, initial RMSNorm, all 64 target layers, and the full LM head into one
compute encoder. One commit/wait produces resident `TargetLogits`; no
vocabulary-sized host readback occurs. Encoding, command-buffer, and test-only
KV-verifier failures restore the complete target-layer transaction. This does
not yet execute the MTP transition, draft/verify loop, sampler, or final
barrier, so it is not the complete 645-step token executor.

The full-attention path now applies the previously missing Qwen K RMSNorm
before partial RoPE. Because K must remain in one shared-arena slot, a dedicated
256-wide per-head kernel retains all eight values per lane across the variance
reduction and is safe when input and output alias. A Golden assertion reads the
test-only pre-pack snapshot and compares all four normalized K heads with the
CPU equation. This corrects target and future MTP attention semantics without
adding a production K activation buffer.

`PreparedMappedMetalMtpCore` is the corresponding atomic load boundary for the
native one-layer MTP graph. It admits the shared embedding plus the canonical
restricted LM-head rows as offset-only views, both pre-FC norms, `mtp.fc`,
input norm, complete MTP attention/MLP resource set, packed KV state, and
target-selector scratch only when every tensor and mapping identity matches.
The draft vocabulary must be non-empty, strictly increasing, unique, and
within the full LM-head row range.

Target selection can now feed MTP embedding without returning the token to the
host. `qwen_argmax_f32_final` leaves the selected ID in its compact result
buffer; `q2_b64_dynamic_embedding_row` and
`q4_b64_dynamic_embedding_row` consume that buffer directly. Mixed embedding
segments each carry a fixed 32-byte range/stride parameter block and only the
segment containing the selected row writes the shared output. A same-command
Golden crosses the Q2/Q4 boundary and matches loader-resolved static rows.
The complete target core can now consume that same selector ABI as well. Its
shared validator admits the frozen embedding/norm/64-layer/LM-head graph once,
then the selector-driven verifier encodes the whole target transition without
a host token-to-embedding handoff. It validates the compact selector status
before committing all target states, so a non-finite source selection restores
the speculative transition.
`qwen_concat_f32` now supplies the next native MTP frontend primitive: it joins
the normalized selected-token embedding and retained target hidden state into
one caller-owned device view, with a fixed 16-byte ABI and no host staging.
The standalone verifier checks exact ordering and fails closed on empty or
non-finite inputs. `MetalMtpWorkspacePlan` expands the single scheduled MTP
operation into 20 typed scratch values and aliases only disjoint live
intervals: one 180,224-byte, 256-byte-aligned allocation replaces 536,576 bytes
of independent buffers while remaining physically separate from live target
logits and the final draft. The prepared frontend now encodes offset-aware
target argmax, dynamic Q2/Q4 embedding, both pre-FC norms, concatenate,
`mtp.fc`, and the MTP input norm in one encoder. The same arena now carries the
complete native MTP transformer layer as well: Q/K/V fan-out, Q/K norms and
RoPE, packed paged-GQA append, gated output projection, both residual norms,
and the Q2/Q4 SwiGLU FFN. Its bring-up boundary checkpoints the independent MTP
KV owner, commits only after a completed finite result, and restores cache
metadata on every failure. A graph-I/O gathered Q2/Q4 head then writes exactly
the canonical restricted rows into the main arena's `MtpDraft` view in the
same encoder; its offset Golden proves it allocates no activation input/output.
The gathered kernels share one canonical row-ID buffer across mixed Q2/Q4
segments. A second finite-checking argmax selects the restricted local row and
`qwen_argmax_index_to_token` maps it in place to the global token through that
same buffer, without an intermediate host read. Only the verifier reads draft
logits. `dispatch_prepared_mapped_initial_mtp_target_verifier` supplies the
causal first pair: MTP and target advance from the same real input token, and
MTP consumes the retained pre-target hidden state before the target core
overwrites that arena slot. A selector-driven complete target transition and
its full-vocabulary argmax share the encoder with the MTP draft. The
accepted-only `dispatch_prepared_mapped_greedy_mtp_target_verifier` then
permits another pair only when the previous compact verification record has
`accepted=true`. Both paths require target state to enter exactly one token
ahead of MTP, check that the relation is restored after both transitions, and
commit both checkpoint sets only after `qwen_greedy_mtp_verify` has written a
four-u32 target/draft/accept/status record and the single completion wait
validates it. The host no longer reads or compares separate selector buffers.
Encoding, GPU, non-finite, alignment, and verifier failures restore both
branches. This is one completion wait per pair, but its compact result
allocation and offset-safe verifier dispatch now retain four independent
16-byte records for the causal-prefix decision. A one-lane
`qwen_greedy_mtp_prefix` kernel validates all records, returns the accepted
prefix, and selects the first mismatching target token or the final resident
bonus token without separate selector reads. The native MTP4 tail dispatcher
now queues records 1–3 plus that reduction in one command buffer after the
accepted initial record. Full acceptance commits the three-step state branch;
partial acceptance restores the entire tail and returns four exact replay
records while resetting selectors to record zero. The complete MTP4 wrapper
replays only the accepted tail through the ordinary full target/MTP verifier
and rejects any record divergence by poisoning both graphs. Full-artifact
device evidence and production executor integration remain pending. The fused
from-token wrapper now places record zero, records 1-3, and prefix reduction in
one command buffer, eliminating the separate initial-record completion wait on
the 4/4 path. A partial branch restores the pre-initial transaction, replays
the mandatory initial transition even when its draft was rejected, and then
replays only the accepted tail; every replay record must match the speculative
record exactly.

The continuation path now preserves each canonical mapped draft in a distinct
full-vocabulary candidate selector with a two-word device copy before the
restricted selector is overwritten. MTP embedding and target embedding both
consume that candidate. This keeps speculative inputs draft-driven even when
later verification rejects them; the host still reads only the final compact
verification result.

Paged-GQA dispatch no longer implicitly hard-wires its owner's mutable
descriptor and parameter buffers. An explicit metadata-binding encoder accepts
immutable per-step snapshots while retaining the same single Q2/Q4 KV arenas,
pack parameters, and append plan. The standalone paged-GQA oracle now executes
through snapshot buffers and still matches the quantized scalar reference
through page demotion. The complete target graph and native MTP layer now bind
the same immutable dispatch plans rather than the owners' mutable metadata
buffers. A two-depth test proves the first page view and token count survive
planning the second append without copying KV payloads.

Each full-attention dispatch plan now owns immutable Query/Key RoPE tables and
the key-RoPE parameter block together with its GQA metadata. The ordinary
target and MTP graphs bind those snapshots directly. Planning derives the
position from the pre-append KV token count rather than mutable owner tables,
so successive queued branch steps receive distinct causal positions. A device
regression changes the reusable owner to position zero after planning a
nonzero position and proves the queued dispatch still matches the nonzero
scalar oracle. Every prepared full-attention layer now owns a bounded
four-slot metadata pool for exactly one MTP4 branch. Planning fills the slot at
`position % 4` and retains its existing MTLBuffer handles in the immutable
dispatch plan; it performs no replacement GPU allocation. Resident-state
accounting includes the pool, and a regression proves consecutive speculative
positions use distinct preallocated slots before rollback restores the exact
KV prefix.

Paged GQA now has a bounded append-only transaction as well. Begin records a
constant-size cache prefix marker and small page-to-Q4/free-slot vectors. While
active, appends retain all pre-branch Q4 pages and use the memory-plan boundary
slot, so no referenced packed page is demoted or overwritten. Restore
truncates page metadata (and the test-only oracle), restores descriptors, and
leaves unreferenced device bytes harmless; it never snapshots the Q2/Q4
arenas. A four-token branch restores to the exact eight-token prefix and its
replay produces bit-identical Metal attention outputs. The CPU packed mirror
is compiled only under `cfg(test)`; release owners contain metadata and packed
device arenas only.

`PreparedMetalSpeculativeTransaction` composes those primitives into one
all-or-nothing operation for the frozen decode graph: the final normalized
target-hidden arena slot, exactly 17 target/MTP attention owners, and exactly
48 paired convolution/GatedDelta owners. Begin rejects a wrong resource count,
wrong target-hidden width, poisoned owner, active checkpoint, or missing Q4
boundary slot before taking the first snapshot. A begin failure restores only
the already-started prefix in reverse ownership order. Explicit reject restores
all recurrence, convolution, attention, and target-hidden state in reverse
order and poisons the coordinator if any restore fails. Commit first validates
the complete active set, then consumes every checkpoint. Its Apple-device test
mutates all four state classes, proves graph-wide restoration, repeats the same
branch, and proves graph-wide commit. This closes atomic speculative-state
orchestration, not per-step encoder binding or the full target+MTP executor.

`PreparedMetalDecodeAttempt` joins the real-buffer program, exact binding-plan
cursor, and graph-wide state transaction under one lifetime. Admission checks
the requested token position against the committed/context bounds before
opening checkpoints. Every successful kernel encoding must advance the exact
bound `(index, layer, operation)` tuple. Dropping an incomplete attempt, or
consuming it with an early final-commit request, restores all active state
owners automatically. Explicit abort reports restore errors; implicit drop
leaves the coordinator poisoned if restoration fails. Commit first proves that
the next step is the sole final barrier, then consumes all checkpoints and
returns the cursor's next committed position. The Apple-device test exercises
wrong-position rejection, partial-drop rollback, early-commit rollback, and a
complete 644-operation plus final-barrier commit. Actual kernel dispatch for
the complete step set remains executor work.

## Entry points

| Kernel | DType | Block layout |
|---|---|---|
| `q2_b64_fused_matvec` | Q2_B64 | 18 bytes: fp16-LE scale + 16 code bytes (4 x 2-bit values per byte) |
| `q4_b64_fused_matvec` | Q4_B64 | 34 bytes: fp16-LE scale + 32 code bytes (2 x 4-bit values per byte) |
| `q2_b64_swiglu_matvec` | Q2_B64 | fused `SiLU(gate) * up` in registers plus recovered down projection |
| `q4_b64_swiglu_matvec` | Q4_B64 | fused `SiLU(gate) * up` in registers plus recovered down projection |
| `q2_b64_sigmoid_gate_matvec` | Q2_B64 | fused `attention * sigmoid(gate)` in registers plus recovered output projection |
| `q4_b64_sigmoid_gate_matvec` | Q4_B64 | fused `attention * sigmoid(gate)` in registers plus recovered output projection |
| `q2_b64_recovered_row` | Q2_B64 embedding row | one packed byte/four corrected outputs per thread |
| `q4_b64_recovered_row` | Q4_B64 embedding row | one packed byte/two corrected outputs per thread |
| `qwen_rms_norm_1p_f32` | FP16 weight, f32 activation | one 32-wide simdgroup per row |
| `qwen_residual_rms_norm_1p_f32` | FP16 weight, two f32 inputs/two f32 outputs | fused residual add plus Qwen RMSNorm, one 32-wide simdgroup per row |
| `qwen_rms_norm_gated_f32` | FP16 weight, f32 core/gate | one 32-wide simdgroup per value head |
| `qwen_partial_rope_f32` | f32 Q/K heads in place | one thread per non-interleaved rotary pair |
| `qwen_query_gate_norm_rope_f32` | mmap FP16 Q norm, interleaved f32 Query/Gate | one 32-wide simdgroup per query head; deinterleave + Qwen RMSNorm + partial RoPE |
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
- Fused SwiGLU buffers: 0 weights, 1 gate, 2 up, 3 s_in, 4 s_out,
  5 bias, 6 output, 7 the same `FusedMatVecParams`. The gate/up product is
  never written to memory.
- No threadgroup scratch allocation: every output row is reduced wholly inside
  one simdgroup.
- Fused semantics identical to the CPU oracle:
  `y[r] = act(s_out[r] * (sum_c w[r,c] * x[c] * s_in[c] + bias[r]))`,
  activation 0 = identity, 1 = SiLU (`x / (1 + exp(-x))`).
- `s_in` and `s_out` stay byte-identical to the FP16 CTOXQ recovery tensors;
  kernels widen individual values in registers. No startup expansion to f32
  or duplicate scale allocation is permitted.
- SwiGLU-down buffers are 0 weights, 1 gate, 2 up, 3 `s_in`, 4 `s_out`, 5
  bias, 6 output, and 7 params. The intermediate `SiLU(gate) * up` values are
  formed only in registers and never occupy another arena slot.

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

The schedule-bound form removes even those reusable operation-local hidden and
projection outputs. Embedding, RMSNorm, and projection preparation retain only
mmap-backed tensor offsets plus fixed bias/parameter blocks. The first three
frozen decode steps bind `HiddenA`, `Normalized`, `LinearQkv`, `LinearZ`,
`LinearA`, and `LinearB` directly to typed offsets in the single 1,173,760-byte
arena. All four Q2/Q4 projections accept distinct artifact offsets only when
their packed FP16 `s_in` bytes are identical. Slot, size, mapping, arena-owner,
or correction mismatch fails before submission. The following causal
convolution consumes and overwrites the exact `LinearQkv` view while retaining
only its mmap-backed FP16 weights, FP16 history/checkpoint, and parameter
block. The next fused kernel consumes that convolved QKV plus `LinearA` and
`LinearB`, repeats Q/K from 16 to 48 heads, and writes `Query`, `Key`, `Value`,
`LogDecay`, and `Beta` directly into their five schedule slots. `A_log` and
`dt_bias` stay at mmap offsets and only a 16-byte parameter block is allocated.
The recurrent kernel then reads those exact step-5 views, mutates only its
checkpointed 1,572,864-byte FP16 state, and writes the 6,144-value
`AttentionOutput` arena slot. Its graph preparation retains only FP16 state,
an equally sized rollback checkpoint, and a 16-byte parameter block.
Direct-weight gated RMSNorm then consumes `AttentionOutput` plus `LinearZ` and
updates `AttentionOutput` in place. Its graph preparation retains only the
mmap-backed FP16 weight and a 16-byte parameter block. The recovered Q2/Q4
linear output projection consumes that exact view and writes `MixerOutput`
without owning either activation endpoint. A fused residual-add/Qwen-RMSNorm
kernel consumes `HiddenA`, `MixerOutput`, and the mmap-backed post-attention
norm weight, writes the exact `HiddenB` residual, and writes the next
`Normalized` view without allocating activation endpoints. The Apple-device
Golden test then executes the mixed-Q2/Q4 FFN gate/up fan-out from that exact
view into `FfnGate` and `FfnUp`, followed by a fused Q2/Q4 SwiGLU-down
projection into `FfnDown`. Steps 0-10 use one command encoder and one wait;
the same fused residual/Qwen-RMSNorm kernel then combines `HiddenB` and
`FfnDown`, writes the complete layer result to `HiddenA`, and writes the next
layer input to `Normalized`. Steps 0-11 therefore execute the complete first
linear-attention transformer layer in one command encoder and one wait; failure
leaves state poisoned and recoverable through the active device checkpoint.
Because the arena intentionally aliases dead slots, the verifier uses
checkpointed reruns when inspecting earlier and final views.

The same ten layer operations are now encoded by a reusable linear-attention
layer entry point. Its input is the exact schedule slice returned by
`linear_attention_layer_steps(layer)`. Before encoding it resolves canonical
Qwen tensor identities back into the admitted mmap and checks weight, `s_in`,
`s_out`, convolution, `A_log`, `dt_bias`, both RMSNorm weights, and the
layer-owned recurrence state. The Golden test proves that a Layer-1 slice with
Layer-0 resources is rejected before poisoning state, then executes Layer 0
through the reusable path and matches the complete-layer scalar oracle.
`prepare_mapped_linear_attention_layer` now resolves that complete canonical
resource set directly from the admitted artifact. The all-layer loader walks
the frozen topology and must return exactly 48 owners; one absent tensor drops
the partial vector and fails the model load rather than leaving a reduced graph.

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
`prepare_partial_rope_graph` omits the operation-local activation buffer and
binds the same kernel directly to an offset in the shared decode arena. The
Layer-3 Full-Attention Golden test validates the real `KeyRope` schedule view,
including preservation of the unused tail in the aliased `Key` slot. The
decode program also exposes a fail-closed ten-step full-attention slice for all
17 target/MTP owners, parallel to the reusable linear-layer slice.

The decode-only grouped-query-attention candidate keeps persistent K/V pages
packed on the Metal device. Logical pages have deterministic Q2 arena slots;
a bounded Q4 arena holds only sink pages, recent pages, and one
precision-boundary page. One 16-byte descriptor per logical page selects the
arena and physical slot. Stable three-pass softmax decodes Q2/Q4 blocks
directly into registers and never creates an f32 K/V device cache. An append
now packs only the new K/V token into its Q4 slot on device; a page crossing
the retention boundary is demoted directly from its Q4 slot into the fixed Q2
arena before that slot is reused.

A device-only transition candidate now packs resident K/V inputs directly to
canonical Q4_B64 and demotes Q4 blocks to canonical Q2_B64 in the same command
encoder. Its Apple-device Golden test compares every output byte against the
Rust Q4 quantizer and the exact Q4-dequantize-to-Q2 oracle. The paged-cache
verifier now uses these kernels for its actual device pages and verifies every
resident page byte against the independent CPU oracle. The CPU packed mirror
exists only in test builds and drives reference policy/quality checks; release
builds cannot allocate it.

`prepare_paged_gqa_decode_graph` additionally omits the reusable standalone
query/output buffers and binds one cache owner to its canonical full-attention
layer. `append_and_dispatch_paged_gqa_views` validates the exact `PagedKvAppend`
and `PagedGqa` schedule pair, then packs `Key`/`Value`, performs any Q4-to-Q2
demotions, reads `Query`, and writes `AttentionOutput` using offsets in the one
shared decode arena. Release builds do not upload or read back an activation on
this edge. The same private encoder helper serves the standalone verifier and
graph path, preventing their quantization or attention dispatches from drifting.

For the frozen 24-query-head/4-KV-head/256-wide topology, a token contains
2,048 combined K/V values: 576 Q2 bytes or 1,088 Q4 bytes. A 128K layer with
128-token pages, 128 sink tokens, and 256 recent tokens reserves 75,497,472
Q2-arena bytes, 557,056 Q4-arena bytes, and 16,384 descriptor bytes:
76,070,912 bytes (72.546875 MiB) per GQA layer and 1,217,134,592 bytes
(1.133545 GiB) across all 16 attention layers. The production owner uses a
metadata-only page policy and the packed arenas above. A full CPU packed mirror
is present only in test builds and is excluded from release residency by
construction.

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
- `full_attention_fanout_writes_shared_query_gate_key_and_value_views` loads
  canonical Layer-3 Q2/Q4 Q/K/V projections, compares every one of the 14,336
  logical arena outputs with independent recovered-row oracles, proves unused
  alias tails remain zero, and rejects the Layer-7 schedule view.
- `query_gate_norm_rope_deinterleaves_directly_in_shared_arena` checks all
  6,144 normalized/rotated Query values and all 6,144 untouched Gate values at
  position 12,345, reuses the same owner at position zero, and rejects a
  different layer before encoding.
- `attention_sigmoid_gate_projects_mixed_q2_q4_without_product_buffer` verifies
  a mixed Q2/Q4 Layer-3 output projection against the scalar oracle for all
  5,120 rows, proves the 6,144-value gated tensor is not owned, reuses the same
  resources for a zero-input dispatch, and rejects Layer-7 views.
- `complete_full_attention_layer_reaches_next_normalized_view` executes the
  exact ten-step Layer-3 schedule through the Layer-4 normalized arena view,
  compares both final residual and normalization with an independent scalar
  oracle, proves one packed KV token is committed, and rejects Layer 7 before
  cache mutation.
- `paged_q2q4_gqa_decode_matches_quantized_oracle_and_demotes_pages` forces a
  Q4-to-Q2 page transition, compares every decode step with the scalar GQA
  oracle using the identical quantized cache, verifies bounded arena byte
  counts, and proves reset/reuse without an f32 device cache.
- `paged_gqa_consumes_full_attention_shared_arena_without_owned_io` runs seven
  exact-Qwen Layer-3 tokens from shared `Query`/`Key`/`Value` views through the
  persistent packed cache into shared `AttentionOutput`, covers Q4-to-Q2 page
  demotion, compares every output with the quantized scalar oracle, proves the
  graph owner has no local activation buffers, and rejects Layer-7 views before
  mutating cache state.
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
- `mapped_q2_q4_swiglu_down_matches_scalar_oracle_without_product_buffer`
  verifies both fused Q2 and fused Q4 entry points against scalar
  `SiLU(gate) * up` plus recovered-matvec oracles and rejects malformed input
  widths; the product vector exists only in the verifier oracle, never in the
  Metal dispatch.
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
  verifier candidates. GQA release builds now keep only metadata plus packed
  device arenas, while tests retain an independent CPU oracle; neither attention
  path has controlled performance evidence. Target-side embedding through the
  full LM head now executes in one command buffer; the MTP block and sampling
  do not exist yet. A deterministic shared decode arena and its single Metal
  buffer
  exist, and target-hidden plus per-owner linear-state checkpoint/restore is
  available together with bounded paged-KV rollback and one graph-wide atomic
  target+MTP state transaction. All 645 steps have real shared-buffer views;
  exact kernel dispatch now covers steps 0-11 (embedding, layer-0 RMSNorm, all
  four linear-attention projections, in-place causal convolution, and the
  five-output GatedDelta preparation, recurrent FP16-state update, and in-place
  direct-weight gated RMSNorm followed by the recovered Q2/Q4 linear output
  projection, then fused residual-add/Qwen RMSNorm into `HiddenB` and the next
  `Normalized` view, then mixed-Q2/Q4 FFN gate/up fan-out and fused SwiGLU-down
  projection, then fused post-FFN residual-add/next-layer Qwen RMSNorm). The
  complete first linear-attention layer is therefore wired. Full-attention
  schedule slices and the shared-arena Layer-3 fan-out, Query/Gate
  normalization/RoPE, per-head Key RMSNorm/RoPE, combined KV-append/paged-GQA,
  and fused
  sigmoid-gated mixed-Q2/Q4 output projection edges are also executable. The
  complete first full-attention mixer and its residual/FFN tail are therefore
  wired through the next normalized layer input in one encoder and one wait.
  Later layer execution, the prefill arena, release-build memory/high-watermark
  evidence, and complete model-graph execution remain unfinished. The CPU KV
  mirror is verifier-only and is not present in release builds.
- Per `docs/PROMOTION_GATES.md`, all promotion evidence is required before any state change;
  the backend therefore remains fail-closed.
