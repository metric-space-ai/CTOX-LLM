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
| CUDA | complete resident load-graph candidate plus per-op verifiers | none |
| Metal | contract plus direct same-device Q2/Q4 verifier | none |
| Snapdragon HTP/Vulkan | contract | none |

The status table is intentionally conservative. Update it only with verifier
and benchmark artifacts.

## First native artifact

The first direct BF16-to-native pack is independently verified at 7.7695 GiB
including its manifest. It contains text plus resident MTP and produces a
9.5822 GiB corrected calculated 128K Fold plan. Its recovery scales are still identity
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

[`docs/RECOVERY_INITIALIZER_V1.json`](docs/RECOVERY_INITIALIZER_V1.json)
records the first complete 506-matrix positive channel-scale fit and a native
checkpoint carrying those exact FP16 corrections. All matrices improve on its
activation-weighted objective, the packed text+MTP artifact remains below 7.8
GiB, and the Rust loader verifies every tensor checksum. It remains a smoke
initializer: end-to-end KL/CE/hidden/MTP recovery and the release-size
multilingual quality gates have not run.

The expanded initializer in
[`docs/RECOVERY_INITIALIZER_V2.json`](docs/RECOVERY_INITIALIZER_V2.json) merges
167 unique Nemotron, German, Agentic, and genuine long-context samples with
823,996 observed tokens. All 506 matrices, including embedding, LM head, and
resident MTP, are activation weighted. The resulting 127-Q4/377-Q2/two-mixed
assignment and trained channel-scale initializer produce a fully checksummed
8,373,658,112-byte CTOXQ artifact. Its no-MTP FP32-state 128K plan is 9.6110
GiB; the corrected active-MTP4 FP16 replay plan is 9.6815 GiB. These remain
calculated profiles rather than device measurements. It is still not the
release checkpoint: end-to-end KL, cross-entropy,
hidden-state, and MTP distillation plus held-out quality gates remain pending.

The final quality-filtered corpus evidence in
[`docs/RECOVERY_CORPUS_V4.json`](docs/RECOVERY_CORPUS_V4.json) freezes 2,328
recovery-training and 642 held-out samples with zero identity and complete
payload overlap. It balances ordinary chat, coding, mathematics/STEM,
Agentic/tool use, German, twelve additional language strata, and genuine
32K/64K/128K retrieval. All 36 service domains, ten domain families, and 15
language strata pass independent train/evaluation gates. The earlier v3
candidate is retained only as superseded evidence because its Nemotron-v1
`chat` portion contained empty user turns. Five verified teacher batches cover
593 final identities; the missing 1,735 BF16/MTP targets, end-to-end recovery,
and held-out model-quality gates remain pending.

The exact final-cache subtraction and disk-admitted execution plan are frozen
in [`docs/TEACHER_CACHE_FINAL_PLAN_V1.json`](docs/TEACHER_CACHE_FINAL_PLAN_V1.json).
Five verified batches contribute 593 unchanged final identities; the remaining
1,735 samples require 16 token-aware batches and a projected 18.5263 GiB. The
subtraction preserves the complete 36-domain/15-language corpus contract rather
than selecting a coding-only or English-only cache subset.

The release activation selection in
[`docs/ACTIVATION_CALIBRATION_V1.json`](docs/ACTIVATION_CALIBRATION_V1.json)
binds 256 final-training identities and 917,704 tokens. It covers every one of
the 36 primary domains, 15 language strata, and 14 service modes, with explicit
coding, agentic/tool-calling, ordinary chat, mathematics, and genuine 32K/128K
quotas. Three over-96K examples are isolated in token-bounded batches. The
previous 167-sample activation artifact overlaps only 15 final-training
identities and six primary domains, so it is superseded as a release assignment
basis. All six immutable all-506-matrix collection batches ran on GPU1+2 and
their verifier reports passed. The frozen result in
[`docs/ACTIVATION_CALIBRATION_RESULT_V1.json`](docs/ACTIVATION_CALIBRATION_RESULT_V1.json)
uses the complete 256-sample/917,704-token basis to assign 381 full-Q2, 123
full-Q4, and two mixed matrices. Embedding keeps 857 of 970 row groups at Q2;
the measured LM-head error keeps all 970 row groups at Q4. Six iterations of
fixed-code channel-scale fitting reduce the global activation-weighted error by
48.27%, and the fully checksummed text+MTP pack is 8,373,658,112 bytes. This is
the release-corpus initializer, not the final KL/CE/hidden/MTP-trained model.

The signed, backend-neutral release and memory-admission schema is implemented
in `src/release.rs` and documented in
[`docs/RELEASE_MANIFEST_V2.md`](docs/RELEASE_MANIFEST_V2.md). It binds one
logical Q2/Q4+MTP identity across differently packed CUDA, Metal, CPU, and
Snapdragon artifacts. No final release manifest can be sealed until recovery
and held-out qualification freeze the actual logical checkpoint.

The pinned pure-Rust tokenizer and text-only chat/reasoning/tool renderer are
implemented in `src/tokenizer.rs`. They refuse changed tokenizer/template
bytes, preserve the model's exact special-token IDs, and match the offline
Transformers reference on multilingual text, reasoning modes, assistant
history, and a complete tool-call round trip. Exact hashes and Golden evidence
are recorded in
[`docs/TOKENIZER_VERIFICATION_V1.json`](docs/TOKENIZER_VERIFICATION_V1.json).
The concurrent `EngineServer` now owns this frontend for unbuffered multilingual
Responses/tool streaming; constructing it in the public binary remains gated
on a promoted production executor and signed release.

The embeddable Rust lifecycle is implemented in `src/engine.rs` and documented
in [`docs/ENGINE_ABI_V1.md`](docs/ENGINE_ABI_V1.md). It provides signed loading,
warmup, single-session prefill/decode, engine-owned MTP verification,
cancellation, reset, health/capabilities, metrics, and fail-closed zero-residue
unload. The backend table remains unchanged: no complete decoder executor has
yet passed the production promotion gates.

The loader now validates the complete model-specific tensor topology before a
backend receives the artifact: exactly 506 quantized text+MTP matrices, 1,012
paired recovery-scale tensors, and 360 frozen float tensors. Missing, extra,
wrongly shaped, or wrongly typed graph inputs fail closed. The scalar oracle
also covers the pinned Qwen normalization, RoPE, grouped-query attention,
GatedDeltaNet decode/prefill recurrence, convolution state, and SwiGLU
equations. Direct recovered quantized projections and both token-mixer types
are now composed into a stateful target-decoder correctness path.

The mmap-backed target graph composes recovered embedding, Qwen RMSNorm, both
attention types, gate/up/SwiGLU/down residual MLPs, final norm, and recovered LM
head without copying or repacking model tensors. Full attention includes the
interleaved query/gate projection, head norms, partial RoPE, causal GQA, KV
state, output gate, projection, and residual. Linear attention includes causal
convolution, Q/K head repetition, beta/decay, recurrent state, gated RMSNorm,
projection, and residual. The stateful layer iterator follows the frozen
three-linear/one-full pattern for all configured layers. The native MTP path
normalizes the selected-token embedding and prior target final hidden, applies
`mtp.fc`, its cached full-attention/MLP block, `mtp.norm`, and the shared LM
head. It remains a correctness executor: production paged-Q2/Q4 KV, fused
token-mixer kernels, chunked prefill, and optimized backend integration remain.

Metal now has a verified no-copy CTOXQ ownership primitive: one shared Metal
buffer wraps the complete immutable file mapping, while projections bind
weights and packed recovery scales by validated offsets. The fixture survives
the original loader handles being dropped and reports zero copied model bytes.
Mixed Q2/Q4 matrices dispatch their original contiguous row groups through the
same mapping and one command encoder, with no backend-specific repacking.
The restricted MTP LM head also gathers arbitrary canonical token rows from
that mapping in one batched Q2/Q4 command path instead of copying or expanding
the vocabulary matrix. Embedding lookup resolves one pure or mixed Q2/Q4 row
and decodes it with its packed FP16 `s_in`/`s_out` corrections directly from
the same mapping; only the resulting hidden vector is transient.
Qwen `(1 + weight)` RMSNorm is also a direct Metal candidate: its FP16 weight
stays mapping-backed, while reusable f32 input/output graph buffers support
both one-row decode and multi-row prefill.
Decode RMSNorm and a following mixed Q2/Q4 projection can now be encoded in
one command encoder: the projection reads the norm output directly and omits
its otherwise duplicated activation allocation.
Qwen's non-interleaved partial RoPE is a native in-place Metal candidate for
the exact 24-query/4-key-head, 256-wide, 64-rotary-dimension topology. Query
and key transforms share one command encoder and one synchronization.
Complete-graph residency and unload measurements on the 7.8-GiB artifact are
still required before this changes backend promotion state.
Metal target selection now also has a finite-checking full-vocabulary argmax
candidate. It returns only the selected token and invalid-count words, matches
the engine's larger-token tie rule on Apple Silicon, and rejects non-finite
logits. A composed final RMSNorm -> recovered Q2/Q4 LM-head -> argmax verifier
now binds the selector directly to resident logits in one command encoder; the
complete Metal decoder executor remains open. Its host assembly now has a
frozen 645-step decode schedule covering all 64 layers, target LM head, native
MTP transition, and exactly one final command-buffer wait. The contract proves
16 paged-attention and 48 linear-attention layers, both residual norms per
layer, and every KV/convolution/recurrent state mutation before runtime binding
is admitted. The matching artifact-resource plan now binds every schedule step
to the exact backend-neutral ownership set: all 505 non-embedding projections,
262 recovery activation groups, 48 linear mixers, 17 full-attention owners,
four regular norms, and 130 residual norms. This closes the logical binding
contract; it does not yet provide the complete Metal executor or hardware
promotion evidence. A fail-closed execution cursor admits only the current
committed token position, records those bound operations in exact order, and
returns the next committed position only after all 645 steps. A deterministic
liveness pass now packs all 21 named f32 decode activation slots into one
256-byte-aligned arena: the frozen 40,000-row MTP-draft profile needs 1,173,760
bytes instead of 1,633,280 bytes with independent buffers. Aliasing is admitted
only for non-overlapping produced-value intervals; target and MTP logits remain
simultaneously live and therefore distinct. `MetalCandidateRuntime` now
materializes that plan as exactly one shared Metal buffer, exposes only the
validated buffer/offset pairs, and passes write/read plus drop/recreate device
tests. Every logical read and write of all 645 bound decode steps now resolves
to a typed view of that same real buffer and its exact schedule-derived offset;
the final barrier retains target and MTP logits as explicit reads. The first
exact decode chain now binds those views directly: embedding writes `HiddenA`,
layer-0 RMSNorm writes `Normalized`, and all four linear-attention projections
write `LinearQkv`, `LinearZ`, `LinearA`, and `LinearB` through one command
encoder; the layer-0 causal convolution then updates `LinearQkv` in place in
that same encoder, then a fused preparation kernel expands Q/K to 48 heads and
writes `Query`, `Key`, `Value`, `LogDecay`, and `Beta` into their exact arena
slots. The recurrent GatedDelta kernel consumes those five views, mutates only
its checkpointed FP16 state, and writes `AttentionOutput`; direct-weight gated
RMSNorm then consumes that view plus `LinearZ` and updates `AttentionOutput`
in place. The recovered Q2/Q4 linear output projection consumes that exact view
and writes `MixerOutput` before the final wait. `A_log`, `dt_bias`, and the FP16
norm weight remain mmap-backed.
These graph preparations retain no operation-local input/output activation
buffers; separately stored recovery inputs must be byte-identical. The
remaining 637 schedule steps and the complete executor remain open. A bounded
f32 checkpoint can now snapshot and restore an
exact arena slot through a Metal device-to-device blit with no host mirror. It
is single-use and fail-closed across snapshot/restore/commit, providing the
target-hidden primitive needed by MTP replay. FP16 causal-convolution and
GatedDelta state owners now have the same device-only snapshot/restore/commit
lifecycle and account one checkpoint equal to their active-state bytes. Paged
KV now uses a constant-size append marker plus small page-slot metadata: an
active four-token branch suppresses Q4 demotion and consumes the already
budgeted boundary slot, so restore never copies the full Q2/Q4 arenas. Metal
replay reproduces the original branch outputs exactly. A graph-wide Metal
transaction now coordinates the final normalized target hidden, all 17
target/MTP attention owners, and all 48 paired causal-convolution/GatedDelta
owners. Begin prevalidates the entire resource set before changing any owner;
reject restores state in reverse order, while commit consumes every checkpoint
as one logical operation. The Apple-device verifier proves both all-owner
rollback and commit. An RAII decode-attempt owner now couples this transaction
to the exact 645-step execution cursor and real arena-view program: wrong token
positions fail before snapshot, incomplete or early-commit attempts restore on
drop, and only the sole completed final barrier returns the next committed
token position. Kernel encoder dispatch and the complete target+MTP executor
remain open.

CUDA SM86 now has an isolated exact-Qwen paged-GQA candidate in addition to
the projection and token-mixer candidates. Q4 append quantization and
Q4-to-Q2 page demotion execute on device, persistent storage contains no
expanded FP16/FP32 KV cache, and attention uses one-pass online softmax over
the packed pages. A context-bound borrowed device-view API now accepts
device-resident Q/K/V and returns device-resident attention output; only its
standalone verifier wrapper uploads and reads f32 host slices. Partial RoPE
likewise transforms producer-owned Q/K views in place, completing that
device-resident edge into GQA. Shared A8 Q/K/V fan-out now also consumes a
producer-owned activation and exposes its projection outputs as sliceable
device views. Qwen RMSNorm and the post-attention gated RMSNorm accept the same
device-view contract, closing the tensor-transfer edges around this decode
subgraph. The canonical head-wise query/gate layout now has an unpromoted fused
deinterleave/Q-norm/RoPE candidate with a composite verifier; residual fusion
now combines the sublayer update with the following Qwen RMSNorm while
retaining both device outputs; final scheduler assembly remains. Causal convolution and the FP16
GatedDelta recurrence also accept producer-owned device views. An exact-Qwen
preparation candidate now repeats compact 16-head Q/K to 48 heads and applies
the A/B-to-decay/beta transforms on device, then feeds those views directly
into the recurrence. Its expanded scalar-oracle verifier is queued on physical
GPU 2 after the BF16 recovery pipeline; the production backend remains
fail-closed until that evidence and
the complete projection/RoPE/attention/output graph wiring exist.
For the FFN path, a fused CUDA candidate now combines `SiLU(gate) * up`, the
down-projection recovery input scale, and A8 block quantization without a
separate f32 SwiGLU allocation. Hardware numerical evidence remains required
before it can be used by the production executor.
The CUDA assembly contract is now a frozen 645-step device schedule covering
all 64 target layers, final logits, and MTP verification with exactly one host
barrier at token completion. It validates device-slot dataflow and rejects any
topology other than the exact Qwen3.8-27B configuration. An exact binding plan
now resolves all 645 steps to the resident artifact-backed resources and proves
unique coverage of every 505 projection, 262 activation owner, 48 linear mixer,
17 full-attention state, and 134 norm operator. The CUDA
projection loader now validates and uploads pure or mixed Q2/Q4 matrices and
packed FP16 recovery scales directly from mmap-backed `RecoveredMatrixView`
objects, without widening/repacking model state or allocating a redundant f32
input buffer per matrix. A deterministic artifact graph covers all 505
non-embedding target/MTP projections with exactly 262 activation owners and a
full-load/unload verifier. The packed 248,320-row embedding is resident and
selected in place without per-token weight upload. The graph also binds every
linear-attention parameter and persistent FP16 state plus context-sized Q2/Q4
KV arenas for 16 target and one MTP full-attention layer; the queued GPU3 run
uses 128K and is a residency gate, not an end-to-end inference claim. All 134
target/MTP normalization operators are also owned by the load graph: four
standalone input norms and 130 fused residual/RMSNorm edges. The previously
open full-attention gate edge has a fused sigmoid-gate/recovery/A8/output-
projection CUDA candidate and dedicated hardware verifier. Dispatching the
bound schedule has now reached a first complete target-token candidate:
embedding, all 64 hybrid layers, final normalization, and the LM head pass
device views directly with no tensor readback before the token boundary. The
complete graph now defers operator-local driver barriers and commits each
target or MTP transition with one context synchronization. The dedicated SM86
verifier records attempted/committed submissions, deferred barriers, and the
explicit verifier readbacks; its corrected target-one-ahead checkpoint hardware
result is still pending. The CUDA
graph now also owns exactly one FP16 checkpoint for every linear recurrent and
convolution state, one target-hidden checkpoint, and retained Q4 KV boundary
capacity. It can restore a speculative branch without copying state through
the host; the verifier requires the replayed target logits to be bit-identical.
The one-layer MTP draft is now connected to the final normalized target hidden
state through a device-only concatenation buffer and reuses the same embedding,
attention, FFN, norm, and LM-head operators. Its hardware run and subsequent
target verification use a second complete target transition and report either
an accepted draft or the target fallback without hiding rejection. A
verifier-only `CudaModelExecutor` now drives load, warmup, layer-major prefill,
chained MTP4 target verification, bounded device checkpointing, full-branch
commit, accepted-prefix restore/replay, reset, allocation accounting, and unload
through the shared Rust ABI. The final accepted draft advances MTP state without
reading either LM head, so a fully accepted block retains the already computed
target branch with no replay. `qwen38-cuda-executor-verify` binds that lifecycle to the exact
artifact, CUDA module, and canonical release draft-vocabulary hashes. Its
MTP proposals use a Q2/Q4-aware gathered projection over exactly the canonical
40,000 draft rows while the full target head remains resident for verification;
only 320,000 bytes of row IDs and compact logits are added to the graph. Its
hardware verifier replays an identical checkpoint through both head variants
and requires all 40,000 logits to match bit-for-bit. Its complete hardware run,
quality gates, and roofline promotion remain open. Greedy MTP decisions now
come from a finite-checking device argmax and are compared with the host oracle;
the server ABI returns only compact draft/target/bonus token decisions while an
explicit hardware-evidence mode retains complete logit readbacks. Each compact
decision is still causally checked by the engine before commit. A
pinned-TensorRT-derived top-k/top-p candidate now accepts canonical
caller-supplied RNG draws without
host logit readback and is bound to ordinary target selection in the CUDA
executor; its same-device primitive/lifecycle runs, unrestricted top-p,
on-device RNG state, and stochastic MTP rejection sampling remain open. For IPC
verification, a sendable adapter
owns this deliberately thread-affine CUDA executor on one dedicated worker;
the socket threads exchange typed commands and never move driver objects.

Chunked CUDA prefill now also has a layer-major 645-step schedule contract and
an exact resident-resource binding plan. Before execution is enabled, that
plan resolves the complete prompt program to the same 505 projections, 262
shared activation owners, 48 linear mixers, 17 full-attention states, and 134
norm operators used by decode; reordered residual producers or missing MTP
resources fail closed. The loaded graph now owns exactly one reusable
73,533,440-byte frontend and full-attention workspace pool for a 512-token
chunk: a 2,048-byte device token-ID list, 10,485,760-byte token-major
embedding output, hidden and K normalization, shared RoPE tables, Query/Gate
output, and causal GQA output. Its planned and allocated byte counts must agree
exactly, and the pool does not scale with the 16 target attention layers plus
MTP. Projection workspaces remain separate from that immutable ownership
contract. The graph now also owns one reusable 84,082,688-byte linear-attention pool for the same
512-token chunk: causal-convolution output, prepared GatedDelta Q/K/V/decay/
beta inputs, recurrent output, and batched gated-RMSNorm output. Those buffers
are shared by all 48 linear-attention layers rather than multiplied by layer
count, and planned versus allocated bytes fail closed. The three graph-owned
prefill pools therefore total 240,584,704 bytes before executor-specific
scratch. The schedule batches every large
Q2/Q4 projection, retains causal device scans for paged GQA, convolution,
GatedDelta recurrence, and MTP state, computes the target LM head only for the
last token of the final prompt chunk, and exposes one cancellation/commit
barrier per bounded chunk. Intermediate chunks commit all target/MTP state
while skipping the otherwise unused 248,320-row head read. Batched
activation/output workspaces are separate from resident matrix
owners, so enabling a 512-token chunk does not duplicate model weights. The
frozen projection arena now proves that all 504 chunk-wide target/MTP matrices
fit four conflict-free output slots plus one maximum-width A8 encoding slot:
82,968,576 planned bytes at 512 tokens. The 248,320-row LM head remains outside
that arena because prefill consumes only its final prompt row. The graph now
allocates this arena once and binds compact offset views to the MMQ dispatcher;
a mixed-Q2/Q4 A4500 run using a workspace twice as wide as the active matrix is
bit-identical to the established MMQ graph path. A fail-closed execution cursor
now admits a chunk only at the exact committed position and releases its new
commit position only after all 645 bound operations and the final barrier have
completed in order. The target-only executor now dispatches that complete
program, and `qwen38-cuda-model-prefill-verify` compares target-only or MTP-
enabled execution with the sequential 64-layer device path after an explicit
reset. Prompts above 512 tokens traverse multiple bounded chunks and therefore
exercise the retained target-hidden MTP boundary. Complete-graph SM86 hardware
evidence remains open. The fused FFN SwiGLU/A8 and full-attention sigmoid-
gate/A8 candidates now use `grid.y` for all 512 prompt rows while retaining the
single-row decode ABI. On the RTX A4500, selected rows were bit-exact to the
sequential CUDA path, CPU-equation scale error stayed below `1.12e-8`, and all
109,051,904 observed bytes were reclaimed. Evidence is in
`benchmarks/cuda/sm86-batched-fused-a8-512-20260826.json`; graph-wide binding
now exposes all three arena-backed projection forms, while the 645-step
target-only executor transaction now dispatches them layer-major with one
barrier per chunk. MTP-enabled prefill uses the same batched projections and
causal attention path. Its chunk contract proves the first-token omission,
prior-chunk hidden boundary, and one-step KV/RoPE offset, while a direct
`cuMemcpy2D_v2` primitive provides device-only strided row assembly without
another CUDA kernel. Complete-model hardware evidence for both paths remains
open. A
pinned-`get_rows`-structured batched embedding
candidate now keeps FP16 `s_in`/`s_out` resident and gathers a whole token-ID
chunk in at most one launch per canonical Q2/Q4 segment. The final 857-Q2/
113-Q4 embedding assignment is bit-identical to sequential CUDA row lookup on
the A4500 and differs from the CPU oracle by at most `7.45e-9`; all
398,458,880 verifier-owned bytes are reclaimed. Executor binding remains the
next gate. The
standalone MMQ verifier exercises this same graph-facing path and the shared
two-buffer batched RMSNorm workspace. The first causal-convolution scan now
matches sequential CUDA output and final FP16 state bit-for-bit across a
17-token hardware fixture. The upstream-structured GatedDelta scan likewise
matches 11 sequential CUDA launches bit-for-bit for every output and the final
FP16 recurrent state, while keeping one persistent state owner. Its batched
Qwen input preparation now expands compact Q/K, copies V, and transforms raw
A/B in one launch while sharing the immutable A_log/dt_bias allocation. The
direct causal paged-GQA prefill scan is also exact against sequential all-Q4
decode and stays within `5.97e-8` of the mixed Q2/Q4 scalar oracle. Its new
production-shaped KV page packer consumes token-major device K/V views and
submits one two-dimensional launch per crossed page rather than one launch per
token. On the 40-token SM86 fixture, five page launches produce bit-identical
all-Q4 attention and the same 24-Q2/16-Q4 mixed cache as the scalar oracle;
the pack kernel uses 16 registers with no spills. Batched partial RoPE now
builds one compact position table on device and shares it across token-major
query and key views; at position 131,071 both remain within `5.97e-8` of the
sequential CUDA path and preserve every non-rotary tail value bit-for-bit.
The batched Query/Gate fusion now consumes that same table, deinterleaves all
prompt rows, applies the resident Q RMSNorm, and rotates Q in one launch per
chunk; Query differs by at most `2.39e-7` from sequential CUDA and Gate is
bit-identical. Executor replacement of the current sequential loop remains
open. The schedule names batched key RoPE and persistent KV append
explicitly before every causal GQA scan; these state mutations can no longer
be hidden by a nominal attention step.

An isolated mixed-Q2/Q4 split-KV attention candidate now covers the five
causal tail queries used by MTP4 verification. Sixteen KV segments expose
1,920 partial blocks on SM86 and a second kernel combines their online-softmax
state entirely on device. The byte-identical Apache-2.0 upstream patch is
pinned beside the existing CUDA references. Scheduler integration remains
closed until the GPU3 numerical/lifecycle verifier and representative-context
latency sweep pass; the verifier now reports a one-barrier comparison with
five sequential full-cache attention launches at short, 1,536-token, and
16,384-token contexts. Long-context fixtures use the same device pack/demotion
path but skip per-append attention so benchmark setup remains linear.

The Metal linear-attention candidate set now also covers FP16 causal-
convolution history, FP16 recurrent GatedDelta state, and the direct-weight
gated RMSNorm with fused `SiLU(z)`. Each operation matches its scalar state or
numerical oracle and keeps immutable float weights mapping-backed. A shared
full-layer scheduler still has to connect their device buffers without host
intermediates before the backend can be promoted.

`CpuCorrectnessExecutor` connects that target graph to the stable embeddable
`ModelExecutor` lifecycle: shared-`Arc<Mmap>` load, warmup, sequential prefill,
incremental decode, cancellation, reset, allocation reporting, and zero-residue
unload are exercised together. Its promotion state is permanently `Verifier`,
and production admission rejects it because attention, MTP attention, and
GatedDeltaNet still use scalar oracle composition. Its MTP verifier is enabled
only for greedy sampling: the executor returns unverified draft logits and the
engine itself compares them with the target argmax, without double-counting
the accepted token in context state.

[`docs/MEMORY_PLAN_CORRECTION_V2.json`](docs/MEMORY_PLAN_CORRECTION_V2.json)
records the 7.5-MiB causal-convolution state that the earlier calculated Fold
figures omitted. The subsequent active-MTP correction in
[`docs/MEMORY_PLAN_CORRECTION_V3.json`](docs/MEMORY_PLAN_CORRECTION_V3.json)
adds the independent 72.1875-MiB MTP KV cache and speculative target-state
strategy. The paged implementation in
[`docs/MEMORY_PLAN_CORRECTION_V4.json`](docs/MEMORY_PLAN_CORRECTION_V4.json)
also attributes page metadata, a worst-case Q4 boundary page, and Q4-to-Q2
conversion scratch. At the 7.8-GiB weight ceiling, only FP16 state plus
replay-on-reject keeps MTP4 below 9.7 GiB: 9.68562 GiB with 14.73 MiB calculated
headroom. This still requires numerical and Android PSS/accelerator-memory
evidence. Historical artifact hashes and byte counts are unchanged.

[`docs/PAGED_KV_NOTES.md`](docs/PAGED_KV_NOTES.md) defines the canonical
128-token page layout and the zero-copy page-view/update contract used to wire
the same logical Q2/Q4 cache into CUDA, Metal, and Snapdragon Vulkan kernels.

[`docs/WIRE_PROTOCOL_V1.md`](docs/WIRE_PROTOCOL_V1.md) defines the matching
versioned Unix-socket/named-pipe control and token-stream contract. Artifact
bring-up negotiates and reports health but remains fail-closed with
`engine_not_ready`. An explicit signed-release CPU-verifier mode now exercises
the real loader, tokenizer, token-ID and Responses generation, cancellation,
MTP token ordering, reset, and unload through the same reusable server adapter.
Its verifier promotion state cannot pass production admission. With the
`cuda` feature, the same server binary accepts `--verification-cuda` plus the
signed release inputs and exact CUDA module; it still selects verifier policy
and cannot be mistaken for a promoted service. CUDA now has an assembled
verifier executor and IPC path awaiting complete-model hardware evidence;
Metal executor assembly remains unfinished.
