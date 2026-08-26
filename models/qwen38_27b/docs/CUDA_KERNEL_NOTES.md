# CUDA SM86 Q2/Q4 kernel notes

Status: **Verifier candidate**. A direct CUDA Driver API runtime now loads an
explicit SM86 cubin and executes isolated Q2_B64/Q4_B64 fused matvecs. The
public `CudaBackend` remains `Contract` and fail-closed: the candidate is not a
production kernel and no full graph or promotion is claimed.

The serving evidence and non-transferable benchmark distinctions from the
user-supplied RTX 3090 project are pinned separately in
[`RTX3090_UPSTREAM_NOTES.md`](RTX3090_UPSTREAM_NOTES.md). In particular, its
chained MTP, split-KV verify attention, FP16 recurrent state, and hybrid prefix
cache inform CTOX candidates but do not promote a CTOX kernel.

## Vendored baseline

Pinned llama.cpp revision `ef2d770117db45b05aa7ecd1b0acca36370c5470` (MIT,
(c) 2023-2026 The ggml authors). Files, SHA-256 digests, purposes, and the
required upstream includes deliberately not vendored are recorded in
`models/qwen38_27b/vendor/cuda/UPSTREAM.json`. Vendored files are unmodified
reference material; they are not compiled into the crate.

The adapted candidate source is
`kernels/cuda/q2q4_fused_matvec_sm86.cu`. It retains `// ref:` anchors to the
pinned dequantization, unpack, warp-reduction, and MMQ organization while
implementing CTOX's different 64-value blocks. It is compiled independently;
the vendored ggml sources are never linked into the runtime.

The minimum 2-bit/4-bit reference set:

| File | Ground truth used for |
|---|---|
| `ggml/src/ggml-common.h` | upstream packed block layouts (block_q2_K 288-299, block_q4_K 317-328) |
| `ggml/src/ggml-cuda/dequantize.cuh` | scale fetch + nibble/bit extraction (25-38) |
| `ggml/src/ggml-cuda/vecdotq.cuh` | integer unpack + dp4a accumulation (7-32, 115-137) |
| `ggml/src/ggml-cuda/mma.cuh` | Ampere-gated mma.sync / ldmatrix wrappers (923-1022, 1162-1224) |
| `ggml/src/ggml-cuda/mmq.cuh` | MMQ tile sizes (186-229), tile loaders (1628+, 2093+), kernel entry (3542) |
| `ggml/src/ggml-cuda/mmq.cu` | stream-k and per-arch eligibility gating (121-122, 267-378) |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | register-sharded recurrent update, warp reductions, prefill traversal, rollback slots |
| `ggml/src/ggml-cuda/norm.cu` | RMSNorm block reduction and fused multiply organization |
| `ggml/src/ggml-cuda/rope.cu` | non-interleaved/NeoX rotary pairing and launch organization |
| `ggml/src/ggml-cuda/ssm-conv.cu` | width-4 depthwise convolution and fused SiLU organization |
| `ggml/src/ggml-cuda/fattn*.{cu,cuh}` | online-softmax, vector, tile, MMA and WMMA flash-attention organization |

All twenty-three vendored files are byte-identical to the same immutable
revision and are verified by `training/verify_vendor_manifest.py`. The pinned
operator families are reference baselines, not compiled ggml dependencies.
Their CTOX adaptations must preserve `// ref:` anchors while changing state
storage to FP16 where the signed memory profile requires it, keeping immutable
float weights mapping-/pack-owned, and exposing only the frozen Qwen geometry.

Layout divergence: upstream Q2_K/Q4_K use 256-value super-blocks with
sub-scales; this crate uses Q2_B64/Q4_B64 (64 values, one f16 scale, no
zero-point). Only Q2_B64 and Q4_B64 exist; there is no Q3. The vendored files
pin the dp4a/mma techniques and launch geometry, not the block format.

## SM86 kernel ABI (see `src/backend/cuda.rs`)

- Target compute capability 8.6, warp size 32, LP64 device pointers.
- Module must export at least:
  - `ctox_q2_b64_fused_matvec_sm86` (18-byte blocks: f16 scale + 16 code bytes)
  - `ctox_q4_b64_fused_matvec_sm86` (34-byte blocks: f16 scale + 32 code bytes)
- The same verifier cubin additionally exports eighteen explicitly unpromoted
  candidates: an A8 quantizer, two A8/dp4a projections, two recovered-row
  decoders, the persistent-state GatedDelta recurrence, causal convolution,
  and gated RMSNorm:
  - `ctox_quantize_a8_b64_sm86`
  - `ctox_quantize_swiglu_a8_b64_sm86`
  - `ctox_quantize_sigmoid_gate_a8_b64_sm86`
  - `ctox_q2_b64_a8_matvec_sm86`
  - `ctox_q4_b64_a8_matvec_sm86`
  - `ctox_q2_b64_recovered_row_sm86`
  - `ctox_q4_b64_recovered_row_sm86`
  - `ctox_qwen_gated_delta_prepare_f32_sm86`
  - `ctox_gated_delta_recurrent_f16_sm86`
  - `ctox_causal_conv_silu_f16_sm86`
  - `ctox_gated_rms_norm_f16_sm86`
  - `ctox_qwen_rms_norm_f16_sm86`
  - `ctox_qwen_residual_rms_norm_f16_sm86`
  - `ctox_partial_rope_f32_sm86`
  - `ctox_qwen_query_gate_norm_rope_f32_sm86`
  - `ctox_pack_paged_kv_q4_f32_sm86`
  - `ctox_demote_paged_kv_q4_to_q2_sm86`
  - `ctox_paged_q2q4_gqa_decode_f32_sm86`
  These symbols are intentionally excluded from the production module ABI
  until their quality and complete-graph gates pass.
- One launch fuses dequant, dot product, `s_in`, `s_out`, bias, and
  activation (Identity/Silu). Parameter buffer is 60 bytes, tightly packed:
  six device pointers (weights, input, s_in, s_out, bias, output) then
  `rows`, `columns`, `activation` as u32.
- Validation is fail-closed: wrong compute capability, wrong block geometry,
  a malformed parameter layout, or any missing symbol rejects the module.
  There is no scalar fallback in the CUDA path.

## Reproducible candidate build and verifier

On a CUDA 12 toolchain:

```sh
./scripts/build_cuda_sm86.sh
cargo run --release --features cuda --bin qwen38-cuda-bench -- \
  --module target/cuda-sm86/q2q4_fused_matvec_sm86.cubin \
  --dtype q2 --rows 5120 --columns 5120
cargo run --release --features cuda --bin qwen38-cuda-gated-delta-verify -- \
  --module target/cuda-sm86/q2q4_fused_matvec_sm86.cubin --device 0
```

The build script resolves the canonical `nvcc` target rather than trusting a
misdirecting symlink, emits a native `sm_86` cubin, records its SHA-256, and
captures `cuobjdump` resource usage. The Rust verifier dynamically loads only
the NVIDIA Driver API, requires compute capability 8.6, keeps the projection
buffers resident across launches, and compares device output with the scalar
CPU oracle before timing.

## Frozen decode schedule

`src/backend/cuda_schedule.rs` now freezes the complete single-token target
schedule as 645 explicit steps over all 64 layers: 16 paged-GQA layers, 48
GatedDeltaNet layers, two residual/norm fusions per layer, FFN fan-out/down
edges, final LM head, MTP draft/target verification, and one final token
barrier. A dataflow validator rejects reads from unavailable device slots,
non-frozen topology, missing residual fusions, or any intermediate host
barrier. The fused attention-gate/A8/output-projection edge is implemented as
an unpromoted CUDA candidate rather than hidden behind a CPU fallback.

The recurrent candidate accepts only Qwen3.8-27B's frozen 48-head,
128-key-dimension, 128-value-dimension profile. Its state is exactly 1,572,864
bytes of FP16 and has no FP32 shadow; query, key, value, decay, beta, and output
add 98,688 reusable transient bytes. One 128-thread block owns one head and
one thread owns one value column. Decay and update stores round immediately to
FP16, matching the Rust and Metal oracle. CUDA 12.6 compiled the candidate for
SM86 with 24 registers, 40 bytes shared memory, and zero stack/spill bytes
(current unified cubin SHA-256
`060efe2ac64615b62a24e854360c7ca8fb21225fc65fff3fbf7c6c91d2362094`).
The numerical verifier is built for a later physical-GPU-2 run after the
teacher/evaluation/activation pipeline releases GPU 1+2; GPU 0 remains
reserved for Greppy. No numerical or performance promotion is claimed yet.

The same unpromoted token-mixer set now contains exact-profile CUDA
candidates for the 10,240-channel, width-4 causal convolution with fused SiLU
and for the 48x128 direct-weight gated RMSNorm. Convolution weight and history
remain FP16 (81,920 bytes each per layer); the gated-norm weight remains 256
bytes of FP16. CUDA 12.6 reports 22 and 31 registers respectively, with zero
stack and spill bytes for both. `qwen38-cuda-linear-ops-verify` checks six
stateful convolution steps through producer-owned device views, exact FP16
history, reset, gated-norm output, and buffer reclamation against the Rust
oracles. Its physical-GPU-2 run is chained
after the GatedDelta verifier; neither job may use GPU 0.

The FP16 GatedDelta recurrence now has an equivalent device-view entry point
for Q, K, V, log-decay, and beta, returning its device-resident output while
retaining the state-poison/reset contract. A preceding exact-profile candidate
consumes the convolution's compact `[Q:16x128, K:16x128, V:48x128]` view plus
the two 48-value A/B projection views, repeats Q/K to the 48 recurrence heads,
and computes `-exp(A_log) * softplus(A + dt_bias)` and `sigmoid(B)` entirely on
device. Its verifier compares all four prepared outputs with the Rust oracle
before feeding them directly into the recurrence; graph execution itself does
not use verifier staging owners. CUDA 12.6 compiles this preparation kernel
with 18 registers and no stack or spill traffic. Physical-GPU numerical
evidence is still required before either candidate can be promoted.

The Qwen FFN down-projection path also has an unpromoted fused activation
quantizer. It consumes the producer-owned 17,408-value gate and up projection
views, computes `SiLU(gate) * up`, applies the down-projection's packed FP16
`s_in`, and emits A8 codes/scales in one launch. This avoids materializing and
rereading a 69,632-byte f32 SwiGLU tensor for every token and layer. Its
verifier compares every A8 code and block scale with the Rust oracle before
the candidate may feed the existing Q2/Q4 dp4a down projection. CUDA 12.6
reports 18 registers, 12 bytes shared memory, and no stack or spill traffic.
The device graph now enqueues this quantizer and its identity-bound Q2/Q4 down
projection consecutively, synchronizing only once after the projection set.
The verifier requires both exact A8 codes/scales and a device-resident down
projection output, so the fused edge is not merely tested in isolation.

Full-attention output uses the parallel fused path: packed GQA output is
multiplied by `sigmoid(attention_gate)`, corrected by the output projection's
FP16 `s_in`, and quantized to A8 before the Q2/Q4 projection without a
6,144-value f32 gated-attention allocation. Its verifier checks every code and
scale plus the directly chained projection output on the same device. CUDA
12.6 reports 18 registers, 12 bytes shared memory, and no stack or spill
traffic.

The general Qwen `(1 + weight)` RMSNorm candidate uses one 256-thread block
per row and an eight-warp reduction, so hidden width 5,120 does not serialize
through a single warp. It accepts positive 32-aligned widths, keeps its learned
weight FP16, and has no f32 weight expansion. SM86 compilation uses 16
registers, 36 bytes shared memory, and zero stack/spill bytes. The same chained
linear-op verifier now feeds both this operation and the direct-weight gated
RMSNorm from producer-owned device views, includes a two-row 5,120-wide oracle
comparison, and accounts for model, transient, and explicit verifier-staging
buffers in the unload proof.

The residual variant fuses `residual + sublayer_update` with the following
Qwen RMSNorm in the same 256-thread block. It writes both the updated residual
and normalized activation, so the scheduler can preserve the skip connection
while feeding the next projection without a standalone add kernel or a second
read of the sum. The expanded linear-op verifier requires an exact residual
sum plus the existing RMSNorm tolerance before this edge may be promoted.
CUDA 12.6 reports 16 registers, 36 bytes shared memory, and no stack or spill
traffic for the fused residual/RMSNorm kernel.

Qwen partial RoPE is implemented as an in-place non-interleaved/NeoX-pairing
candidate anchored to the newly pinned upstream `rope.cu`/`rope.cuh`. Query
(24x256) and key (4x256) profiles share the 64-value rotary prefix and leave
the remaining 192 values per head byte-identical. Only two 32-value f32
trigonometric tables are prepared per position. Its context-bound device-view
entry point mutates projection-owned Q/K allocations in place and returns the
same borrowed view, so the production edge into paged GQA needs no staging
copy. SM86 compilation uses 18
registers with no stack, spills, or shared memory; the chained verifier checks
this device-view path at position 131,071 against the Rust oracle and requires
an exactly unchanged tail.

The first packed paged-GQA correctness candidate fixes Qwen3.8-27B's exact
24-query-head, 4-KV-head, 256-wide geometry. Persistent device storage consists
only of canonical Q2_B64/Q4_B64 page arenas plus 16-byte page descriptors; it
never allocates an expanded FP16/FP32 KV cache. Sink, recent, and current pages
remain Q4 while completed middle pages are demoted to Q2. Q4 append
quantization now runs directly on the GPU, and completed pages are converted
Q4-to-Q2 without an f32 intermediate or host packed-cache mirror. Explicit RN
arithmetic pins Q4 codes to the Rust canonical formula. The attention kernel
directly decodes packed K/V and performs numerically stable online-softmax plus
value accumulation in one cache scan rather than separate max, denominator and
output scans. CUDA 12.6 reports 15 registers/76 bytes shared memory for Q4
packing, 16 registers/no shared memory for demotion, and 128 registers/no
shared memory for GQA; all three have zero stack/spill bytes. GQA's
16-warps-per-SM launch bound is explicit. The current unified cubin SHA-256 is
`060efe2ac64615b62a24e854360c7ca8fb21225fc65fff3fbf7c6c91d2362094`.
Its numerical/demotion/reset/unload verifier is queued on physical GPU 2 after
the teacher, evaluation, activation and earlier verifier chain; GPU 0 is never
eligible. The CPU `PagedKvCache` exists only in the separate verifier as an
oracle. The Rust runtime now exposes a lifetime- and context-bound device-view
entry point: it consumes device-resident Q/K/V pointers, performs append,
demotion and attention without host copies, and returns a borrowed device view
of the result. The slice-based upload/readback wrapper remains verifier-only.
The complete decoder scheduler still has to connect projection, RoPE, GQA,
gate and output-projection views, and the online-softmax path still needs
controlled roofline evidence.

The first GPU3 run is recorded in
`benchmarks/cuda/sm86-q2q4-fused-matvec-20260826.json`. It proved both formats
numerically and measured a 5120x5120 projection. Q2 used 56 registers with no
spills and reached 110.82 packed-weight GB/s; Q4 used 44 registers with no
spills and reached 210.95 GB/s. These are useful baseline measurements, not
promotion evidence: another BF16 job occupied the other GPUs, clocks and
thermals were uncontrolled, allocator overhead and hardware counters were not
captured, and both kernels remain below the required practical roofline. The
near-identical Q2/Q4 latency shows that scalar unpack/FMA instruction
throughput, not weight bandwidth, is the baseline Q2 limiter.

An explicit symmetric A8_B64 activation path is now implemented as the next
verifier candidate. It applies packed FP16 `s_in` first, deterministically
quantizes each 64-value activation block to signed int8 with one f32 scale,
and runs half-warp-per-row Q2/Q4 `dp4a` matvecs. The logical Q2/Q4 weight codes
are unchanged; there is no backend-specific weight requantization, and A8
buffers are transient rather than serialized model state. The current prepared
object retains that activation only for one projection. A separate
shared-activation dispatcher now owns one corrected input, one transient A8
code/scale pair, and matrix-local Q2/Q4 projections. It refuses a projection
unless the column count, CUDA context, and SHA-256 identity of the exact packed
FP16 `s_in` bytes match. Its device-view entry point consumes a producer-owned
activation directly and returns borrowed projection-output views. K and V can
feed the subsequent normalization/RoPE path directly. The canonical Q
projection remains head-wise `[query, gate]` interleaved. A new verifier-only
kernel therefore fuses the exact per-head deinterleave, Qwen `(1 + weight)` Q
normalization, and partial RoPE into contiguous query and gate device buffers.
It consumes the Q projection's borrowed output directly; no backend-specific
weight permutation or host copy is introduced. CUDA 12.6 reports 21 registers,
36 bytes shared memory, and zero stack/spill bytes. Its composite numerical
run remains queued, so this candidate is not yet promoted. The loader
independently checks all 130 frozen Qwen
fan-out groups (373 logical `s_in` tensors) when the checkpoint carries the
`qwen38_fanout_s_in_v1` contract. The host contract, Rust/Python group digest,
compile path, and negative identity test are validated; the exact Q/K/V-shaped
GPU verifier has not run yet because GPU 0 is reserved for Greppy and the BF16
teacher occupies GPU 1+2. No performance evidence is claimed for this new
dispatcher until that verifier runs on a released device.
Canonical `MixedQ2Q4B64` tensors use the same transient activation across all
manifest row groups. The host validates exact contiguous row/byte coverage,
uploads the original mixed payload once, offsets device pointers into each
homogeneous group, and synchronizes only after every Q2/Q4 segment has run.
The production-loader preparation API now accepts `RecoveredMatrixView`
directly. It validates packed FP16 `s_in`/`s_out`, pure payload length, or the
complete mixed row/byte partition before allocating device state, then copies
the immutable mapped weight and scale ranges straight to their long-lived
CUDA allocations. It never constructs a matrix-sized `Vec`, widens recovery
scales, repacks quantization codes, or allocates the legacy per-matrix f32
input buffer. Producer-owned device views supply activations during graph
execution. Unit fixtures cover exact pure Q2, exact mixed Q2/Q4, and rejection
of non-finite recovery data; complete-artifact residency remains a hardware
gate.

The evidence in
`benchmarks/cuda/sm86-a8-dp4a-20260826.json` separates two errors that must not
be conflated. The CUDA implementation differs from a CPU A8 oracle by at most
`2.63e-5` on the 5120x5120 cases, while A8 itself differs from the exact f32
activation oracle by as much as `0.04284` (Q2) and `0.03769` (Q4) for this
synthetic fixture. Five uncontrolled replicates reached median amortized rates
of 264.45 GB/s for Q2 and 390.98 GB/s for Q4 when one activation quantization
was reused for 20 launches of the same projection. With one quantization per
launch,
the medians were 139.19 GB/s and 236.44 GB/s. These remain application-level
packed-byte ratios under a concurrent teacher workload, not roofline claims.
The 50/50 mixed 5120x5120 verifier reached a median 284.30 GB/s over five
amortized replicates and 177.01 GB/s when one quantization served one complete
mixed projection. Its maximum CUDA-vs-A8-oracle error was `2.63e-5`.

The A8 path is therefore computationally validated but not promoted. Its
quality must be measured after recovery on full-model logits and the held-out
multilingual, general, coding, agentic, tool-calling, and long-context suite.

The loader-resolved embedding-row candidate now decodes one canonical Q2 or
Q4 row and fuses packed FP16 `s_in` plus scalar `s_out` on device. The
5120-column verifier matched the scalar recovered-row oracle exactly for Q2
and within `5.97e-8` for Q4. Five repeated-launch replicates had median kernel
intervals of 2.96 microseconds (Q2) and 3.02 microseconds (Q4). The verifier
copies output back for comparison; production graph wiring must instead keep
the activation device-resident. Full evidence is in
`benchmarks/cuda/sm86-recovered-row-20260826.json`.

## Runtime ownership and unload

Prepared CUDA graph objects now own a reference to a private, thread-affine
driver context instead of borrowing `CudaCandidateRuntime`. This permits one
model executor to own the runtime and every resident graph object without a
self-referential Rust structure. Each device buffer calls `cuMemFree` before
releasing its context owner; the last owner unloads the module and destroys
the context. No process-global CUDA allocator cache exists in this path.

GPU3 driver evidence recorded in
`benchmarks/cuda/sm86-owned-context-unload-20260826.json` observed 2 MiB of
driver allocation for a 32,160-byte prepared fixture and the exact same 2 MiB
returned immediately after `drop`, without process exit or cache trimming.
The final daemon must drive the context on one dedicated executor thread.
Complete-model high-watermark and an external unload measurement remain
promotion gates.

## Promotion evidence still required (per `docs/PROMOTION_GATES.md`)

1. Full-model A8 activation-quality gates after recovery, including the
   multilingual/domain-balanced held-out suite and end-to-end logits.
2. Controlled sustainable-bandwidth and typed-throughput roofline probes on
   the exact GPU3 device/profile.
3. Complete same-device verifier: every production fused op compared against the scalar/BF16
   oracle within recorded tolerances (packed round-trip, decoder block,
   end-to-end logits).
4. Same-device benchmark: pinned reference, identical artifact/prompt/
   context/batch/sampler/thermal state; no prefill or decode regression and
   at least one improved by >= 10%.

Until all four exist, `CudaBackend::promotion_state()` remains `Contract` and
`fused_matvec` returns `UnsupportedOperation`. The isolated
`CudaCandidateRuntime` cannot be selected by production dispatch.
