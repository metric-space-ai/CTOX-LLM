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

Layout divergence: upstream Q2_K/Q4_K use 256-value super-blocks with
sub-scales; this crate uses Q2_B64/Q4_B64 (64 values, one f16 scale, no
zero-point). Only Q2_B64 and Q4_B64 exist; there is no Q3. The vendored files
pin the dp4a/mma techniques and launch geometry, not the block format.

## SM86 kernel ABI (see `src/backend/cuda.rs`)

- Target compute capability 8.6, warp size 32, LP64 device pointers.
- Module must export at least:
  - `ctox_q2_b64_fused_matvec_sm86` (18-byte blocks: f16 scale + 16 code bytes)
  - `ctox_q4_b64_fused_matvec_sm86` (34-byte blocks: f16 scale + 32 code bytes)
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
```

The build script resolves the canonical `nvcc` target rather than trusting a
misdirecting symlink, emits a native `sm_86` cubin, records its SHA-256, and
captures `cuobjdump` resource usage. The Rust verifier dynamically loads only
the NVIDIA Driver API, requires compute capability 8.6, keeps the projection
buffers resident across launches, and compares device output with the scalar
CPU oracle before timing.

The first GPU3 run is recorded in
`benchmarks/cuda/sm86-q2q4-fused-matvec-20260826.json`. It proved both formats
numerically and measured a 5120x5120 projection. Q2 used 56 registers with no
spills and reached 110.82 packed-weight GB/s; Q4 used 44 registers with no
spills and reached 210.95 GB/s. These are useful baseline measurements, not
promotion evidence: another BF16 job occupied the other GPUs, clocks and
thermals were uncontrolled, allocator overhead and hardware counters were not
captured, and both kernels remain below the required practical roofline. The
near-identical Q2/Q4 latency shows that scalar unpack/FMA instruction
throughput, not weight bandwidth, is the current Q2 limiter. The next
candidate therefore needs the pinned upstream Q8-activation plus `dp4a`/MMQ
organization rather than more launch tuning.

## Promotion evidence still required (per `docs/PROMOTION_GATES.md`)

1. Controlled sustainable-bandwidth and typed-throughput roofline probes on
   the exact GPU3 device/profile.
2. Complete same-device verifier: every production fused op compared against the scalar/BF16
   oracle within recorded tolerances (packed round-trip, decoder block,
   end-to-end logits).
3. Same-device benchmark: pinned reference, identical artifact/prompt/
   context/batch/sampler/thermal state; no prefill or decode regression and
   at least one improved by >= 10%.

Until all three exist, `CudaBackend::promotion_state()` remains `Contract` and
`fused_matvec` returns `UnsupportedOperation`. The isolated
`CudaCandidateRuntime` cannot be selected by production dispatch.
