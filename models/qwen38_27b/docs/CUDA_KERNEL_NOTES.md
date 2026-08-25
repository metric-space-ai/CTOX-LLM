# CUDA SM86 Q2/Q4 kernel notes

Status: **Contract**. No production CUDA kernel is authored in this change and
no benchmark numbers are claimed. This note records the vendored baseline, the
ABI the future kernel must satisfy, and the evidence required to promote.

## Vendored baseline

Pinned llama.cpp revision `ef2d770117db45b05aa7ecd1b0acca36370c5470` (MIT,
(c) 2023-2026 The ggml authors). Files, SHA-256 digests, purposes, and the
required upstream includes deliberately not vendored are recorded in
`models/qwen38_27b/vendor/cuda/UPSTREAM.json`. Vendored files are unmodified
reference material; they are not compiled into the crate.

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

## Promotion evidence still required (per `docs/PROMOTION_GATES.md`)

1. A root build of the SM86 kernel module on GPU3.
2. Same-device verifier: each fused op compared against the scalar/BF16
   oracle within recorded tolerances (packed round-trip, decoder block,
   end-to-end logits).
3. Same-device benchmark: pinned reference, identical artifact/prompt/
   context/batch/sampler/thermal state; no prefill or decode regression and
   at least one improved by >= 10%.

Until all three exist, `CudaBackend::promotion_state()` remains `Contract`
and `fused_matvec` returns `UnsupportedOperation`.
