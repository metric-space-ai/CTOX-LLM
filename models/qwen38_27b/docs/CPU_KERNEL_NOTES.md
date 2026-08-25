# CPU packed Q2/Q4 kernel notes

Scope: `src/backend/cpu.rs` fused matvec for `TensorDType::Q2B64` and
`TensorDType::Q4B64`. Only these two dtypes exist; there is no Q3 path.

## What changed

The previous SIMD path dequantized every 64-value block into a temporary
`[f32; 64]` on the stack and rebuilt an `s_in`-scaled `[f32; 64]` input copy
for every block of every row. The current implementation:

1. Applies `s_in` exactly once per operation, producing the corrected input
   `x[i] = input[i] * s_in[i]` (a single `Vec<f32>` per call, borrowed
   directly when `s_in` is absent). Per-element products are identical to the
   old code, so the scalar oracle is numerically unchanged.
2. Reads block scales and code bytes straight from the packed weight slice
   and decodes them in SIMD registers. No heap allocation and no
   dequantization temporary exists inside the block loop.
3. Dispatches per profile through `packed_q2_dot` / `packed_q4_dot`:
   - `ScalarVerifier`: sequential f32 accumulation of
     `scale * Q2_CODEBOOK[code] * x` (Q2) and
     `scale * ((code - 7.5) / 7.5) * x` (Q4), byte-for-byte the same
     arithmetic as the old oracle (`Q2Block64::dequantize` + `scalar_dot`).
   - `Avx2` (x86_64, runtime-detected) and `Neon` (aarch64): direct packed
     kernels.

## Packed decode schemes

Both SIMD families dequantize per element as
`scale * (2c - 3) / 3` (Q2) and `scale * (2c - 15) / 15` (Q4). These are
bit-identical to the scalar forms `scale * Q2_CODEBOOK[c]` and
`scale * (c - 7.5) / 7.5`: numerators and denominators are exactly
representable and both sides are the correctly rounded result of the same
real quotient. Only the accumulation order differs from the oracle.

- AVX2 Q2: broadcast each little-endian u32 code word (16 codes at bits `2l`)
  and extract eight codes per vector with `_mm256_srlv_epi32`.
- AVX2 Q4: even codes are low nibbles, odd codes high nibbles; inputs are
  deinterleaved with `shuffle_ps` + `permutevar8x32_ps` lane fix, codes
  widened with `cvtepu8_epi32`.
- NEON Q2: same word broadcast scheme with per-lane negative shifts
  (`vshlq_u32`).
- NEON Q4: nibble split with `vandq_u8`/`vshrq_n_u8`, widening via `vmovl`,
  input pairs deinterleaved by `vld2q_f32`; accumulation uses `vfmaq_f32`.

Non-finite block scales are rejected with `InvalidArtifact` before decoding,
matching the previous `decode` behavior. `bias`, `s_out`, and the activation
remain fused per row after the packed accumulation. The scalar oracle is
never selected under `ExecutionPolicy::Production` on unsupported hardware;
`detect` still fails closed.

## Correctness evidence and tolerance

Tests in `src/backend/cpu.rs` compare the detected SIMD profile against the
scalar oracle deterministically: Q2 and Q4, single- and multi-block rows,
multiple rows, non-identity `s_in`/`s_out`, bias, Identity and SiLU
activations, plus arbitrary-code packed decode checks against
`Q2Block64`/`Q4Block64` dequantization.

SIMD-vs-oracle differences come only from lane-wise reassociation and NEON
FMA contraction (per-element products are bit-identical, see above). For
O(1) terms over at most a few hundred products per row this stays far below
the documented absolute tolerance of 2e-4; the `qwen38-kernel-bench`
verification gate keeps its tighter 1e-4 maximum-absolute-error check.

## Measured benchmark evidence

Independently re-measured after patch review on 2026-08-25 with
`cargo build --release`, `qwen38-kernel-bench`, fused
non-identity `s_in`/`s_out`, bias, and SiLU:

| host/profile | dtype | rows x columns | iterations | scalar (ms) | SIMD (ms) | speedup | max abs err |
|---|---|---|---|---|---|---|---|
| Apple M5 / NEON | q2_b64 | 512 x 512 | 10 | 1.003 | 0.359 | 2.79x | 7.6e-6 |
| Apple M5 / NEON | q4_b64 | 512 x 512 | 10 | 2.631 | 1.028 | 2.56x | 8.1e-6 |
| Intel i5-13400F / AVX2 | q2_b64 | 512 x 512 | 50 | 9.800 | 2.552 | 3.84x | 6.7e-6 |
| Intel i5-13400F / AVX2 | q4_b64 | 512 x 512 | 50 | 15.925 | 2.719 | 5.86x | 6.8e-6 |

The full test suite and arbitrary-code decoder tests passed on both hosts.
Per `docs/PROMOTION_GATES.md`, the CPU SIMD profiles remain `Experimental`:
promotion to `optimized` still requires a pinned same-hardware reference run
plus the model/Fold quality gates.
