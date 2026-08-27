# CPU packed Q2/Q4 kernel notes

Scope: `src/backend/cpu.rs` fused matvec for `TensorDType::Q2B64`,
`TensorDType::Q4B64`, and manifest-defined `MixedQ2Q4B64` row groups. The
mixed form dispatches only those same Q2 and Q4 kernels; there is no Q3 path.

## What changed

The previous SIMD path dequantized every 64-value block into a temporary
`[f32; 64]` on the stack and rebuilt an `s_in`-scaled `[f32; 64]` input copy
for every block of every row. The current implementation:

1. Applies `s_in` exactly once per operation, producing the corrected input
   `x[i] = input[i] * s_in[i]` (a single `Vec<f32>` per call, borrowed
   directly when `s_in` is absent). Per-element products are identical to the
   old code, so the scalar oracle is numerically unchanged. Production scales
   are read directly from their little-endian FP16 mmap payload and widened
   per value; no persistent f32 scale copy is created.
   The model graph now emits gate/up, full-attention Q/K/V, and all four
   linear-attention input projections through an explicit fan-out operation.
   When their input allocation and exact `s_in` representation match, the CPU
   backend builds this corrected vector once for the complete fan-out. An
   independent-recovery checkpoint with different scales keeps exact
   per-projection semantics instead of incorrectly sharing the correction.
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
4. Executes mixed embedding/LM-head payloads directly by their contiguous
   manifest row groups. Segment indices, row coverage, byte offsets, lengths,
   and Q2/Q4 dtypes are revalidated before execution. The path neither
   dequantizes nor repacks the complete tensor.
5. Resolves embedding lookup to exactly one Q2 or Q4 packed row, including a
   row inside a mixed tensor, and composes block scale, FP16 `s_in`, and the
   single selected FP16 `s_out` value while producing the 5,120-element hidden
   vector. It never materializes or scans the complete vocabulary matrix.
6. Hoists the fixed Q2/Q4 normalization divisor out of the inner SIMD loop.
   AVX2 and NEON compute one `scale / 3` or `scale / 15` factor per packed
   block; the hot loop then uses multiplication, and NEON feeds the centered
   code and scaled activation directly to FMA. This removes every vector
   division from packed decode without changing the logical Q2/Q4 codes or
   allocating another activation buffer.

## Packed decode schemes

Both SIMD families implement the logical per-element values
`scale * (2c - 3) / 3` (Q2) and `scale * (2c - 15) / 15` (Q4). They evaluate
these as `(scale / 3) * (2c - 3)` and `(scale / 15) * (2c - 15)` so that the
division happens once per block rather than in every SIMD group. This changes
only floating-point association; the verifier bounds the resulting difference
against the sequential scalar oracle.

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
`Q2Block64`/`Q4Block64` dequantization. Mixed-row tests compare one combined
payload against independent pure-Q2 and pure-Q4 dispatches and prove malformed
segment coverage fails closed. The fan-out test compares shared Q2/Q4 output
against independent dispatches, exercises the unequal-scale path, and rejects
an empty fan-out.
The mmap-to-kernel test also executes a recovered packed matrix and embedding
row through the CPU backend and checks the composed numerical result.

SIMD-vs-oracle differences come only from lane-wise reassociation, hoisting the
constant normalization factor, and NEON FMA contraction. For
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

The normalization-hoisting change was also measured directly against its
parent commit on the same Apple M5 host. Each median below is from eight
alternating old/new runs of 500 iterations, so process warm-up does not always
favor the candidate:

| dtype | parent median (ms) | hoisted median (ms) | kernel-time reduction | max abs err vs scalar |
|---|---:|---:|---:|---:|
| q2_b64 | 35.779 | 31.793 | 11.1% | 8.6e-6 |
| q4_b64 | 40.595 | 31.855 | 21.5% | 7.3e-6 |

The x86_64 Linux target cross-check also compiles the AVX2 implementation.
These isolated 512-by-512 measurements are kernel evidence, not the required
same-hardware full-model promotion benchmark.
