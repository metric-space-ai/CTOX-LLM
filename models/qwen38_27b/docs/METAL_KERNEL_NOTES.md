# Metal Q2/Q4 fused matvec candidate notes

Status: **candidate, not promoted**. The backend reports
`PromotionState::Contract` and its production `fused_matvec` entry point fails
closed with `EngineError::UnsupportedOperation`. A separately named
`MetalCandidateRuntime` now performs direct same-device verifier dispatch, but
it cannot be selected as a production `Backend` until resident-memory and
benchmark evidence exists. No MLX/MPSGraph usage, no scalar fallback.

## Files

- Kernel source: `kernels/metal/q2q4_fused_matvec.metal`
- Rust ABI/dispatch contract: `src/backend/metal.rs`
- Direct verifier runtime: `src/backend/metal_runtime.rs`

## Entry points

| Kernel | DType | Block layout |
|---|---|---|
| `q2_b64_fused_matvec` | Q2_B64 | 18 bytes: fp16-LE scale + 16 code bytes (4 x 2-bit values per byte) |
| `q4_b64_fused_matvec` | Q4_B64 | 34 bytes: fp16-LE scale + 32 code bytes (2 x 4-bit values per byte) |

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
- `qwen38-metal-bench` performs synchronous warmups and repeated dispatches on
  those resident buffers, reports the exact requested buffer bytes, and keeps
  its output marked `verifier_only_not_promotion_evidence`.
- The complete suite also covers ABI constants against `src/quant.rs`, invalid
  shape/buffer rejection, dispatch-name checks, and an in-test `xcrun metal`
  compilation of the source.

Independent review corrected the generated candidate's positive Q2 code order
to `{+1/3, +1}` for codes `{2, 3}` and removed its per-thread 64-float
dequantization array before this source was accepted.

## Not yet done (promotion blockers)

- The verifier currently creates shared staging buffers for an isolated
  operation. It is not evidence for resident model tensors or the required
  no-duplicate loader ownership contract.
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
- No full embedding, attention, GatedDeltaNet, MTP, sampling, or model-graph
  Metal execution exists yet.
- Per `docs/PROMOTION_GATES.md`, all promotion evidence is required before any state change;
  the backend therefore remains fail-closed.
