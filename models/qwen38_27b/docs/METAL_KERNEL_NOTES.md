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
- threadgroup(0): 8 f32 scratch slots for the cross-simdgroup reduction.
- Fused semantics identical to the CPU oracle:
  `y[r] = act(s_out[r] * (sum_c w[r,c] * x[c] * s_in[c] + bias[r]))`,
  activation 0 = identity, 1 = SiLU (`x / (1 + exp(-x))`).
- `s_in` and `s_out` stay byte-identical to the FP16 CTOXQ recovery tensors;
  kernels widen individual values in registers. No startup expansion to f32
  or duplicate scale allocation is permitted.

## Organization

One threadgroup per output row (grid = rows); threads stride over the row's
64-value blocks and decode directly into the dot product without a per-thread
dequantization array. Per-thread partials are reduced with `simd_sum` plus a
threadgroup scratch pass; lane 0 applies bias/s_out/activation. Rows and
columns are bounds-checked; trailing partial blocks contribute only their
valid prefix.

## Validation evidence (this worktree, Apple M5)

- `xcrun -sdk macosx metal -c kernels/metal/q2q4_fused_matvec.metal -o target/fleet-metal.air` — compiles clean.
- The `metal` Cargo feature links only the native `metal-rs` driver binding and
  compiles the in-crate MSL source with fast math disabled. It does not link an
  inference framework.
- `cargo test --features metal q2_and_q4_device_results_match_scalar_oracle`
  dispatches Q2 and Q4 on the 10-core Apple M5 GPU. Eleven 192-column rows use
  non-identity packed-FP16 `s_in`/`s_out`, bias, and SiLU, and pass the scalar
  CPU oracle tolerance for every output row.
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
- No sustained device-resident benchmark or roofline evidence exists yet.
- No full embedding, attention, GatedDeltaNet, MTP, sampling, or model-graph
  Metal execution exists yet.
- Per `docs/PROMOTION_GATES.md`, all promotion evidence is required before any state change;
  the backend therefore remains fail-closed.
