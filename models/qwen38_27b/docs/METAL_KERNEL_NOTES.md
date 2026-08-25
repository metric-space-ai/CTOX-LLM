# Metal Q2/Q4 fused matvec candidate notes

Status: **candidate, not promoted**. The backend reports
`PromotionState::Contract` and `fused_matvec` fails closed with
`EngineError::UnsupportedOperation` until same-device verifier and benchmark
evidence exists. No MLX/MPSGraph usage, no scalar fallback.

## Files

- Kernel source: `kernels/metal/q2q4_fused_matvec.metal`
- Rust ABI/dispatch contract: `src/backend/metal.rs`

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

## Validation evidence (this worktree, Apple Silicon)

- `xcrun -sdk macosx metal -c kernels/metal/q2q4_fused_matvec.metal -o target/fleet-metal.air` — compiles clean.
- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo test` on macOS — complete crate suite passes, including ABI constant checks against
  `src/quant.rs`, invalid shape/buffer rejection, dispatch-name checks, and an
  in-test `xcrun metal` compile of the kernel source.

Independent review corrected the generated candidate's positive Q2 code order
to `{+1/3, +1}` for codes `{2, 3}` and removed its per-thread 64-float
dequantization array before this source was accepted.

## Not yet done (promotion blockers)

- No same-device numeric comparison against the CPU oracle (no Metal runtime
  dispatch exists in-crate yet).
- No benchmark evidence against the pinned CPU reference.
- Per `docs/PROMOTION_GATES.md`, both are required before any state change;
  the backend therefore remains fail-closed.
