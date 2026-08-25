# Embeddable engine lifecycle v1

`src/engine.rs` is the model-local Rust ownership and lifecycle boundary. The
local Unix-socket/Windows-named-pipe server is intended to be a thin owner of
this same type; it must not grow a second inference implementation.

## Loading and admission

`Engine::load_signed` requires a trusted Ed25519 key, validates the complete
release manifest, opens the selected CTOXQ pack with all tensor checksums,
binds its original embedded-manifest digest to the selected backend pack, and
checks the selected memory profile before calling the executor. A development
entry point exists only for a release already authenticated by a containing
trusted bundle.

Production admission additionally requires:

- an `optimized` executor matching the signed backend and hardware profile;
- complete cancellation and session-reset support;
- an explicit guarantee of no hidden fallbacks;
- executor context capacity at least as large as the selected memory profile.

Progress events distinguish signature verification, artifact opening,
artifact admission, and backend loading. Cold-load and warmup times are
reported separately.

## Session contract

The current acceptance profile owns exactly one active session. The stable
operations are:

- `warmup`;
- `prefill` over token IDs;
- incremental `decode`;
- cooperative cancellation through a thread-safe token;
- `reset_session`;
- `health`, `capabilities`, and timing metrics;
- explicit `unload`.

An executor returns target logits and, when enabled, unverified MTP draft
logits. It cannot claim acceptance itself: sampling belongs to the engine.
Qwen3.8 has one native MTP layer, so v1 admits at most one draft distribution
per decode. The engine verifies its argmax against the target-selected token,
then reports proposed, verified, and accepted counts. The accepted draft is
the one returned token, not an additional context transition, and is never
counted twice. MTP output is rejected when the session disabled MTP.

Sampling is owned by this shared engine rather than by a particular embedding
or wire server. `prefill` constructs one sampler from the explicit
temperature, top-k, top-p, and seed values; every subsequent `decode` advances
that same state. Native-library and IPC callers therefore use the same seeded
random stream. The current MTP verifier is deliberately restricted to
temperature zero, where target/draft argmax equality is exact. Non-greedy MTP
fails closed until a probability-correct rejection sampler is implemented.

An executor error, cancellation after partial execution, malformed logits, or
an invalid MTP contract resets the entire session before another request is
allowed. This avoids continuing from partially advanced recurrent or KV state.

## Unload guarantee

The executor reports model, graph, session, scratch, and process-global cache
allocations independently. `unload` succeeds only when every counter is zero;
otherwise the engine enters `unload_failed` and exposes the residue through
health. This is the contract required by model-TTL and process-TTL owners.

The CPU correctness executor now composes the complete target and native MTP
graphs for sequential prefill/decode, including independent MTP KV state and
target-final-hidden handoff. It remains a scalar verifier rather than a
production executor. CUDA, Metal, CPU SIMD token mixers, and Snapdragon still
need optimized full graph implementations before production admission.

[`WIRE_PROTOCOL_V1.md`](WIRE_PROTOCOL_V1.md) maps this lifecycle onto versioned
JSON Lines. It carries distinct request, operation, and session identities so
streaming and cancellation do not collapse into an ambiguous single RPC.
