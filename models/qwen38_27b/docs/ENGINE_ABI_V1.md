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

MTP output records proposed, target-verified, and accepted drafts separately.
The engine rejects a step unless every proposed draft was verified, and it
refuses any MTP output when the session disabled MTP. Context accounting
includes accepted drafts.

An executor error, cancellation after partial execution, malformed logits, or
an invalid MTP contract resets the entire session before another request is
allowed. This avoids continuing from partially advanced recurrent or KV state.

## Unload guarantee

The executor reports model, graph, session, scratch, and process-global cache
allocations independently. `unload` succeeds only when every counter is zero;
otherwise the engine enters `unload_failed` and exposes the residue through
health. This is the contract required by model-TTL and process-TTL owners.

The lifecycle implementation does not imply that a complete decoder executor
exists. CPU, CUDA, Metal, and Snapdragon still need full graph implementations
before any production load can pass their promotion gate.
