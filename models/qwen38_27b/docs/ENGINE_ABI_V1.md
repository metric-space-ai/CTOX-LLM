# Embeddable engine lifecycle v1

`src/engine.rs` is the model-local Rust ownership and lifecycle boundary. The
local Unix-socket/Windows-named-pipe server is intended to be a thin owner of
this same type; it must not grow a second inference implementation.

## Loading and admission

`Engine::load_signed` requires the release installation root and a trusted
Ed25519 key. It validates the complete release manifest, resolves both the
selected CTOXQ pack and restricted MTP vocabulary beneath that root, rehashes
the complete token-ID file and every declared chunk, opens the pack with all
tensor checksums, binds its original embedded-manifest digest to the selected
backend pack, and checks the selected memory profile before calling the
executor. A development entry point exists only for a release already
authenticated by a containing trusted bundle.

Production admission additionally requires:

- an `optimized` executor matching the signed backend and hardware profile;
- complete cancellation and session-reset support;
- an explicit guarantee of no hidden fallbacks;
- executor context capacity at least as large as the selected memory profile.

Progress events distinguish signature verification, draft-vocabulary
verification, artifact opening, artifact admission, and backend loading.
Cold-load and warmup times are reported separately.

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

An executor returns target logits and, when enabled, a candidate block, the
target verification distribution at every candidate position, and the target
bonus distribution after the complete block. It cannot claim acceptance
itself: sampling and causal-prefix verification belong to the engine. The
draft distribution may be complete or a release-bound restricted list of
global token IDs and scores. Restricted IDs must be strictly increasing,
unique, in-vocabulary, and paired one-to-one with finite scores. Target and
bonus distributions always remain full-vocabulary; consequently an omitted
draft token causes rejection/fallback rather than changing target semantics.
The signed release binds the exact ID file and multilingual/domain coverage.
The executor retains the corresponding speculative state branch until the
engine calls `commit_speculative(accepted_prefix_len)`. Rejection retains only
the already processed input and discards the branch; acceptance commits exactly
the accepted target prefix and advances the MTP cache through the same prefix.
An error or cancellation invalidates the complete session rather than leaving
an ambiguous partially committed state.

The current correctness executor chains the native MTP layer to the signed
admitted depth, up to four drafts. When the complete block is accepted, the
engine emits `accepted_draft_tokens` first and `token_id` as the target bonus;
on rejection, that list contains only the accepted causal prefix and `token_id`
is the target fallback at the rejected position. The resident context advances
by one input token plus the accepted prefix length. The full verification
execution span is admitted before dispatch, including target work for every
reserved draft. MTP output is rejected when the session disabled MTP or
exceeds the signed memory profile's draft depth.

Sampling is owned by this shared engine rather than by a particular embedding
or wire server. `prefill` constructs one sampler from the explicit
temperature, top-k, top-p, and seed values; every subsequent `decode` advances
that same state. Native-library and IPC callers therefore use the same seeded
random stream. The current MTP verifier is deliberately restricted to
temperature zero, where target/draft argmax equality is exact. Non-greedy MTP
fails closed until a probability-correct rejection sampler is implemented.
Production MTP will chain the native module for several drafts using the same
two-phase ABI; the one-layer checkpoint does not impose a one-token scheduler
limit.

An executor error, cancellation after partial execution, malformed logits, or
an invalid MTP contract resets the entire session before another request is
allowed. This avoids continuing from partially advanced recurrent or KV state.

The model-local gathered-row contract applies packed `s_in` once to the MTP
hidden vector, resolves each canonical token ID directly to its Q2/Q4 row, and
applies that row's packed `s_out`. It neither expands the full matrix nor keeps
a second restricted LM-head copy. CPU is the current correctness oracle;
accelerator backends fail closed until a fused gathered-row kernel passes their
own verifier and roofline gates.

## Unload guarantee

The executor reports model, graph, session, scratch, and process-global cache
allocations independently. `unload` succeeds only when every counter is zero;
otherwise the engine enters `unload_failed` and exposes the residue through
health. This is the contract required by model-TTL and process-TTL owners.

The CPU correctness executor composes the complete target and native MTP
graphs for sequential prefill/decode, including independent MTP KV state,
target-final-hidden handoff, chained MTP4 target verification, and tested
partial-prefix replay/commit transitions. It remains a scalar verifier rather
than a production executor. CUDA, Metal, CPU SIMD token mixers, and Snapdragon
still need optimized full graph implementations before production admission.

[`WIRE_PROTOCOL_V1.md`](WIRE_PROTOCOL_V1.md) maps this lifecycle onto versioned
JSON Lines. It carries distinct request, operation, and session identities so
streaming and cancellation do not collapse into an ambiguous single RPC.
