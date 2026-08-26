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

Progress events distinguish signature verification, tokenizer/template
verification, draft-vocabulary verification, artifact opening, artifact
admission, and backend loading.
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

An executor returns target logits unless it advertises and fulfills
`resident_target_selection`; omission then forces successful executor-owned
selection and may never fall back to the scalar sampler. When MTP is enabled,
it also returns one of two mutually exclusive candidate-block representations.
The correctness/evidence form
contains the draft distribution, the complete target verification distribution
at every candidate position, and the complete target bonus distribution. The
compact accelerator form contains device-selected draft tokens, matching
device-selected complete-target tokens, and the complete-target bonus token.
It is accepted only when the executor advertises
`compact_greedy_mtp_verification`; the engine still compares every causal pair
and stops at the first mismatch, so the backend cannot claim acceptance
itself. The logit draft distribution may be complete or a release-bound
restricted list of global token IDs and scores. Restricted IDs must be strictly
increasing, unique, in-vocabulary, and paired one-to-one with finite scores.
Target and bonus distributions always remain full-vocabulary; consequently an
omitted draft token causes rejection/fallback rather than changing target
semantics.
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

Sampling policy and RNG state are owned by this shared engine rather than by a
particular embedding or wire server. `prefill` constructs one sampler from the
explicit temperature, top-k, top-p, and seed values; every subsequent `decode`
advances that same state. An accelerator may implement `select_target_token`
over its most recent resident complete distribution. The engine supplies the
canonical PCG draw, validates the returned vocabulary ID, and uses the scalar
sampler only when the executor explicitly delegates with `None`. Native-library
and IPC callers therefore retain one seeded random stream without requiring a
full CUDA/Metal logit readback. The current MTP verifier is deliberately
restricted to temperature zero, where target/draft argmax equality is exact.
CUDA server execution uses resident target selection plus the compact form and
therefore avoids copying the target vocabulary, four target-verification
vocabularies, and four 40,000-row gathered draft distributions to the host. The
dedicated hardware verifier explicitly requests the evidence form so it can
still hash and compare those distributions. Non-greedy MTP fails closed until
a probability-correct rejection sampler is implemented.
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

The CUDA candidate now uses independently owned prepared objects backed by one
private, thread-affine driver context. Device buffers free themselves before
the last context owner unloads the module and destroys the context; there is
no process-global CUDA allocator cache. The model daemon must keep these
objects on a dedicated CUDA executor thread. A GPU3 fixture demonstrated exact
return of the complete observed driver allocation on `drop`; production
admission still requires the same evidence for the complete resident graph.

The CPU correctness executor composes the complete target and native MTP
graphs for sequential prefill/decode, including independent MTP KV state,
target-final-hidden handoff, chained MTP4 target verification, and tested
partial-prefix replay/commit transitions. It remains a scalar verifier rather
than a production executor. The CUDA SM86 candidate now implements the same
`ModelExecutor` lifecycle over its complete resident graph, including MTP4
checkpoint/restore and accepted-prefix replay with target state exactly one
token ahead of MTP state. The direct lifecycle verifier exists, but its
complete-model hardware result and promotion gates remain pending. Metal, CPU
SIMD token mixers, and Snapdragon still need optimized full graph
implementations before production admission.

CUDA driver ownership remains thread-affine. The sendable server adapter owns
the actual CUDA executor on one dedicated worker and serializes the same typed
ABI calls over in-process channels; it does not mark the context or its `Rc`
graph owners as `Send`. Cancellation tokens retain their shared atomic flag, so
a socket control request can cancel a running worker operation without moving
CUDA state. Worker shutdown performs reset and unload before joining.

`EngineServer<E>` now maps this exact lifecycle onto the v1 wire contract. It
does not own alternate sampling or model state: the server mutex owns one
`Engine<E>`, streams accepted MTP tokens before the target bonus/fallback,
allows cancellation from a separate connection, and reports unload residue
through the engine health contract. The Responses text renderer/detokenizer and
the pinned chat-template frontend are implemented. The server binary exposes a
fully signed CPU-verifier assembly for ABI tests. A `cuda`-feature build also
exposes an explicitly named `--verification-cuda` signed assembly using the
dedicated worker; verifier promotion state still prevents production admission.
Constructing a promoted CUDA or Metal executor remains open.
`EngineServer::load_signed` is the single production assembly boundary: it
verifies one manifest trust root and loads the selected backend pack, memory
profile, MTP vocabulary, model container, tokenizer, and chat template from
that same signed release. Callers cannot combine an authenticated engine with
frontend bytes from another installation.

[`WIRE_PROTOCOL_V1.md`](WIRE_PROTOCOL_V1.md) maps this lifecycle onto versioned
JSON Lines. It carries distinct request, operation, and session identities so
streaming and cancellation do not collapse into an ambiguous single RPC.
