# Local engine wire protocol v1

The local owner uses UTF-8 JSON Lines over Unix sockets and, on Windows, named
pipes. TCP loopback is not a new managed-inference transport. Every line is one
`WireRequest` or `WireResponse` and carries `protocol_version` plus a caller
chosen `request_id`.

Version 1 declares these requests:

- protocol negotiation, health, model listing, and capabilities;
- load by signed `release_id`, backend `pack_id`, and `memory_profile_id`;
- warmup, token-ID prefill, and incremental decode;
- cancellation by `operation_id` and reset by `session_id`;
- unload;
- the higher-level Responses-shaped generation entry point.

Prefill fixes sampling parameters and seed for the session. Decode retains that
state instead of accepting silent per-token sampling changes. MTP is selected
per session. Sampling is executed by the shared Rust engine, not duplicated in
the server, so embedded and IPC integrations advance identical seeded sampler
state. Request IDs correlate transport calls; operation IDs correlate
long-running load/generation streams; session IDs bind recurrent and KV state.

One request may emit several response lines with the same request and operation
identity. Events include load progress, ordinary or MTP-accepted tokens, final
completion with timing metrics, acknowledgements, health, capabilities, and
typed errors. This allows another connection to deliver cancellation while a
generation stream remains active.

An unsupported protocol version is rejected before method dispatch. Invalid
JSON receives request ID zero because no trustworthy caller identity could be
decoded.

`EngineServer<E>` is the reusable adapter around any admitted
`Engine<ModelExecutor>`. It implements health, models, capabilities, identity
checked load acknowledgement, warmup, token-ID prefill/decode, ordered MTP
prefix streaming, cancellation, session reset, and unload. Active operations
own distinct cancellation tokens, and the Unix listener handles connections
concurrently so a cancel request does not wait behind the inference call. The
adapter emits empty token text deliberately; detokenization belongs to the
still-open Responses frontend binding.

The current `qwen38-server` binary remains the artifact-inspection bring-up
owner and returns `engine_not_ready`: it is not wired to `EngineServer` until a
complete backend passes promotion and a signed release exists. Thus the wire
adapter is implemented, but its presence is not evidence that a production
executor is ready.
