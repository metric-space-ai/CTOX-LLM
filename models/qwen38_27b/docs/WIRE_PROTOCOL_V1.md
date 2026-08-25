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
per session. Request IDs correlate transport calls; operation IDs correlate
long-running load/generation streams; session IDs bind recurrent and KV state.

One request may emit several response lines with the same request and operation
identity. Events include load progress, ordinary or MTP-accepted tokens, final
completion with timing metrics, acknowledgements, health, capabilities, and
typed errors. This allows another connection to deliver cancellation while a
generation stream remains active.

An unsupported protocol version is rejected before method dispatch. Invalid
JSON receives request ID zero because no trustworthy caller identity could be
decoded.

The current server binary implements negotiation and bring-up health/model
responses. All inference/control methods deliberately return
`engine_not_ready` until it owns a complete promoted `Engine<ModelExecutor>`.
The protocol contract being present is not evidence that generation works.
