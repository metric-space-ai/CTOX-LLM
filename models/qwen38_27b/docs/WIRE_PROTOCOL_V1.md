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
low-level token-ID methods emit empty token text deliberately.

The transport calls the streaming service hook with a write-and-flush sink.
The default implementation emits an ordinary response vector, while a
Responses generation owner can override the hook and publish each token before
the following decode begins. This prevents a superficially streaming API that
buffers a complete generation in memory.

The Responses streaming override is bound to the pinned tokenizer and Qwen
template. It accepts plain text or ordered system/developer, user, assistant,
and tool history; text content arrays; OpenAI-shaped tool calls and tool
definitions; reasoning effort; thinking enablement; deterministic seed; and
optional verified MTP. Vision content fails closed because it belongs to the
separate vision package. The request ID is also the operation/session ID, so a
second connection cancels the stream using that value. Generation is currently
greedy, matching the exact MTP verifier contract. Incremental ByteLevel decode
buffers incomplete UTF-8 suffixes and emits each token exactly once; a
completion record follows `stop`, length, or cancellation. Normal completion
resets the session before the server accepts another generation.

`qwen38-server --artifact ...` remains the artifact-inspection bring-up owner
and returns `engine_not_ready`. An explicit `--verification-cpu` mode now
wires the complete signed-release loader, pinned tokenizer, Responses frontend,
engine lifecycle, and correctness decoder into the same Unix-socket service.
It requires the release root, manifest, CPU pack/profile identity, expected
signing-key ID, and a raw or lowercase-hex Ed25519 public-key file. The mode is
loaded with `ExecutionPolicy::Verifier`; the correctness executor's permanent
`Verifier` promotion state and hidden scalar token mixers make production
admission impossible. This supplies an end-to-end ABI integration target
without representing the unfinished optimized backends as ready.

Example verifier launch:

```sh
cargo run --manifest-path models/qwen38_27b/Cargo.toml \
  --bin qwen38-server -- \
  --socket /tmp/qwen38.sock \
  --verification-cpu \
  --release-root /models/qwen38-release \
  --release-manifest /models/qwen38-release/release.json \
  --pack-id cpu-avx2 \
  --memory-profile-id cpu-verifier-4k \
  --expected-key-id metric-space-release-v1 \
  --trusted-public-key /etc/ctox/model-release.pub
```
