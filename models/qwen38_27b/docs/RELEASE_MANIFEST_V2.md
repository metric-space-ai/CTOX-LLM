# Qwen3.8 release-manifest v2 contract

`src/release.rs` is the admission boundary shared by the embedded Rust engine,
the local IPC server, CTOX, Greppy, and the Android owner. It describes one
canonical logical model and any number of deterministic physical backend
packs. It is not a model checkpoint and does not make the current incomplete
engine release-ready.

The binding `ctox.model-release.v2` identity includes:

- the BF16 repository, immutable revision, and verified root digest;
- the logical Q2/Q4 checkpoint, logical tensor root, fixed-code recovery, and
  resident MTP identity;
- tokenizer files, special-token IDs, chat template, reasoning format, and
  tool-call format;
- separately selectable text+MTP and vision packages;
- each backend artifact, its tensor-manifest digest, exact bytes, resumable
  chunks, hardware profile, and loader ownership policy;
- context/session-specific load, prefill, and decode memory formulas.

Q3 is unrepresentable in the release quantization enum. Every text pack must
carry MTP and reference the same logical checkpoint and tensor-root digests.
The validator rejects backend requantization, more than one resident full-model
copy, a retained full CPU copy, unsafe paths, chunk gaps, duplicate IDs,
unprofiled packs, or a calculated peak above the declared hard limit.

## Integrity and trust

The manifest stores a SHA-256 over its complete semantic body. An optional
`ed25519-sha256` envelope signs those 32 digest bytes. Production activation
must call `verify_signature` with a trusted public key selected outside the
downloaded manifest; an embedded public key from the same download would not
establish trust. Any changed tokenizer, template, model identity, package,
chunk, loader policy, or memory number invalidates the digest and signature.
After installation, `admit_artifact` also compares the opened CTOXQ file size,
the SHA-256 of its original embedded manifest bytes, model/revision, physical
target, trained-recovery digest, and fixed-qcode marker with the signed pack.
Signing one manifest therefore cannot authorize a different local pack.

## Memory interpretation

Each profile exposes exact components, not only artifact size:

- resident model and MTP bytes;
- persistent backend graph and runtime bytes;
- linear-attention state per session;
- KV fixed bytes, bytes per token, and Q4 sink/recent delta;
- prefill/decode scratch peaks;
- loader transient peak and unattributed accelerator reserve.

The Rust methods calculate steady, load, prefill, decode, and maximum peaks
with checked arithmetic. Admission happens before download, load, or session
creation. A future measured profile replaces estimates without changing the
schema; estimates may not be labeled as measurements.

## Backend equivalence

`verify_backend_pack_equivalence` validates the entire manifest and then proves
that two named packs reference the same logical checkpoint and tensor root.
Physical artifact hashes and sizes may differ because CUDA, Metal, CPU, and
Snapdragon use different alignments and tile order. Logical codes, recovery
corrections, tokenizer, template, and MTP may not differ.
