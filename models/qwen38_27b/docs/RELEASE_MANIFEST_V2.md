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
- a canonical restricted MTP draft vocabulary: exact u32-LE token-ID file,
  teacher-cache-set hash, observed-token count, and overall/code/minimum-domain/
  minimum-language coverage;
- tokenizer files, special-token IDs, chat template, reasoning format, and
  tool-call format;
- separately selectable text+MTP and vision packages;
- each backend artifact, its tensor-manifest digest, exact bytes, resumable
  chunks, hardware profile, and loader ownership policy;
- context/session-specific load, prefill, and decode memory formulas.

Q3 is unrepresentable in the release quantization enum. Every text pack must
carry MTP and reference the same logical checkpoint and tensor-root digests.
Restricted MTP rows are proposal-only: every proposed token is compared with
the full-vocabulary target distribution, so an uncovered token causes a normal
speculative rejection and cannot change greedy model semantics. Token IDs must
be strictly increasing at runtime and their file is signed as part of the
canonical model identity. `training/build_mtp_draft_vocab.py` constructs that
file only from the exact, fully rehashed teacher-cache recovery cohort and
fails closed on coding, domain, language, or overall coverage gaps.
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
- signed FP16/FP32 linear-attention state dtype and base bytes per session;
- MTP draft depth plus `disabled`, `replay_on_reject`, or `aligned_pages`
  target-state strategy and its additional bytes per session;
- separate target and MTP KV fixed bytes, bytes per token, and Q4 sink/recent
  deltas—the MTP cache may not be hidden inside an unattributed reserve;
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
