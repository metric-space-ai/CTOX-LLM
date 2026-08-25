# Vendor policy

The CUDA directory contains the first pinned upstream reference set. Every
kernel import must add, in the same change:

- `UPSTREAM.json` (or an equivalent `SOURCE.version`) containing the repository
  URL, immutable commit, per-file purpose, and SHA-256 digests;
- the upstream license text;
- an unmodified upstream correctness/reference baseline;
- Rust dispatcher comments of the form `// ref: path:line-range`;
- verifier and benchmark evidence.

Reference-only sources with omitted framework includes must enumerate every
missing include and may not be represented as directly buildable. They do not
promote a backend without a compiled model-format kernel and same-device
evidence.

CI runs `training/verify_vendor_manifest.py` so a local source change without a
matching reviewed pin fails immediately.

Qualcomm SDK files never enter this directory.
