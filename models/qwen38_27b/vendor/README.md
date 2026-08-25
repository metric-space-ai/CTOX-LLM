# Vendor policy

No source is vendored yet. A kernel import must add, in the same change:

- `SOURCE.version` containing repository URL and immutable commit;
- the upstream license text;
- an unmodified correctness baseline;
- Rust dispatcher comments of the form `// ref: path:line-range`;
- verifier and benchmark evidence.

Qualcomm SDK files never enter this directory.
