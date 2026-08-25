# Contributing

Kernel changes require all of the following evidence in the same pull request:

1. an immutable upstream/reference pin or an explicit original-work statement;
2. a scalar or BF16 correctness comparison;
3. a same-hardware benchmark with raw JSON output;
4. a promotion decision recording tolerances and regressions;
5. license and NOTICE updates for newly vendored material.

Do not add a general inference framework to obtain an end-to-end shortcut.
Production code must report unsupported operations rather than silently execute
them through the scalar verifier.
