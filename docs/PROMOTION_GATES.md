# Backend promotion gates

A backend moves through `unavailable`, `verifier`, `experimental`, and
`optimized`. These states are reported by the binary and are not marketing
labels.

## Correctness

- Packed Q2/Q4 blocks round-trip deterministically.
- Each fused operation is compared with the scalar/BF16 oracle.
- A decoder block and end-to-end logits meet recorded absolute and relative
  tolerances.
- Unsupported operations fail closed. Production policy never permits the
  scalar oracle.

## Performance

`optimized` requires a pinned reference on the same hardware, model artifact,
prompt, context, batch, sampler, and thermal state. The candidate must not lose
either prefill or decode throughput and must improve at least one by 10 percent.
Peak-only results are insufficient for mobile: the Fold profile uses a
30-minute warm run.

Beating the pinned reference is necessary but no longer sufficient. Every
production-reachable phase and shape sweep must pass `ROOFLINE_GATES.md`: at
least 85 percent of the measured sustainable bandwidth/compute/dispatch
roofline, complete traffic accounting, and no unexplained result above 105
percent of the calculated ceiling. A backend remains `experimental` while a
material roofline gap is unmeasured or unexplained.

## Model and Fold gates

- weighted quality >= 95% of BF16;
- no primary category below 90% of BF16;
- recovery closes >= 30% of the direct-quantization gap;
- agentic tool-error recovery immediately converts an actionable diagnostic
  into the exact corrected call (for example a qualified symbol or the
  repository's actual filename) without an unrelated search branch;
- 128K retrieval >= 90% of BF16;
- Fold text+MTP pack <= 7.8 GiB;
- Fold visible-process target <= 9.7 GiB, hard ceiling 10 GiB;
- sustained Fold decode >= 5 token/s.
