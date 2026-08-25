# RTX 3090 Qwen3.8 serving evidence

The user-supplied [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090)
repository was inspected at immutable revision
[`60daef8255b6757d9791955a44bce27df1658ea6`](https://github.com/syv-ai/qwen38-27b-rtx3090/commit/60daef8255b6757d9791955a44bce27df1658ea6).
The repository is Apache-2.0. It is valuable same-device evidence, but it is a
patched vLLM deployment rather than a standalone custom engine and does not by
itself prove that every operation reaches the RTX 3090 roofline.

## Findings that change CTOX-LLM

- Qwen3.8 has separate embedding and LM-head matrices. The upstream deployment
  recovers roughly 2.6 GB by quantizing both instead of retaining their public
  BF16 forms (`docs/optimizations.md`, item 1). CTOXQ already quantizes both and
  must preserve this property.
- FP16 GatedDelta recurrent state is a first-class memory/performance profile,
  not just an implementation detail (`docs/optimizations.md`, item 3). CTOX now
  models FP16 and FP32 state separately; FP16 still needs numerical promotion
  evidence.
- The single native MTP module is chained for several speculative tokens
  (`single-user/start_qwen.sh`, `DRAFT_TOKENS`, default 4). A one-layer
  checkpoint therefore does not imply a one-token production scheduler.
- Multi-query target verification benefits from a split-KV attention path
  because ordinary FlashAttention underutilizes SM86 for the short verify
  block (`docs/optimizations.md`, item 6).
- Speculative depth consumes additional recurrent-state pages. Keeping all
  aligned states is fast but conflicts with the Fold memory target. CTOX's Fold
  profile therefore admits an FP16 replay-on-reject checkpoint first; aligned
  pages remain a larger-memory CUDA profile.
- Hybrid prefix caching must resume both attention KV and recurrent state. It
  cannot cache only transformer KV and claim an equivalent second turn.
- KVarN-style K4/V2 long-context storage is credible evidence for 240K-class
  retrieval, but the published RTX 3090 measurements also show a substantial
  long-context decode penalty. CTOX retains Q2 plus a Q4 sink/recent tier as an
  explicit candidate until its own quality and speed gates decide the format.

## Non-transferable benchmark claims

The reported W4A8/Marlin, vLLM scheduler, FlashInfer/Triton, DFlash2, and RTX
3090 results are not CTOX Q2/Q4 measurements. They may define candidates and
regression scenarios, but they cannot promote a CTOX kernel. Every CTOX backend
still requires exact packed-code comparison, full-graph golden tests, measured
bytes moved, device bandwidth/compute ceilings, and same-hardware benchmarks.

## CUDA work derived from the evidence

1. Implement optimized chained MTP draft scheduling plus block target
   verification with rollback/replay semantics; retain the scalar MTP4 replay
   path as the oracle.
2. Add an SM86 split-KV verify-attention candidate with an immutable upstream
   anchor before promotion.
3. Make recurrent-state dtype and speculative state strategy signed memory
   profile fields rather than ambient runtime toggles.
4. Benchmark projection, recurrent update, attention, and whole-token traffic
   separately. No aggregate tokens/s result may hide an operation below its
   roofline gate.
