# Scalar operator oracle

`src/reference.rs` is the non-production numerical oracle for the model-local
graph and future CUDA, Metal, CPU, HTP, and Vulkan kernels. Production dispatch
is forbidden from selecting it.

The equations and committed golden values are bound to:

- model: `Qwen/Qwen3.8-27B`;
- model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- Transformers source commit:
  `a353632607c59463e6ced86a44c2de3c2cd62d5e`;
- source module: `src/transformers/models/qwen3_5/modeling_qwen3_5.py`;
- source license: Apache-2.0.

Current oracle coverage includes Qwen's `(1 + weight)` RMSNorm convention,
GatedDeltaNet's directly weighted gated RMSNorm, SwiGLU, 25-percent partial
RoPE, causal depthwise-convolution decode-state updates, and the exact
single-token recurrent gated-delta rule with Q/K L2 normalization. Causal
grouped-query attention and its post-attention sigmoid query gate are covered
as well. A sequential prefill oracle exercises the same recurrence expected
from future chunked kernels. Tests use golden values emitted by that pinned
Python implementation, including nonzero recurrent state so decay is exercised.

The Fold memory profile additionally has explicit FP16-resident convolution
and recurrent-state oracles. They widen arithmetic to FP32 and round every
persistent write to IEEE binary16, matching the intended accelerator storage
boundary. A deterministic 512-step recurrence/conv test compares that storage
mode with the FP32 oracle and rejects unbounded drift. This is a synthetic
implementation check only: promotion of the FP16 Fold profile still requires
captured model activations, full-sequence logits, held-out tokens, and
long-context state-stability evidence.

The mmap-backed correctness executor now composes quantized projections,
residual blocks, both token mixers, the target graph, and up to four drafts by
chaining the one native MTP layer. Partial acceptance replays the exact prefix
through both target and MTP state. Optimized chunked prefill, an optimized MTP
block verifier, and full-artifact BF16 golden logits remain required before any
production backend can be promoted.
