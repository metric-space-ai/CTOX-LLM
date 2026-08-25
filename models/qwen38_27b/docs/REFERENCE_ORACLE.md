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
single-token recurrent gated-delta rule with Q/K L2 normalization. Tests use
golden values emitted by that pinned Python implementation, including nonzero
recurrent state so decay is exercised.

Full grouped-query attention, chunked prefill delta recurrence, quantized
projection composition, residual block execution, MTP, and end-to-end logits
remain required before this becomes a complete decoder oracle.
