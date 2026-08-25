# Hardware roofline gates

CTOX-LLM does not call a backend `optimized` merely because it beats another
runtime. Every promoted hardware profile must also explain its distance from
the fastest physically sustainable execution on that exact device.

The practical ceiling is measured, not copied from a vendor data sheet:

- sustainable device-memory bandwidth with the same access direction,
  allocation type, thermals, and working-set size as inference;
- sustainable compute throughput with the actual accumulator and input types;
- dispatch/launch latency measured through the production driver path.

Vendor theoretical bandwidth and compute remain in the report for context.
They are not substituted for the sustainable ceiling because protocol,
refresh, cache policy, instruction mix, and thermal limits make 100 percent of
the data-sheet number physically unavailable to application code.

## Exact accounting

Each timed interval records accepted target tokens, elapsed time, bytes read,
bytes written, FLOPs, and dispatch floor. Bytes come from hardware counters
where the platform exposes trustworthy counters. Otherwise they come from the
exact tensor schedule and include weights, activations, KV pages, recurrent
state, scale metadata, intermediate spills, and host/device transfers. MTP
drafts count as throughput only after target verification accepts them.

The executable `qwen38-roofline-gate` computes:

```text
memory floor   = total bytes / sustainable bandwidth
compute floor  = total FLOPs / sustainable compute throughput
practical floor = max(memory floor, compute floor, dispatch floor)
efficiency      = practical floor / measured elapsed time
```

An `optimized` candidate must reach at least 85 percent of this practical
roofline. A computed efficiency above 105 percent is not celebrated: it proves
that byte, FLOP, dispatch, or ceiling accounting is incomplete and fails the
gate. The existing same-hardware reference and numerical gates remain
additional requirements.

## Required sweep

Promotion evidence covers all production-reachable residues and not only a
favorable matrix shape:

- batch-1 decode, MTP off and on;
- prefill at 512, 4K, 32K, and the backend maximum context;
- MTP verification for every supported draft length and acceptance bucket;
- KV attention at short, medium, 128K, sink/recent-page boundaries, and page
  residues;
- GatedDeltaNet state update and recurrent-state restore;
- production batch sizes and matrix row/column residues;
- cold, warm, and sustained thermal operation;
- load, cancellation, reset, and unload traffic outside the steady-state
  interval.

CUDA, Metal, CPU, HTP, and Vulkan keep separate ceiling records. Snapdragon
promotion also accounts for NPU/GPU interconnect and AHardwareBuffer ownership;
moving bytes between accelerators cannot disappear from the model.

## Optimization implications

The public RTX-3090 Qwen3.8 work demonstrates several model-specific levers
that transfer to our independent engine: quantized embeddings, reduced
GatedDeltaNet state precision, quantized MTP, verify-specialized attention,
small-set sampling, residue sweeps, and exact context drafting. We use those as
design evidence, not as a vLLM/Triton runtime dependency and not as permission
to copy dependency code without its own license record.

References:

- <https://github.com/syv-ai/qwen38-27b-rtx3090>
- <https://github.com/syv-ai/qwen38-27b-rtx3090/blob/main/docs/optimizations.md>
- <https://github.com/syv-ai/qwen38-27b-rtx3090/blob/main/docs/gotchas.md>
