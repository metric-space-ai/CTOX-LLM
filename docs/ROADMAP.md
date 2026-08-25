# Qwen3.8-27B implementation roadmap

This roadmap is the execution contract for the first complete CTOX-LLM model.
It deliberately distinguishes measured evidence from estimates. A phase is not
complete because its types, stubs, or kernels compile; it is complete only when
the listed exit evidence exists and passes `PROMOTION_GATES.md`.

## Current baseline

The model and inference engine are **not finished**. The repository currently
proves the container, memory-planning, calibration, and early per-operation
paths needed to build them.

| Area | Current evidence | Missing before release |
|---|---|---|
| BF16 origin | `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Frozen tokenizer, template, special-token, and model-file digests in one release manifest |
| Native baseline | 8,342,484,480-byte CTOXQ file; 8,342,086,656 resident bytes; text plus MTP; SHA-256 `02d38cc877ad2ae8bea244bc11d4572ca0a8c84e757bdfbcb27ebbb9ed8c47f6` | Recovery scales are identity values; this is not the quality checkpoint |
| Candidate assignment | 154 Q4 and 352 Q2 matrices; exact planned layout 8,372,888,576 bytes (7.797860 GiB) | Pack it, train recovery, evaluate it, and prove the final logical digest |
| Recovery data | 80 Nemotron samples/130,994 capped tokens plus a materialized 80-sample German supplement | German contribution, real long-context samples, MTP/head coverage, teacher cache, recovery training, and release-quality evaluation |
| CPU | Scalar oracle plus experimental packed AVX2/NEON Q2/Q4 matvec | Complete graph operations, ISA profiles, end-to-end correctness, and production benchmark |
| CUDA | Pinned upstream reference sources and SM86 ABI contract | CTOX Q2_B64/Q4_B64 kernels, graph execution, verifier, and GPU3 benchmark |
| Metal | Compilable Q2/Q4 MSL candidate | Runtime dispatch, same-device numerical evidence, full graph, and benchmark |
| Snapdragon | QNN/Vulkan/AHardwareBuffer contract | Exact Fold SoC support, compiled HTP/Vulkan graph, shared-memory proof, and device measurements |
| Runtime | Loader, memory planner, wire types, and bring-up server | Stable `Engine` ABI, tokenizer, full decoder, MTP verification, streaming, cancellation, session reset, and complete unload |

The reported 9.5748-GiB baseline and 9.6035-GiB candidate 128K figures are
verified calculations, not measured RSS/PSS/VRAM peaks. Likewise, the planned
9.6976-GiB vision phase is not yet Android device evidence.

## Frozen release invariants

These constraints apply to every phase:

- CUDA, Metal, CPU, and Snapdragon consume the same logical Q2/Q4 codes,
  recovery corrections, tokenizer, chat template, special tokens, and MTP
  weights. Backend packs are deterministic physical reorderings only.
- No backend-dependent requantization or silent CPU fallback is allowed.
- Text plus resident MTP and recovery data must be no larger than 7.8 GiB.
- The Fold limit applies to the entire visible inference process, including
  code, Java/JNI, native heaps, graph state, workspaces, KV state, and
  accelerator allocations: 9.7 GiB operating target and 10 GiB hard refusal.
- The loader may never retain file mapping, full CPU copy, and full device copy
  at the same time.
- Vision remains a separate phase-resident package. Explicit package unmapping
  and remapping is allowed; correctness may not depend on zRAM or UFS swap.
- `unload` must free all model, session, graph, scratch, and device allocations.
  Process-global accelerator caches that make this impossible are forbidden.

## Phase 0: freeze model, artifact, and runtime contracts

**Work**

1. Introduce a v2 release manifest with a canonical `release_id`, BF16 source
   revision, logical-checkpoint digest, tokenizer/template metadata, MTP
   membership, file/chunk hashes, tensor offsets, alignment, endianness, and
   backend pack identity.
2. Specify the embeddable Rust `Engine` lifecycle: `load`, `warmup`, `prefill`,
   incremental `decode`, `cancel`, `reset_session`, `health`, `capabilities`,
   `unload`, and progress/allocator telemetry.
3. Version the same operations over Unix sockets and Windows named pipes for
   CTOX/Greppy process isolation. GGUF remains reference evidence, not the
   production container.
4. Make admission control consume exact artifact sizes and context/session
   formulas before download, load, and session creation.

**Exit evidence**

- Manifest fixtures round-trip in Rust and reject changed tensors or metadata.
- A backend-pack verifier proves equal logical tensor digests.
- ABI lifecycle tests include cancelled loads and repeated load/unload cycles.

## Phase 1: finish the recovery corpus and teacher evidence

**Work**

1. Run activation collection for the pinned German cohort and merge it with the
   existing Nemotron statistics by exact observed token counts.
2. Add release-eligible agentic/tool-use, structured-output, bilingual code,
   mathematics, and genuine long-context/RAG samples. Record repository,
   immutable revision, license, record hash, language, domain, and length.
3. Build 32K, 64K, and 128K calibration/evaluation examples that exercise
   retrieval positions rather than merely padding short prompts.
4. Capture BF16 top-64 logits, residual mass, selected hidden states, and the
   activation statistics required by the recovery losses. Cover embedding,
   LM head, and every resident MTP matrix explicitly.
5. Keep Nemotron v2 quarantined until its public derivative-use decision is
   documented. Track all GPU work in the 240-GPU-hour ledger.

**Exit evidence**

- Immutable provenance and cohort manifests with no unresolved release-license
  finding.
- Reproducible teacher-cache hashes and coverage report for every planned loss.
- Separate calibration, recovery-training, and held-out evaluation splits.

## Phase 2: train and freeze the final Q2/Q4 checkpoint

**Work**

1. Recompute activation-weighted Q2-vs-Q4 error with the complete corpus.
   Allocate Q4 where measured error reduction per byte is greatest; keep Q2
   elsewhere. Q3 remains absent.
2. Extend assignment granularity for embedding and LM-head storage to
   independently scored row/block groups. Frequently used or sensitive groups
   may use Q4 while the remainder stays Q2; neither tensor is forced to INT8 or
   promoted wholesale to Q4 without measured justification.
3. Keep quantization codes fixed and train `s_in`/`s_out`, compatible
   Hadamard/incoherence transforms, and recovery parameters using logit KL,
   cross-entropy, hidden-state, and activation reconstruction losses.
4. Run ablations for correction scales, transforms, layer allocation, LM head,
   embedding, and MTP. Reject improvements that only move the calibration set.
5. Fold inference-time corrections into scales and fused-kernel metadata, then
   emit one backend-neutral logical checkpoint.
6. Produce deterministic CPU, CUDA, Metal, and Snapdragon packs from that
   checkpoint without changing logical values.

**Exit evidence**

- Final text+MTP resident bytes at or below 8,375,186,227 bytes (7.8 GiB).
- Weighted benchmark score at least 95% of BF16; no primary category below 90%.
- Recovery closes at least 30% of the direct-Q2/Q4-to-BF16 quality gap.
- Agentic/tool-calling, German, code, and MTP gates pass on held-out samples.
- 128K retrieval reaches at least 90% of the BF16 reference.

## Phase 3: complete the model-local Rust reference engine

**Work**

1. Implement the tokenizer, canonical chat/tool/reasoning template, sampling,
   deterministic seeds, and token streaming.
2. Execute the complete 64-layer hybrid graph: embeddings, full attention,
   GatedDeltaNet/linear attention, FFN, normalization, RoPE, residual paths,
   LM head, and MTP draft/target verification.
3. Implement paged Q2 KV with Q4 sink/recent pages, exact linear-attention
   state, bounded prefill/decode arenas, cancellation, and one active session.
4. Make the scalar path the test oracle only. Production policy fails closed
   if any graph operation lacks a promoted backend implementation.
5. Implement deterministic ownership so `unload` returns allocator usage to
   the pre-load baseline.

**Exit evidence**

- Per-op, decoder-block, and end-to-end logit comparisons against BF16.
- Golden greedy tokens for agentic, tool, German, code, and long-context cases,
  with MTP enabled and disabled.
- Repeated cancel/reset/unload tests with zero residual model allocation.

## Phase 4: CUDA SM86 production backend on GPU3

**Work**

1. Adapt the pinned, licensed CUDA correctness baseline to CTOX Q2_B64/Q4_B64
   layouts while retaining source anchors and provenance.
2. Implement fused projection/FFN, normalization, full attention, linear
   attention, MTP, KV, and sampling-adjacent operations through the CUDA Driver
   API. Do not introduce a general inference framework runtime.
3. Tune an explicit SM86 profile for the RTX A4500 and record allocator,
   prefill, decode, cold-start, warm-start, and unload measurements.

**Exit evidence**

- Same-device scalar/BF16 verifier and full golden suite pass.
- No fallback is observed in operation traces.
- Prefill and decode do not regress versus the pinned reference, and at least
  one improves by 10% or more under identical conditions.

## Phase 5: production CPU and Metal backends

**CPU work**

- Extend the real packed path to the complete graph and add AVX-512/VNNI,
  DotProd, and I8MM profiles. Runtime dispatch selects only verified profiles.

**Metal work**

- Bind the MSL candidate through direct Rust/Metal APIs, then implement and
  tune the remaining model operations for Apple Silicon. Keep candidates
  isolated until numerical and performance evidence promotes them.

**Exit evidence**

- Each backend passes the same logical-digest, golden-token, lifecycle, and
  same-hardware promotion gates as CUDA.
- CUDA and Metal expose no backend-dependent model-quality delta beyond the
  documented floating-point accumulation tolerance.

## Phase 6: Snapdragon/Fold backend

**Host gate before buying the Fold**

1. Verify the exact Fold SoC, Android version, QNN/HTP target identifier,
   supported A8W2/A8W4 operations, context/graph limits, SDK access, and Adreno
   Vulkan capabilities from official documentation.
2. Convert the frozen logical checkpoint with a no-quality-change packer and
   demonstrate that every large operation has an HTP or Vulkan implementation.
3. Refuse the purchase recommendation if the official stack cannot execute the
   required Q2/Q4 graph or if the conservative complete-process budget fails.

**Device work after the host gate**

- Connect the phone to Linux over USB/ADB; compile, deploy, profile, and collect
  QNN/Vulkan traces directly on the device.
- Use HTP for large projections/FFN/embedding/MTP and Vulkan for paged
  long-context attention, KV compression, RoPE, and stateful operations.
- Share buffers through AHardwareBuffer/DMA-BUF and prove that transient copies
  are released before the next phase allocation.

**Exit evidence**

- Measured whole-process PSS/SwapPSS plus unattributed accelerator allocations
  are at most 9.7 GiB at 128K; admission refuses any plan above 10 GiB.
- Active weights and KV pages have zero dependency on swap.
- Thirty-minute warm decode sustains at least 5 token/s without process death
  or thermal collapse.
- The separately packed vision phase stays under the same measured limit.

## Phase 7: CTOX and Greppy integration

**Work**

1. Publish the model crate as an embeddable library with optional CPU, CUDA,
   Metal, and Snapdragon features. The local server is a thin owner of the same
   `Engine`, not a separate inference implementation.
2. Implement resumable post-install download, exact preflight sizing, atomic
   release activation, health/capabilities, load progress, streaming, cancel,
   reset, TTL-driven unload, and failure recovery over `LocalTransport`.
3. Keep text+MTP as the default download; vision is a separately selected
   artifact and never an implicit dependency.

**Exit evidence**

- CTOX and Greppy consume the same release manifest and protocol fixtures.
- Cold start, warm start, cancel, reset, daemon restart, and model-TTL unload
  pass on CUDA and Metal without leaked accelerator allocations.

## Phase 8: qualification and release

**Work and exit evidence**

- Reproduce all artifacts from pinned inputs and publish code, licenses,
  notices, provenance, manifest, hashes, benchmark commands, and result files.
- Publish large model packs in a dedicated Hugging Face model repository.
- Run the full quality, backend, memory, lifecycle, and thermal matrix.
- Mark a backend `optimized` only after all of its evidence is committed.

## Execution order and immediate work

The critical path is:

`complete corpus -> final sensitivity -> recovery training -> final logical checkpoint -> complete Rust graph -> CUDA reference backend -> Metal/CPU -> Snapdragon device backend -> integrations -> release`

Contract work in Phase 0 may proceed alongside the data pipeline, but neither
backend tuning nor Greppy integration may invent a model identity before the
logical checkpoint and manifest contract are frozen.

The immediate execution batch is:

1. collect German activation statistics;
2. merge German and Nemotron statistics;
3. add genuine long-context and explicit MTP/head calibration coverage;
4. recompute the byte-exact Q2/Q4 assignment;
5. begin the first bounded recovery-training ablation;
6. implement the v2 manifest and `Engine` lifecycle fixtures in parallel with
   the GPU runs.
