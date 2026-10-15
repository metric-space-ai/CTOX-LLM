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
| BF16 origin | `Qwen/Qwen3.8-27B` revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; 23 files/55,575,959,504 bytes verified under root SHA-256 `c63bc259cd8d18b0a701983a226867927b4ec19b376a684b218c3fa572754524` | Bind tokenizer, template, special-token, model-file, logical-checkpoint, and backend-pack digests in the final release manifest |
| Native baseline | 8,342,484,480-byte CTOXQ file; 8,342,086,656 resident bytes; text plus MTP; SHA-256 `02d38cc877ad2ae8bea244bc11d4572ca0a8c84e757bdfbcb27ebbb9ed8c47f6` | Recovery scales are identity values; this is not the quality checkpoint |
| Release calibration initializer | 256 final-corpus samples/917,704 observed tokens across all 36 domains, 15 language strata, and 14 service modes; all 506 matrices covered; 123 Q4, 381 Q2, and two mixed matrices; fixed-code channel scales reduce the activation-weighted error by 48.27%; 8,373,052,416 resident bytes; fully checksummed 8,373,658,112-byte CTOXQ pack | Complete teacher cache, end-to-end KL/CE/hidden/MTP recovery, held-out evaluation, and final logical digest |
| Recovery data | The existing 2,328 training/642 held-out cohort is complete pipeline and regression evidence only. The release policy now requires at least 1,000,000 training, 50,000 calibration, and 50,000 held-out records with exact domain, language, context, provenance, license, payload, and semantic-cluster gates. The scalable evidence builder, admission audit, ordered token sidecar, and million-bound recovery-plan contract are implemented. | Materialize and semantically deduplicate the million-scale partitions, run the 10,000-sample GPU1+2 throughput probe, complete BF16 teacher caches, then train and evaluate recovery. |
| CPU | Scalar Qwen oracle, mmap-bound recovered target+MTP decoder correctness graph, engine-owned greedy MTP4 verification with partial-prefix replay/commit, release-bound restricted LM-head row execution, and experimental packed AVX2/NEON Q2/Q4/mixed projections plus embedding gather | Production token-mixer kernels, chunked prefill, ISA expansion, full-artifact golden run, and roofline/reference benchmark |
| CUDA | Pinned llama.cpp, TensorRT-LLM, and syv-ai split-KV reference sources, SM86 ABI, direct Driver API runtime, resident pure/mixed embedding plus a 505-projection/262-activation mmap-backed target+MTP graph, all 48 linear token-mixer parameter/state groups, context-sized Q2/Q4 KV for 16 target plus one MTP full-attention state, all 134 target/MTP norm operators, fused residual/RMSNorm, SwiGLU/A8/down and attention-gate/A8/output paths, exact prepared-resource bindings for all 645 frozen decode and 645 layer-major prefill steps, compiled target-token, one-layer MTP draft, and target-verification device chains, one transactional commit barrier per complete target/MTP step, bounded FP16/retained-KV checkpoint/restore, chained greedy MTP4 partial-prefix restore/replay in `CudaModelExecutor`, a Q2/Q4-aware gathered 40,000-row draft head, finite-checking deterministic device argmax, resident ordinary-target selection through a bounded top-k/top-p candidate with canonical engine draws, an isolated five-query/16-segment mixed-Q2/Q4 split-KV verifier candidate, a verified upstream-derived SM86 Q2/Q4 MMQ tile, single-copy batched projection and shared RMSNorm workspaces, exact sequential-equivalent causal-convolution, fully prepared GatedDelta, direct mixed-Q2/Q4 paged-GQA scans, a verified one-launch-per-page persistent batched KV packer, shared-table batched partial RoPE, batched Query/Gate+Q-RMSNorm+RoPE fusion, a bit-exact mixed-Q2/Q4 batched embedding gather, a graph-owned 73,533,440-byte embedding/frontend/full-attention pool shared across every 512-token target/MTP chunk, an allocated 82,968,576-byte four-slot projection arena covering all 504 chunk-wide matrices while keeping LM-head last-row-only with bit-exact mixed-Q2/Q4 A4500 offset-view evidence, and an allocated 84,082,688-byte causal-convolution/GatedDelta/gated-RMSNorm pool shared across all 48 linear layers (240,584,704 total graph-owned chunk workspaces), a frozen 645-step layer-major chunked-prefill schedule with explicit key-RoPE/KV-append state mutations, a fail-closed cursor that admits only the exact committed start and returns a new commit position only after all 645 ordered steps plus the final barrier, and a dedicated driver-owning server thread | Pass the queued target/MTP, gathered-head/argmax, stochastic-sampling primitive/lifecycle, split-KV numerical/latency, barrier-count, bit-exact replay, threaded lifecycle, and 128K hardware runs; hardware-verify the batched gated-RMSNorm/complete linear pool, integrate all verified chunk operators into the executor, add device RNG, probability-correct stochastic MTP, and unrestricted top-p; replace sequential prefill with the chunk schedule; run the controlled roofline sweep and full-model golden/unload run |
| Metal | Direct MSL compilation/dispatch, no-copy mmap artifact ownership, recovered embedding/projection/token-mixer candidates, Q2/Q4 same-device oracle comparison, shape-local simdgroup tuning, a frozen 645-step decode schedule, exact binding of all 505 projection/262 activation resources, one allocated 1,173,760-byte alias-safe shared decode-activation arena plus a separate 180,224-byte liveness-packed MTP scratch arena, real buffer/offset views for every read/write of all 645 steps, a native target-argmax/dynamic-embedding/pre-FC-norm/concat/`mtp.fc`/input-norm frontend followed by the complete one-layer Q/K/V/RoPE/paged-GQA/gated-output/residual/SwiGLU MTP transformer with append rollback, an offset-view canonical restricted Q2/Q4 draft head, device-side restricted argmax plus local-row-to-global-token mapping, a causally aligned initial MTP/target pair that advances both graphs from the same real token, an accepted-only one-command continuation with device-side compact acceptance/status preserving target-one-ahead state without a host token-to-embedding handoff, a fixed 64-byte four-record verifier history, device-side causal-prefix/target-fallback/true post-draft bonus-token reduction, a three-record queued MTP4 tail with partial-prefix restore and exact ordinary-verifier replay, a fused from-token four-record branch whose fifth transition consumes the fourth accepted draft without a separate completion wait, immutable causal RoPE/GQA plans backed by a resident five-slot metadata pool, single-use device-only target-hidden plus FP16 convolution/GatedDelta checkpoints, bounded append-only paged-KV rollback without arena duplication, one fail-closed atomic transaction across the final target hidden plus all 17 attention and 48 paired linear-state owners, ordinary/continuation/target-only complete-token entry points, a five-transition block cursor that publishes input plus four accepted drafts only after the shared GPU barrier, an embeddable verifier `MetalModelExecutor` owning load, serial causally shifted prefill, resident greedy selection, MTP4/variable-depth decode, reset, allocation reporting, and complete owned-resource unload, plus a safe dedicated-thread adapter and signed-profile `qwen38-server --verification-metal` Unix-socket path | Add the chunked prefill arena and stochastic resident sampler; hardware-prove fused MTP4 and hidden-state rollback on the full artifact, remove the verifier CPU KV mirror, prove allocator high-watermark/unload behavior, capture stable controlled roofline/reference evidence and a full-model golden run, then promote |
| Snapdragon | QNN/Vulkan/AHardwareBuffer contract | Exact Fold SoC support, compiled HTP/Vulkan graph, shared-memory proof, and device measurements |
| Runtime | Checksummed v1/v2 container, corrected memory planner, graph ownership plan, stable lifecycle/wire contracts, pinned pure-Rust tokenizer plus text/reasoning/tool template, signed/rehashed multilingual MTP draft-vocabulary contract, two-phase target+MTP verification/commit composition, fail-closed greedy MTP verification, concurrent/cancellable token-ID `EngineServer`, unbuffered multilingual Responses/tool frontend, CPU correctness executor, and threaded CUDA verifier executor reachable through the same server ABI | Promoted production executor binding, probability-correct non-greedy MTP, optimized streaming prefill/decode, and measured full-artifact unload evidence |

The CUDA full-artifact verifier now also records the driver-visible free-memory
baseline, sampled executor residency peak, reclaimed bytes, and exact
post-`unload` drift for both evidence and compact server modes. These fields are
plumbing evidence only until they have been measured on the final checkpoint.
The Metal executor now also binds a bounded resident top-k/top-p kernel to the
same LM-head output as greedy selection. Its explicit engine draw is
decision-equivalent to the canonical Rust sampler on Apple Silicon; on-device
RNG state and probability-correct stochastic MTP remain separate release gates.

The corrected 9.5822-GiB baseline and 9.6110-GiB initializer 128K figures are
verified no-MTP calculations, not measured RSS/PSS/VRAM peaks. The v2 correction adds
the previously omitted 7.5-MiB causal-convolution state to the 144-MiB
GatedDeltaNet recurrent state and is recorded in
`models/qwen38_27b/docs/MEMORY_PLAN_CORRECTION_V2.json`. Active MTP additionally
requires a 72.1875-MiB independent KV cache and speculative target-state
storage. The admitted calculated MTP4 profile uses FP16 recurrent state plus
one replay checkpoint. At the 7.8-GiB weight ceiling it totals 9.68562 GiB,
leaving only 14.73 MiB to the 9.7-GiB target after paged-KV metadata, boundary
retention, and requantization scratch are counted; aligned MTP4 state pages do
not fit. The MTP correction is recorded in
`models/qwen38_27b/docs/MEMORY_PLAN_CORRECTION_V3.json` and the paged-KV
correction in `models/qwen38_27b/docs/MEMORY_PLAN_CORRECTION_V4.json`. Likewise,
the planned 9.6976-GiB vision phase is not yet Android device evidence.

The final training cache set now covers all 2,328 admitted identities and is
content-bound to the BF16 teacher revision. Activation collection is likewise
complete at 2,328 samples and 13,971,665 observed sequence tokens across all
506 recoverable modules. The held-out 642-sample cache is the active GPU1+2
stage; it remains excluded from quality claims until all eight batches and the
cache-set manifest are terminal and rehashed. No smoke cache or failed/OOM
directory counts toward these totals.

## Frozen release invariants

These constraints apply to every phase:

- CUDA, Metal, CPU, and Snapdragon consume the same logical Q2/Q4 codes,
  recovery corrections, tokenizer, chat template, special tokens, and MTP
  weights. Backend packs are deterministic physical reorderings only.
- No backend-dependent requantization or silent CPU fallback is allowed.
- The complete text+MTP package must be no larger than 7.8 GiB. Its resident
  tensor plan reserves 2 MiB for the manifest and release metadata; fitting
  tensor bytes while the final file exceeds the ceiling is a hard failure.
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

## Phase 1: freeze the million-sample recovery corpus and finish teacher evidence

**Status:** the 2,328/642 pilot is preserved as regression evidence. It cannot
admit the final release. `training/MILLION_RECOVERY_POLICY.json`,
`build_million_corpus_evidence.py`, and `audit_million_corpus.py` now define the
release boundary for 1,000,000/50,000/50,000 disjoint records. No million-scale
teacher-cache or recovery-quality claim exists yet.

**Work**

1. Preserve the 2,328/642 materialized identities as regression inputs only;
   neither their teacher cache nor their scales may be relabeled as final.
2. Materialize at least 1,000,000 recovery-training, 50,000 calibration, and
   50,000 held-out identities under the frozen mix and source-license policy.
3. Produce ordered domain, service-mode, token-count, provenance, and semantic
   cluster sidecars. The evidence builder rejects changed order and the final
   audit requires zero within- or cross-partition duplicates.
4. Run a 10,000-sample throughput probe on physical GPU1+2 with at most 14 GiB
   weight placement per A4500 before admitting the full teacher-cache schedule.
   GPU0 remains reserved for Greppy.
5. Capture BF16 top-64 logits, residual mass, selected hidden states, and MTP
   targets under the same settings as the existing verified batches.
6. Assemble every passed partition into its own content-addressed cache-set
   manifest and rehash every artifact before recovery admission.
7. Keep Nemotron v2 quarantined until its public derivative-use decision is
   documented. Track all GPU work in the 240-GPU-hour ledger.

**Exit evidence**

- Immutable provenance and cohort manifests with no unresolved release-license
  finding; 36/36 domains and all required language strata pass in the training,
  calibration, and held-out partitions.
- At least 1,000,000/50,000/50,000 unique, verified artifacts under one teacher
  revision, provenance digest, loss-position contract, and partition root hash.
- Reproducible teacher-cache hashes and coverage report for every planned loss;
  missing, duplicate, and extra cache identities are all hard failures.
- Separate calibration, recovery-training, and held-out evaluation splits.

## Phase 2: train and freeze the final Q2/Q4 checkpoint

**Work**

1. Recompute activation-weighted Q2-vs-Q4 error with the complete admitted
   50,000-sample calibration partition.
   Allocate Q4 where measured error reduction per byte is greatest; keep Q2
   elsewhere. Q3 remains absent.
2. Extend assignment granularity for embedding and LM-head storage to
   independently scored row/block groups. Frequently used or sensitive groups
   may use Q4 while the remainder stays Q2; neither tensor is forced to INT8 or
   promoted wholesale to Q4 without measured justification.
3. Keep quantization codes fixed and train `s_in`/`s_out`, compatible
   Hadamard/incoherence transforms, and recovery parameters using logit KL,
   cross-entropy, hidden-state, and activation reconstruction losses.
   Compare independent corrections against the plan-bound
   `qwen38_fanout_s_in_v1` candidate. The tied candidate may share corrected
   A8 activations only when every exported FP16 `s_in` in the declared group
   is byte-identical and its held-out quality gates pass.
4. Run ablations for correction scales, transforms, layer allocation, LM head,
   embedding, and MTP. Reject improvements that only move the calibration set.
5. Fold inference-time corrections into scales and fused-kernel metadata, then
   emit one backend-neutral logical checkpoint.
6. Produce deterministic CPU, CUDA, Metal, and Snapdragon packs from that
   checkpoint without changing logical values.

**Exit evidence**

- Final text+MTP resident bytes at or below 8,375,186,227 bytes (7.8 GiB).
- Weighted benchmark score at least 95% of BF16; no primary category below 90%.
- Recovery closes at least 50% of the direct-Q2/Q4-to-BF16 quality gap.
- Agentic/tool-calling, German, code, and MTP gates pass on held-out samples.
- Held-out agent traces include actionable tool failures and require immediate
  exact recovery: qualify an ambiguous symbol or use the observed repository
  filename, then resume the original task without speculative search fan-out.
- 128K retrieval reaches at least 90% of the BF16 reference.

## Phase 3: complete the model-local Rust reference engine

**Work**

1. Implement the tokenizer, canonical chat/tool/reasoning template, sampling,
   deterministic seeds, and token streaming.
2. Execute the complete 64-layer hybrid graph: embeddings, full attention,
   GatedDeltaNet/linear attention, FFN, normalization, RoPE, residual paths,
   LM head, and chained MTP draft/block-target verification. The scalar MTP4
   replay path remains the oracle; production MTP4 must replay the accepted
   prefix from one FP16 target-state checkpoint on the Fold profile.
3. Build the canonical restricted MTP draft vocabulary from the final teacher
   cohort, require overall/coding/per-domain/per-language coverage, and gather
   only those LM-head rows for proposals. Full target verification remains
   mandatory and preserves exact greedy output.
4. Implement paged Q2 KV with Q4 sink/recent pages, exact linear-attention
   state, bounded prefill/decode arenas, cancellation, and one active session.
5. Make the scalar path the test oracle only. Production policy fails closed
   if any graph operation lacks a promoted backend implementation.
6. Implement deterministic ownership so `unload` returns allocator usage to
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
4. Measure sustainable memory bandwidth, typed tensor-core throughput, and
   dispatch latency on the same device, then close every material gap reported
   by `ROOFLINE_GATES.md` across the required shape/residue sweep.

`qwen38-cuda-e2e-bench` now implements the release-bound measurement contract
for item 3 through the production `Engine`/executor ABI, including real prompt
prefill, incremental decode, MTP accounting, deterministic repetition, and
zero-residue unload. Results remain pending until it runs against the final
trained CUDA pack; no kernel microbenchmark or projection substitutes for that
evidence.

**Exit evidence**

- Same-device scalar/BF16 verifier and full golden suite pass.
- No fallback is observed in operation traces.
- Prefill and decode do not regress versus the pinned reference, and at least
  one improves by 10% or more under identical conditions.
- Every production-reachable CUDA phase reaches at least 85% of its measured
  practical roofline; evidence above 105% is rejected as incomplete accounting.

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
- CPU and Metal each carry their own sustainable ceiling measurements and pass
  the full roofline shape/residue sweep rather than inheriting CUDA evidence.

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

`complete final teacher cache -> final activation/sensitivity -> recovery training -> held-out quality loop -> final logical checkpoint -> complete Rust graph -> CUDA reference backend -> Metal/CPU -> Snapdragon device backend -> integrations -> release`

Contract work in Phase 0 may proceed alongside the data pipeline, but neither
backend tuning nor Greppy integration may invent a model identity before the
logical checkpoint and manifest contract are frozen.

The `ctox.model-release.v2` Rust schema, signed integrity envelope, chunk and
loader contract, backend logical-equivalence verifier, and checked memory-peak
formulas are implemented. The actual release instance, tokenizer/template
digests and measured platform profiles remain open until the final logical
checkpoint exists. The shared embeddable lifecycle ABI now covers signed load,
warmup, incremental prefill/decode, cancellation, MTP accounting, reset,
telemetry, and verified zero-residue unload; complete model executors and the
thin IPC executor binding remain open. The versioned JSON-lines request,
operation, session, progress, token-stream, cancellation, reset, and unload
wire contract is implemented, and the bring-up server rejects inference until
it can own a promoted executor.

Metal parallel bring-up has now replaced operation-local activations for the
first exact decode chain. Frozen steps 0-11 dispatch embedding, layer-0 RMSNorm,
and all four linear-attention projections directly through typed views of the
single 1,173,760-byte arena, then update `LinearQkv` in place through the
stateful causal convolution, prepare expanded Q/K/V plus LogDecay/Beta arena
views, execute the recurrent update against checkpointed FP16 state, and apply
direct-weight gated RMSNorm in place to `AttentionOutput`, and project that
view through the recovered Q2/Q4 linear output matrix into `MixerOutput`, with
one fused residual-add/Qwen-RMSNorm dispatch writing `HiddenB` and the next
`Normalized` view, then project that view through the mixed-Q2/Q4 FFN gate/up
fan-out, then fuse SwiGLU directly into the Q2/Q4 down projection without a
materialized product vector, then fuse the post-FFN residual add and next-layer
Qwen RMSNorm. The complete first linear-attention transformer layer now runs in
one command encoder/wait. A reusable ten-step encoder now admits any frozen
linear-attention layer only after canonical weight/recovery, convolution,
GatedDelta-parameter, recurrence-owner, and norm identity checks. This is
paired with an atomic all-48 resource loader; each layer becomes one closed
owner for mmap-backed parameters and its two persistent state classes. This is
verified on Apple Silicon but is not backend promotion: complete 48-layer
iteration, the remaining 633 graph steps, complete
target+MTP execution, prefill arena, lifecycle measurements, and Golden suite
remain open.

The immediate execution batch is:

1. emit an exact final-minus-verified 1,735-record cohort and a report binding
   the final corpus hash to the five reused verification hashes;
2. produce its exact teacher-cache size and token-aware batch plan, then run and
   verify each missing BF16/MTP batch;
3. assemble and rehash the full 2,328-record teacher cache set;
4. collect and verify the frozen 256-sample, 917,704-token activation cohort
   across all 36 domains, 15 languages, and 14 service modes, then recompute
   the all-506-matrix Q2/Q4 assignment under the 7.8-GiB package ceiling;
5. run the implemented complete packed fixed-qcode `train_recovery.py` student
   trainer over the admitted cache set, consuming KL, CE, hidden and MTP
   losses while retaining bounded layer fitting as an initializer tool;
6. run the content-addressed held-out numerical evaluator and direct-vs-trained
   fixed-qcode comparison, then generation/tool/long-context/MTP ablations until
   every separate release gate passes and freeze the logical checkpoint and
   release manifest;
7. in parallel where it does not consume the recovery GPU, complete the
   embeddable Rust `Engine` lifecycle and scalar full-graph oracle. Accelerator
   tuning begins only after logical checkpoint identity is frozen.
