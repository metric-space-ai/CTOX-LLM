# Recovery training

Training is offline tooling, not an inference-runtime dependency. The release
pipeline has four immutable stages:

1. `build_manifest.py` streams source datasets and emits provenance records.
2. `materialize_prompts.py` re-streams the pinned revisions, verifies each
   complete prompt/answer or conversation hash, and stores the records in an
   access-controlled cache; materialized text is never committed.
3. `cache_teacher.py` runs the pinned BF16 teacher and stores top-64 logits,
   residual probability mass, and selected hidden states.
4. `train_recovery.py` freezes Q2/Q4 codes and trains channel correction scales.

Before assigning scarce Q4 capacity, `collect_activation_stats.py` hooks the
real BF16 text graph and stores only per-channel input/output mean squares for
quantized linear modules. These diagonal statistics support activation-weighted
Q2/Q4 error scoring without retaining source prompts or token activations.
Long calibration runs use `--start-sample`/`--max-samples` batches. Successful
batches are immutable and `merge_activation_stats.py` combines their channel
means using exact token counts, so a transient GPU failure never invalidates
already completed work.
`score_quant_sensitivity.py` then reuses the packer's canonical Q2/Q4 code
construction and estimates each matrix's output error under a diagonal input
covariance. Q4 optimization consumes the resulting measured quality gain per
additional byte.
The optimizer recomputes the complete aligned native layout for every accepted
Q4 candidate, including non-quantized tensors and recovery scales. Its immutable
assignment can be passed back to `build_quant_plan.py --assignment`; a raw sum
of weight bytes is never accepted as the Fold memory gate.

Nemotron v2 is quarantined by default. Research may opt into the cohort, but a
public checkpoint cannot claim release eligibility until a legal decision is
recorded in the manifest.

German calibration uses the pinned, CC-BY-4.0
`Beko2210/German-Instruct-Dataset` source. Its optional RAG context is folded
into the user turn before hashing, so context changes invalidate provenance in
the same way as prompt or answer changes. Attribution remains required for any
released derivative.

Sample identities cover the complete recovery payload, including reference
answers. Changing an answer therefore changes both the payload hash and sample
identity; stale source coordinates cannot silently enter a teacher cache.
`select_manifest.py` produces reproducible hash-ranked samples per input
manifest, preventing source order from biasing smoke tests and calibration
cohorts.

Agentic manifests pin source-specific splits and reviewed upstream revisions.
Their complete tool schemas are part of both the payload hash and materialized
teacher input; changing a function name or JSON schema therefore invalidates
the sample identity. Dataset-card and per-record licenses are recorded
separately and unreviewed license identifiers fail closed.

`generate_long_context.py` builds Apache-2.0 procedural retrieval dossiers at
32K, 64K, and 128K. Every context consists of distinct structured records, and
the answer requires following a link between two records inserted at recorded
token offsets. It does not repeat or pad a short prompt. Calibration and
evaluation invocations must use different seeds and splits. NVIDIA's
`ChatQA2-Long-SFT-data` is useful external evidence that 131K training examples
exist, but its non-commercial terms exclude it from the public CTOX checkpoint.

Teacher caching renders a normalized copy of source messages. OpenAI-shaped
tool calls are converted to Qwen's flat chat-template form without changing the
hashed source payload. Only requested transformer layers are captured, and the
LM head is evaluated in bounded token chunks so full-vocabulary FP32 logits are
never resident for an entire sequence.

The 240 GPU-hour ceiling is cumulative across teacher generation, sensitivity
runs, ablations, final recovery, and evaluation. Every command appends its GPU
count and elapsed time to `run-ledger.jsonl`.

`build_quant_plan.py` reads only safetensor metadata and calculates exact packed
bytes, including alignment and recovery scales. It excludes vision, includes
resident MTP, rejects a plan above 7.8 GiB, and emits the immutable assignment
consumed by target packers.
The resident tensor plan reserves 2 MiB below that package ceiling for the
container manifest and release metadata. `pack_checkpoint.py` independently
rejects a final file above the full 7.8-GiB package limit; fitting tensor bytes
alone is not accepted as evidence.

`pack_checkpoint.py` performs the direct BF16 conversion. It memory-maps all
source shards, slices matrices by rows, quantizes those chunks on one GPU,
writes an aligned temporary data region, then creates the final manifest and
checksummed `.ctoxq` artifact. It refuses to overwrite files and refuses plans
above the Fold limit.

Vision remains a separate phase-resident package. `build_vision_plan.py`
zero-pads matrix columns to 64-value storage blocks so non-aligned vision MLPs
can still use Q2/Q4 kernels. `plan_vision_residency.py` then selects whole
text/MTP bundles to unmap before vision, and refuses a phase plan above the
9.7-GiB operating target. Decoder workspaces are reused rather than duplicated.
