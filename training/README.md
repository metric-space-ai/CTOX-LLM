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

Before end-to-end distillation, `fit_recovery_scales.py` creates a complete
deterministic initializer for every planned matrix. It regenerates the exact
logical Q2/Q4 codes (including mixed row layouts) from the pinned BF16 source,
keeps those codes fixed, and fits only positive `s_in`/`s_out` with
activation-weighted alternating least squares. Final error metrics use the
FP16-rounded scales that will actually be packed. The output tensor names must
match every recovery placeholder in the plan or a complete run fails closed.
This initializer does not substitute for the later logit-KL, cross-entropy,
hidden-state, and MTP end-to-end recovery stage.

Before assigning scarce Q4 capacity, `collect_activation_stats.py` hooks the
real BF16 text graph and stores only per-channel input/output mean squares for
quantized linear modules. These diagonal statistics support activation-weighted
Q2/Q4 error scoring without retaining source prompts or token activations.
Long calibration runs use `--start-sample`/`--max-samples` batches. Successful
batches are immutable and `merge_activation_stats.py` combines their channel
means using exact token counts, so a transient GPU failure never invalidates
already completed work.
Merged artifacts retain only semantic model/statistics metadata at top level.
Weight placement, allocator configuration, Torch/CUDA versions, and measured
CUDA peaks remain an ordered `source_runtime_profiles` list, so batches made
with different safe offload profiles cannot be mislabeled as one runtime.
Nested merges preserve the original leaf profiles, including maximum sequence
length and selected sample range; they never replace them with the merge
process's own empty runtime fields.
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

The multilingual general-purpose supplement uses the human-curated, Apache-2.0
`CohereLabs/aya_dataset` at its reviewed immutable revision. `build_manifest.py
--language` selects normalized `language_code` strata before the record limit is
applied; this prevents a high-resource language at the start of a stream from
silently consuming the multilingual quota. Machine-translated bulk from the
larger Aya Collection is not part of the initial recovery mix.

Ordinary English dialogue gaps use the MIT-licensed
`HuggingFaceH4/ultrachat_200k` source at reviewed revision
`8049631c405ae6576f93f445c6b8166f76f5505a`. Train and held-out candidates
come from its already separate `train_sft` and `test_sft` splits. This source
was admitted only after the materialized Nemotron-v1 `chat` selection was found
to contain empty user turns; those unconditioned records are excluded rather
than counted as general-domain evidence.

Sample identities cover the complete recovery payload, including reference
answers. Changing an answer therefore changes both the payload hash and sample
identity; stale source coordinates cannot silently enter a teacher cache.
`select_manifest.py` produces reproducible hash-ranked samples per input
manifest, preventing source order from biasing smoke tests and calibration
cohorts.
`split_manifests.py` performs one fail-closed pass over every selected source
manifest and emits deterministic, globally disjoint recovery-training and
held-out evaluation cohorts. Public-checkpoint splits reject quarantined or
otherwise release-ineligible records by default. `--stratify-field language`
applies the requested quota independently to each selected language instead of
letting corpus frequency determine the multilingual mix.
`--exclude-manifest` removes every exploratory calibration or previously
materialized sample before ranking, so the held-out release gate cannot inherit
examples already seen during Q2/Q4 assignment or teacher-cache smoke tests.
`merge_manifests.py` then combines the independently sized strata, deduplicates
by complete sample identity, rejects release-ineligible records, and can enforce
the exact total before materialization.
`filter_recovery_cohort.py` rejects empty conditioning, missing targets,
duplicate payloads, and payloads already used by a denied partition while
keeping materialized records and semantic tags in exact lockstep.
`audit_corpus.py` independently re-verifies every materialized payload hash,
rejects internal and train/evaluation payload overlap, and reports exact
language, capability, source, prompt-length, assistant-target, multi-turn,
structured-output, and tool-schema distributions before GPU work is admitted.
`DOMAIN_RUBRIC.json` is the subsequent semantic gate. Version 2 expands the
coarse 16-label pilot into 36 independently gated service domains grouped into
ten required families. It separates, for example, medicine from biology,
physics from chemistry, software development from systems and cybersecurity,
and finance from business operations and law. Major general, coding, agentic,
writing, mathematics, and structured-data domains have higher primary quotas;
the long tail still requires unambiguous train and held-out examples. Source
split names are not accepted as domain evidence.
`classify_domains.py` applies that rubric with a revision-pinned multilingual
NLI classifier. Threshold labels are supplemented only by deterministic source
facts (for example, an actual tool schema or the pinned code/math split). Its
tags guide cohort selection and are never used as target labels during recovery.
Hard source facts also override the primary label for genuine tool/agentic,
code, math-split, and procedural long-context records. The original classifier
primary and all 36 scores remain in the tag, while the effective primary records
`primary_source=source_fact` and confidence 1.0. `apply_primary_overrides.py`
upgrades already frozen score files without rerunning the NLI model and emits a
new immutable hash; free-form Chat and STEM records never receive an override.
The domain gate has two independent views. Multi-label counts prove that a
domain is present, while `minimum_primary_train`/`minimum_primary_evaluation`
require enough records whose clearest semantic assignment is that domain.
`audit_domain_tags.py` can apply a stricter policy to frozen classifier output
without rerunning the revision-pinned NLI model. A multi-label pass never
overrides a primary-domain gap.
`select_primary_domain_supplement.py` closes only those declared gaps from a
disjoint candidate pool. It requires the target to be the candidate's primary
label above a configurable confidence floor, adds a safety margin, and ranks
confidence before token cost; broad low-confidence multi-label matches cannot
enter merely to satisfy a quota.
`LANGUAGE_RUBRIC.json` independently fixes English, German, mixed German-English,
and twelve additional language minima. `audit_selection_coverage.py` joins the
frozen semantic tags back to the exact materialized records and rejects a mix
when a language contains only translation tasks, too few distinct primary
domains, or when the combined non-English cohort omits any required semantic
family. Language count alone is therefore not accepted as multilingual
coverage.
`select_coverage_supplement.py` performs the corresponding deterministic
gap-closing selection. It greedily credits only high-confidence candidates for
the exact still-open multi-label, primary-domain, language, non-translation,
per-language diversity, and aggregate non-English-family cells, preferring
broader coverage and then lower token cost. It fails with the unresolved cells
when the candidate pool is insufficient and re-runs both complete gates over
the combined cohort before writing release evidence.
Because independent NLI scores can produce unstable argmax choices between
sibling domains, the selector may assign one candidate to a non-argmax primary
only when that domain itself exceeds the rubric confidence and lies within the
declared selection-specific score tolerance of the classifier maximum. The
final quality-filtered selection records tolerances of 0.04 for recovery and
0.03 for held-out evaluation. The emitted tag keeps
the pre-assignment primary, target score, exact margin, and
`primary_source=near_tie_coverage_assignment`; one sample can close only one
primary cell. The effective supplement tags are a separately hashed output.
Large candidate pools may be classified as independent GPU shards.
`merge_domain_tags.py` accepts them only when their union is an exact,
duplicate-free match for the materialized candidate IDs, restores materialized
order, recomputes both quota views, and emits a new content hash.

`plan_teacher_cache.py` tokenizes the complete frozen cohort with the same
assistant, hidden-state, marker-window, and MTP position rules as the cache
writer. It accounts for every persisted tensor dtype and adds a conservative
per-file safetensors/header allowance. A release-size GPU cache run is admitted
only after this plan fits the target volume while preserving separate recovery
workspace headroom; raw corpus byte size is not a valid disk estimate.
When the tokenizer is loaded from a staged local model, the planner requires
the same verified model-provenance document as the teacher cache itself.
`verify_teacher_cache.py` is the recovery-side admission gate. It binds the
cache to its ordered source slice and teacher provenance, checks every tensor
name, dtype, shape, MTP count, hidden layer, and payload hash, and emits a
content-addressed artifact inventory. Recovery never consumes an unchecked
directory merely because it contains safetensors files.
`select_teacher_smoke.py` chooses the lowest-cost eligible records that cover
every required primary semantic domain and every frozen language stratum. This
gates the complete cache path on broad behavior rather than validating only an
English coding or agentic example.
`plan_teacher_batches.py` partitions the frozen source order into immutable
batches bounded simultaneously by sample count, input tokens, and projected
output bytes. Each batch uses `cache_teacher.py --start-sample/--max-samples`
and passes the same verifier independently, so a late host or GPU failure does
not invalidate earlier teacher work.
`select_uncached_teacher_records.py` subtracts only passed, revision- and
provenance-matched teacher verifications from a final materialized cohort. It
requires identical hidden-layer, hidden-size, top-k, and MTP settings and
rejects any cached identity outside the final cohort. This permits verified
batches from a superseded cohort ordering to be reused by exact payload
identity without treating failed or smoke directories as evidence.
`run_teacher_batches.py` executes that plan one batch at a time and immediately
runs the content verifier. It skips existing work only when the completed run,
source slice, teacher revision, provenance hash, and verification sample count
all agree; partial directories stop the orchestrator for inspection.
An inspected OOM prefix can be continued only with the explicit
`--resume-incomplete` path. `cache_teacher.py --resume` requires the existing
index to be a non-empty exact prefix of the same source slice, rejects missing
or unindexed artifacts, keeps all semantic teacher/cache settings fixed, and
records differing memory/device layouts as runtime profiles. The normal path
continues to reject every pre-existing output directory, and full content
verification still runs after the resumed suffix completes.

The end-to-end loss contract uses the teacher's exact top-k probabilities plus
one residual-vocabulary mass bucket; dropping the residual would not be a valid
KL approximation. Cross entropy targets the recorded `p -> token[p+1]`
positions, while hidden reconstruction is signal-normalized and includes a
directional penalty. The same sparse KL contract applies to every verifiable
MTP draft target.
`end_to_end_recovery_loss` composes six explicit, independently reported
families: base KL, base CE, multi-layer hidden reconstruction, MTP KL, MTP CE,
and MTP hidden reconstruction. No aggregate loss may silently omit MTP or
collapse multilingual/domain selection into a target label.
`teacher_cache_dataset.py` is the training-side content boundary. It accepts
only passed verification documents with one teacher revision and provenance,
rejects duplicate sample identities and unsafe paths, and rechecks each
artifact's exact byte length and SHA-256 when that sample is opened.
After the final batch, `build_teacher_cache_set.py` resolves either one batch
plan or an explicit list of verification documents, rehashes every artifact by
default, and emits the single content-addressed cache-set manifest accepted by
the end-to-end trainer. With `--expected-input`, its cached identity union must
equal the final cohort exactly. Missing, extra, or settings-incompatible
batches fail closed.
The trainer reopens that set only with its expected manifest SHA-256, then
reconstructs and compares the sample count, artifact bytes, content root, and
every underlying batch-verification hash before the first optimization step.
`ctox_artifact.py` provides the corresponding offline Python reader for the
native Rust container. It validates the 64-byte header, version, endianness,
manifest bounds, alignment, Q2/Q4 or mixed-row byte formulas, non-overlapping
ranges, and optional per-tensor SHA-256 before exposing a read-only memoryview.
Its row decoder reads only the requested Q2/Q4 or mixed-row slice, reconstructs
the native FP16 block scale and bit ordering exactly, and leaves the canonical
packed codes immutable. This bounded decoder is the correctness path for the
future autograd/fused-kernel recovery adapter, not a second quantizer.
`packed_recovery_ops.py` supplies the first bounded autograd primitive. Its
forward decodes only an output-row chunk at a time; backward re-decodes those
same immutable codes and analytically accumulates gradients for input,
`s_in`, `s_out`, and bias. It therefore does not retain a dequantized BF16
matrix between passes, and its gradients are tested against the dense oracle.
Native FP16/FP32 tensors, including the initial recovery scales, are decoded by
the same artifact reader with exact manifest shapes; the trainer never needs a
parallel scale file after a checkpoint has been packed.
`PackedRecoveryLinear` wraps that primitive as a trainable module whose only
parameters are logarithmic positive channel corrections. Packed weights and
bias stay frozen, and export returns the exact `<weight>.s_in`/`.s_out` FP16
names required by the native manifest.
`PackedRecoveryRegistry` scans the native artifact for every quantized matrix,
requires an exact FP16 scale pair with matching channel shapes, and constructs
modules on demand. This prevents the trainer from optimizing a partial tensor
set or silently initializing a missing correction to one.
The packed embedding module decodes only unique token rows (coalescing adjacent
IDs), then applies the same column and row corrections as the logical matrix.
Gradients flow only into those corrections; the vocabulary matrix is never
expanded or retained in BF16.
`packed_student_model.py` replaces the meta-initialized Qwen embedding and
linear modules with those packed operators, loads every remaining frozen
FP16/FP32 norm and recurrent parameter from the same container, excludes MTP
for its separate graph, and fails if any meta parameter or unmatched module
remains. Only logarithmic recovery scales retain gradients.
The separate MTP installer constructs the single full-attention draft layer on
meta, maps its eight native packed matrices through the pinned checkpoint-name
contract, loads its frozen norms from the same container, and preserves the
main graph's shared packed embedding and LM head.

Agentic manifests pin source-specific splits and reviewed upstream revisions.
Their complete tool schemas are part of both the payload hash and materialized
teacher input; changing a function name or JSON schema therefore invalidates
the sample identity. Dataset-card and per-record licenses are recorded
separately and unreviewed license identifiers fail closed.
The NVIDIA Agentic sources are streamed from their revision-pinned raw JSONL
files. This preserves heterogeneous tool schemas that the generic Arrow reader
otherwise attempts to coerce into one incompatible struct.

`generate_long_context.py` builds Apache-2.0 procedural retrieval dossiers at
32K, 64K, and 128K. Every context consists of distinct structured records, and
the answer requires following a link between two records inserted at recorded
token offsets. It does not repeat or pad a short prompt. Calibration and
evaluation invocations must use different seeds and splits. NVIDIA's
`ChatQA2-Long-SFT-data` is useful external evidence that 131K training examples
exist, but its non-commercial terms exclude it from the public CTOX checkpoint.
The generator writes provenance-only and materialized files separately.
`materialize_prompts.py --local-materialized` accepts the latter only when its
source tuple and complete payload hash match the requested manifest, preventing
locally generated records from being mistaken for a remote dataset revision.

Teacher caching renders a normalized copy of source messages. OpenAI-shaped
tool calls are converted to Qwen's flat chat-template form without changing the
hashed source payload. Only requested transformer layers are captured, and the
LM head is evaluated in bounded token chunks so full-vocabulary FP32 logits are
never resident for an entire sequence.
Local staged BF16 teachers are accepted only with
`--local-model-provenance` generated by `verify_local_model.py`. That verifier
compares every model shard and required tokenizer/config file byte-for-byte
against the pinned Hugging Face commit; a merely supplied revision string is
not accepted as provenance.
For long-context records, `cache_teacher.py --target-mode assistant` stores
logits only for the final supervised assistant response. Hidden-state targets
cover a bounded uniform sample of that response, bounded windows around
recorded retrieval markers, and a uniform full-sequence sample. MTP sparse
logits still cover every verifiable draft while MTP hidden-state storage uses
the same bounded positions. Hooks slice and move those positions immediately;
they never persist five full 128K hidden-state tensors. Teacher and activation
inputs fail rather than silently truncating a record above `--max-length`.
On memory-constrained teacher hosts, `--gpu-weight-memory-gib` reserves GPU
headroom and allows Accelerate to place the remaining frozen BF16 layers in
host RAM. `--use-fla-kernel` selects only the FLA Gated-Delta implementation at
the immutable revision in `teacher_runtime.py`. It deliberately does not use
Transformers' generic kernel switch: that switch also requests a hub
causal-convolution build unavailable for GPU3's pinned Torch/CUDA pair. The
selected FLA revision and exact device map are stored in the teacher artifact.
`--prefill-chunk-tokens` executes a long prompt as a stateful causal prefill:
GatedDelta carries its recurrent state and full-attention layers carry their KV
pages, while QKV and FLA workspaces are bounded by the current chunk. Selected
hidden/logit positions are reassembled in global token order and verified
before writing the sample.
The same option on `collect_activation_stats.py` accumulates diagonal base and
MTP statistics chunk by chunk. `--mtp-device cuda:N` can keep the frozen MTP
block on a GPU when the last text layers and shared LM head are deliberately
offloaded to CPU; activation collection bypasses the unused MTP vocabulary
projection but executes every frozen MTP matrix and its persistent attention
cache.
Passing `cache_teacher.py --mtp-device cuda:N` additionally loads the pinned
resident MTP checkpoint fail-closed and caches its selected BF16 hidden states,
top-64 draft logits, residual probability, and exact base-position semantics.
The final base position is excluded because it has no verifiable `p+2` token.

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
checksummed `.ctoxq` artifact. Passing `--recovery-scales` requires a complete
`ctox.recovery.channel-scales.v2` file whose model, revision, plan hash, tensor
names, shapes, dtypes, and fixed-code declaration match exactly. The manifest
records the scale artifact, plan, activation-statistics, and report hashes.
Omitting the option remains available only for explicitly marked identity
baselines. The packer refuses to overwrite files and refuses plans above the
Fold limit.

Vision remains a separate phase-resident package. `build_vision_plan.py`
zero-pads matrix columns to 64-value storage blocks so non-aligned vision MLPs
can still use Q2/Q4 kernels. `plan_vision_residency.py` then selects whole
text/MTP bundles to unmap before vision, and refuses a phase plan above the
9.7-GiB operating target. Decoder workspaces are reused rather than duplicated.
