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

Nemotron v2 is quarantined by default. Research may opt into the cohort, but a
public checkpoint cannot claim release eligibility until a legal decision is
recorded in the manifest.

Sample identities cover the complete recovery payload, including reference
answers. Changing an answer therefore changes both the payload hash and sample
identity; stale source coordinates cannot silently enter a teacher cache.
`select_manifest.py` produces reproducible hash-ranked samples per input
manifest, preventing source order from biasing smoke tests and calibration
cohorts.

The 240 GPU-hour ceiling is cumulative across teacher generation, sensitivity
runs, ablations, final recovery, and evaluation. Every command appends its GPU
count and elapsed time to `run-ledger.jsonl`.

`build_quant_plan.py` reads only safetensor metadata and calculates exact packed
bytes, including alignment and recovery scales. It excludes vision, includes
resident MTP, rejects a plan above 7.8 GiB, and emits the immutable assignment
consumed by target packers.

`pack_checkpoint.py` performs the direct BF16 conversion. It memory-maps all
source shards, slices matrices by rows, quantizes those chunks on one GPU,
writes an aligned temporary data region, then creates the final manifest and
checksummed `.ctoxq` artifact. It refuses to overwrite files and refuses plans
above the Fold limit.
