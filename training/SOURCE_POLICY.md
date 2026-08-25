# Recovery source policy

This table is the reviewed input boundary for the Qwen3.8-27B recovery run.
`build_manifest.py` defaults to the immutable revisions below and fails closed
when the pinned dataset card has no machine-readable license or a record
introduces an unreviewed license identifier.

Review date: 2026-08-25

| Source | Reviewed revision | Included splits | Card license | Public checkpoint status |
|---|---|---|---|---|
| [Nemotron Post-Training v1](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1) | `74e23eb6f830fef4a9e96a92f6f6262214cbb9a8` | `chat`, `code`, `math`, `stem`, `tool_calling` | CC-BY-4.0 | eligible with attribution |
| [Nemotron Agentic v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | `650d590978ca35c8f1ecea2faf136e5fac421b62` | `interactive_agent`, `tool_calling` | CC-BY-4.0 | eligible with attribution |
| [Nemotron SFT Agentic v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v2) | `7c804833427f633ccd53b582dbf02525fd680f78` | `interactive_agent`, `search`, `tool_calling` | CC-BY-4.0 plus listed Apache-2.0/MIT components | eligible with attribution and component notices |
| [German Instruct Dataset](https://huggingface.co/datasets/Beko2210/German-Instruct-Dataset) | `4456bdf1b82f906a70fb9e5431530d2e9d1c565b` | `train` | CC-BY-4.0 | eligible with attribution |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` | `train` (language-stratified) | Apache-2.0 | eligible; human-curated multilingual supplement |
| [Nemotron Post-Training v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | `5c89e01dd720ae0f4058445ed49c5fb68a03c76e` | selected English and German splits | CC-BY-4.0; gated access | quarantined pending a documented derivative-use decision |

Tool definitions are training input, not incidental metadata. For Agentic v1
and v2, the full `tools` array is included in the source payload hash,
materialized record, and Qwen chat-template rendering. A changed function JSON
schema therefore creates a different sample identity.

## Long-context decision

[NVIDIA ChatQA2 Long SFT](https://huggingface.co/datasets/nvidia/ChatQA2-Long-SFT-data)
contains a `NarrativeQA_131072` configuration and is useful evidence for the
shape of a 128K RAG example. Its dataset card restricts the release to
non-commercial use and notes upstream OpenAI terms, so CTOX-LLM does not ingest
it into either recovery training or public checkpoint evaluation.

Instead, `generate_long_context.py` creates Apache-2.0 procedural dossiers with
distinct facts and cross-record links. It records the actual Qwen-token offsets
of both retrieval records and produces separate `calibration` and `evaluation`
splits from different seeds. Accepted samples must fall within the configured
token band below 32K, 64K, or 128K; repeated filler and tokenizer padding are
not generated.

This is an engineering source review, not a substitute for Metric Space's
final legal release approval. Generated manifests retain every applicable
license and source revision for that approval.
