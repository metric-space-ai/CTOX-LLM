# Qwen3.8-27B recovery corpus mix

This is the release-candidate sampling contract, not an assertion that a finite
corpus literally contains every human domain. Coverage is enforced across
capability families, application domains, languages, prompt lengths, and output
shapes. Training and evaluation identities are disjoint before prompt text is
materialized.

## Final quality-filtered cohort

| Capability stratum | Train | Held-out | Purpose |
|---|---:|---:|---|
| General dialogue | 866 | 247 | ordinary questions, explanation, writing, summarization, practical help, culture, society, and professional communication |
| Coding | 503 | 142 | generation, debugging, review, tests, systems, databases, and multiple programming languages |
| Mathematics and logic | 523 | 137 | arithmetic through formal reasoning plus quantitative problem solving |
| Agentic/tool/search | 381 | 96 | planning, tool schemas, tool results, search, and multi-step execution |
| Procedural long context | 55 | 20 | disjoint 32K/64K/128K retrieval and cross-record linking |
| **Total** | **2,328** | **642** | one active recovery run and a payload-disjoint release gate |

The original Nemotron-v1 `chat` records selected for the fixed cohort were
rejected after the exact materialized payload audit found 524 training and 138
evaluation records without any conditioning content. They are not counted as
normal prompts. Valid pinned Aya and UltraChat records replenish the general,
writing, creative, interpersonal, and multilingual gaps. UltraChat is pinned at
revision `8049631c405ae6576f93f445c6b8166f76f5505a` under MIT; Aya remains pinned
at `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` under Apache-2.0.

The final source counts are:

| Source | Train | Held-out |
|---|---:|---:|
| Nemotron Post-Training v1 (valid non-chat strata) | 1,033 | 286 |
| Nemotron Agentic v1 | 125 | 32 |
| Nemotron SFT Agentic v2 | 192 | 48 |
| German Instruct | 264 | 73 |
| Aya | 458 | 118 |
| UltraChat 200k | 232 | 73 |
| CTOX procedural long context | 24 | 12 |

Only deterministic, gap-closing subsets enter the final mix, so tool use,
coding, or mathematics cannot displace normal language. English and German
remain the highest-weight languages because they match the initial CTOX use
cases; twelve additional language strata cover Latin, Cyrillic, Arabic,
Devanagari, Han, Kana, and Hangul scripts. Exact per-language and per-domain
counts and artifact hashes are frozen in
`models/qwen38_27b/docs/DOMAIN_COVERAGE_V2.json`.

## Application-domain coverage

The general, German, and multilingual strata are tagged and audited against a
36-domain matrix before a release candidate is admitted. The compact list below
is a readable summary; `DOMAIN_RUBRIC.json` contains the binding quotas:

- everyday assistance, communication, rewriting, summarization, and education;
- software, data, cybersecurity, hardware, and engineering;
- mathematics, physics, chemistry, biology, environment, and medicine;
- law, public administration, business, economics, accounting, and finance;
- history, geography, politics, culture, philosophy, and social science;
- creative writing, media, travel, food, home, and personal organization;
- translation, localization, structured extraction, classification, and tables;
- safety-sensitive refusal/calibration cases and uncertainty-aware answers.

No single source is trusted to prove this coverage. A committed audit reports
counts from source metadata plus the frozen multi-label classifier contract in
`DOMAIN_RUBRIC.json`, and a stratum that misses its minimum is replenished from
its source pool before training. Medical,
legal, and financial records preserve general assistance and uncertainty
behavior; they are not treated as a substitute for expert-reviewed knowledge.

The semantic gate is hierarchical: every one of ten families must be present,
and every one of its 36 leaf domains must independently meet both a multi-label
minimum and a clear-primary minimum. This prevents broad labels such as STEM,
business, or humanities from hiding a missing field. Coding, agentic work, and
mathematics retain higher quotas without being allowed to displace ordinary
language, professional, scientific, societal, creative, and daily-life tasks.

Multilingual admission is joint rather than a raw language count. Each of the
15 declared language strata must contain several distinct primary domains and
enough non-translation tasks. Across all non-English records, every semantic
family—including software/data, safety, science, business/law, and daily life—
must have primary examples in both recovery and held-out evaluation.

## Length and output-shape gates

- At least 20% of ordinary samples are multi-turn.
- At least 15% of ordinary samples require structured or constrained output.
- Short, medium, and 8K+ prompts are all represented; long-context examples are
  budgeted separately so repeated long records do not dominate token count.
- Code is audited by programming language and task type, not just the `code`
  split label.
- Train/evaluation overlap is exactly zero by complete payload SHA-256.
- Source revision, license, language, category, prompt hash, and generator are
  retained in every manifest record.

The matrix is a pre-training gate. It does not by itself establish retained
quality; BF16-vs-quant evaluations remain separate for every major capability,
domain family, and language family.
