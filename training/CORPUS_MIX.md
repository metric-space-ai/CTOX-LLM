# Qwen3.8-27B recovery corpus mix

This is the release-candidate sampling contract, not an assertion that a finite
corpus literally contains every human domain. Coverage is enforced across
capability families, application domains, languages, prompt lengths, and output
shapes. Training and evaluation identities are disjoint before prompt text is
materialized.

## Fixed cohort quotas

| Capability/source stratum | Train | Held-out | Purpose |
|---|---:|---:|---|
| General conversation and knowledge (Nemotron v1 `chat`) | 384 | 96 | ordinary questions, explanation, writing, summarization, practical help |
| Coding (Nemotron v1 `code`) | 320 | 80 | generation, debugging, review, tests, multiple programming languages |
| Mathematics and logic (Nemotron v1 `math`) | 192 | 48 | arithmetic through formal reasoning |
| STEM and technical (Nemotron v1 `stem`) | 192 | 48 | natural sciences, engineering, computing, data analysis |
| Agentic/tool/search pool (six pinned Agentic strata) | 384 | 96 | planning, tool schemas, tool results, search, multi-step execution |
| German general/professional (German Instruct) | 192 | 48 | normal dialogue, business, administration, RAG, technical German |
| Multilingual human-curated dialogue (Aya; twelve strata) | 384 | 96 | 32/8 each for French, Spanish, Italian, Portuguese, Dutch, Polish, Chinese, Japanese, Arabic, Hindi, Russian, Korean |
| Procedural long context | 24 | 12 | disjoint 32K/64K/128K retrieval and cross-record linking |
| **Total** | **2,072** | **524** | one active recovery run and a separate release gate |

The Agentic 768/192 materialization is a candidate pool. Only a deterministic
384/96 subset enters the final mix, so tool use cannot dominate normal language
or coding. English and German remain the highest-weight languages because they
match the initial CTOX use cases; the other twelve languages are equal-weighted
across Latin, Cyrillic, Arabic, Devanagari, Han, Kana, and Hangul scripts. This
avoids defining multilingual as only French and Spanish translation.

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
`DOMAIN_RUBRIC.json`, and a stratum that
misses its minimum is replenished from its source pool before training. Medical,
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
