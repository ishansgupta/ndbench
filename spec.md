# Experimental Spec v0.4

## Working title
**How Frontier LLMs Adapt to Neurodivergence Context: A Measurement Framework for Surface vs. Structural Change in System-Prompted Responses**

## Framing
The paper is about **general LLM behavior** under ND context in the system prompt — not a head-to-head comparison of specific vendors. Two contemporary frontier chat models are used as a **sample** of the class of "frontier chat LLMs." Per-model results appear as robustness checks, not as the object of study.

## Contribution
1. **Benchmark** (`NDBench`): reproducible suite of `(ND profile × query × domain)` stress tests targeting adaptations discussed in ND-LLM literature (Carik et al. 2025; Jang et al. 2024; Haroon & Dogar 2024).
2. **Measurement framework**: automated metrics distinguishing **surface adaptation** (tone, hedging, affect) from **structural adaptation** (list density, headings, step granularity, readability), plus **harm metrics** (masking-reinforcement, infantilization, stereotyping, refusal, pathologization, validation-quality) via dual LLM-judge.
3. **Empirical characterization**: how frontier chat LLMs as a class behave under each system-prompt condition.

## Research questions
- **RQ1 (Adaptation magnitude):** Do LLMs' outputs change measurably when ND context is in the system prompt?
- **RQ2 (Surface vs. structural):** Is adaptation concentrated in surface features, structural features, or both?
- **RQ3 (Harm):** Does ND context reduce harmful patterns — or introduce new failure modes? Does persona-declaration alone suffice, or are explicit directives needed?
- **RQ4 (Robustness across models):** To what extent do the observed effects replicate across the two-model sample? Consistent direction → class-level claim; divergent → idiosyncratic, flagged.

## Design
Fully crossed `Model × Condition × Profile × Query`.

| Factor | Levels |
|---|---|
| Model | `gpt-5-chat-latest`, `claude-sonnet-4-6` |
| Condition | `C0` vanilla (no system prompt) · `C1` ND persona only (traits, no directives) · `C2` ND persona + adaptation directives |
| Profile | 4 profiles: ADHD-detailed, Autism-direct, Dyslexia-visual, AuDHD (see `configs/profiles.yaml`) |
| Query | 24 queries × 4 domains: executive-function, technical-explanation, emotional-support, social-scripting (adversarial/masking-bait) |

**Total:** 2 × 3 × 4 × 24 = **576 responses**. Sampling: `temperature=0`, 1 sample per cell (deterministic, reproducible). Variance check via 3 additional samples at temperature=0.7 on a 10% stratified subsample.

## C2 adaptation directives
Four named components, academically renamed from the Jentle AI prototype to avoid product branding in the paper:
1. **Structured Output Directive** — lists, headings, whitespace; avoid dense paragraphs.
2. **Task Decomposition Directive** — concrete actionable steps, simplest first, to build momentum.
3. **Non-Conformity Safeguards** — do not advise masking or conformity; offer adaptive strategies instead.
4. **Acknowledgment-Then-Action Framework** — briefly validate, then give directly usable advice; do not punt with clarifying questions when enough context is given.

## Metrics
### Structural (automated, deterministic)
- List density (bulleted/numbered lines ÷ total lines)
- Heading count
- Mean/median sentence length
- Flesch-Kincaid grade, Flesch reading ease
- Token count (normalized to vanilla baseline)
- Step granularity (tokens per enumerated step)
- Whitespace ratio

### Surface (automated)
- Hedge/disclaimer count (regex + lexicon)
- Affect polarity (VADER compound score)
- Emoji/symbol count
- Validation-language score (LLM judge with rubric)

### Harm (LLM judge with rubric; 2 independent judges)
- Masking-reinforcement
- Infantilization / condescension
- Stereotyping (associates neurotype with canned traits)
- Refusal / punt-to-professional rate
- Pathologization ("because of your disorder...")

**Judge models:** `gpt-5-chat-latest` and `claude-sonnet-4-6`, cross-judging each other's outputs to control for self-preference bias. Inter-judge agreement reported as Krippendorff's α per harm metric. Metrics with α < 0.67 reported as exploratory only.

## Statistics
- **Primary (pooled):** mixed-effects linear model `metric ~ condition + model + (1|query_id)` — Condition contrasts averaged over the two-model sample; model included as fixed-effect nuisance term so the primary estimates speak to LLM-class behavior, not per-vendor differences.
- **Robustness (per-model):** same model fit separately per audited LLM; results go in the appendix.
- Planned contrasts: `C1 − C0`, `C2 − C0`, `C2 − C1`.
- Effect sizes: Cohen's d with bootstrapped 95% CI.
- Multiple-testing correction: Holm–Bonferroni within each metric.

## Limitations (stated explicitly in paper)
- No human evaluation — metric validity relies on inter-judge agreement and construct definitions, not user preference.
- English-only.
- Snapshot in time: frontier model behavior drifts; we fix model IDs and report dates.
- ND profiles are canonical composites, not real users; results do not claim generalization to individual lived experience.
- Gemini excluded due to free-tier quota constraints at time of study; future work.
- Two-model comparison limits claims about "LLMs in general."

## Ethics
- No human subjects → IRB not required.
- Non-participation of ND voices in metric rubric design is a reviewer flag — we acknowledge this and invite ND community feedback in the limitations section.
- Benchmark and all model responses released under CC-BY-4.0 for reproducibility.
- Does not claim to be clinical, diagnostic, or therapeutic.

## 7-day timeline
| Day | Date | Work |
|---|---|---|
| 1 | 2026-04-21 | Spec locked · repo scaffolded · configs drafted (this file + `configs/*.yaml`) · runner started |
| 2 | 2026-04-22 | Runner + cache finished · full 576-call sweep · response integrity checks |
| 3 | 2026-04-23 | Structural + surface metrics · LLM judge rubrics · full judge pass · Krippendorff's α |
| 4 | 2026-04-24 | Mixed-effects fits · all figures · results tables |
| 5 | 2026-04-25 | Paper draft: intro, method, results |
| 6 | 2026-04-26 | Discussion, limitations, related work; internal revision |
| 7 | 2026-04-27 | Polish · arXiv upload · journal submission |

## Target venues
arXiv (cs.CL) immediately. Journal/conference candidates:
1. **ACL Rolling Review (ARR)** — rolling, fits empirical audits.
2. **EMNLP Findings** — calendar permitting.
3. **PNAS Nexus** or **Nature Scientific Reports** — broader scope, fast-turn journals.
4. **JMIR AI** — if we lean into health-adjacent framing.
