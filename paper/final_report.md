# User Signal Mining Agents: Multi-Step Grounded Pipelines for Extracting Actionable Insights from Customer Reviews

## Abstract

We investigate whether decomposing a founder's natural-language question into structured sub-tasks—intent decomposition, multi-query retrieval, grounded synthesis, and evidence verification—produces more actionable focus points from customer reviews than a single-step zero-shot baseline. Using an LLM-as-judge evaluation across 20 prompts spanning three domains (restaurants, SaaS, ecommerce), we find that the multi-step pipeline is preferred on 17 of 20 prompts, with overall preference improving from 3.50 to 4.65 on a 5-point scale (+1.15). Gains are largest on groundedness (+0.65) and relevance (+0.80), confirming that explicit retrieval planning and evidence grounding reduce hallucination and improve specificity. Human annotation with two reviewers over the same 20 tasks reveals weak calibration: interannotator exact agreement on winner selection is 40% (Cohen's κ = -0.19), and annotator-versus-judge agreement ranges from 35% to 70%. We release the full system, evaluation framework, and dataset pipeline as open-source software.

## 1. Introduction

Founders routinely need to extract actionable signals from noisy, unstructured customer feedback. While large language models (LLMs) can summarize text, zero-shot prompting often produces generic, ungrounded outputs that lack the specificity founders need for decision-making (Ji et al., 2023). Recent work on agentic AI systems—where LLMs are orchestrated as planning, retrieval, and reasoning components in multi-step pipelines—suggests that task decomposition can improve output quality on complex information tasks (Yao et al., 2023; Shinn et al., 2023).

We formalize this as a controlled comparison: given a founder's natural-language question and a corpus of real customer reviews, does a multi-step grounded pipeline outperform a zero-shot baseline at producing actionable, evidence-backed focus points? Our contribution sits at the intersection of **evaluation** and **implementation**: we build an open-source agent system with a structured evaluation framework and measure its effectiveness across domains.

**Research questions.** (1) Does multi-step decomposition (intent → retrieval → synthesis → verification) improve the quality of extracted focus points over a single-step baseline? (2) How well does an LLM-as-judge evaluation align with human preferences for this task?

## 2. System Design

### 2.1 Baseline: Zero-Shot Retrieval-Augmented Generation

The baseline retrieves top-K evidence snippets using hybrid search (dense + BM25 with reciprocal rank fusion), then passes them with the founder's question to an LLM in a single call, prompting for 3–5 structured focus points.

### 2.2 Pipeline: Multi-Step Grounded Agents

The pipeline decomposes the task into four sequential agents:

1. **Intent Agent.** Parses the founder's statement into keywords, target user, counter-hypotheses, and 3–5 retrieval queries written in customer-voice language.
2. **Evidence Filter.** Executes all generated queries against the hybrid retrieval index, deduplicates results, and reranks by relevance.
3. **Synthesis Agent.** Produces 3–5 focus points grounded in retrieved evidence, with inline quotes, counter-signals, and validation questions.
4. **Evidence Verifier.** Checks each supporting snippet against the source corpus to flag hallucinated or misattributed quotes.

Both systems use the same retrieval index (Sentence Transformers `all-MiniLM-L6-v2` embeddings + BM25), the same LLM (`gemini-3.1-flash-lite-preview`), and the same output schema (`FocusPoint` with label, rationale, supporting snippets, counter-signal, and next validation question).

### 2.3 A/B-Debiased LLM Judge

We evaluate outputs using an LLM-as-judge (Zheng et al., 2023) that scores both systems on four dimensions: **relevance** (does the output address the founder's intent?), **groundedness** (are claims traceable to evidence?), **distinctiveness** (are focus points non-repetitive?), and **overall preference** (all-things-considered utility for decision-making). To mitigate positional bias, the judge randomly assigns systems to "System A" and "System B" for each prompt, then maps scores back.

## 3. Experimental Setup

**Dataset.** We use the Yelp Open Dataset, filtering to restaurant reviews and chunking into sentence-window snippets (window size 2). For domain transfer, we construct SaaS and ecommerce prompt sets that query the same restaurant corpus, testing whether the pipeline handles domain-mismatched retrieval gracefully.

**Prompts.** 20 founder prompts across three domains: restaurants (10), SaaS (5), ecommerce (5). Each prompt poses a specific operational question (e.g., "What frustrates guests during busy weekend brunch visits?").

**Evaluation.** Each prompt runs through both systems; the judge scores both outputs. We report per-prompt deltas and aggregate statistics. A human annotation study with 20 blinded A/B tasks provides a calibration signal against the automated judge.

## 4. Results

### 4.1 Aggregate Performance

| Metric | Baseline | Pipeline | Δ |
|---|---:|---:|---:|
| Relevance | 4.05 | 4.85 | +0.80 |
| Groundedness | 3.90 | 4.55 | +0.65 |
| Distinctiveness | 3.95 | 4.70 | +0.75 |
| Overall Preference | 3.50 | 4.65 | +1.15 |

The pipeline wins on 17/20 prompts. The largest gains appear in relevance (+0.80) and overall preference (+1.15), while the baseline wins on 3 restaurant prompts where simpler queries matched the evidence corpus well.

### 4.2 Domain Transfer

| Domain | n | Pipeline Wins | Δ Overall |
|---|---:|---:|---:|
| Ecommerce | 5 | 5/5 | +2.20 |
| SaaS | 5 | 5/5 | +1.40 |
| Restaurants | 10 | 7/10 | +0.50 |

The pipeline's advantage is largest on out-of-domain prompts (ecommerce, SaaS), where intent decomposition helps the system formulate better queries against a mismatched corpus. On in-domain restaurant prompts, the baseline's simpler retrieval sometimes suffices.

### 4.3 Failure Analysis

A failure taxonomy tags each evaluation with failure categories and severity (1–5). The baseline accumulates 11 of 13 `overall_preference_gap` tags and 7 of 9 `groundedness_gap` tags—its primary failure mode is producing ungrounded, generic claims. The pipeline's 3 regression cases (date-night-ambiance, loyalty-repeat-visits, local-discovery-standout) involve prompts where broader retrieval introduced noise that diluted focus.

### 4.4 Human–Judge Alignment

Two annotators scored the same 20 blinded A/B tasks. Interannotator exact agreement on winner selection is 40% (Cohen's κ = -0.19), indicating poor consistency on the pairwise preference label. Judge alignment is highly annotator-dependent: reviewer_01 matches the LLM judge on 70% of tasks (Cohen's κ = 0.34), while reviewer_02 matches on 35% (Cohen's κ = -0.29). Rather than providing a clean validation signal, this study suggests the current rubric and task framing still leave substantial room for subjective interpretation, so LLM-as-judge calibration claims should be treated cautiously.

## 5. Discussion

**Why does decomposition help?** The intent agent generates customer-voice retrieval queries that surface evidence the baseline's single query misses. The synthesis agent's grounding constraints (inline quotes, counter-signals) force specificity. The evidence verifier catches hallucinated attributions before they reach the final output.

**Limitations.** Our evaluation uses a single LLM as both generator and judge, risking self-preference bias. The retrieval benchmark uses only 2 labeled queries—too sparse for confident IR evaluation. Human annotation now includes two annotators, but interannotator agreement is low, indicating that the rubric and review protocol need refinement before they can serve as a strong calibration signal. Domain transfer uses restaurant data with non-restaurant prompts, which is a deliberate stress test but not a realistic deployment scenario.

**Connection to course concepts.** This work applies several core course themes: *agentic architecture* (decomposing a monolithic LLM call into specialized sub-agents), *tool use* (agents querying a retrieval index as an external tool), *evaluation methodology* (LLM-as-judge with debiasing, human annotation, failure taxonomy), and *grounding* (enforcing evidence traceability to reduce hallucination).

## 6. Conclusion

Multi-step grounded pipelines meaningfully outperform zero-shot baselines for extracting actionable insights from customer reviews. The key drivers are retrieval query planning and evidence-grounded synthesis. Our open-source implementation provides a reproducible framework for founder-oriented review mining and a reusable evaluation harness for comparing agentic system variants.

## References

- Ji, Z., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS*.
- Yao, S., et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR*.
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS*.
