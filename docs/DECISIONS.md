# Design decisions

Recorded as they are made, so the write-up can cite them. Newest at the bottom.

## 2026-10-23

**Base model:** Qwen3-4B-Instruct-2507 (non-thinking), Qwen3-1.7B as fallback if CPU latency
is unworkable. To be re-confirmed against current Hugging Face releases before M3.

**Local inference:** arms 1-4 and the local run of arm 5 all use the same llama.cpp/Ollama
build and the same quantization (Q8 by default), so differences come from the approach,
not the runtime.

**Gold test set:** 150 hand-verified items = 50 question groups x 3 languages. Language
comparisons are paired (same question in each language). Trade-off: only 50 distinct
questions, so per-language 95% CIs will be wide; this is reported as a limitation.

**Splits:** primary split by scheme (about 70/30). Secondary split by question group within
training schemes, used only to test whether SFT adds knowledge about seen schemes.

**Judge:** a different, stronger Claude model than the one that generates the dataset.
Code-based scoring wherever possible; the judge is used only for free-text correctness,
faithfulness, and clarifying-question appropriateness, and is validated against human
labels (Cohen's kappa; below about 0.6 means revise the rubric before use). Generator and
judge share a vendor, so some shared bias remains; stated in limitations.

**Corpus scope:** about 60 central-government schemes, spread across agriculture,
education/scholarships, health, women and child, pensions/social security, housing,
employment/skills, and financial inclusion. Hand-coded eligibility rules for about 20
schemes with crisp, checkable criteria (age, income, land, occupation, category). State
schemes are out of scope for v1.

## 2026-10-23 (build)

**No myScheme scraping.** myScheme's Terms of Use prohibit bots/scrapers without written
permission, and its API needs a frontend key. The fetcher blocks `myscheme.gov.in`; sources
come from a curated list (official ministry/scheme sites, guideline PDFs, hand-saved pages).

**Newer base model exists, not adopted.** Qwen3.5-4B (2026-03) covers more languages, but on
a free T4 Unsloth forces float32 for it and it needs transformers 5.x and a vision loader.
Qwen3-4B-Instruct-2507 stays the default for a reliable notebook; switching is one setting.

**One GGUF converter for both models.** Base and fine-tuned models are converted with the
same llama.cpp checkout (commit recorded), so arms 1-2 vs 3-4 differ only in weights.

**SFT data from `train_seen.jsonl`.** Training on all of train would leak the seen-split
test questions.

**Tool calls as validated structured outputs,** not provider-native tool calling: identical
on Claude and Ollama, every call validated and logged, tool correctness scored by code.

**Self-check is code-only** (citations exist in context, verdict matches the rules, answer
script matches the user's language), with one revision.

**Eligibility rules are drafted by Claude and hand-verified;** unverified files are refused.
Not written from memory, because criteria change and memory may be wrong.

**Judge: claude-fable-5-1, generator: claude-opus-5-5.** Judge is the stronger model, as
decided earlier. It is also the largest cost item; Opus 5.5 as judge would cost ~40% as much.

**Claude-backed runs are not deterministic.** The API no longer accepts temperature; local
runs use temperature 0 and a fixed seed.

## 2026-11-09 ($0 path)

**No paid API on the default path.** The project had no budget for the Claude API, so every
step that needed it got a free replacement, and Claude became an optional extra (`claude`)
instead of a core dependency. Audit: docs/AUDIT_FREE_PATH.md. Reasons: anyone can reproduce
the benchmark without a key, and nothing in the results depends on a paid service.

**Test set = hand-written questions,** written from the official passages and validated in
code (passage exists and belongs to the scheme; every number in the answer is in the
passage; enough word overlap when the scripts match), then hand-verified with
`saathi-verify --groups any`. Smaller than the planned 150-item generated set, but every
item is human-written. Schemes with hand-written questions are forced into the test split.

**Training data** comes from an open dataset with a clear license, or from a free local
model that is not Qwen (Llama/Gemma via Ollama), or more hand-written rows. Each source is
recorded in data/qa/manifest.jsonl. Datasets with unclear licenses, or scraped from
myscheme.gov.in, are refused by code. Most scheme datasets found on Hugging Face (Nov 2026)
appear to be scraped from myScheme, so none is adopted yet.

**Headline metric is code-scored:** `numbers_exact` (every amount, age, year of the gold
answer appears in the answer), with refusal, citation and retrieval metrics. The judge is
optional, may be any backend but must not be Qwen (refused unless overridden), and its
metrics appear in the report only if results/judge_validation.json shows kappa >= 0.6
overall and >= 0.4 per language. This replaces the earlier Claude judge decision.

**Arm-5 conversations are scripted by default:** template openings and a template user
simulator in en/hi/Hinglish answer exactly the fields the agent asks for. Ground truth is
still the rules engine. Adds code metrics asked_missing_field, tool_calls_valid,
citation_correct. Trade-off: templates are cleaner than real users (stated in Limitations).

**Eligibility rules are written by hand** from a template and checked by
`saathi.agent.rules_tool validate`; drafting with an LLM is optional. Unverified files are
still refused by the agent.

**Romanized Hindi is optional and flagged.** Rule-based transliteration
(indic-transliteration, MIT, pure Python, installs on Windows and Colab) with casual
spelling rules; items are `machine_made` and share their source's group. Not natural
Hinglish; reported separately. AI4Bharat IndicXlit/IndicTrans2 were not used: they need
heavy dependencies (fairseq/torch models) that are hard to install on Windows.

## 2026-11-10 (smaller base model)

**Base model switched to Qwen3-1.7B** (the fallback named on 2026-10-23). On the author's
laptop the 4B model was too slow to run the benchmark on CPU. All arms, the SFT run and the
GGUF conversion use the same 1.7B model, so the comparison stays fair. Qwen3-1.7B is a
hybrid thinking model; every local call sets `think=False` (non-thinking mode) for speed
and so that structured outputs are not preceded by reasoning text. Trade-off: a smaller
model answers worse in absolute terms; the benchmark measures the differences between
approaches on the same model, which is still meaningful. Ollama models are named
`qwen3-1.7b-{base,saathi}:{q8_0,q4_k_m,f16}`.

## 2026-11-10 (scope of v1)

**v1 evaluates arms 1 and 2 only: does retrieval (RAG) stop a small model from making up
amounts, in English, Hindi and Hinglish?** Fine-tuning (arms 3-4) and the agent (arm 5) are
built and unit-tested but not evaluated in v1: they need a Colab fine-tuning run, training
data and many hours of CPU time, and the author's laptop is slow. They remain in the repo
as future work. Planned comparison: arm2-rag-base-q8 vs arm1-prompt-base-q8 on the same
test questions, overall and per language; headline metric `numbers_exact` (code-scored).

## 2026-11-10 (no reranker)

**Cross-encoder reranker turned off** (`rerank: false` in every RAG config; the agent eval
uses it only with `--rerank`). On the author's laptop the reranker (bge-reranker-v2-m3,
~2 GB in memory) next to the Ollama model ran out of memory (Windows "paging file is too
small") and made RAG answers take 3-6 minutes each. Retrieval is still hybrid: BM25 +
multilingual dense search (e5-small), fused with RRF. Trade-off: the top-5 passages may be
slightly less precise; retrieval_recall@k in the report measures retrieval quality directly.

## 2026-11-10 (arm 2b: query translation)

**New arm 2b = arm 2 + query translation.** v1 showed RAG gave no gain for Hindi because
retrieval failed (recall@5 = 0% for Devanagari questions against English documents). In arm
2b, Hindi and Hinglish questions are first rewritten by the same Qwen3-1.7B model into a
short English search query; retrieval uses that query, and the answer prompt still gets the
original question, so the answer stays in the user's language. English questions are not
rewritten, so they run exactly as in arm 2. The rewrite call's tokens and latency are counted.
Same model rather than a separate translator (e.g. IndicTrans2): no extra memory on the
laptop and no hard-to-install dependency. Comparison: arm2b vs arm2 on the same 61 questions.
