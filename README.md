# SchemeSaathi Bench

A reproducible comparison of five ways to answer questions about Indian government schemes
in **English, Hindi, and Hinglish**, all built on the same small open model
(Qwen3-1.7B), with honest evaluation: cited test
questions, code-based scoring as the main metric, an optional judge validated against human
labels, 95% confidence intervals, and paired comparisons.

**Runs for $0:** no API key is needed anywhere on the default path. Everything runs on local
open models (Ollama) and the free Colab T4 GPU. A paid Claude backend is still available
as an optional extra.

| Arm | Approach | Framework |
|---|---|---|
| 1 | Prompting only | none |
| 2 | RAG: ChromaDB + BM25 hybrid search, RRF fusion (cross-encoder reranker optional, off in v1 for memory) | none |
| 3 | LoRA SFT (Unsloth + TRL, Colab T4) | none |
| 4 | RAG + the fine-tuned model | none |
| 5 | Eligibility agent: language detection, profile extraction, clarifying questions (human-in-the-loop interrupt), retriever tool, deterministic eligibility-rules tool, cited answer, self-check, SQLite memory, tracing. Runs on the fine-tuned local model (optionally also on Claude). | LangGraph |

> **Status (November 2026): v1 = arms 1, 2 and 5 (prompting, RAG, agent).** The question it answers:
> *does giving a small model the official documents (RAG) stop it from making up amounts, in
> English, Hindi and Hinglish?* Arm 5 (the LangGraph agent) is evaluated on 18
> eligibility conversations. Arms 3-4 (fine-tuning) are built and unit-tested but **not
> evaluated** in v1 (see [docs/DECISIONS.md](docs/DECISIONS.md)).
> The whole path runs for $0 on a laptop. No number in this repository is made up; the
> results section is filled only by `saathi.eval.report` from committed raw outputs.

## Why

Millions of people are eligible for central-government schemes they never claim, often
because the information is scattered, in English, and written in bureaucratic language. A
small model that runs on a laptop CPU would be cheap to deploy, but small models hallucinate
amounts and eligibility rules, and most evaluations ignore Hindi and Hinglish. This project
measures, per language, what each approach actually buys: retrieval, fine-tuning, both, or
an agent that asks before it answers and leaves eligibility decisions to deterministic code.

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it), and
[Ollama](https://ollama.com). Works on Windows (PowerShell or Git Bash), macOS, Linux.

```bash
uv sync --extra models                    # core + embedder/reranker (torch, CPU)
uv run pytest                             # offline: no network, no keys, no GPU
cp .env.example .env                      # optional; every key in it is optional
```

Optional extras: `demo` (Gradio), `translit` (romanized Hindi, pure Python), `open-data`
(Hugging Face datasets), `claude` (paid backend). Example: `uv sync --extra models --extra demo`.

Windows: run `setx PYTHONUTF8 1` once (then open a new terminal) so Hindi prints correctly.

### Model backends

| `--backend` | What | Cost | Needs |
|---|---|---|---|
| `ollama` | local models via Ollama (all five arms) | $0 | Ollama running |
| `openai_compat` | any OpenAI-compatible endpoint: Ollama's `/v1`, llama.cpp server, vLLM, free hosted tiers | $0 unless you set a price | `SAATHI_LLM_BASE_URL`, `SAATHI_LLM_MODEL`, optional `SAATHI_LLM_API_KEY` |
| `claude` | Claude API (optional) | paid | `uv sync --extra claude`, `ANTHROPIC_API_KEY` |

Free hosted endpoints have rate limits; the client retries with backoff and every long
command can be re-run to resume. If a hosted model has a price, set `SAATHI_LLM_PRICE_IN`/
`_OUT` so the `--max-usd` guards see it.

## Reproduce for $0

Any model used to write, judge or simulate is recorded in `data/qa/manifest.jsonl`, and
the tools warn (the judge refuses) if it is from the Qwen family (same family as the model
under test).

### 1. Corpus

myScheme's Terms of Use prohibit bots and scrapers without written permission, so the
fetcher refuses `myscheme.gov.in`. Documents come only from `data/sources.yaml`: official
ministry/scheme pages and guideline PDFs, or pages you save by hand.

```bash
uv run python -m saathi.data.ingest fetch --limit 5     # pilot: polite, robots.txt-aware
uv run python -m saathi.data.ingest process             # -> data/processed/{schemes,passages}.jsonl
```
Every document is recorded in `data/raw/manifest.jsonl` (URL, fetch time, sha256, terms).

### 2. Questions

**Test set: hand-written.** Copy `data/qa/handwritten_template.csv` to
`data/qa/handwritten.csv` and write questions with answers taken from one passage (same
`group_id` for translations of one question). Then:

```bash
uv run python -m saathi.data.ingest_handwritten data/qa/handwritten.csv   # validates every row
uv run saathi-verify --qa data/qa/handwritten.jsonl --passages data/processed/passages.jsonl \
    --labels data/labels/verification.jsonl --n 60 --groups any          # hand-verify
uv run saathi-verify --qa data/qa/handwritten.jsonl --passages data/processed/passages.jsonl \
    --labels data/labels/verification.jsonl --n 60 --groups any \
    --build-gold data/qa/test_gold.jsonl
```

**Training data (arms 3-4) for the other schemes**, from any of:
- an open dataset with a clear license (refused otherwise):
  `uv run python -m saathi.data.import_open_dataset configs/datasets/<name>.yaml`
  (copy `configs/datasets/TEMPLATE.yaml`);
- a free local model that is not Qwen:
  `ollama pull llama3.1:8b` then
  `uv run python -m saathi.gen.generate_qa --backend ollama --model llama3.1:8b`;
- more hand-written rows.

```bash
uv run python -m saathi.data.build_splits --test-frac 0.3 --seed 13 \
    --extra-test data/qa/handwritten.jsonl    # hand-written schemes always go to test
```
The split is by scheme and checks that no scheme, cited passage or question group appears
on both sides.

Optional, romanized Hindi variants (machine-made, kept in the same group, flagged):
```bash
uv sync --extra translit
uv run python -m saathi.data.romanize make --qa data/qa/handwritten.jsonl
uv run python -m saathi.data.romanize sample --qa data/qa/handwritten-romanized.jsonl --n 30
```

### 3. Eligibility rules (arm 5)

```bash
uv run python -m saathi.agent.rules_tool template --scheme pm-kisan   # then fill it in
uv run python -m saathi.agent.rules_tool validate
```
Check each rule against its cited passage, then set `verified: true`. Unverified rule files
are refused by the agent. (Optional: `saathi.agent.draft_rules --backend ... --model ...`
drafts a file for you to check.)

### 4. Fine-tuning (arms 3, 4)

```bash
uv run --extra models python -m saathi.finetune.export_sft   # from train_seen.jsonl only
git add data/sft && git commit -m "SFT data" && git push
```
Open `notebooks/finetune_colab.ipynb` in Colab (free T4) or in VS Code with the Colab
extension, check `BRANCH`, and run all cells. It trains, then converts both the fine-tuned
and the untouched base model to GGUF (F16, Q8_0, Q4_K_M) with the same llama.cpp checkout,
and writes `run_info.json`. Download the `gguf/` folder, then:

```bash
uv run python -m saathi.deploy.ollama_models gguf/   # qwen3-1.7b-{base,saathi}:{f16,q8_0,q4_k_m}
```

### 5. Evaluation

```bash
uv run --extra models saathi-eval predict configs/runs/arm1-prompt-base-q8.yaml   # each config
uv run saathi-eval score arm1-prompt-base-q8

uv run saathi-agent-eval scenarios --n 80                     # template openings, $0
uv run --extra models saathi-agent-eval run --run-id conv-sft-q8 --backend ollama \
    --model qwen3-1.7b-saathi:q8_0                              # scripted user simulator, $0
uv run saathi-agent-eval report --runs conv-sft-q8

uv run python -m saathi.eval.report \
    --runs arm1-prompt-base-q8 arm2-rag-base-q8 arm3-sft-q8 arm4-rag-sft-q8 \
           arm5-agent-sft-q8 quant-arm4-rag-sft-q4 quant-arm4-rag-sft-f16 \
    --compare arm2-rag-base-q8:arm1-prompt-base-q8 arm3-sft-q8:arm1-prompt-base-q8 \
              arm4-rag-sft-q8:arm2-rag-base-q8 arm4-rag-sft-q8:arm3-sft-q8 \
              arm5-agent-sft-q8:arm4-rag-sft-q8 quant-arm4-rag-sft-q4:arm4-rag-sft-q8 \
              quant-arm4-rag-sft-f16:arm4-rag-sft-q8
```

**Optional judge** for free-text correctness and faithfulness, with an open model from a
different family:
```bash
uv run saathi-eval judge arm1-prompt-base-q8 --backend ollama --judge-model llama3.1:8b
uv run saathi-label-judge label --n 100     # your labels, verdict hidden
uv run saathi-label-judge kappa             # -> results/judge_validation.json
```
The report shows judge metrics only if kappa >= 0.6 overall and >= 0.4 in every language;
otherwise they are left out with a note. Without a judge, the report still has every
code-scored metric.

All runs are resumable; raw predictions and scores in `results/runs/` are committed so every
number can be traced to model outputs.

**Optional, paid:** `configs/optional-paid/arm5-agent-claude.yaml`, `--backend claude` on
any command (needs `uv sync --extra claude` and `ANTHROPIC_API_KEY`). Claude commands print
a cost estimate first and stop at `--max-usd`.

### 6. Demo

```bash
uv run --extra models --extra demo python -m saathi.demo.app --backend ollama \
    --model qwen3-1.7b-saathi:q8_0
```
**Q&A** shows the cited answer with its source passages (URL and fetch date). **Agent chat**
shows every graph step as it runs; clarifying questions appear in the chat.

## Results (v1: arms 1 and 2)

Qwen3-1.7B (Q8_0, Ollama, CPU, non-thinking mode), 61 test questions on 5 schemes held out
of all training data (drafted with an AI assistant from the official passages, code-checked, not
hand-verified) (34 English, 13 Hindi, 14 Hinglish; 9 unanswerable). Code-scored; mean
with 95% cluster-bootstrap CI. Full tables: [results/report.md](results/report.md); raw
outputs: `results/runs/`. RAG = BM25 + multilingual dense retrieval (e5-small) fused with
RRF, top 5 passages, no reranker.

| Metric | Arm 1: prompting only | Arm 2: RAG | Paired difference |
|---|---|---|---|
| Amounts exactly right (`numbers_exact`), all | 2.4% [0.0, 8.1] | 52.4% [36.2, 68.9] | +50.0 pp [+33.3, +65.9], p_holm < 0.001 |
| &nbsp;&nbsp;English | 4.5% | 63.6% | +59.1 pp |
| &nbsp;&nbsp;Hinglish | 0.0% | 72.7% | +72.7 pp |
| &nbsp;&nbsp;Hindi (Devanagari) | 0.0% | 0.0% | 0 |
| Answerable questions refused (lower is better) | 53.7% | 3.7% | |
| Unanswerable questions refused (higher is better) | 71.4% | 42.9% | |
| Retrieval found a gold passage (recall@5), all / Hindi | n/a | 33.3% / 0.0% | |
| Latency per answer (s) | 17.9 | 115.1 | |

Findings (arms 1-2):
- **RAG mostly stops the model from inventing amounts** in English and Hinglish: exact-amount
  accuracy rises from ~2% to ~52% overall.
- **RAG does not help Hindi questions, because retrieval fails for them** (recall@5 = 0%):
  the documents are English, BM25 cannot match Devanagari words to English ones, and the
  small multilingual embedder does not bridge the gap. Hinglish works because it shares
  Roman-script words (amounts, "premium", "pension") with the documents. Arm 2b below
  tests query translation.
- **Trade-off:** with passages in front of it the model refuses less, including on questions
  the documents cannot answer (71% -> 43%).
- Small sets: per-language intervals are wide (9-22 items per cell). LLM-judge metrics are
  not reported (no validated judge). Arms 3-4 (fine-tuning) are built but not evaluated in v1.


### Arm 2b: RAG + query translation (follow-up experiment)

Hindi and Hinglish questions are first rewritten by the same model into an English search
query; the answer still uses the original question; English questions are unchanged. Same
61 questions as arm 2.

| Metric | Language | Arm 2 (RAG) | Arm 2b (RAG + translation) |
|---|---|---|---|
| Retrieval recall@5 | Hindi | 0.0% | 18.2% |
| | Hinglish | 42.3% | 34.6% |
| `numbers_exact` | Hindi | 0.0% | 22.2% |
| | Hinglish | 72.7% | 54.5% |
| | all | 52.4% | 52.4% |
| `number_recall` | Hindi | 7.1% | 56.0% |

Translation helps Hindi (the documents are English) but hurts Hinglish, which already
worked; the 1.7B model sometimes invented scheme names while translating (e.g. PM-KMY ->
"Pradhan Mantri Kisan Vikram Yojana"). None of the arm 2b vs arm 2 differences is
statistically significant (9-11 items per language; p_holm = 1). Next step: translate only
Devanagari questions, with a stronger translator.

### Arm 5: the LangGraph agent (eligibility conversations)

Same model (Qwen3-1.7B, base, no reranker). 18 conversations generated from the verified
rules for 3 schemes (apy, pm-kmy, pmsby): 6 per language, half eligible and half not. A
scripted user (templates) answers the agent's clarifying questions from a hidden profile;
the correct verdict comes from the rules engine. 1 conversation (Hindi) crashed on invalid
model output and is scored as a failure. Full table: [results/agent_report.md](results/agent_report.md).

| Metric | All (n=18) | English | Hindi | Hinglish |
|---|---|---|---|---|
| Final eligibility verdict correct (`task_success`) | 66.7% [44.4, 88.9] | 66.7% | 50.0% | 83.3% |
| Asked a question exactly when one was needed | 83.3% [66.7, 100] | 83.3% | 83.3% | 83.3% |
| Asked for a missing fact (when needed) | 72.7% [45.5, 100] | 66.7% | 80.0% | 66.7% |
| Asked again for a fact already given | 0.0% | 0.0% | 0.0% | 0.0% |
| Tool calls valid | 100% | 100% | 100% | 100% |
| Checked the right scheme | 94.4% [83.3, 100] | 100% | 83.3% | 100% |
| Cited the right rule passage | 55.6% [33.3, 77.8] | 66.7% | 33.3% | 66.7% |
| Profile facts extracted correctly | 81.7% [66.1, 94.4] | 90.0% | 61.7% | 93.3% |
| User turns / latency per conversation | 1.6 / 308 s | | | |

Findings: with eligibility decided by code, the 1.7B agent reached the correct verdict in
2 of 3 conversations, never re-asked for known facts and made no invalid tool calls. Its weak
points are the model's parts: extracting facts from Hindi messages (62%), citing the rule
passage (56%), and one Hindi answer that ran past the output limit. n=6 per language, so the
per-language differences are not conclusive.

## Cost

| Step | $0 path | Optional paid alternative |
|---|---|---|
| Test questions | drafted from official passages, checked by code (v1: not hand-verified) | none |
| Training questions | open dataset, or a local non-Qwen model via Ollama | Claude generation |
| Eligibility rules | hand-written, validated | Claude drafts, still hand-checked |
| Arms 1-5 | local Ollama | arm 5 on Claude |
| Fine-tuning | Colab free T4 | none |
| Agent conversations | template openings + scripted user | LLM openings/simulator |
| Judge | optional, local open model of another family | Claude judge |

The real costs on the $0 path are your time (writing and checking questions and rules) and
laptop CPU time (several hours for all runs; runs can be stopped and resumed).

## Design notes

- **Fairness across arms:** one prompt module renders every arm's prompt and the SFT data;
  arms 1-4 and the local agent use the same Ollama runtime and quantization; base and
  fine-tuned GGUFs come from the same converter.
- **No leakage:** split by scheme with checks on schemes, cited passages, and question groups
  (translations stay together). Schemes with hand-written test questions are forced into the
  test split. SFT trains on `train_seen.jsonl`, never on `test_seen.jsonl`.
- **Eligibility is code, not the model:** the agent extracts facts; `check_eligibility`
  applies hand-verified rules and cites their source passages.
- **Tool calls are structured outputs validated in code,** so they behave the same on every
  backend and tool-call correctness is scored without a judge.
- **Code-scored first:** the headline metric is `numbers_exact` (every amount, age and year
  in the gold answer appears in the model's answer), plus refusal, citation and retrieval
  metrics. The judge is optional and gated on agreement with human labels.
- **Provenance:** every question set has a manifest record (license, source, how it was
  made, which model). Unclear licenses and myscheme.gov.in data are refused.
- More in [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/EVAL_METRICS.md](docs/EVAL_METRICS.md).

## Limitations

- **Test questions AI-drafted, not hand-verified (v1):** the 61 test questions were drafted
  with an AI assistant (not from the Qwen family under test; the model is recorded in
  `data/qa/manifest.jsonl`) from the official passages, and checked only by code (every
  number in an answer appears in its cited passage; word overlap for English). They were
  not hand-verified, so some may be imperfect, especially Hindi/Hinglish wording.
- **Small test set:** 61 questions, so per-language confidence
  intervals are wide; the overall paired comparisons carry the conclusions.
- **Question provenance:** test questions are written by the author from the official
  documents, not collected from real users; real users ask messier questions. Training
  questions may be machine-generated; their source is in `data/qa/manifest.jsonl`.
- **Same-family bias risk:** if any generator or judge model is from the Qwen family it
  could favour Qwen-style answers; the tools warn about this and the judge refuses it, and
  the manifest records every model used.
- **`numbers_exact` is a partial check:** it confirms amounts, ages and years, not whether
  the rest of the answer is right. Answers without numbers rely on citations, refusal, and
  (optionally) the validated judge.
- **Machine-made Hinglish:** optional romanized variants are rule-based transliterations of
  Hindi, flagged `machine_made`. That is romanized Hindi, not natural Hinglish (which mixes
  in English words and spells freely); report it separately (`--exclude-machine-made`).
- **Scripted simulator:** in arm-5 conversations the user is played by templates. They are
  cleaner than real users, so the eval measures whether the agent asks for and uses the
  right facts, not robustness to messy replies.
- **Rules cover a subset:** only schemes whose criteria are AND-combinations over the profile
  fields are in the agent's rules; OR across fields and document requirements are out of scope.
- **Language detection** is a transparent heuristic (script + Hinglish word list).
- **Snapshot in time:** scheme rules change; every passage carries its fetch date.
- **Not advice:** a research benchmark, not a service for real applications.

## Repository layout

```
configs/runs/          one YAML per evaluation run (all free, local)
configs/optional-paid/ clearly marked paid examples (Claude)
configs/datasets/      open-dataset import mappings (TEMPLATE.yaml)
data/sources.yaml      the curated source list (only these are collected)
data/raw/              fetched documents + manifest.jsonl
data/processed/        schemes.jsonl, passages.jsonl
data/qa/               handwritten template, generated/, splits, test_gold, manifest.jsonl
data/rules/            eligibility rules (verified: true after hand-checking)
data/labels/           your verification and judge labels
data/sft/              fine-tuning data
docs/                  DECISIONS.md, EVAL_METRICS.md, AUDIT_FREE_PATH.md
notebooks/             finetune_colab.ipynb
src/saathi/            data, gen, llm, retrieval, arms, finetune, agent, eval, deploy, demo
results/               runs/, report.md, plots/
examples/              a fictional toy scheme for trying the CLI
tests/
```

## Development history

This repository is the cleaned, final version. The step-by-step development history
(pipeline, the switch to a $0 setup, each experiment) is kept in a separate private
repository; design decisions and their reasons are recorded in
[docs/DECISIONS.md](docs/DECISIONS.md).

## Data and licensing

Code: MIT. Scheme text belongs to its publishers; each passage records its source URL and
fetch date, and each source's terms are recorded in `data/sources.yaml` and the manifest.
Check that those terms allow redistribution before publishing `data/processed/`. Imported
datasets keep their own licenses, recorded in `data/qa/manifest.jsonl`.
