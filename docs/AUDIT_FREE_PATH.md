# Audit: what needs the paid Claude API, and the free replacement

> **Status 2026-11-09:** all rows below are implemented (phases 1-8). This file is the
> original Phase 0 audit, kept as written; paths may have moved (e.g. the Claude run
> config is now `configs/optional-paid/arm5-agent-claude.yaml`).

Phase 0 of the "$0 reproduction" plan. Audit only: no code was changed. Written 2026-11-09
from the code as it was before the $0 work started.

## 1. Where the paid API is used

All Claude calls go through one class, `ClaudeChat` in `src/saathi/llm/claude.py`. It is
imported lazily (inside `main()` or a branch), so most modules work without it. The free
local backend `OllamaChat` (`src/saathi/llm/local.py`) already implements the same
`ChatModel` interface, which makes swapping cheap.

| # | File | Function / command | Why it needs Claude today | Proposed free replacement (phase) |
|---|---|---|---|---|
| 1 | `src/saathi/gen/generate_qa.py` | `main()` -> `python -m saathi.gen.generate_qa` | Hard-codes `ClaudeChat`, default `--model claude-opus-5-5`; uses `count_input_tokens` (Claude-only endpoint) for the cost estimate | Primary: open dataset importer (P2) + hand-written questions (P3). Optional: `--backend openai_compat` with a non-Qwen open model; estimate skipped / $0 for free backends (P1) |
| 2 | `src/saathi/agent/draft_rules.py` | `main()` -> `python -m saathi.agent.draft_rules` | Hard-codes `ClaudeChat(effort="high")` | Hand-written rule template + validator (P7); drafting becomes optional with a `--backend` flag |
| 3 | `src/saathi/eval/run.py` | `saathi-eval judge` (`main()`, judge branch) | Hard-codes `ClaudeChat`, default `--judge-model claude-fable-5-1`, Claude token counting for the estimate | Judge becomes optional (P5). Main results are code-scored. Optional judge via OpenAI-compatible backend, different family from Qwen3, still gated by kappa >= 0.6 |
| 4 | `src/saathi/eval/run.py` | `RunConfig.backend: Literal["ollama","claude"]`, `make_model()` | Claude is one of two backends (fine), but there is no third, free hosted option | Add `openai_compat` backend (P1) |
| 5 | `src/saathi/eval/report.py` | `item_metrics()` | **accuracy** is `None` for every answered item unless a judge verdict exists, so without a judge the headline metric is empty | Add code-scored accuracy proxies as headline: number/amount/age match, citation correctness, abstention (P5) |
| 6 | `src/saathi/agent/conversations.py` | `saathi-agent-eval scenarios` (`make_scenarios` -> `write_opening`) | Opening user message written by `ClaudeChat(effort="low")` | Deterministic template openings in en / hi / Hinglish (P6). Profile sampling is already deterministic (`sample_profile`, `plan_scenario`) |
| 7 | `src/saathi/agent/conversations.py` | `saathi-agent-eval run` (`Simulator`) | User simulator is `ClaudeChat(model=--sim-model)`, default `claude-opus-5-5`, even when the agent itself runs on Ollama | Scripted template simulator; `--simulator scripted|llm`, default scripted (P6) |
| 8 | `configs/runs/arm5-agent-claude.yaml` | `saathi-eval predict` | Arm 5 on Claude | Keep as an optional example, renamed/clearly marked; default arm 5 is `arm5-agent-sft-q8.yaml` (Ollama) (P8) |
| 9 | `src/saathi/demo/app.py` | `--backend ollama|claude` | Claude is optional; default is already `ollama` | No change needed (maybe add `openai_compat`) |
| 10 | `src/saathi/llm/claude.py` | `ClaudeChat` | The paid client itself | Keep, untouched, behind an optional extra (P1) |

Not paid, but worth knowing:

| File | What | Note |
|---|---|---|
| `src/saathi/agent/tracing.py` | LangSmith / Langfuse | Already optional: only on if keys are set. Both have free tiers; nothing needed. |
| `src/saathi/retrieval/models.py` | `intfloat/multilingual-e5-small` (MIT), `BAAI/bge-reranker-v2-m3` (Apache-2.0) | Free; downloaded from Hugging Face once. Reranker is 568M params: slow on CPU but works. |
| `notebooks/finetune_colab.ipynb` | Unsloth + TRL on Colab T4 | Free tier. No Claude. Note: its `BRANCH` setting pointed at an old development branch (since fixed). |

## 2. Every place that reads `ANTHROPIC_API_KEY`

- **No code reads it by name.** `anthropic.Anthropic()` in `src/saathi/llm/claude.py:39`
  reads it implicitly from the environment.
- `.env.example:2` lists it (blank). `load_dotenv()` in each CLI loads `.env`.
- `README.md` Setup section tells users to fill it in.
- `anthropic>=2.1.0` is a **core** dependency in `pyproject.toml`, so every install pulls
  the SDK even when it is never used. Proposal: move it to an optional extra `claude`.

## 3. The `--max-usd` budget guard

| File | Flag / default | Behaviour |
|---|---|---|
| `src/saathi/gen/generate_qa.py` | `--max-usd 5.0`, `--estimate-only` | Pre-run estimate via Claude `count_tokens`; refuses if over budget; stops mid-run (`BudgetExceeded`) |
| `src/saathi/agent/draft_rules.py` | `--max-usd 1.0` | Stops when spend reaches budget |
| `src/saathi/eval/run.py` (judge) | `--max-usd 5.0`, `--estimate-only` | Pre-run estimate via Claude `count_tokens`; stops mid-run |
| `src/saathi/agent/conversations.py` (scenarios) | `--max-usd 2.0` | Stops mid-run |
| `src/saathi/agent/conversations.py` (run) | `--max-usd 5.0` | Stops mid-run; counts agent + simulator spend |

Cost comes from `cost_usd()` in `src/saathi/llm/base.py`: known Claude models use the price
table; unknown `claude*` names raise; **any other name is $0**. That rule already treats
free backends as $0, but it would also silently treat a *paid* non-Claude model as free.
Proposal (P1): price lookup per backend, free backends explicitly marked $0, and the
pre-run estimate skipped (not faked) when the backend has no token-count endpoint.

## 4. Default configs and CLI defaults that name a Claude model

| Where | Value |
|---|---|
| `configs/runs/arm5-agent-claude.yaml` | `backend: claude`, `model: claude-opus-5-5` |
| `generate_qa.py --model` | `claude-opus-5-5` |
| `eval/run.py judge --judge-model` | `claude-fable-5-1` |
| `conversations.py run --sim-model` | `claude-opus-5-5` |
| `conversations.py scenarios` | `ClaudeChat(effort="low")` (default model `claude-opus-5-5`) |
| `draft_rules.py` | `ClaudeChat(effort="high")` (default model `claude-opus-5-5`) |
| `llm/claude.py ClaudeChat.__init__` | `model="claude-opus-5-5"` |
| `llm/base.py PRICES_PER_MTOK` | price table (fine to keep) |
| `README.md` | Reproduce section, cost table, Limitations ("same vendor for data and judge") |
| `docs/DECISIONS.md`, `docs/EVAL_METRICS.md` | judge = `claude-fable-5-1`, generator = `claude-opus-5-5` |

The other 9 run configs (arms 1-4, arm 5 local, quantisation, seen-split) already use
`backend: ollama` and need no key.

## 5. Other findings (problems you should know about)

1. **No processed corpus is committed.** `data/raw/` and `data/processed/` contain only
   `.gitkeep`; there is no `data/raw/manifest.jsonl`. Only two hand-saved PDFs are in
   `data/manual/`. So "collection verified on the first schemes" happened on your machine
   but its output is not in the repo. A fresh clone must run `ingest fetch` + `process`
   first (needs network to government sites). Every later phase depends on
   `data/processed/passages.jsonl`.
2. **`data/rules/` is empty.** No verified eligibility rules exist yet, so arm 5's
   scenario generator (P6) has nothing to sample from until rules are written (P7).
3. **Hand-verification assumes 3 languages per group.** `saathi-verify`'s
   `sample_groups()` only picks groups that exist in en, hi and Hinglish and takes
   `n // 3` groups. Hand-written English-only questions would be skipped. Phase 3 must
   relax this.
4. **Headline accuracy depends on the judge** (row 5 above). Without a judge the README's
   main table would be empty, not wrong; P5 fixes this.
5. **Same-family risk:** if an open model is used to generate or judge, it must not be
   Qwen (the model under test). Many free hosted models are Qwen; the manifest must record
   the model and warn (rule 9).
6. **Notebook branch** is hard-coded to an old branch (see table above).
7. **Python version:** `requires-python = ">=3.12,<3.13"`; the system Python here is 3.13,
   so `uv` installs 3.12. Fine, but worth noting for Colab (currently 3.12).

## 6. Baseline test run

Run on 2026-11-09, Linux, Python 3.12 (installed by uv), **no `ANTHROPIC_API_KEY`, no GPU**:

```bash
uv sync
env -u ANTHROPIC_API_KEY uv run pytest -q
```

Result: **216 passed, 1 skipped** in about 2.5 s. The skip is `tests/test_demo.py:86`
(`gradio` is in the optional `demo` extra and was not installed). `ruff check` and
`ruff format --check` both pass. So the existing tests already need no key, no network and
no GPU; the new phases must keep it that way.
