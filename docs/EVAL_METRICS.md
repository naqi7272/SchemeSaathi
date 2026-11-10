# Evaluation metrics (DRAFT, awaiting sign-off)

> **Update 2026-11-09 ($0 path):** the gold set is hand-written (60-100 items, groups may
> have 1-3 languages); the headline metric is the code-scored `numbers_exact`; the judge is
> optional, must not be Qwen, and is shown only when validated. See "Changes for the $0
> path" at the end. The sections below describe the original plan where not overridden.

Nothing has been run. These definitions are implemented in `src/saathi/eval/score.py`,
`judge.py`, `report.py`, and `agent/conversations.py`, with tests for every scoring function.
Please review and approve or change them before any evaluation run.

## Test sets

| Set | What | Used for |
|---|---|---|
| `test_gold.jsonl` | 150 hand-verified items = 50 question groups x 3 languages, unseen schemes | **headline numbers** |
| `test.jsonl` | all generated items for unseen schemes (not verified) | secondary, larger-n check |
| `test_seen.jsonl` | held-out question groups from training schemes | does SFT add knowledge? (arms 1 vs 3 only) |
| `scenarios.jsonl` | 60-100 multi-turn eligibility conversations | arm 5 only |

## Q&A metrics (all arms), per item

"N/A" items are excluded from that metric's average (never counted as 0 or 1).

| Metric | Definition | Scored by | N/A when |
|---|---|---|---|
| **accuracy** | answerable item: judge says `correct` against the gold answer. An answerable item the model refused counts as wrong. `partially_correct` counts as wrong. | LLM judge + code | unanswerable item |
| **faithfulness** | judge says every claim is supported by the source passages: the retrieved context for RAG arms, the gold passages for closed-book arms (i.e. supported by the official source) | LLM judge | unanswerable, or refused |
| **refused_unanswerable** | unanswerable item: the model set `refused=true` | code | answerable item |
| **false_refusal** | answerable item: the model set `refused=true` | code | unanswerable item |
| **citation_precision** | share of cited passage IDs that are gold | code | unanswerable, refused, or nothing cited |
| **citation_recall** | share of gold passage IDs that were cited | code | unanswerable or refused |
| **citations_grounded** | every cited ID was actually in the context shown (catches invented IDs) | code | nothing cited |
| **retrieval_recall@k** | share of gold passages in the retrieved top-k | code | no retrieval, or unanswerable |
| **number_recall** | share of numbers in the gold answer (amounts, ages, years) that appear in the answer; Devanagari digits and Indian comma grouping normalised | code | gold answer has no numbers, refused, unanswerable |
| **script_match** | Hindi question answered mostly in Devanagari; English/Hinglish mostly in Latin script (majority of letters) | code | answer has no letters |
| **parse_error** | model output was not valid JSON for the answer schema (counted as a wrong, non-refused answer) | code | never |
| **latency_s** | wall clock per question, including retrieval | code | never |
| **output_tokens, cost_usd** | from API/Ollama usage; local models cost 0 | code | never |

Closed-book arms (1, 3) have nothing to cite, so citation metrics are N/A for them, not 0.

## Judge

- Model: `claude-fable-5-1` (stronger than the generator `claude-opus-5-5`), effort medium.
- Sees: question, gold answer, gold passages, source passages, answer. Labels:
  correctness `correct | partially_correct | incorrect`, faithful `yes | no`.
- **Validation:** you label about 100 judged answers (balanced by language) without seeing the
  verdict (`saathi-label-judge label`). Cohen's kappa per dimension, overall and per language.
- **Decision rule:** a judge dimension is reported only if kappa >= 0.6 overall and no
  language below 0.4; otherwise the rubric is revised and re-validated on new items.

## Agent conversations (arm 5)

Ground truth from code: hidden profiles are sampled from verified rules (eligible, or exactly
one rule violated); the opening message reveals a random subset of facts.

| Metric | Definition |
|---|---|
| **task_success** | final `eligibility_verdict` equals the rules' verdict on the hidden profile |
| **clarify_appropriate** | asked a clarifying question if and only if the revealed facts left the rules undecided |
| **asked_known_field** | asked for a fact the user had already stated (lower is better) |
| **eligibility_tool_correct** | first `check_eligibility` call used the right scheme |
| **invalid_tool_calls** | tool calls rejected by validation (e.g. unknown scheme ID) |
| **profile_accuracy** | share of hidden-profile facts the agent extracted correctly (2% / 0.05 tolerance for converted units) |
| **user_turns** | user messages until the final answer (cap 4) |
| **latency_s, cost_usd** | per conversation |

## Statistics

- **95% CIs:** percentile bootstrap, 10,000 resamples, **resampling whole question groups**
  (the three translations of a question are correlated). Conversations are independent.
- **Paired comparisons** (same items, two runs): cluster-bootstrap CI for the difference and a
  two-sided bootstrap p-value (primary); exact McNemar for binary metrics (secondary, ignores
  clustering). **Holm** correction across all comparisons in a report.
- Planned comparisons: arm2 vs arm1 (RAG effect), arm3 vs arm1 (SFT effect), arm4 vs arm2
  (SFT on top of RAG), arm4 vs arm3 (RAG on top of SFT), arm5 vs arm4, Q4 vs Q8 vs F16, and
  seen-split arm3 vs arm1. Each for accuracy and refusal, overall and per language.

## Open questions for sign-off

1. **partially_correct = wrong?** Strict, and simple to explain. The alternative is 0.5 credit.
2. **Bootstrap at 0% or 100%:** gives a zero-width CI, which overstates certainty with 50
   items per language. Option: report Wilson intervals for proportions instead (no clustering).
3. **Gold set size:** 50 groups means per-language CIs of roughly ±13 points. Comparisons
   within a language will often be inconclusive; the overall (n=150, paired) numbers carry
   the conclusions.
4. **Faithfulness for closed-book arms** is judged against the gold passages, which is really
   "factually consistent with the official source". OK to report it in the same column?
5. **Run-to-run variance:** local models run greedy (temperature 0, fixed seed); the Claude API
   no longer accepts temperature, so Claude-backed runs vary. Run the Claude agent twice and
   report both?

## Changes for the $0 path (2026-11-09)

| Metric | Definition | Scored by | N/A when |
|---|---|---|---|
| **numbers_exact** (headline) | answerable item: every number in the gold answer (amounts, ages, years; Devanagari digits and Indian grouping normalised) appears in the answer. Refusing an answerable item whose gold answer has numbers counts as 0. | code | unanswerable, or gold answer has no numbers |
| accuracy, faithful | as above (judge) | optional LLM judge | as above; **hidden from the report unless kappa >= 0.6 overall and >= 0.4 per language** |

Judge model: any backend; must not be from the Qwen family (the model under test). The
judge command refuses a Qwen model unless `--allow-same-family` is given.

Agent conversations: openings are templates by default and the user is a scripted
simulator (`--simulator scripted`). New code-scored metrics:

| Metric | Definition |
|---|---|
| **asked_missing_field** | when a clarifying question was needed: the agent asked for at least one rule field the opening did not state (N/A when not needed) |
| **tool_calls_valid** | no tool call was rejected by validation |
| **citation_correct** | the final answer cites only passages of the right scheme, including at least one of the rule passages |

Gold set: hand-written, so groups may exist in one, two or three languages. Per-language
numbers are then not fully paired; paired comparisons between runs (same items) are
unaffected. Machine-made romanized items can be left out with `--exclude-machine-made`.
