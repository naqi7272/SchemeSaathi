"""Generate cited Q&A in English, Hindi, and Hinglish with an LLM (any backend).

On the $0 path this is used mainly for TRAINING data (SFT) for the training schemes, with
a free local model that is NOT Qwen (e.g. Llama or Gemma via Ollama); the test set comes
from hand-written questions. The model is recorded in data/qa/manifest.jsonl and a
same-family warning is printed. Claude remains possible with the `claude` extra.

One call per scheme. The model sees only that scheme's passages (with IDs) and returns
question groups: the same question and answer in all three languages, with the gold
passage IDs. Every group is validated in code; invalid groups are dropped and logged.

Spending is guarded twice: an estimate before the run (exact input tokens via
count_tokens, output tokens estimated) must be under --max-usd, and the run stops if the
actual spend reaches --max-usd.

Usage:
    uv run python -m saathi.gen.generate_qa --backend ollama --model llama3.1:8b --limit 2
    uv run python -m saathi.gen.generate_qa --backend claude --model claude-opus-5-5 \
        --max-usd 5 --estimate-only                                   # optional, paid
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from saathi.io import read_jsonl
from saathi.llm.base import ChatModel, Usage, cost_usd
from saathi.schemas import Language, Passage, QAItem, QuestionType, Scheme

# Question groups to request per scheme, by type.
MIX = {
    QuestionType.ELIGIBILITY: 2,
    QuestionType.BENEFITS: 2,
    QuestionType.DOCUMENTS: 2,
    QuestionType.HOW_TO_APPLY: 2,
    QuestionType.UNANSWERABLE: 2,
}
# Output tokens per group (three languages plus thinking). Deliberately pessimistic;
# compare against generation_log.jsonl after the pilot and adjust.
EST_OUTPUT_TOKENS_PER_GROUP = 1200

SYSTEM = """You write evaluation data for a question-answering system about Indian \
government schemes. Users ask in English, Hindi (Devanagari script), or Hinglish \
(Hindi-English code-mixed, written in Roman script, as people type on WhatsApp).

Rules:
- Use ONLY the passages provided. Never add facts from memory.
- Each group is ONE question asked naturally in three languages, with the answer in the \
same language as the question. Write each language as a native speaker would; do not \
translate word for word. Keep scheme names, amounts, and document names exact.
- Answers are short (1-4 sentences) and contain only what the cited passages support.
- "citations" lists every passage ID needed to answer, and no others.
- For "unanswerable" groups: ask something a real applicant might plausibly ask about \
THIS scheme that the passages do NOT answer (for example a helpline number, a deadline, \
a state-specific detail, or an amount that is not stated). The answer says politely that \
the provided information does not cover it. citations must be empty.
- Vary wording, user situations, and difficulty. Do not copy sentences from passages."""


class GenText(BaseModel):
    question: str
    answer: str


class GenGroup(BaseModel):
    qtype: QuestionType
    citations: list[str]
    en: GenText
    hi: GenText
    hinglish: GenText


class GenOutput(BaseModel):
    groups: list[GenGroup]


KEY_TERMS = re.compile(
    r"Rs\.?|₹|\bage\b|eligib|document|apply|enrol|premium|benefit|pension|cover|subsid",
    re.IGNORECASE,
)


def select_passages(passages: list[Passage], max_n: int | None) -> list[Passage]:
    """At most max_n passages, preferring ones with amounts, ages, eligibility, documents...
    Keeps document order. Small local models cannot read hundreds of passages at once."""
    if max_n is None or len(passages) <= max_n:
        return passages
    keyed = [p for p in passages if KEY_TERMS.search(p.text)]
    rest = [p for p in passages if not KEY_TERMS.search(p.text)]
    if len(keyed) > max_n:  # spread the choice over the whole document
        step = len(keyed) / max_n
        keyed = [keyed[int(i * step)] for i in range(max_n)]
    chosen = {p.passage_id for p in (keyed + rest)[:max_n]}
    return [p for p in passages if p.passage_id in chosen]


def build_user_prompt(scheme: Scheme, passages: list[Passage]) -> str:
    mix = ", ".join(f"{n} {qtype.value}" for qtype, n in MIX.items())
    body = "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in passages)
    return (
        f"Scheme: {scheme.name}\nSource: {scheme.source_url}\n\n"
        f"Passages:\n\n{body}\n\n"
        f"Write {sum(MIX.values())} question groups: {mix}."
    )


def to_items(
    scheme_id: str, groups: list[GenGroup], valid_ids: set[str]
) -> tuple[list[QAItem], list[str]]:
    """Validate groups and turn them into QAItems. Returns (items, reasons for dropped groups)."""
    items, dropped = [], []
    for i, g in enumerate(groups):
        group_id = f"{scheme_id}-{i:02d}"
        unknown = [c for c in g.citations if c not in valid_ids]
        if unknown:
            dropped.append(f"{group_id}: unknown citations {unknown}")
            continue
        answerable = g.qtype != QuestionType.UNANSWERABLE
        if answerable and not g.citations:
            dropped.append(f"{group_id}: answerable group without citations")
            continue
        if not answerable and g.citations:
            dropped.append(f"{group_id}: unanswerable group with citations")
            continue
        texts = {Language.EN: g.en, Language.HI: g.hi, Language.HINGLISH: g.hinglish}
        if any(not t.question.strip() or not t.answer.strip() for t in texts.values()):
            dropped.append(f"{group_id}: empty question or answer")
            continue
        for lang, t in texts.items():
            items.append(
                QAItem(
                    qa_id=f"{group_id}-{lang}",
                    group_id=group_id,
                    scheme_id=scheme_id,
                    language=lang,
                    qtype=g.qtype,
                    question=t.question.strip(),
                    answer=t.answer.strip(),
                    citations=sorted(set(g.citations)),
                    answerable=answerable,
                )
            )
    return items, dropped


def estimate_cost(model_name: str, input_tokens: int, n_schemes: int) -> float:
    est_output = n_schemes * sum(MIX.values()) * EST_OUTPUT_TOKENS_PER_GROUP
    return cost_usd(model_name, input_tokens, est_output)


class BudgetExceeded(RuntimeError):
    pass


def generate(
    model: ChatModel,
    schemes: list[Scheme],
    passages_by_scheme: dict[str, list[Passage]],
    out_dir: Path,
    max_usd: float,
    max_tokens: int = 32000,
) -> list[QAItem]:
    """Generate per scheme, skipping schemes already done (resumable). Stops at max_usd."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generation_log.jsonl"
    spent = 0.0
    all_items: list[QAItem] = []
    for scheme in schemes:
        out_path = out_dir / f"{scheme.scheme_id}.jsonl"
        if out_path.exists():
            all_items.extend(read_jsonl(out_path, QAItem))
            continue
        if spent > 0 and spent >= max_usd:
            raise BudgetExceeded(f"spent ${spent:.2f} >= budget ${max_usd:.2f}; stopping")
        passages = passages_by_scheme[scheme.scheme_id]
        result = model.chat(
            system=SYSTEM,
            messages=[{"role": "user", "content": build_user_prompt(scheme, passages)}],
            schema=GenOutput,
            max_tokens=max_tokens,
        )
        spent += result.usage.cost_usd
        groups = GenOutput.model_validate_json(result.text).groups
        items, dropped = to_items(scheme.scheme_id, groups, {p.passage_id for p in passages})
        origin = f"llm:{model.name}"
        items = [it.model_copy(update={"origin": origin}) for it in items]
        out_path.write_text("".join(it.model_dump_json() + "\n" for it in items), encoding="utf-8")
        _log(log_path, scheme.scheme_id, result.usage, len(groups), dropped)
        all_items.extend(items)
        print(
            f"{scheme.scheme_id}: {len(items)} items, dropped {len(dropped)}, "
            f"${result.usage.cost_usd:.3f} (total ${spent:.2f})"
        )
    return all_items


def _log(path: Path, scheme_id: str, usage: Usage, n_groups: int, dropped: list[str]) -> None:
    record = {"scheme_id": scheme_id, "groups": n_groups, "dropped": dropped, **usage.model_dump()}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    from saathi.data.provenance import MANIFEST, ManifestRecord, now, record
    from saathi.llm.factory import BACKENDS, make_chat, warn_same_family

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--schemes", type=Path, default=Path("data/processed/schemes.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/generated"))
    ap.add_argument("--backend", choices=BACKENDS, default="ollama")
    ap.add_argument("--model", required=True, help="generator model; prefer a non-Qwen model")
    ap.add_argument("--effort", default="high", help="Claude only")
    ap.add_argument("--max-tokens", type=int, default=8192, help="per scheme (Claude: 32000)")
    ap.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="passages shown per scheme (default: 20 for local backends, all for Claude)",
    )
    ap.add_argument("--only", nargs="+", metavar="SCHEME_ID", help="only these schemes")
    ap.add_argument(
        "--per-type", type=int, default=2, help="question groups per question type (default 2)"
    )
    ap.add_argument("--limit", type=int, help="only the first N schemes (pilot)")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    schemes = read_jsonl(args.schemes, Scheme)
    if args.only:
        unknown = set(args.only) - {s.scheme_id for s in schemes}
        if unknown:
            raise SystemExit(f"not in {args.schemes}: {sorted(unknown)}")
        schemes = [s for s in schemes if s.scheme_id in args.only]
    schemes = schemes[: args.limit]
    by_scheme: dict[str, list[Passage]] = defaultdict(list)
    for p in read_jsonl(args.passages, Passage):
        by_scheme[p.scheme_id].append(p)
    max_passages = args.max_passages or (None if args.backend == "claude" else 20)
    for sid in list(by_scheme):
        by_scheme[sid] = select_passages(by_scheme[sid], max_passages)
    for qtype in MIX:
        MIX[qtype] = args.per_type
    todo = [s for s in schemes if not (args.out / f"{s.scheme_id}.jsonl").exists()]

    family_warning = warn_same_family(args.model, "generator")
    model = make_chat(args.backend, args.model, effort=args.effort)
    if args.backend == "claude":
        input_tokens = sum(
            model.count_input_tokens(
                SYSTEM,
                [{"role": "user", "content": build_user_prompt(s, by_scheme[s.scheme_id])}],
            )
            for s in todo
        )
        estimate = estimate_cost(args.model, input_tokens, len(todo))
        print(
            f"{len(todo)} schemes to generate, {input_tokens} input tokens, "
            f"estimated ${estimate:.2f} (budget ${args.max_usd:.2f})"
        )
        if args.estimate_only:
            return
        if estimate > args.max_usd:
            raise SystemExit("estimate exceeds budget; raise --max-usd only after approval")
    else:
        print(f"{len(todo)} schemes to generate with {args.model} ({args.backend}, $0)")
    items = generate(model, schemes, by_scheme, args.out, args.max_usd, args.max_tokens)
    record(
        ManifestRecord(
            name=f"generated-{args.model}",
            kind="llm_generated",
            license="project-own",
            source=args.passages.as_posix(),
            provenance="synthetic",
            generator_model=f"{args.backend}:{args.model}",
            generator_family_warning=family_warning,
            out_file=args.out.as_posix(),
            n_rows_in=len(schemes),
            n_items_out=len(items),
            n_dropped=0,
            warnings=["synthetic questions: use for training; hand-verify before any test use"],
            created_at=now(),
        ),
        MANIFEST,
    )


if __name__ == "__main__":
    main()
