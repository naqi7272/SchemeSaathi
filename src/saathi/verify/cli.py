"""Hand-verification CLI for test questions.

Samples whole question groups. By default only groups in all three languages (en, hi,
hinglish), so language comparisons are paired; `--groups any` also accepts groups in one
or two languages (hand-written sets of 60-100 items).

Each decision is appended to the labels file immediately, so a session can be stopped and
resumed. The gold test set is built from the original items plus the latest label per item.

Usage:
    uv run saathi-verify --qa data/qa/test.jsonl --passages data/processed/passages.jsonl \
        --labels data/labels/verification.jsonl --n 150
    uv run saathi-verify ... --build-gold data/qa/test_gold.jsonl
    uv run saathi-verify --qa data/qa/test.jsonl ... --n 80 --groups any
"""

import argparse
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from saathi.io import append_jsonl, read_jsonl, write_jsonl
from saathi.schemas import Language, Passage, QAItem, QuestionType


class Status(StrEnum):
    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class VerificationLabel(BaseModel):
    """One reviewer decision. Stores the final field values so gold can be rebuilt from it."""

    qa_id: str
    status: Status
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    answerable: bool
    note: str = ""
    verified_at: datetime


def sample_groups(
    items: list[QAItem], n_items: int, seed: int, complete_only: bool = True
) -> list[QAItem]:
    """Sample whole question groups and return their items.

    complete_only=True (default): n_items // 3 groups that exist in all three languages, so
    language comparisons are paired.
    complete_only=False: any groups (e.g. hand-written questions in one or two languages),
    added whole until n_items is reached without going over. Use n=60-100 for a small set.
    """
    by_group: dict[str, list[QAItem]] = defaultdict(list)
    for it in items:
        by_group[it.group_id].append(it)
    lang_order = list(Language)
    if complete_only:
        all_langs = set(Language)
        pool = sorted(g for g, its in by_group.items() if {i.language for i in its} == all_langs)
        n_groups = n_items // len(all_langs)
        if n_groups > len(pool):
            raise ValueError(f"asked for {n_groups} groups, only {len(pool)} are complete")
        random.Random(seed).shuffle(pool)
        picked = pool[:n_groups]
    else:
        pool = sorted(by_group)
        total = sum(len(v) for v in by_group.values())
        if n_items > total:
            raise ValueError(f"asked for {n_items} items, only {total} exist")
        random.Random(seed).shuffle(pool)
        picked, count = [], 0
        for g in pool:
            if count + len(by_group[g]) <= n_items:
                picked.append(g)
                count += len(by_group[g])
            if count == n_items:
                break
    chosen = []
    for g in picked:
        chosen.extend(sorted(by_group[g], key=lambda i: lang_order.index(i.language)))
    return chosen


def latest_labels(labels: list[VerificationLabel]) -> dict[str, VerificationLabel]:
    """Later lines override earlier ones (re-reviewing an item appends a new label)."""
    return {lab.qa_id: lab for lab in labels}


def build_gold(items: list[QAItem], labels: list[VerificationLabel]) -> list[QAItem]:
    """Accepted and edited items, with edits applied. Rejected and unreviewed items are dropped."""
    latest = latest_labels(labels)
    gold = []
    for it in items:
        lab = latest.get(it.qa_id)
        if lab is None or lab.status == Status.REJECT:
            continue
        qtype = it.qtype
        if not lab.answerable:
            qtype = QuestionType.UNANSWERABLE
        elif qtype == QuestionType.UNANSWERABLE:
            raise ValueError(f"{it.qa_id}: marked answerable; regenerate it with a real qtype")
        gold.append(
            it.model_copy(
                update={
                    "question": lab.question,
                    "answer": lab.answer,
                    "citations": lab.citations,
                    "answerable": lab.answerable,
                    "qtype": qtype,
                }
            )
        )
    return [QAItem.model_validate(g.model_dump()) for g in gold]  # re-run validators


def _show(item: QAItem, passages: dict[str, Passage], out: Callable[[str], None]) -> None:
    out("=" * 72)
    out(f"{item.qa_id}  [{item.language}]  {item.qtype}  scheme={item.scheme_id}")
    out(f"Q: {item.question}")
    out(f"A: {item.answer}")
    if not item.citations:
        out("(no citations: check that the answer is a refusal and the docs really lack this)")
    for pid in item.citations:
        p = passages.get(pid)
        out(f"--- {pid}")
        out(p.text if p else "!! passage ID not found")


def _edit(item: QAItem, ask: Callable[[str], str], scheme_passages: list[str]) -> dict[str, object]:
    question = ask("question [Enter=keep]: ").strip() or item.question
    answer = ask("answer [Enter=keep]: ").strip() or item.answer
    raw_cites = ask("citations, comma-separated [Enter=keep, '-'=none]: ").strip()
    if raw_cites == "-":
        citations = []
    elif raw_cites:
        citations = [c.strip() for c in raw_cites.split(",") if c.strip()]
    else:
        citations = item.citations
    unknown = [c for c in citations if c not in scheme_passages]
    if unknown:
        raise ValueError(f"unknown passage IDs for this scheme: {unknown}")
    raw_ans = ask("answerable y/n [Enter=keep]: ").strip().lower()
    answerable = item.answerable if raw_ans == "" else raw_ans == "y"
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "answerable": answerable,
    }


def review(
    items: list[QAItem],
    passages: dict[str, Passage],
    labels_path: Path,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Interactive loop. Returns the number of labels written this session."""
    done = (
        set(latest_labels(read_jsonl(labels_path, VerificationLabel)))
        if labels_path.exists()
        else set()
    )
    todo = [it for it in items if it.qa_id not in done]
    out(f"{len(done)} already labelled, {len(todo)} to go")
    written = 0
    for n, item in enumerate(todo, 1):
        scheme_pids = sorted(p for p, v in passages.items() if v.scheme_id == item.scheme_id)
        _show(item, passages, out)
        while True:
            cmd = ask(f"[{n}/{len(todo)}] (a)ccept (e)dit (r)eject (v)iew scheme (s)kip (q)uit: ")
            cmd = cmd.strip().lower()
            if cmd == "q":
                return written
            if cmd == "s":
                break
            if cmd == "v":
                for pid in scheme_pids:
                    out(f"--- {pid}\n{passages[pid].text}")
                continue
            if cmd not in ("a", "e", "r"):
                continue
            fields = {
                "question": item.question,
                "answer": item.answer,
                "citations": item.citations,
                "answerable": item.answerable,
            }
            if cmd == "e":
                try:
                    fields = _edit(item, ask, scheme_pids)
                except ValueError as err:
                    out(f"!! {err}")
                    continue
            status = {"a": Status.ACCEPT, "e": Status.EDIT, "r": Status.REJECT}[cmd]
            note = ask("note [optional]: ").strip()
            append_jsonl(
                labels_path,
                VerificationLabel(
                    qa_id=item.qa_id,
                    status=status,
                    note=note,
                    verified_at=datetime.now(UTC),
                    **fields,
                ),
            )
            written += 1
            break
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--groups",
        choices=["complete", "any"],
        default="complete",
        help="complete: only groups in all 3 languages (n//3 groups); any: whole groups of "
        "any size up to n items (hand-written sets)",
    )
    ap.add_argument("--build-gold", type=Path, help="write gold set here instead of reviewing")
    args = ap.parse_args()

    items = sample_groups(
        read_jsonl(args.qa, QAItem), args.n, args.seed, complete_only=args.groups == "complete"
    )
    if args.build_gold:
        labels = read_jsonl(args.labels, VerificationLabel)
        gold = build_gold(items, labels)
        write_jsonl(args.build_gold, gold)
        print(f"wrote {len(gold)} gold items to {args.build_gold}")
        return
    passages = {p.passage_id: p for p in read_jsonl(args.passages, Passage)}
    review(items, passages, args.labels)


if __name__ == "__main__":
    main()
