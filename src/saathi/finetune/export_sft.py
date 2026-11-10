"""Build the LoRA SFT dataset (chat format) from the training split.

Input is data/qa/train_seen.jsonl, NOT train.jsonl: the question groups held out in
test_seen.jsonl must not be trained on, or the "does SFT add knowledge about seen schemes"
test would be contaminated. Test-scheme passages are never used.

Each training item gives two examples, rendered with the same prompt functions the arms use:
  closed-book  question -> answer, no citations       (teaches arm 3)
  RAG          retrieved passages + question -> cited answer   (teaches arm 4)
For RAG examples the gold passages are inserted into the retrieved set if retrieval missed
them, at a seeded random position, so the target answer is always supported by the context.
Unanswerable items become refusals in both formats.

A 5% slice of question groups is written to val.jsonl for validation loss.

Usage:
    uv run --extra models python -m saathi.finetune.export_sft
"""

import argparse
import json
import random
from pathlib import Path
from typing import Protocol

from saathi.arms.prompts import closed_book_messages, rag_messages
from saathi.data.split import split_by_group
from saathi.io import read_jsonl
from saathi.schemas import Answer, Passage, QAItem


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Passage]: ...


def target(item: QAItem, with_citations: bool) -> str:
    answer = Answer(
        text=item.answer,
        citations=item.citations if (with_citations and item.answerable) else [],
        refused=not item.answerable,
    )
    return answer.model_dump_json()


def context_with_gold(
    item: QAItem, retrieved: list[Passage], by_id: dict[str, Passage], rng: random.Random
) -> list[Passage]:
    """Make sure every gold passage is in the context, keeping its size fixed."""
    context = list(retrieved)
    ids = {p.passage_id for p in context}
    for gold_id in item.citations:
        if gold_id in ids:
            continue
        # replace the lowest-ranked non-gold passage, then move gold to a random position
        for i in range(len(context) - 1, -1, -1):
            if context[i].passage_id not in item.citations:
                context.pop(i)
                break
        context.insert(rng.randrange(len(context) + 1), by_id[gold_id])
        ids = {p.passage_id for p in context}
    return context


def examples_for(
    item: QAItem, retriever: Retriever, by_id: dict[str, Passage], rng: random.Random, k: int
) -> list[dict]:
    out = []
    system, messages = closed_book_messages(item.question)
    out.append(
        {
            "messages": [
                {"role": "system", "content": system},
                *messages,
                {"role": "assistant", "content": target(item, False)},
            ]
        }
    )
    context = context_with_gold(item, retriever.retrieve(item.question, k=k), by_id, rng)
    system, messages = rag_messages(item.question, context)
    out.append(
        {
            "messages": [
                {"role": "system", "content": system},
                *messages,
                {"role": "assistant", "content": target(item, True)},
            ]
        }
    )
    return out


def export(
    items: list[QAItem],
    passages: list[Passage],
    retriever: Retriever,
    out_dir: Path,
    k: int = 5,
    seed: int = 0,
) -> dict[str, int]:
    item_schemes = {it.scheme_id for it in items}
    if any(p.scheme_id not in item_schemes for p in passages):
        raise ValueError("passages must be restricted to the training schemes")
    by_id = {p.passage_id: p for p in passages}
    train_items, val_items = split_by_group(items, test_frac=0.05, seed=seed)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, subset in (("train", train_items), ("val", val_items)):
        rows = [ex for it in subset for ex in examples_for(it, retriever, by_id, rng, k)]
        rng.shuffle(rows)
        path = out_dir / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
        counts[name] = len(rows)
    return counts


def main() -> None:
    from saathi.retrieval.models import CrossEncoderReranker, E5Embedder
    from saathi.retrieval.search import HybridRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("data/qa/train_seen.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/sft"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    items = read_jsonl(args.train, QAItem)
    schemes = {it.scheme_id for it in items}
    passages = [p for p in read_jsonl(args.passages, Passage) if p.scheme_id in schemes]
    retriever = HybridRetriever(passages, E5Embedder(), CrossEncoderReranker())
    print(export(items, passages, retriever, args.out, k=args.k))


if __name__ == "__main__":
    main()
