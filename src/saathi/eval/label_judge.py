"""Validate the LLM judge against human labels.

You label a stratified sample of judged answers WITHOUT seeing the judge's verdict; then
Cohen's kappa is computed per dimension (correctness: 3 labels; faithful: yes/no), overall
and by language. Decision rule (docs/EVAL_METRICS.md): a judge dimension is reported only
if kappa >= 0.6; otherwise the rubric is revised and re-validated.

    uv run saathi-label-judge label --runs arm1-... arm2-... --n 100
    uv run saathi-label-judge kappa
"""

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from saathi.eval.judge import Judgment
from saathi.eval.run import RUNS_DIR, Prediction, load_run_config
from saathi.eval.stats import cohen_kappa
from saathi.io import append_jsonl, read_jsonl
from saathi.schemas import Passage, QAItem

LABELS = Path("data/labels/judge_labels.jsonl")


class HumanLabel(BaseModel):
    run_id: str
    qa_id: str
    correctness: str  # correct | partially_correct | incorrect
    faithful: str  # yes | no


def sample(
    judgments: list[Judgment], items: dict[str, QAItem], n: int, seed: int
) -> list[Judgment]:
    """Round-robin over (language) strata so each language gets about n/3 items."""
    strata: dict[str, list[Judgment]] = defaultdict(list)
    for j in sorted(judgments, key=lambda j: (j.run_id, j.qa_id)):
        strata[items[j.qa_id].language].append(j)
    rng = random.Random(seed)
    for group in strata.values():
        rng.shuffle(group)
    picked, keys = [], sorted(strata)
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    return picked


def label(
    todo: list[Judgment],
    items: dict[str, QAItem],
    preds: dict[tuple[str, str], Prediction],
    passages: dict[str, Passage],
    path: Path,
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    done = {(h.run_id, h.qa_id) for h in read_jsonl(path, HumanLabel)} if path.exists() else set()
    todo = [j for j in todo if (j.run_id, j.qa_id) not in done]
    correctness = {"c": "correct", "p": "partially_correct", "i": "incorrect"}
    written = 0
    for n, j in enumerate(todo, 1):
        item, pred = items[j.qa_id], preds[(j.run_id, j.qa_id)]
        sources = pred.output.retrieved or item.citations
        out("=" * 72)
        out(f"[{n}/{len(todo)}] {j.run_id} {item.qa_id} [{item.language}]")
        out(f"Q: {item.question}\nGOLD: {item.answer}")
        for pid in sources:
            out(f"--- {pid}\n{passages[pid].text}")
        out(f">>> ANSWER: {pred.output.answer.text}")
        c = ""
        while c not in correctness:
            c = ask("correctness (c)orrect (p)artial (i)ncorrect, or (q)uit: ").strip().lower()
            if c == "q":
                return written
        f = ""
        while f not in ("y", "n"):
            f = ask("faithful to the passages shown? (y/n): ").strip().lower()
        append_jsonl(
            path,
            HumanLabel(
                run_id=j.run_id,
                qa_id=j.qa_id,
                correctness=correctness[c],
                faithful="yes" if f == "y" else "no",
            ),
        )
        written += 1
    return written


def agreement(
    humans: list[HumanLabel], judgments: dict[tuple[str, str], Judgment], items: dict[str, QAItem]
) -> dict:
    """Kappa per dimension, overall and by language."""
    result: dict = {}
    subsets: dict[str, list[HumanLabel]] = {"all": humans}
    for h in humans:
        subsets.setdefault(str(items[h.qa_id].language), []).append(h)
    for name, hs in subsets.items():
        js = [judgments[(h.run_id, h.qa_id)].verdict for h in hs]
        result[name] = {
            "n": len(hs),
            "kappa_correctness": cohen_kappa(
                [h.correctness for h in hs], [v.correctness for v in js]
            ),
            "kappa_faithful": cohen_kappa([h.faithful for h in hs], [v.faithful for v in js]),
        }
    return result


def _load(run_ids: list[str]):
    judgments, preds, items, passages = {}, {}, {}, {}
    for run_id in run_ids:
        run_dir = RUNS_DIR / run_id
        cfg = load_run_config(run_id)
        items.update({it.qa_id: it for it in read_jsonl(Path(cfg.test_file), QAItem)})
        passages.update({p.passage_id: p for p in read_jsonl(Path(cfg.passages_file), Passage)})
        for j in read_jsonl(run_dir / "judgments.jsonl", Judgment):
            judgments[(j.run_id, j.qa_id)] = j
        for p in read_jsonl(run_dir / "predictions.jsonl", Prediction):
            preds[(p.run_id, p.qa_id)] = p
    return judgments, preds, items, passages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["label", "kappa"])
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_ids = args.runs or sorted(
        p.name for p in RUNS_DIR.iterdir() if (p / "judgments.jsonl").exists()
    )
    judgments, preds, items, passages = _load(run_ids)
    if args.cmd == "label":
        todo = sample(list(judgments.values()), items, args.n, args.seed)
        label(todo, items, preds, passages, LABELS)
        return
    humans = read_jsonl(LABELS, HumanLabel)
    result = agreement(humans, judgments, items)
    out = Path("results/judge_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
