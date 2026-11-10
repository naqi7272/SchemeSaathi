"""Run an arm over a test set, score it, and judge it. Every step is resumable.

    uv run saathi-eval predict configs/runs/arm2-rag-base-q8.yaml
    uv run saathi-eval score arm2-rag-base-q8
    uv run saathi-eval judge arm2-rag-base-q8 --backend openai_compat --judge-model <model>

The judge is OPTIONAL: code scores (score) are the main results. A judge model must not be
from the Qwen family (the model under test); its numbers are reported only after
`saathi-label-judge kappa` shows agreement with your labels (see eval/report.py).

Outputs go to results/runs/<run_id>/: config.yaml, predictions.jsonl, scores.jsonl,
judgments.jsonl. These are committed, so every reported number traces back to raw outputs.
"""

import argparse
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from saathi.arms.arms import Arm, ArmOutput
from saathi.eval.judge import SYSTEM as JUDGE_SYSTEM
from saathi.eval.judge import Judgment, build_prompt, judge, needs_judging
from saathi.eval.score import ItemScore, score_item
from saathi.io import append_jsonl, read_jsonl, write_jsonl
from saathi.llm.base import ChatModel, cost_usd
from saathi.schemas import Passage, QAItem

RUNS_DIR = Path("results/runs")
JUDGE_OUTPUT_TOKENS_EST = 800  # verdict + thinking; check against judgments.jsonl after a pilot


class RunConfig(BaseModel):
    run_id: str
    arm: Literal["closed_book", "rag", "agent"]
    backend: Literal["ollama", "openai_compat", "claude"]
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"  # Claude only
    base_url: str | None = None  # openai_compat only; default from SAATHI_LLM_BASE_URL
    k: int = 5
    candidates: int = 20
    rerank: bool = True
    translate_query: bool = False  # rag only: rewrite Hindi/Hinglish questions to English first
    test_file: str = "data/qa/test_gold.jsonl"
    passages_file: str = "data/processed/passages.jsonl"
    rules_dir: str = "data/rules"  # agent only


def load_run_config(run_id: str) -> RunConfig:
    path = RUNS_DIR / run_id / "config.yaml"
    return RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Prediction(BaseModel):
    run_id: str
    qa_id: str
    output: ArmOutput


def make_model(cfg: RunConfig) -> ChatModel:
    from saathi.llm.factory import make_chat

    kwargs = {"base_url": cfg.base_url} if cfg.backend == "openai_compat" and cfg.base_url else {}
    return make_chat(cfg.backend, cfg.model, effort=cfg.effort, **kwargs)


def make_arm(cfg: RunConfig) -> Arm:
    from saathi.arms.arms import ClosedBookArm, RagArm

    model = make_model(cfg)
    if cfg.arm == "closed_book":
        return ClosedBookArm(cfg.run_id, model)
    from saathi.retrieval.models import CrossEncoderReranker, E5Embedder
    from saathi.retrieval.search import HybridRetriever

    passages = read_jsonl(Path(cfg.passages_file), Passage)
    retriever = HybridRetriever(
        passages,
        E5Embedder(),
        CrossEncoderReranker() if cfg.rerank else None,
        candidates=cfg.candidates,
    )
    if cfg.arm == "rag":
        return RagArm(cfg.run_id, model, retriever, k=cfg.k, translate_query=cfg.translate_query)
    from saathi.agent.qa_adapter import AgentArm

    return AgentArm(cfg.run_id, model, retriever, Path(cfg.rules_dir), k=cfg.k)


def predict(run_id: str, arm: Arm, items: list[QAItem], run_dir: Path) -> int:
    """Answer every item not yet answered. Failures are printed and left for the next run."""
    path = run_dir / "predictions.jsonl"
    done = {p.qa_id for p in read_jsonl(path, Prediction)} if path.exists() else set()
    todo = [it for it in items if it.qa_id not in done]
    failed = 0
    for n, item in enumerate(todo, 1):
        try:
            output = arm.answer(item.question)
        except Exception as err:  # keep going; a rerun retries the missing items
            failed += 1
            print(f"[{n}/{len(todo)}] {item.qa_id} FAILED: {err!r}")
            continue
        append_jsonl(path, Prediction(run_id=run_id, qa_id=item.qa_id, output=output))
        print(f"[{n}/{len(todo)}] {item.qa_id} {output.latency_s:.1f}s")
    return failed


def score_run(run_dir: Path, items: list[QAItem]) -> list[ItemScore]:
    by_id = {it.qa_id: it for it in items}
    scores = []
    for pred in read_jsonl(run_dir / "predictions.jsonl", Prediction):
        out = pred.output
        scores.append(
            score_item(
                by_id[pred.qa_id],
                pred.run_id,
                out.answer,
                out.retrieved,
                latency_s=out.latency_s,
                input_tokens=sum(u.input_tokens for u in out.usage),
                output_tokens=sum(u.output_tokens for u in out.usage),
                cost_usd=sum(u.cost_usd for u in out.usage),
                parse_error=out.parse_error is not None,
            )
        )
    missing = set(by_id) - {s.qa_id for s in scores}
    if missing:
        print(f"WARNING: {len(missing)} items have no prediction; rerun predict first")
    write_jsonl(run_dir / "scores.jsonl", scores)
    return scores


def judge_todo(run_dir: Path, items: list[QAItem]) -> list[Prediction]:
    by_id = {it.qa_id: it for it in items}
    path = run_dir / "judgments.jsonl"
    done = {j.qa_id for j in read_jsonl(path, Judgment)} if path.exists() else set()
    return [
        p
        for p in read_jsonl(run_dir / "predictions.jsonl", Prediction)
        if p.qa_id not in done and needs_judging(by_id[p.qa_id], p.output.answer)
    ]


def judge_run(
    run_dir: Path,
    items: list[QAItem],
    passages: list[Passage],
    model: ChatModel,
    max_usd: float,
) -> float:
    by_id = {it.qa_id: it for it in items}
    by_pid = {p.passage_id: p for p in passages}
    spent = 0.0
    for pred in judge_todo(run_dir, items):
        if spent >= max_usd:
            print(f"budget reached (${spent:.2f}); rerun to continue")
            break
        item = by_id[pred.qa_id]
        gold = [by_pid[c] for c in item.citations]
        context = [by_pid[c] for c in pred.output.retrieved]
        j = judge(model, pred.run_id, item, pred.output.answer, gold, context)
        append_jsonl(run_dir / "judgments.jsonl", j)
        spent += j.usage.cost_usd
    return spent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("config", type=Path)
    for name in ("score", "judge"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
    j = sub.choices["judge"]
    j.add_argument(
        "--backend", choices=["openai_compat", "ollama", "claude"], default="openai_compat"
    )
    j.add_argument(
        "--judge-model",
        default=None,
        help="required; e.g. a Llama/Gemma/Mistral model (NOT Qwen). For openai_compat the "
        "default is SAATHI_LLM_MODEL. Claude (paid, optional extra): claude-fable-5-1",
    )
    j.add_argument("--allow-same-family", action="store_true", help="run a Qwen judge anyway")
    j.add_argument("--max-usd", type=float, default=5.0)
    j.add_argument("--estimate-only", action="store_true", help="Claude only: estimate cost")
    args = ap.parse_args()
    load_dotenv()

    if args.cmd == "predict":
        cfg = RunConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
        run_dir = RUNS_DIR / cfg.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
        )
        items = read_jsonl(Path(cfg.test_file), QAItem)
        failed = predict(cfg.run_id, make_arm(cfg), items, run_dir)
        print(f"done; {failed} failed (rerun to retry)")
        return

    run_dir = RUNS_DIR / args.run_id
    cfg = load_run_config(args.run_id)
    items = read_jsonl(Path(cfg.test_file), QAItem)
    if args.cmd == "score":
        scores = score_run(run_dir, items)
        print(f"scored {len(scores)} predictions")
        return

    import os

    from saathi.llm.factory import make_chat, same_family_warning

    judge_model = args.judge_model or (
        os.getenv("SAATHI_LLM_MODEL") if args.backend == "openai_compat" else None
    )
    if not judge_model:
        raise SystemExit("pass --judge-model (a non-Qwen model)")
    warning = same_family_warning(judge_model, "judge")
    if warning:
        print(warning)
        if "same family" in warning and not args.allow_same_family:
            raise SystemExit("refusing a same-family judge; pass --allow-same-family to override")
    passages = read_jsonl(Path(cfg.passages_file), Passage)
    model = make_chat(args.backend, judge_model, effort="medium")
    if args.backend != "claude":
        spent = judge_run(run_dir, items, passages, model, args.max_usd)
        print(f"judged with {judge_model} ({args.backend}); spent ${spent:.2f}")
        return
    todo = judge_todo(run_dir, items)
    by_id = {it.qa_id: it for it in items}
    by_pid = {p.passage_id: p for p in passages}
    input_tokens = sum(
        model.count_input_tokens(
            JUDGE_SYSTEM,
            [
                {
                    "role": "user",
                    "content": build_prompt(
                        by_id[t.qa_id],
                        t.output.answer,
                        [by_pid[c] for c in by_id[t.qa_id].citations],
                        [by_pid[c] for c in t.output.retrieved],
                    ),
                }
            ],
        )
        for t in todo
    )
    estimate = cost_usd(judge_model, input_tokens, len(todo) * JUDGE_OUTPUT_TOKENS_EST)
    print(f"{len(todo)} to judge, estimated ${estimate:.2f} (budget ${args.max_usd:.2f})")
    if args.estimate_only:
        return
    if estimate > args.max_usd:
        raise SystemExit("estimate exceeds budget; raise --max-usd only after approval")
    spent = judge_run(run_dir, items, passages, model, args.max_usd)
    print(f"spent ${spent:.2f}")


if __name__ == "__main__":
    main()
