import pytest

from saathi.arms.arms import ArmOutput
from saathi.eval.judge import Judgment, Verdict
from saathi.eval.label_judge import HumanLabel, agreement, label, sample
from saathi.eval.run import Prediction
from saathi.io import read_jsonl
from saathi.llm.base import Usage
from saathi.schemas import Answer, Language
from tests.conftest import make_item, make_passage

USAGE = Usage(
    requested_model="j", served_model="j", input_tokens=1, output_tokens=1, latency_s=0, cost_usd=0
)


def judgment(qa_id, correctness="correct", faithful="yes", run_id="r1"):
    v = Verdict(
        correctness=correctness,
        faithful=faithful,
        unsupported_claims=["SECRET-CLAIM"],
        reason="SECRET-REASON",
    )
    return Judgment(run_id=run_id, qa_id=qa_id, verdict=v, judge_model="j", usage=USAGE)


ITEMS = {
    it.qa_id: it
    for it in (make_item(f"s{s}", g, lang) for s in range(3) for g in range(3) for lang in Language)
}


def test_sample_balances_languages_and_is_deterministic():
    js = [judgment(q) for q in ITEMS]
    picked = sample(js, ITEMS, n=9, seed=1)
    langs = [ITEMS[j.qa_id].language for j in picked]
    assert sorted(langs.count(lang) for lang in Language) == [3, 3, 3]
    assert [j.qa_id for j in picked] == [j.qa_id for j in sample(js, ITEMS, n=9, seed=1)]
    assert len(sample(js, ITEMS, n=500, seed=1)) == len(js)


def test_label_loop_writes_and_resumes(tmp_path):
    qa = list(ITEMS)[:2]
    js = [judgment(q) for q in qa]
    preds = {
        ("r1", q): Prediction(
            run_id="r1",
            qa_id=q,
            output=ArmOutput(answer=Answer(text="ans"), usage=[USAGE], latency_s=0),
        )
        for q in qa
    }
    passages = {"s0:benefits:0": make_passage("s0")}
    answers = iter(["x", "c", "maybe", "y", "q"])  # invalid inputs are re-asked
    path = tmp_path / "labels.jsonl"
    out = []
    assert label(js, ITEMS, preds, passages, path, ask=lambda _: next(answers), out=out.append) == 1
    assert not any("SECRET" in line for line in out)  # the judge's verdict is never shown
    answers = iter(["i", "n"])
    assert label(js, ITEMS, preds, passages, path, ask=lambda _: next(answers), out=out.append) == 1
    got = read_jsonl(path, HumanLabel)
    assert [(h.correctness, h.faithful) for h in got] == [("correct", "yes"), ("incorrect", "no")]


def test_agreement_perfect_and_by_language():
    qa = list(ITEMS)
    judgments = {}
    humans = []
    for i, q in enumerate(qa):
        c = ["correct", "incorrect", "partially_correct"][i % 3]
        f = ["yes", "no"][i % 2]
        judgments[("r1", q)] = judgment(q, c, f)
        humans.append(HumanLabel(run_id="r1", qa_id=q, correctness=c, faithful=f))
    result = agreement(humans, judgments, ITEMS)
    assert result["all"]["kappa_correctness"] == pytest.approx(1.0)
    assert result["all"]["kappa_faithful"] == pytest.approx(1.0)
    assert set(result) == {"all", "en", "hi", "hinglish"}
