import json

from saathi.arms.arms import ArmOutput
from saathi.eval.judge import Judgment, Verdict, needs_judging
from saathi.eval.run import Prediction, judge_run, judge_todo, predict, score_run
from saathi.io import read_jsonl
from saathi.llm.base import LLMResult, Usage
from saathi.schemas import Answer, Language
from tests.conftest import make_item, make_passage

USAGE = Usage(
    requested_model="m",
    served_model="m",
    input_tokens=10,
    output_tokens=5,
    latency_s=0.1,
    cost_usd=0.5,
)


class FlakyArm:
    """Answers with refusals for 'refuse' questions, fails once on 'boom'."""

    name = "fake-arm"

    def __init__(self):
        self.failed_once = False

    def answer(self, question):
        if "boom" in question and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("model crashed")
        refused = "refuse" in question
        ans = Answer(
            text="Rs 6000", citations=[] if refused else ["s0:benefits:0"], refused=refused
        )
        return ArmOutput(answer=ans, retrieved=["s0:benefits:0"], usage=[USAGE], latency_s=0.2)


def items():
    a = make_item("s0", 0, Language.EN)
    b = make_item("s0", 1, Language.HI).model_copy(update={"question": "boom"})
    c = make_item("s0", 2, Language.EN).model_copy(update={"question": "refuse"})
    return [a, b, c]


def test_predict_resumes_after_failure(tmp_path):
    arm = FlakyArm()
    assert predict("r1", arm, items(), tmp_path) == 1
    assert len(read_jsonl(tmp_path / "predictions.jsonl", Prediction)) == 2
    assert predict("r1", arm, items(), tmp_path) == 0  # only the failed item is retried
    preds = read_jsonl(tmp_path / "predictions.jsonl", Prediction)
    assert sorted(p.qa_id for p in preds) == sorted(it.qa_id for it in items())


def test_score_run_sums_usage_and_writes_file(tmp_path):
    arm = FlakyArm()
    predict("r1", arm, items(), tmp_path)
    predict("r1", arm, items(), tmp_path)
    scores = score_run(tmp_path, items())
    assert len(scores) == 3 and scores[0].cost_usd == 0.5
    assert (tmp_path / "scores.jsonl").exists()


class FakeJudge:
    name = "judge"

    def __init__(self):
        self.calls = 0

    def chat(self, system, messages, schema=None, max_tokens=0):
        self.calls += 1
        v = Verdict(correctness="correct", faithful="yes", unsupported_claims=[], reason="ok")
        return LLMResult(text=v.model_dump_json(), usage=USAGE)


def test_judge_skips_refusals_resumes_and_respects_budget(tmp_path):
    arm = FlakyArm()
    predict("r1", arm, items(), tmp_path)
    predict("r1", arm, items(), tmp_path)
    passages = [make_passage("s0")]
    judge_model = FakeJudge()
    # 2 judgeable items (the refusal is skipped); budget 0.5 stops after the first
    judge_run(tmp_path, items(), passages, judge_model, max_usd=0.5)
    assert judge_model.calls == 1
    assert len(judge_todo(tmp_path, items())) == 1
    judge_run(tmp_path, items(), passages, judge_model, max_usd=5)
    assert judge_model.calls == 2 and judge_todo(tmp_path, items()) == []
    judged = read_jsonl(tmp_path / "judgments.jsonl", Judgment)
    assert {j.qa_id for j in judged} == {items()[0].qa_id, items()[1].qa_id}


def test_needs_judging():
    item = make_item("s0", 0, Language.EN)
    assert needs_judging(item, Answer(text="x"))
    assert not needs_judging(item, Answer(text="x", refused=True))
    assert not needs_judging(make_item("s0", 3, Language.EN, False), Answer(text="x"))


def test_prediction_round_trips_as_json():
    p = Prediction(run_id="r", qa_id="q", output=FlakyArm().answer("x"))
    assert Prediction.model_validate(json.loads(p.model_dump_json())) == p
