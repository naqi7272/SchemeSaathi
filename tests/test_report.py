import json
import random

import pytest

from saathi.eval.judge import Judgment, Verdict
from saathi.eval.report import build_report, compare, item_metrics, summarize
from saathi.eval.score import ItemScore
from saathi.llm.base import Usage
from saathi.schemas import Language
from tests.conftest import make_item

USAGE = Usage(
    requested_model="j", served_model="j", input_tokens=1, output_tokens=1, latency_s=0, cost_usd=0
)


def score(qa_id, lang, answerable=True, refusal_correct=True):
    return ItemScore(
        qa_id=qa_id,
        arm="r",
        language=lang,
        answerable=answerable,
        refusal_correct=refusal_correct,
        citation_precision=None,
        citation_recall=None,
        citations_grounded=None,
        retrieval_recall=None,
        number_recall=None,
        script_match=True,
        parse_error=False,
        latency_s=1.0,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0,
    )


def judgment(qa_id, correct=True, faithful=True):
    v = Verdict(
        correctness="correct" if correct else "incorrect",
        faithful="yes" if faithful else "no",
        unsupported_claims=[],
        reason="",
    )
    return Judgment(run_id="r", qa_id=qa_id, verdict=v, judge_model="j", usage=USAGE)


def test_item_metrics_accuracy_rules():
    assert item_metrics(score("a", "en"), judgment("a"))["accuracy"] == 1.0
    assert item_metrics(score("a", "en"), judgment("a", correct=False))["accuracy"] == 0.0
    refused = score("a", "en", refusal_correct=False)  # answerable but refused
    assert item_metrics(refused, None)["accuracy"] == 0.0
    assert item_metrics(refused, None)["false_refusal"] == 1.0
    assert item_metrics(score("a", "en"), None)["accuracy"] is None  # not judged yet
    unans = item_metrics(score("u", "en", answerable=False), None)
    assert unans["accuracy"] is None and unans["refused_unanswerable"] == 1.0


ITEMS = {
    it.qa_id: it
    for it in (make_item(f"s{s}", g, lang) for s in range(4) for g in range(5) for lang in Language)
}


def fake_run(p_correct, seed):
    rng = random.Random(seed)
    per_item = {}
    for q, it in ITEMS.items():
        per_item[q] = item_metrics(score(q, it.language), judgment(q, rng.random() < p_correct))
    return per_item, ITEMS


def test_summarize_has_every_language_and_na():
    per_item, items = fake_run(0.8, 0)
    s = summarize(per_item, items)
    assert set(s) == {"all", "en", "hi", "hinglish"}
    assert s["all"]["accuracy"].n == len(ITEMS)
    assert s["en"]["accuracy"].n == len(ITEMS) // 3
    assert s["all"]["citation_precision"] is None  # no applicable items


def test_compare_includes_mcnemar_for_binary_metrics():
    a, items = fake_run(0.9, 1)
    b, _ = fake_run(0.3, 2)
    res = compare(a, b, items, "accuracy")
    assert res["diff"] > 0 and "p_mcnemar" in res
    assert compare(a, b, items, "latency_s").get("p_mcnemar") is None


def test_build_report_writes_files_and_holm(tmp_path):
    runs = {"good": fake_run(0.9, 1), "bad": fake_run(0.3, 2)}
    result = build_report(["good", "bad"], [("good", "bad")], tmp_path, loader=lambda r: runs[r])
    report = (tmp_path / "report.md").read_text()
    assert "## accuracy" in report and "| good |" in report and "Paired comparisons" in report
    assert (tmp_path / "plots" / "accuracy.png").stat().st_size > 1000
    comps = result["comparisons"]
    assert len(comps) == 2 * 4  # 2 metrics x 4 language slices
    assert all(c["p_holm"] >= c["p"] for c in comps)
    assert json.loads((tmp_path / "summary.json").read_text())["summaries"]["good"]


def test_build_report_unknown_pair_raises(tmp_path):
    runs = {"good": fake_run(0.9, 1)}
    with pytest.raises(KeyError):
        build_report(["good"], [("good", "missing")], tmp_path, loader=lambda r: runs[r])
