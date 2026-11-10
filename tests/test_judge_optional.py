"""Phase 5: the report works without a judge, and judge metrics are gated on kappa."""

import json
import sys

import pytest
import yaml

from saathi.eval import run as run_mod
from saathi.eval.report import build_report, item_metrics, judge_gate, markdown
from saathi.eval.score import score_item
from saathi.schemas import Answer, Language
from tests.conftest import make_item
from tests.test_report import fake_run, score


def test_numbers_exact_is_code_scored():
    item = make_item("s", 1, Language.EN).model_copy(update={"answer": "Rs 6,000 at age 60"})

    def m(text, refused=False):
        s = score_item(item, "r", Answer(text=text, refused=refused), [], 0, 0, 0, 0, False)
        return item_metrics(s, None)["numbers_exact"]

    assert m("You get ₹6000 from age 60.") == 1.0
    assert m("You get ₹6000.") == 0.0  # one number missing
    assert m("", refused=True) == 0.0  # refusing an answerable item is wrong
    no_numbers = item.model_copy(update={"answer": "Apply at the bank."})
    s = score_item(no_numbers, "r", Answer(text="bank"), [], 0, 0, 0, 0, False)
    assert item_metrics(s, None)["numbers_exact"] is None  # nothing to check


def test_old_scores_without_new_field_still_load():
    assert score("x", Language.EN).has_gold_numbers is False


def write_validation(path, overall, per_lang):
    data = {"all": {"n": 100, "kappa_correctness": overall, "kappa_faithful": overall}}
    for lang, k in per_lang.items():
        data[lang] = {"n": 30, "kappa_correctness": k, "kappa_faithful": 0.9}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_judge_gate(tmp_path):
    path = tmp_path / "v.json"
    assert set(judge_gate(path)) == {"accuracy", "faithful"}  # missing file: hide both
    write_validation(path, 0.75, {"en": 0.8, "hi": 0.7, "hinglish": 0.5})
    assert judge_gate(path) == {}
    write_validation(path, 0.55, {"en": 0.8})
    assert set(judge_gate(path)) == {"accuracy", "faithful"}
    write_validation(path, 0.7, {"en": 0.8, "hi": 0.3})
    hidden = judge_gate(path)
    assert set(hidden) == {"accuracy"} and "hi" in hidden["accuracy"]


def test_report_without_judge_hides_judge_metrics(tmp_path):
    per_item, items = fake_run(0.9, 1)
    for metrics in per_item.values():  # a run with no judgments at all
        metrics["accuracy"] = None
        metrics["faithful"] = None
    hidden = {"accuracy": "no judge", "faithful": "no judge"}
    result = build_report(["r"], [], tmp_path, loader=lambda r: (per_item, items), hidden=hidden)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## accuracy" not in report and "## faithful" not in report
    assert "`accuracy` (LLM judge) not reported: no judge." in report
    assert "## numbers_exact" in report
    assert "accuracy" not in result["summaries"]["r"]["all"]
    assert not (tmp_path / "plots" / "accuracy.png").exists()


def test_markdown_code_metrics_come_first():
    text = markdown({}, [])
    assert text.index("## numbers_exact") < text.index("## accuracy")


# ------------------------------------------------------------- judge CLI
@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "RUNS_DIR", tmp_path)
    d = tmp_path / "r1"
    d.mkdir()
    cfg = {
        "run_id": "r1",
        "arm": "rag",
        "backend": "ollama",
        "model": "m",
        "test_file": str(tmp_path / "t.jsonl"),
        "passages_file": str(tmp_path / "p.jsonl"),
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "t.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "p.jsonl").write_text("", encoding="utf-8")
    (d / "predictions.jsonl").write_text("", encoding="utf-8")
    return d


def cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["saathi-eval", *argv])
    monkeypatch.setattr(run_mod, "load_dotenv", lambda: None)
    run_mod.main()


def test_judge_refuses_same_family(run_dir, monkeypatch):
    with pytest.raises(SystemExit, match="same-family"):
        cli(monkeypatch, "judge", "r1", "--judge-model", "qwen2.5:32b")


def test_judge_needs_a_model(run_dir, monkeypatch):
    monkeypatch.delenv("SAATHI_LLM_MODEL", raising=False)
    with pytest.raises(SystemExit, match="judge-model"):
        cli(monkeypatch, "judge", "r1")


def test_judge_with_open_model_runs_without_key(run_dir, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cli(monkeypatch, "judge", "r1", "--judge-model", "llama3.1:8b")  # nothing to judge
    assert "judged with llama3.1:8b (openai_compat); spent $0.00" in capsys.readouterr().out
