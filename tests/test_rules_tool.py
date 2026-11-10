import subprocess
import sys

import pytest
import yaml

from saathi.agent.eligibility import load_rules
from saathi.agent.rules_tool import check_file, template_text, validate_dir
from saathi.io import write_jsonl
from tests.conftest import make_passage

PASSAGES = {p.passage_id: p for p in [make_passage("pmk", "eligibility"), make_passage("apy")]}


def write(tmp_path, name, data):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def rules(**over):
    base = {
        "scheme_id": "pmk",
        "verified": True,
        "rules": [
            {"field": "occupation", "op": "==", "value": "farmer",
             "passage_id": "pmk:eligibility:0"},
            {"field": "is_income_tax_payer", "op": "==", "value": False,
             "passage_id": "pmk:eligibility:0"},
        ],
    }  # fmt: skip
    return base | over


def test_valid_file(tmp_path):
    scheme, errors = check_file(write(tmp_path, "pmk", rules()), PASSAGES)
    assert errors == [] and scheme.verified


@pytest.mark.parametrize(
    "rule, message",
    [
        ({"field": "caste", "op": "==", "value": "x"}, "unknown profile field"),
        ({"field": "age", "op": ">=", "value": "eighteen"}, "needs a number"),
        ({"field": "gender", "op": "in", "value": ["female", "woman"]}, "'woman' is not a valid"),
        ({"field": "age", "op": ">=", "value": 18, "passage_id": "pmk:eligibility:9"},
         "not in passages.jsonl"),
        ({"field": "age", "op": ">=", "value": 18, "passage_id": "apy:benefits:0"},
         "another scheme"),
    ],
)  # fmt: skip
def test_errors(tmp_path, rule, message):
    rule = {"passage_id": "pmk:eligibility:0"} | rule
    _, errors = check_file(write(tmp_path, "pmk", rules(rules=[rule])), PASSAGES)
    assert any(message in e for e in errors), errors


def test_schema_and_filename_errors(tmp_path):
    _, errors = check_file(write(tmp_path, "pmk", {"scheme_id": "pmk", "rules": []}), PASSAGES)
    assert errors and errors[0].startswith("schema")
    _, errors = check_file(write(tmp_path, "other", rules()), PASSAGES)
    assert any("does not match" in e for e in errors)


def test_validate_dir_reports_unverified(tmp_path, capsys):
    write(tmp_path, "pmk", rules(verified=False))
    bad, unverified = validate_dir(tmp_path, PASSAGES)
    assert bad == 0 and unverified == ["pmk"]
    assert "NOT verified" in capsys.readouterr().out


def test_template_is_valid_yaml_refused_until_verified(tmp_path):
    path = tmp_path / "pmk.yaml"
    path.write_text(template_text("pmk", ["pmk:eligibility:0"]), encoding="utf-8")
    scheme, errors = check_file(path, PASSAGES)
    assert errors == [] and scheme.verified is False
    with pytest.raises(ValueError, match="not hand-verified"):
        load_rules(path)  # the agent still refuses unverified rules
    text = path.read_text(encoding="utf-8")
    assert "#   pmk:eligibility:0" in text and "female | male | other" in text


def test_drafted_notes_are_ignored(tmp_path):
    path = write(tmp_path, "pmk", rules(unsupported=["x"], drafted_by="openai_compat:llama"))
    assert check_file(path, PASSAGES)[1] == []
    assert load_rules(path).scheme_id == "pmk"


def test_cli(tmp_path):
    write_jsonl(tmp_path / "p.jsonl", PASSAGES.values())
    rules_dir = tmp_path / "rules"
    base = [sys.executable, "-m", "saathi.agent.rules_tool"]
    opts = ["--rules", str(rules_dir), "--passages", str(tmp_path / "p.jsonl")]
    res = subprocess.run([*base, "template", "--scheme", "pmk", *opts], capture_output=True)
    assert res.returncode == 0 and (rules_dir / "pmk.yaml").exists()
    res = subprocess.run([*base, "template", "--scheme", "pmk", *opts], capture_output=True)
    assert res.returncode != 0  # never overwrites
    res = subprocess.run([*base, "validate", *opts], capture_output=True, text=True)
    assert res.returncode == 0 and "unverified: ['pmk']" in res.stdout
    write(rules_dir, "bad", {"scheme_id": "bad", "rules": []})
    res = subprocess.run([*base, "validate", *opts], capture_output=True, text=True)
    assert res.returncode == 1 and "FAIL" in res.stdout
