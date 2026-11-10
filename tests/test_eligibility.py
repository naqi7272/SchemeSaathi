import pytest
import yaml

from saathi.agent.eligibility import (
    Rule,
    SchemeRules,
    Status,
    UserProfile,
    check_rule,
    evaluate,
    load_all_rules,
    load_rules,
    validate_rules,
)

# Fictional scheme: small farmers, not income-tax payers, 18-60, any category but general.
TOY = SchemeRules(
    scheme_id="toy",
    verified=True,
    rules=[
        Rule(field="occupation", op="==", value="farmer", passage_id="toy:eligibility:0"),
        Rule(field="land_hectares", op="<=", value=2.0, passage_id="toy:eligibility:0"),
        Rule(field="is_income_tax_payer", op="==", value=False, passage_id="toy:eligibility:0"),
        Rule(field="age", op=">=", value=18, passage_id="toy:eligibility:1"),
        Rule(field="age", op="<=", value=60, passage_id="toy:eligibility:1"),
        Rule(
            field="social_category",
            op="in",
            value=["obc", "sc", "st"],
            passage_id="toy:eligibility:1",
        ),
    ],
)

FULL = dict(
    occupation="farmer", land_hectares=1.5, is_income_tax_payer=False, age=40, social_category="sc"
)


@pytest.mark.parametrize(
    "op,value,actual,expected",
    [
        ("==", "farmer", "farmer", True),
        ("==", "farmer", "student", False),
        ("!=", "farmer", "student", True),
        ("<", 2, 2, False),
        ("<=", 2, 2, True),
        (">", 18, 18, False),
        (">=", 18, 18, True),
        ("in", ["sc", "st"], "st", True),
        ("in", ["sc", "st"], "general", False),
        ("not_in", ["general"], "obc", True),
        ("==", False, False, True),
        ("==", False, True, False),
    ],
)
def test_check_rule_operators(op, value, actual, expected):
    rule = Rule(field="age", op=op, value=value, passage_id="toy:x:0")
    assert check_rule(rule, actual) is expected


def test_eligible_when_all_rules_hold():
    result = evaluate(UserProfile(**FULL), TOY)
    assert result.status == Status.ELIGIBLE
    assert result.failed == [] and result.missing_fields == []


@pytest.mark.parametrize(
    "override,failed_field",
    [
        ({"land_hectares": 2.01}, "land_hectares"),
        ({"is_income_tax_payer": True}, "is_income_tax_payer"),
        ({"age": 17}, "age"),
        ({"age": 61}, "age"),
        ({"social_category": "general"}, "social_category"),
        ({"occupation": "student"}, "occupation"),
    ],
)
def test_not_eligible_names_the_failing_rule(override, failed_field):
    result = evaluate(UserProfile(**(FULL | override)), TOY)
    assert result.status == Status.NOT_ELIGIBLE
    assert [r.field for r in result.failed] == [failed_field]


def test_boundaries_are_inclusive_where_written():
    assert (
        evaluate(UserProfile(**(FULL | {"land_hectares": 2.0, "age": 60})), TOY).status
        == Status.ELIGIBLE
    )
    assert evaluate(UserProfile(**(FULL | {"age": 18})), TOY).status == Status.ELIGIBLE


def test_unknown_lists_each_missing_field_once():
    result = evaluate(UserProfile(occupation="farmer", land_hectares=1.0), TOY)
    assert result.status == Status.UNKNOWN
    assert result.missing_fields == ["is_income_tax_payer", "age", "social_category"]


def test_known_failure_beats_missing_fields():
    result = evaluate(UserProfile(occupation="student"), TOY)
    assert result.status == Status.NOT_ELIGIBLE
    assert "age" in result.missing_fields  # still reported, but status is decided


def test_empty_profile_is_unknown():
    assert evaluate(UserProfile(), TOY).status == Status.UNKNOWN


@pytest.mark.parametrize(
    "rule,message",
    [
        (Rule(field="salary", op="<", value=1, passage_id="toy:e:0"), "unknown profile field"),
        (Rule(field="state", op="in", value="Bihar", passage_id="toy:e:0"), "needs a list"),
        (Rule(field="age", op="<", value="18", passage_id="toy:e:0"), "needs a number"),
        (Rule(field="age", op="<", value=True, passage_id="toy:e:0"), "needs a number"),
        (Rule(field="age", op="<", value=18, passage_id="other:e:0"), "another scheme"),
    ],
)
def test_validate_rules_rejects_bad_rules(rule, message):
    with pytest.raises(ValueError, match=message):
        validate_rules(SchemeRules(scheme_id="toy", rules=[rule]))


def test_load_refuses_unverified_rules(tmp_path):
    data = TOY.model_dump() | {"verified": False}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="not hand-verified"):
        load_rules(path)
    assert load_rules(path, allow_unverified=True).scheme_id == "toy"


def test_load_all_rules_round_trip(tmp_path):
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump(TOY.model_dump()), encoding="utf-8")
    assert load_all_rules(tmp_path) == {"toy": TOY}


class FakeDraftModel:
    name = "fake"

    def __init__(self, payload):
        self.payload = payload

    def chat(self, system, messages, schema=None, max_tokens=0):
        from saathi.llm.base import LLMResult, Usage

        usage = Usage(
            requested_model="fake",
            served_model="fake",
            input_tokens=0,
            output_tokens=0,
            latency_s=0,
            cost_usd=0.25,
        )
        return LLMResult(text=self.payload, usage=usage)


def test_draft_rules_are_unverified_and_validated():
    import json

    from saathi.agent.draft_rules import draft

    payload = json.dumps(
        {
            "rules": [
                {"field": "age", "op": ">=", "value": 18, "passage_id": "toy:e:0", "note": ""}
            ],
            "unsupported": ["must hold a BPL card"],
        }
    )
    rules, unsupported, cost = draft(FakeDraftModel(payload), "toy", [])
    assert rules.verified is False and len(rules.rules) == 1
    assert unsupported == ["must hold a BPL card"] and cost == 0.25

    bad = payload.replace('"age"', '"salary"')
    with pytest.raises(ValueError, match="unknown profile field"):
        draft(FakeDraftModel(bad), "toy", [])

    empty = json.dumps({"rules": [], "unsupported": ["x"]})
    assert draft(FakeDraftModel(empty), "toy", [])[0] is None
