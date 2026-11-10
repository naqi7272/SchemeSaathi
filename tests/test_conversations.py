import random
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from saathi.agent.conversations import (
    ConversationResult,
    Scenario,
    Simulator,
    candidates,
    plan_scenario,
    run_conversation,
    sample_profile,
    score_conversation,
)
from saathi.agent.eligibility import Rule, SchemeRules, Status, evaluate
from saathi.agent.graph import AgentDeps, SchemeChoice, build_graph
from saathi.schemas import Language
from tests.test_agent_graph import (
    PASSAGES,
    FixedRetriever,
    ScriptedModel,
    ans,
    understanding,
)
from tests.test_agent_graph import RULES as TOY

PENSION = SchemeRules(
    scheme_id="pension",
    verified=True,
    rules=[
        Rule(field="age", op=">=", value=18, passage_id="pension:e:0"),
        Rule(field="age", op="<=", value=40, passage_id="pension:e:0"),
        Rule(field="is_income_tax_payer", op="==", value=False, passage_id="pension:e:0"),
        Rule(field="social_category", op="in", value=["sc", "st"], passage_id="pension:e:1"),
    ],
)


def test_candidates_cover_thresholds_and_literals():
    rules = [r for r in PENSION.rules if r.field == "age"]
    ages = candidates("age", rules)
    assert {17, 18, 19, 39, 40, 41} <= set(ages)
    assert set(candidates("social_category", [])) == {"general", "obc", "sc", "st"}
    assert set(candidates("is_income_tax_payer", [])) == {True, False}
    assert "farmer" in candidates("occupation", [])


@pytest.mark.parametrize("seed", range(20))
def test_sample_profile_matches_requested_status(seed):
    rng = random.Random(seed)
    for scheme in (PENSION, TOY):
        assert evaluate(sample_profile(scheme, True, rng), scheme).status == Status.ELIGIBLE
        assert evaluate(sample_profile(scheme, False, rng), scheme).status == Status.NOT_ELIGIBLE


def test_sample_profile_sets_only_rule_fields():
    p = sample_profile(PENSION, True, random.Random(0))
    assert {k for k, v in p.model_dump().items() if v is not None} == {
        "age",
        "is_income_tax_payer",
        "social_category",
    }


def test_sample_profile_impossible_rules_raise():
    impossible = SchemeRules(
        scheme_id="x",
        rules=[
            Rule(field="age", op=">=", value=50, passage_id="x:e:0"),
            Rule(field="age", op="<=", value=40, passage_id="x:e:0"),
        ],
    )
    with pytest.raises(ValueError, match="no candidate"):
        sample_profile(impossible, True, random.Random(0))


def test_plan_scenario_needs_clarification_is_computed_by_rules():
    seen = set()
    for seed in range(40):
        profile, revealed, needs = plan_scenario(
            PENSION, "c", Language.EN, True, random.Random(seed)
        )
        # eligible profile: clarification needed exactly when some rule field is hidden
        assert needs == (set(revealed) != {"age", "is_income_tax_payer", "social_category"})
        seen.add(needs)
    assert seen == {True, False}  # both cases occur


def scenario(**kw):
    base = dict(
        conv_id="c000",
        language=Language.HINGLISH,
        scheme_id="toy",
        scheme_name="Toy",
        true_profile={"occupation": "farmer", "land_hectares": 1.5},
        revealed=["occupation"],
        opening="Main farmer hoon, kya toy yojana milegi?",
        expected_status=Status.ELIGIBLE,
        needs_clarification=True,
    )
    return Scenario(**(base | kw))


class FakeSimulator(Simulator):
    def __init__(self):
        self.questions = []

    def reply(self, scenario, question):
        from tests.test_agent_graph import USAGE

        self.questions.append(question)
        return "1.5 hectare", USAGE


def make_app(model):
    deps = AgentDeps(
        model=model,
        retriever=FixedRetriever(),
        rules={"toy": TOY},
        passages_by_id={p.passage_id: p for p in PASSAGES},
        scheme_names={"toy": "Toy"},
    )
    return build_graph(deps, SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))


def test_run_and_score_successful_conversation():
    from saathi.agent.graph import Clarification

    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="farmer"), understanding(land_hectares=1.5)],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "Clarification": [
                Clarification(question="Kitni zameen hai?", fields=["land_hectares"])
            ],
            "AgentAnswer": [ans("eligible")],
        }
    )
    sim = FakeSimulator()
    result = run_conversation(make_app(model), scenario(), sim, "r1")
    assert sim.questions == ["Kitni zameen hai?"] and not result.hit_turn_limit
    s = score_conversation(scenario(), result)
    assert s.task_success and s.asked and s.clarify_appropriate
    assert not s.asked_known_field and s.search_called and s.eligibility_tool_correct
    assert s.profile_accuracy == 1.0 and s.user_turns == 2 and s.invalid_tool_calls == 0


def result_with(steps, verdict="eligible", profile=None, transcript=None):
    return ConversationResult(
        run_id="r",
        conv_id="c000",
        transcript=transcript or [{"role": "user", "content": "x"}],
        final_answer={"eligibility_verdict": verdict},
        final_profile=profile or {},
        steps=steps,
        agent_usage=[],
        simulator_usage=[],
        latency_s=1.0,
        hit_turn_limit=False,
    )


def test_score_flags_unneeded_and_redundant_questions():
    steps = [
        {"node": "clarify", "fields": ["occupation"]},
        {"node": "check_eligibility", "tool": "check_eligibility", "args": {"scheme_id": "other"}},
    ]
    s = score_conversation(scenario(needs_clarification=False), result_with(steps, "unknown"))
    assert not s.task_success and not s.clarify_appropriate
    assert s.asked_known_field  # asked for occupation, which was already stated
    assert not s.eligibility_tool_correct and not s.search_called


def test_score_profile_accuracy_tolerates_unit_rounding():
    s = score_conversation(
        scenario(), result_with([], profile={"occupation": "farmer", "land_hectares": 1.52})
    )
    assert s.profile_accuracy == 1.0
    s = score_conversation(scenario(), result_with([], profile={"occupation": "farmer"}))
    assert s.profile_accuracy == 0.5
