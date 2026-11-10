"""Phase 6: arm-5 evaluation with no LLM in the user role."""

import random

import pytest

from saathi.agent.conversations import (
    make_scenarios,
    run_conversation,
    score_conversation,
)
from saathi.agent.eligibility import PROFILE_FIELDS, UserProfile
from saathi.agent.graph import Clarification, SchemeChoice
from saathi.agent.language import detect_language
from saathi.agent.scripted import (
    ScriptedSimulator,
    fact_sentence,
    fields_from_text,
    template_opening,
)
from saathi.schemas import Language
from tests.test_agent_graph import RULES as TOY
from tests.test_agent_graph import ScriptedModel, ans, understanding
from tests.test_conversations import PENSION, make_app, result_with, scenario

SAMPLE = {
    "age": 35,
    "gender": "female",
    "annual_family_income_inr": 150000,
    "state": "Bihar",
    "area": "rural",
    "occupation": "street_vendor",
    "land_hectares": 1.5,
    "social_category": "sc",
    "is_income_tax_payer": False,
    "has_bank_account": True,
    "is_government_employee": False,
}


def test_every_profile_field_has_a_template_in_every_language():
    assert set(SAMPLE) == set(PROFILE_FIELDS)
    for lang in Language:
        for field, value in SAMPLE.items():
            assert fact_sentence(field, value, lang)
        for value in (True,):
            assert fact_sentence("has_bank_account", value, lang)


@pytest.mark.parametrize("lang", list(Language))
def test_template_openings_are_detected_as_their_language(lang):
    rng = random.Random(0)
    for _ in range(30):
        fields = rng.sample(sorted(SAMPLE), rng.randrange(len(SAMPLE)))
        text, usage = template_opening("PM Kisan", lang, {f: SAMPLE[f] for f in fields}, rng)
        assert detect_language(text) == lang, text
        assert usage.cost_usd == 0.0


def test_opening_states_exactly_the_revealed_facts():
    text, _ = template_opening("Atal Pension", Language.EN, {"age": 35}, random.Random(1))
    assert "35 years old" in text and "Atal Pension" in text
    assert "income" not in text.lower()


def test_simulator_answers_asked_fields_from_hidden_profile():
    sim = ScriptedSimulator()
    sc = scenario(language=Language.HI)  # true profile: farmer, 1.5 ha
    reply, usage = sim.reply(sc, "कितनी ज़मीन है?", ["land_hectares"])
    assert reply == "मेरे पास 1.5 हेक्टेयर ज़मीन है।" and usage.cost_usd == 0
    reply, _ = sim.reply(sc, "Umar?", ["age"])  # not in the hidden profile
    assert reply == "मुझे इसके बारे में पक्का नहीं पता।"
    reply, _ = sim.reply(scenario(), "Aap kaam kya karte ho aur kitni zameen hai?", [])
    assert "kisan" in reply and "1.5 hectare" in reply  # keyword fallback


def test_fields_from_text():
    assert fields_from_text("What is your age and annual income?") == [
        "age",
        "annual_family_income_inr",
    ]
    assert fields_from_text("आपके पास कितनी ज़मीन है?") == ["land_hectares"]


def test_make_scenarios_without_any_model_is_deterministic():
    names = {"toy": "Toy Yojana", "pension": "Pension Yojana"}
    rules = {"toy": TOY, "pension": PENSION}
    a = make_scenarios(rules, names, 12, seed=3)
    b = make_scenarios(rules, names, 12, seed=3)
    assert a == b and len(a) == 12
    assert {s.language for s in a} == set(Language)
    for s in a:
        assert detect_language(s.opening) == s.language, s.opening
        assert UserProfile.model_validate(s.true_profile.model_dump())


def test_full_conversation_with_scripted_simulator_and_new_metrics():
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
    result = run_conversation(make_app(model), scenario(), ScriptedSimulator(), "r1")
    assert result.transcript[2]["content"] == "Mere paas 1.5 hectare zameen hai."
    assert all(u.cost_usd == 0 for u in result.simulator_usage)
    s = score_conversation(scenario(), result, TOY)
    assert s.task_success and s.asked_missing_field and s.tool_calls_valid
    assert s.citation_correct and s.user_turns == 2


def test_new_metrics_failure_cases():
    steps = [
        {"node": "clarify", "fields": ["occupation"]},  # already stated; land was missing
        {"node": "choose_scheme", "args": {"scheme_id": "zzz"}, "error": "invalid scheme_id"},
    ]
    r = result_with(steps)
    r.final_answer["citations"] = ["other:benefits:0"]
    s = score_conversation(scenario(), r, TOY)
    assert s.asked_missing_field is False and not s.tool_calls_valid
    assert s.citation_correct is False
    s = score_conversation(scenario(), r)  # no rules given: N/A, not guessed
    assert s.asked_missing_field is None and s.citation_correct is None
    s = score_conversation(scenario(needs_clarification=False), r, TOY)
    assert s.asked_missing_field is None


def test_crashed_conversation_scores_as_failure():
    from saathi.agent.conversations import ConversationResult

    r = ConversationResult(
        run_id="r", conv_id="c000", transcript=[{"role": "user", "content": "x"}],
        final_answer=None, final_profile={}, steps=[], agent_usage=[], simulator_usage=[],
        latency_s=1.0, hit_turn_limit=False, error="ValidationError: EOF",
    )  # fmt: skip
    s = score_conversation(scenario(), r, TOY)
    assert not s.task_success and s.user_turns == 1 and s.citation_correct is False
