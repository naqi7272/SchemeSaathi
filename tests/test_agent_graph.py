import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from saathi.agent.eligibility import Rule, SchemeRules
from saathi.agent.graph import (
    AgentAnswer,
    AgentDeps,
    Clarification,
    SchemeChoice,
    Understanding,
    build_graph,
    check_answer,
)
from saathi.llm.base import LLMResult, Usage
from tests.conftest import make_passage

USAGE = Usage(
    requested_model="fake",
    served_model="fake",
    input_tokens=10,
    output_tokens=5,
    latency_s=0.1,
    cost_usd=0.0,
)

RULES = SchemeRules(
    scheme_id="toy",
    verified=True,
    rules=[
        Rule(field="occupation", op="==", value="farmer", passage_id="toy:eligibility:0"),
        Rule(field="land_hectares", op="<=", value=2.0, passage_id="toy:eligibility:0"),
    ],
)
PASSAGES = [
    make_passage("toy", "benefits"),
    make_passage("toy", "eligibility"),
    make_passage("other", "benefits"),
]


class ScriptedModel:
    """Replies by output schema, popping scripted replies in order."""

    name = "fake"

    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[str] = []

    def chat(self, system, messages, schema=None, max_tokens=0):
        self.calls.append(schema.__name__)
        reply = self.script[schema.__name__].pop(0)
        return LLMResult(text=reply.model_dump_json(), usage=USAGE)


class FixedRetriever:
    def retrieve(self, query, k=5):
        return PASSAGES[:2]


def understanding(intent="eligibility", **profile):
    return Understanding(profile=profile, intent=intent, search_query="toy scheme")


def ans(verdict, citations=("toy:eligibility:0",), text="You qualify.", refused=False):
    return AgentAnswer(
        text=text, citations=list(citations), refused=refused, eligibility_verdict=verdict
    )


def make_app(model, allow_clarify=True, store=None):
    deps = AgentDeps(
        model=model,
        retriever=FixedRetriever(),
        rules={"toy": RULES},
        passages_by_id={p.passage_id: p for p in PASSAGES},
        scheme_names={"toy": "Toy Scheme"},
        allow_clarify=allow_clarify,
    )
    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    return build_graph(deps, checkpointer=saver, store=store)


def cfg(thread="t1", user=None):
    c = {"configurable": {"thread_id": thread}}
    if user:
        c["configurable"]["user_id"] = user
    return c


def nodes(result):
    return [s["node"] for s in result["steps"]]


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def test_information_question_skips_eligibility():
    model = ScriptedModel(
        {
            "Understanding": [understanding("information")],
            "AgentAnswer": [ans("not_applicable", ["toy:benefits:0"])],
        }
    )
    result = make_app(model).invoke(user("What does the toy scheme give?"), cfg())
    assert nodes(result) == [
        "detect_language",
        "understand",
        "search",
        "choose_scheme",
        "answer",
        "self_check",
    ]
    assert "SchemeChoice" not in model.calls  # no tool decision needed
    assert result["answer"]["eligibility_verdict"] == "not_applicable"
    assert result["messages"][-1] == {"role": "assistant", "content": "You qualify."}


def test_clarification_interrupt_and_resume():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="farmer"), understanding(land_hectares=1.5)],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "Clarification": [
                Clarification(question="Kitni zameen hai?", fields=["land_hectares", "not_a_field"])
            ],
            "AgentAnswer": [ans("eligible")],
        }
    )
    app = make_app(model)
    first = app.invoke(user("Main farmer hoon, kya mujhe toy yojana milegi?"), cfg())
    question = first["__interrupt__"][0].value
    assert question["question"] == "Kitni zameen hai?"
    assert question["fields"] == ["land_hectares"]  # unknown field names dropped
    final = app.invoke(Command(resume="1.5 hectare"), cfg())
    assert final["language"] == "hinglish"
    assert final["clarifications"] == 1
    assert final["profile"] == {"occupation": "farmer", "land_hectares": 1.5}
    assert final["answer"]["eligibility_verdict"] == "eligible"
    roles = [m["role"] for m in final["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    tools = [s for s in final["steps"] if s.get("tool") == "check_eligibility" and "result" in s]
    assert [t["result"]["status"] for t in tools] == ["unknown", "eligible"]
    assert model.calls.count("SchemeChoice") == 1  # scheme not re-chosen after clarifying


def test_no_clarification_when_disabled():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="farmer")],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "AgentAnswer": [ans("unknown")],
        }
    )
    result = make_app(model, allow_clarify=False).invoke(user("Am I eligible for toy?"), cfg())
    assert "__interrupt__" not in result
    assert result["answer"]["eligibility_verdict"] == "unknown"


def test_known_failure_answers_without_asking():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="student")],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "AgentAnswer": [ans("not_eligible")],
        }
    )
    result = make_app(model).invoke(user("I'm a student. Eligible for toy?"), cfg())
    assert "write_clarification" not in nodes(result)
    assert result["eligibility"]["status"] == "not_eligible"


def test_self_check_retries_once_then_accepts():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="student")],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "AgentAnswer": [ans("eligible"), ans("not_eligible")],
        }
    )
    result = make_app(model).invoke(user("Eligible for toy?"), cfg())
    checks = [s for s in result["steps"] if s["node"] == "self_check"]
    assert [c["passed"] for c in checks] == [False, True]
    assert "rules say 'not_eligible'" in checks[0]["problems"][0]
    assert result["answer"]["eligibility_verdict"] == "not_eligible"


def test_self_check_gives_up_after_one_revision():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="student")],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "AgentAnswer": [ans("eligible"), ans("eligible")],
        }
    )
    result = make_app(model).invoke(user("Eligible for toy?"), cfg())
    checks = [s for s in result["steps"] if s["node"] == "self_check"]
    assert [c["retry"] for c in checks] == [True, False]
    assert result["messages"][-1]["role"] == "assistant"  # still answers, check is logged


def test_invalid_scheme_choice_is_logged_not_executed():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="farmer")],
            "SchemeChoice": [SchemeChoice(scheme_id="made-up")],
            "AgentAnswer": [ans("not_applicable")],
        }
    )
    result = make_app(model).invoke(user("Eligible?"), cfg())
    choice = next(s for s in result["steps"] if s["node"] == "choose_scheme")
    assert choice["error"] == "invalid scheme_id"
    assert "check_eligibility" not in nodes(result)


def test_profile_persists_across_conversations():
    conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    model = ScriptedModel(
        {
            "Understanding": [
                understanding(occupation="farmer", land_hectares=1.0),
                understanding(),
            ],  # second conversation states nothing new
            "SchemeChoice": [SchemeChoice(scheme_id="toy")] * 2,
            "AgentAnswer": [ans("eligible")] * 2,
        }
    )
    app = make_app(model, store=store)
    app.invoke(user("I farm 1 hectare. Eligible for toy?"), cfg("t1", "u1"))
    second = app.invoke(user("Eligible for toy?"), cfg("t2", "u1"))
    assert "__interrupt__" not in second  # nothing to ask: profile came from the store
    assert second["profile"]["land_hectares"] == 1.0


def test_check_answer_flags_problems():
    state = {"retrieved": [{"passage_id": "a"}], "eligibility": None, "language": "en"}
    a = AgentAnswer(text="ok", citations=["a", "z"], refused=False, eligibility_verdict="eligible")
    problems = check_answer(a, state)
    assert any("not in the passages" in p for p in problems)
    assert any("not_applicable" in p for p in problems)
    hindi = AgentAnswer(
        text="यह योजना", citations=["a"], refused=False, eligibility_verdict="not_applicable"
    )
    assert any("Hindi script" in p for p in check_answer(hindi, state))
    assert check_answer(hindi, state | {"language": "hi"}) == []


def test_output_from_state_maps_agent_answer():
    from saathi.agent.qa_adapter import output_from_state

    state = {
        "answer": ans("eligible").model_dump(),
        "retrieved": [p.model_dump(mode="json") for p in PASSAGES[:2]],
        "usage": [USAGE.model_dump(), USAGE.model_dump()],
    }
    out = output_from_state(state, 2.0)
    assert out.answer.citations == ["toy:eligibility:0"] and not out.answer.refused
    assert out.retrieved == ["toy:benefits:0", "toy:eligibility:0"]
    assert len(out.usage) == 2 and out.parse_error is None
    assert output_from_state({}, 1.0).parse_error == "agent produced no answer"


def test_run_config_without_tracing_keys(monkeypatch):
    from saathi.agent.tracing import run_config

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert run_config("t", "u") == {"configurable": {"thread_id": "t", "user_id": "u"}}
