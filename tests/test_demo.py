import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from saathi.agent.graph import AgentDeps, Clarification, SchemeChoice, build_graph
from saathi.demo.app import AgentSession, build_ui, format_answer, format_step
from saathi.schemas import Answer
from tests.test_agent_graph import (
    PASSAGES,
    RULES,
    FixedRetriever,
    ScriptedModel,
    ans,
    understanding,
)


def test_format_answer_puts_cited_sources_first():
    head, sources = format_answer(
        Answer(text="Rs 6000", citations=["toy:eligibility:0"]), PASSAGES[:2]
    )
    assert head == "Rs 6000"
    assert (
        sources.index("Cited sources")
        < sources.index("[toy:eligibility:0]")
        < sources.index("Other retrieved passages")
        < sources.index("[toy:benefits:0]")
    )
    assert "https://example.gov.in/toy" in sources


def test_format_answer_refusal_and_empty():
    head, sources = format_answer(Answer(text="Not covered", refused=True), [])
    assert head == "*Not covered*" and sources == "_No passages retrieved._"


def test_format_step_lines():
    upd = {
        "steps": [
            {
                "node": "check_eligibility",
                "args": {"scheme_id": "toy"},
                "result": {"status": "unknown"},
            }
        ]
    }
    assert format_step("check_eligibility", upd).startswith("check_eligibility{'scheme_id'")
    assert format_step("self_check", {"steps": [{"passed": True}]}) == "self_check: passed"
    assert format_step("__interrupt__", {}) == "(waiting for your reply)"


def make_session():
    model = ScriptedModel(
        {
            "Understanding": [understanding(occupation="farmer"), understanding(land_hectares=1.0)],
            "SchemeChoice": [SchemeChoice(scheme_id="toy")],
            "Clarification": [Clarification(question="How much land?", fields=["land_hectares"])],
            "AgentAnswer": [ans("eligible", text="Yes, you qualify.")],
        }
    )
    deps = AgentDeps(
        model=model,
        retriever=FixedRetriever(),
        rules={"toy": RULES},
        passages_by_id={p.passage_id: p for p in PASSAGES},
        scheme_names={"toy": "Toy"},
    )
    app = build_graph(deps, SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))
    return AgentSession(app)


def test_agent_session_clarifies_then_answers():
    session = make_session()
    first = list(session.send("I am a farmer. Am I eligible for toy?"))
    assert first[-1] == (None, "How much land?") and session.waiting
    assert any(line and line.startswith("search_schemes") for line, _ in first)
    second = list(session.send("1 hectare"))
    assert not session.waiting
    assert second[-1][1].startswith("Yes, you qualify.")
    assert "Sources: toy:eligibility:0" in second[-1][1]


def test_build_ui_constructs():
    import pytest

    pytest.importorskip("gradio")
    demo = build_ui(rag_arm=None, session_factory=make_session)
    assert demo is not None
