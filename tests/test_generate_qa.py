import json

import pytest

from saathi.gen.generate_qa import (
    BudgetExceeded,
    GenGroup,
    GenText,
    estimate_cost,
    generate,
    to_items,
)
from saathi.llm.base import LLMResult, Usage
from saathi.schemas import Language, QuestionType, Scheme
from tests.conftest import FETCHED, make_passage

T = GenText(question="q", answer="a")


def group(qtype, citations):
    return GenGroup(qtype=qtype, citations=citations, en=T, hi=T, hinglish=T)


VALID = {"s0:benefits:0", "s0:eligibility:0"}


def test_to_items_expands_group_into_three_languages():
    items, dropped = to_items("s0", [group(QuestionType.BENEFITS, ["s0:benefits:0"])], VALID)
    assert dropped == []
    assert [it.language for it in items] == list(Language)
    assert {it.group_id for it in items} == {"s0-00"}
    assert items[0].qa_id == "s0-00-en"


@pytest.mark.parametrize(
    "g,reason",
    [
        (group(QuestionType.BENEFITS, ["s1:benefits:0"]), "unknown citations"),
        (group(QuestionType.BENEFITS, []), "without citations"),
        (group(QuestionType.UNANSWERABLE, ["s0:benefits:0"]), "unanswerable group with"),
        (
            GenGroup(
                qtype=QuestionType.BENEFITS,
                citations=["s0:benefits:0"],
                en=T,
                hi=GenText(question=" ", answer="a"),
                hinglish=T,
            ),
            "empty",
        ),
    ],
)
def test_to_items_drops_invalid_groups(g, reason):
    items, dropped = to_items("s0", [g], VALID)
    assert items == []
    assert reason in dropped[0]


def test_to_items_keeps_group_numbering_stable_when_dropping():
    groups = [group(QuestionType.BENEFITS, []), group(QuestionType.UNANSWERABLE, [])]
    items, _ = to_items("s0", groups, VALID)
    assert {it.group_id for it in items} == {"s0-01"}
    assert all(not it.answerable for it in items)


def test_estimate_cost_uses_model_price():
    # 10 groups x 1200 tokens x 2 schemes = 24k output tokens at $20/M, 10k input at $4/M
    assert estimate_cost("claude-opus-5-5", 10_000, 2) == pytest.approx(0.04 + 0.48)


class FakeModel:
    name = "claude-opus-5-5"

    def __init__(self, cost=0.5):
        self.calls = 0
        self.cost = cost

    def chat(self, system, messages, schema=None, max_tokens=0):
        self.calls += 1
        payload = {
            "groups": [
                group(QuestionType.BENEFITS, [f"s{self.calls - 1}:benefits:0"]).model_dump(
                    mode="json"
                )
            ]
        }
        usage = Usage(
            requested_model=self.name,
            served_model=self.name,
            input_tokens=1,
            output_tokens=1,
            latency_s=0.0,
            cost_usd=self.cost,
        )
        return LLMResult(text=json.dumps(payload), usage=usage)


def schemes(n):
    return [
        Scheme(scheme_id=f"s{i}", name=f"S{i}", source_url="u", fetched_at=FETCHED, sections={})
        for i in range(n)
    ]


def passages(n):
    return {f"s{i}": [make_passage(f"s{i}")] for i in range(n)}


def test_generate_writes_logs_and_resumes(tmp_path):
    model = FakeModel()
    items = generate(model, schemes(2), passages(2), tmp_path, max_usd=5)
    assert len(items) == 6 and model.calls == 2
    assert len((tmp_path / "generation_log.jsonl").read_text().splitlines()) == 2
    again = generate(model, schemes(2), passages(2), tmp_path, max_usd=5)
    assert model.calls == 2  # nothing regenerated
    assert [it.qa_id for it in again] == [it.qa_id for it in items]


def test_generate_stops_at_budget(tmp_path):
    model = FakeModel(cost=3.0)
    with pytest.raises(BudgetExceeded):
        generate(model, schemes(3), passages(3), tmp_path, max_usd=5)
    assert model.calls == 2  # third call never made after spending $6


def test_select_passages_prefers_key_terms_and_keeps_order():
    from datetime import UTC, datetime

    from saathi.gen.generate_qa import select_passages
    from saathi.schemas import Passage

    def P(i, text):
        return Passage(passage_id=f"s:x:{i}", scheme_id="s", section="x", text=text,
                       source_url="u", fetched_at=datetime(2026, 1, 1, tzinfo=UTC))  # fmt: skip

    ps = [P(i, "intro text" if i % 2 else f"premium Rs. {i}") for i in range(10)]
    chosen = select_passages(ps, 3)
    assert len(chosen) == 3 and all("premium" in p.text for p in chosen)
    assert [p.passage_id for p in chosen] == sorted(
        (p.passage_id for p in chosen), key=lambda x: int(x.split(":")[-1])
    )
    assert select_passages(ps, None) == ps and select_passages(ps, 50) == ps
