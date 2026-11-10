import json
import random

import pytest

from saathi.arms.prompts import SYSTEM_CLOSED_BOOK, SYSTEM_RAG
from saathi.finetune.export_sft import context_with_gold, export, target
from saathi.schemas import Answer, Language
from tests.conftest import make_item, make_passage


def test_target_formats():
    item = make_item("s0", 0, Language.EN)
    assert Answer.model_validate_json(target(item, False)).citations == []
    assert Answer.model_validate_json(target(item, True)).citations == item.citations
    refusal = Answer.model_validate_json(target(make_item("s0", 3, Language.HI, False), True))
    assert refusal.refused and refusal.citations == []


def test_context_with_gold_inserts_missing_gold_keeping_size():
    item = make_item("s0", 0, Language.EN)  # cites s0:benefits:0
    others = [make_passage("s0", "eligibility", i) for i in range(3)]
    by_id = {p.passage_id: p for p in [*others, make_passage("s0")]}
    ctx = context_with_gold(item, others, by_id, random.Random(0))
    assert len(ctx) == 3
    assert "s0:benefits:0" in [p.passage_id for p in ctx]
    assert "s0:eligibility:2" not in [p.passage_id for p in ctx]  # lowest-ranked dropped


def test_context_with_gold_leaves_good_retrieval_alone():
    item = make_item("s0", 0, Language.EN)
    retrieved = [make_passage("s0"), make_passage("s0", "eligibility")]
    by_id = {p.passage_id: p for p in retrieved}
    assert context_with_gold(item, retrieved, by_id, random.Random(0)) == retrieved


class FirstK:
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, query, k=5):
        return self.passages[:k]


def test_export_writes_two_examples_per_item(corpus, tmp_path):
    items = [it for it in corpus if it.scheme_id in ("s0", "s1")]
    passages = [make_passage(s, sec) for s in ("s0", "s1") for sec in ("benefits", "eligibility")]
    counts = export(items, passages, FirstK(passages), tmp_path, k=2)
    assert counts["train"] + counts["val"] == 2 * len(items)
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    systems = {r["messages"][0]["content"] for r in rows}
    assert systems == {SYSTEM_CLOSED_BOOK, SYSTEM_RAG}
    for r in rows:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        answer = Answer.model_validate_json(r["messages"][-1]["content"])
        if r["messages"][0]["content"] == SYSTEM_RAG and not answer.refused:
            for c in answer.citations:  # every cited gold passage is in the context
                assert f"[{c}]" in r["messages"][1]["content"]


def test_export_refuses_test_scheme_passages(corpus, tmp_path):
    items = [it for it in corpus if it.scheme_id == "s0"]
    with pytest.raises(ValueError, match="training schemes"):
        export(items, [make_passage("s9")], FirstK([]), tmp_path)
