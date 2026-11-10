import pytest

from saathi.io import read_jsonl
from saathi.schemas import Language, QuestionType
from saathi.verify.cli import (
    Status,
    VerificationLabel,
    build_gold,
    review,
    sample_groups,
)
from tests.conftest import make_item, make_passage


def scripted(answers):
    it = iter(answers)
    return lambda _prompt: next(it)


@pytest.fixture
def passages():
    ps = [make_passage(f"s{s}", sec) for s in range(10) for sec in ("benefits", "eligibility")]
    return {p.passage_id: p for p in ps}


def test_sample_groups_returns_whole_groups(corpus):
    items = sample_groups(corpus, n_items=30, seed=0)
    assert len(items) == 30
    groups = {it.group_id for it in items}
    assert len(groups) == 10
    assert sum(it.language == Language.HI for it in items) == 10


def test_sample_groups_is_deterministic(corpus):
    a = [it.qa_id for it in sample_groups(corpus, 30, seed=3)]
    assert a == [it.qa_id for it in sample_groups(list(reversed(corpus)), 30, seed=3)]


def test_sample_groups_skips_incomplete_groups(corpus):
    incomplete = [it for it in corpus if not (it.group_id == "s0-0" and it.language == "hi")]
    items = sample_groups(incomplete, n_items=117, seed=0)
    assert "s0-0" not in {it.group_id for it in items}
    with pytest.raises(ValueError, match="only 39"):
        sample_groups(incomplete, n_items=120, seed=0)


def test_review_accept_edit_reject_and_resume(tmp_path, passages):
    items = [make_item("s0", 0, lang) for lang in Language]
    labels = tmp_path / "labels.jsonl"
    out: list[str] = []
    answers = [
        # item 1: accept, no note
        "a",
        "",
        # item 2: edit answer and citation, keep question and answerable, add note
        "e",
        "",
        "fixed answer",
        "s0:eligibility:0",
        "",
        "edited",
        # item 3: quit
        "q",
    ]
    assert review(items, passages, labels, ask=scripted(answers), out=out.append) == 2

    # resume: only the third item is left
    assert review(items, passages, labels, ask=scripted(["r", "bad hinglish"]), out=out.append) == 1
    got = read_jsonl(labels, VerificationLabel)
    assert [lab.status for lab in got] == [Status.ACCEPT, Status.EDIT, Status.REJECT]
    assert got[1].answer == "fixed answer" and got[1].citations == ["s0:eligibility:0"]


def test_review_rejects_unknown_citation_then_retries(tmp_path, passages):
    item = make_item("s0", 0, Language.EN)
    labels = tmp_path / "labels.jsonl"
    out: list[str] = []
    answers = ["e", "", "", "s9:benefits:0", "", "a", ""]  # s9 passage is not in scheme s0
    review([item], passages, labels, ask=scripted(answers), out=out.append)
    assert any("unknown passage IDs" in line for line in out)
    assert read_jsonl(labels, VerificationLabel)[0].status == Status.ACCEPT


def _label(item, status, **update):
    fields = {
        "question": item.question,
        "answer": item.answer,
        "citations": item.citations,
        "answerable": item.answerable,
    } | update
    return VerificationLabel(
        qa_id=item.qa_id, status=status, verified_at="2026-10-23T00:00:00Z", **fields
    )


def test_build_gold_applies_edits_and_drops_rejects():
    en, hi, hl = (make_item("s0", 0, lang) for lang in Language)
    unreviewed = make_item("s1", 0, Language.EN)
    labels = [
        _label(en, Status.ACCEPT),
        _label(hi, Status.EDIT, answer="new"),
        _label(hl, Status.ACCEPT),
        _label(hl, Status.REJECT),  # later label wins
    ]
    gold = build_gold([en, hi, hl, unreviewed], labels)
    assert [g.qa_id for g in gold] == [en.qa_id, hi.qa_id]
    assert gold[1].answer == "new"


def test_build_gold_turning_item_unanswerable_updates_qtype():
    item = make_item("s0", 0, Language.EN)
    gold = build_gold([item], [_label(item, Status.EDIT, citations=[], answerable=False)])
    assert gold[0].qtype == QuestionType.UNANSWERABLE and not gold[0].answerable


def test_build_gold_rejects_invalid_edit():
    item = make_item("s0", 0, Language.EN)
    with pytest.raises(ValueError, match="cite at least one"):
        build_gold([item], [_label(item, Status.EDIT, citations=[])])
