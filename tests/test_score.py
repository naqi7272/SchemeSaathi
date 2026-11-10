import pytest

from saathi.eval.score import (
    citation_precision,
    citation_recall,
    citations_grounded,
    number_recall,
    numbers,
    refusal_correct,
    retrieval_recall,
    score_item,
    script_match,
)
from saathi.schemas import Answer, Language
from tests.conftest import make_item

ANS = make_item("s0", 0, Language.EN)  # gold citation s0:benefits:0
UNANS = make_item("s0", 3, Language.EN, answerable=False)


def answer(text="x", citations=(), refused=False):
    return Answer(text=text, citations=list(citations), refused=refused)


@pytest.mark.parametrize(
    "item,refused,expected",
    [(ANS, False, True), (ANS, True, False), (UNANS, True, True), (UNANS, False, False)],
)
def test_refusal_correct(item, refused, expected):
    assert refusal_correct(item, answer(refused=refused)) is expected


def test_citation_precision_and_recall():
    two = make_item("s0", 0, Language.EN).model_copy(
        update={"citations": ["s0:benefits:0", "s0:benefits:1"]}
    )
    pred = answer(citations=["s0:benefits:0", "s0:other:0"])
    assert citation_precision(two, pred) == 0.5
    assert citation_recall(two, pred) == 0.5
    assert citation_recall(two, answer()) == 0.0  # answered without citing
    assert citation_precision(two, answer()) is None  # nothing cited: precision undefined


def test_citation_metrics_not_applicable_for_refusals_and_unanswerable():
    assert citation_precision(ANS, answer(refused=True)) is None
    assert citation_recall(ANS, answer(refused=True)) is None
    assert citation_recall(UNANS, answer(citations=["a"])) is None


def test_citations_grounded():
    assert citations_grounded(answer(citations=["a"]), ["a", "b"]) is True
    assert citations_grounded(answer(citations=["a", "z"]), ["a", "b"]) is False
    assert citations_grounded(answer(), ["a"]) is None


def test_retrieval_recall():
    assert retrieval_recall(ANS, ["s0:benefits:0", "x"]) == 1.0
    assert retrieval_recall(ANS, ["x"]) == 0.0
    assert retrieval_recall(ANS, []) is None
    assert retrieval_recall(UNANS, ["x"]) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("₹6,000 per year", {"6000"}),
        ("१८ से ४० वर्ष", {"18", "40"}),
        ("Rs 1,20,000 or 2.5 acres", {"120000", "2.5"}),
        ("no numbers", set()),
    ],
)
def test_numbers_normalisation(text, expected):
    assert numbers(text) == expected


def test_number_recall():
    item = ANS.model_copy(update={"answer": "Rs 6,000 in 3 instalments"})
    assert number_recall(item, answer("₹6000, teen kishton mein")) == 0.5
    assert number_recall(item, answer("6000 in 3 parts")) == 1.0
    no_numbers = ANS.model_copy(update={"answer": "Apply at the gram panchayat office"})
    assert number_recall(no_numbers, answer("anything")) is None
    assert number_recall(item, answer("6000", refused=True)) is None


@pytest.mark.parametrize(
    "lang,text,expected",
    [
        (Language.HI, "हर साल ₹6000 मिलते हैं", True),
        (Language.HI, "You get Rs 6000", False),
        (Language.HINGLISH, "Saal ke 6000 milte hain", True),
        (Language.HINGLISH, "हर साल 6000", False),
        (Language.EN, "Rs 6000 per year (PM-KISAN योजना)", True),
        (Language.EN, "यह योजना किसानों के लिए है (for farmers)", False),
        (Language.EN, "6000", None),
    ],
)
def test_script_match(lang, text, expected):
    assert script_match(make_item("s0", 0, lang), answer(text)) is expected


def test_score_item_bundles_everything():
    s = score_item(
        ANS,
        "arm2",
        answer("Rs 1", ["s0:benefits:0"]),
        ["s0:benefits:0"],
        latency_s=1.5,
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.0,
        parse_error=False,
    )
    assert s.refusal_correct and s.citation_precision == 1.0 and s.retrieval_recall == 1.0
    assert s.latency_s == 1.5 and s.arm == "arm2"
