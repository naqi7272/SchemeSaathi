import pytest
from pydantic import ValidationError

from saathi.schemas import Language, QAItem, QuestionType
from tests.conftest import make_item


def test_answerable_item_needs_citation():
    data = make_item("s0", 0, Language.EN).model_dump() | {"citations": []}
    with pytest.raises(ValidationError, match="cite at least one"):
        QAItem.model_validate(data)


def test_unanswerable_qtype_must_match_flag():
    data = make_item("s0", 0, Language.EN).model_dump() | {"qtype": QuestionType.UNANSWERABLE}
    with pytest.raises(ValidationError, match="must match"):
        QAItem.model_validate(data)


def test_unanswerable_without_citations_is_valid():
    item = make_item("s0", 0, Language.HINGLISH, answerable=False)
    assert item.citations == []
    assert item.qtype == QuestionType.UNANSWERABLE
