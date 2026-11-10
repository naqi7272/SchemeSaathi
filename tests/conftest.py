from datetime import UTC, datetime

import pytest

from saathi.schemas import Language, Passage, QAItem, QuestionType

FETCHED = datetime(2026, 10, 1, tzinfo=UTC)


def make_item(
    scheme: str, group: int, lang: Language, answerable: bool = True, section: str = "benefits"
) -> QAItem:
    return QAItem(
        qa_id=f"{scheme}-{group}-{lang}",
        group_id=f"{scheme}-{group}",
        scheme_id=scheme,
        language=lang,
        qtype=QuestionType.BENEFITS if answerable else QuestionType.UNANSWERABLE,
        question=f"q {scheme} {group} {lang}",
        answer=f"a {scheme} {group} {lang}",
        citations=[f"{scheme}:{section}:0"] if answerable else [],
        answerable=answerable,
    )


def make_passage(scheme: str, section: str = "benefits", i: int = 0) -> Passage:
    return Passage(
        passage_id=f"{scheme}:{section}:{i}",
        scheme_id=scheme,
        section=section,
        text=f"text of {scheme} {section} {i}",
        source_url=f"https://example.gov.in/{scheme}",
        fetched_at=FETCHED,
    )


@pytest.fixture
def corpus() -> list[QAItem]:
    """10 schemes x 4 groups x 3 languages; group 3 of each scheme is unanswerable."""
    return [
        make_item(f"s{s}", g, lang, answerable=(g != 3))
        for s in range(10)
        for g in range(4)
        for lang in Language
    ]
