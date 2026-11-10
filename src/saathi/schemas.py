"""Core data records shared by every stage of the pipeline."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Language(StrEnum):
    EN = "en"
    HI = "hi"  # Devanagari Hindi
    HINGLISH = "hinglish"  # romanized, code-mixed Hindi-English


class QuestionType(StrEnum):
    ELIGIBILITY = "eligibility"
    BENEFITS = "benefits"
    DOCUMENTS = "documents"
    HOW_TO_APPLY = "how_to_apply"
    UNANSWERABLE = "unanswerable"


class ManifestEntry(BaseModel):
    """One fetched source document. Written at fetch time, never edited."""

    scheme_id: str
    url: str
    fetched_at: datetime
    sha256: str
    raw_path: str
    terms_note: str  # which terms of use applied when this was fetched


class Scheme(BaseModel):
    scheme_id: str
    name: str
    source_url: str
    fetched_at: datetime
    sections: dict[str, str]  # section name -> cleaned text, e.g. "eligibility" -> "..."


class Passage(BaseModel):
    passage_id: str  # stable: "<scheme_id>:<section>:<index>"
    scheme_id: str
    section: str
    text: str
    source_url: str
    fetched_at: datetime


class QAItem(BaseModel):
    qa_id: str
    group_id: str  # same question in other languages shares a group_id; groups never cross splits
    scheme_id: str
    language: Language
    qtype: QuestionType
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)  # gold passage IDs
    answerable: bool
    # Where the item came from, e.g. "human:handwritten", "open:<dataset>",
    # "llm:<model>". None for items made before provenance was recorded.
    origin: str | None = None
    machine_made: bool = False  # True for machine translations/transliterations

    @model_validator(mode="after")
    def _check_consistency(self) -> "QAItem":
        if self.answerable and not self.citations:
            raise ValueError(f"{self.qa_id}: answerable item must cite at least one passage")
        if (self.qtype == QuestionType.UNANSWERABLE) == self.answerable:
            raise ValueError(f"{self.qa_id}: qtype 'unanswerable' must match answerable=False")
        return self


class Answer(BaseModel):
    """Structured output every arm must produce."""

    text: str
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
