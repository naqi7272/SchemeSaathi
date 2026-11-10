"""Arms 1-4. Framework-free: a model, optionally a retriever, and the shared prompts.

arm 1  prompt-only     ClosedBookArm(base model)
arm 2  RAG             RagArm(base model, retriever)
arm 3  LoRA SFT        ClosedBookArm(fine-tuned model)
arm 4  RAG + SFT       RagArm(fine-tuned model, retriever)
arm 2b RAG + query translation  RagArm(base model, retriever, translate_query=True):
       Hindi/Hinglish questions are first rewritten by the same model into an English
       search query (the documents are English); the answer still uses the original question.
"""

import time
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from saathi.arms.prompts import closed_book_messages, query_rewrite_messages, rag_messages
from saathi.llm.base import ChatModel, Usage
from saathi.schemas import Answer, Passage


class ArmOutput(BaseModel):
    answer: Answer
    retrieved: list[str] = Field(default_factory=list)  # passage IDs shown to the model
    usage: list[Usage]
    latency_s: float  # end to end, including retrieval
    parse_error: str | None = None
    search_query: str | None = None  # set when the query was rewritten before retrieval


class SearchQuery(BaseModel):
    query: str


class Arm(Protocol):
    name: str

    def answer(self, question: str) -> ArmOutput: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> list[Passage]: ...


def parse_answer(text: str) -> tuple[Answer, str | None]:
    """Parse model JSON. On failure, keep the raw text as a non-refusal answer and record why,
    so a format failure is scored as a wrong answer, not silently dropped."""
    try:
        return Answer.model_validate_json(text), None
    except ValidationError as err:
        return Answer(text=text, citations=[], refused=False), str(err)[:300]


class ClosedBookArm:
    def __init__(self, name: str, model: ChatModel, max_tokens: int = 512):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str) -> ArmOutput:
        start = time.perf_counter()
        system, messages = closed_book_messages(question)
        result = self.model.chat(system, messages, schema=Answer, max_tokens=self.max_tokens)
        answer, error = parse_answer(result.text)
        answer.citations = []  # closed-book arms have nothing to cite
        return ArmOutput(
            answer=answer,
            usage=[result.usage],
            latency_s=time.perf_counter() - start,
            parse_error=error,
        )


class RagArm:
    def __init__(
        self,
        name: str,
        model: ChatModel,
        retriever: Retriever,
        k: int = 5,
        max_tokens: int = 512,
        translate_query: bool = False,
    ):
        self.name = name
        self.model = model
        self.retriever = retriever
        self.k = k
        self.max_tokens = max_tokens
        self.translate_query = translate_query

    def search_query(self, question: str) -> tuple[str | None, list[Usage]]:
        """English search query for non-English questions (None = search with the question).
        English questions are never rewritten, so they behave exactly as in arm 2."""
        from saathi.agent.language import detect_language
        from saathi.schemas import Language

        if not self.translate_query or detect_language(question) == Language.EN:
            return None, []
        system, messages = query_rewrite_messages(question)
        result = self.model.chat(system, messages, schema=SearchQuery, max_tokens=128)
        try:
            query = SearchQuery.model_validate_json(result.text).query.strip()
        except ValidationError:
            query = ""
        return (query or None), [result.usage]

    def answer(self, question: str) -> ArmOutput:
        start = time.perf_counter()
        query, usage = self.search_query(question)
        passages = self.retriever.retrieve(query or question, k=self.k)
        system, messages = rag_messages(question, passages)  # the answer sees the original
        result = self.model.chat(system, messages, schema=Answer, max_tokens=self.max_tokens)
        answer, error = parse_answer(result.text)
        return ArmOutput(
            answer=answer,
            retrieved=[p.passage_id for p in passages],
            usage=[*usage, result.usage],
            latency_s=time.perf_counter() - start,
            parse_error=error,
            search_query=query,
        )
