"""LLM judge, used only for what code cannot score:
  correctness   does the answer convey the same facts as the gold answer?
  faithfulness  is every claim supported by the source passages?

Only answerable items where the model did not refuse are judged; refusals are scored by
code. The judge sees the gold answer, the gold passages, and the context the model was
shown (retrieved passages for RAG arms; for closed-book arms there is no context, so
faithfulness is judged against the gold passages, i.e. against the official source).

The judge must be validated against human labels (label_judge.py, Cohen's kappa) before
its numbers are reported.
"""

from typing import Literal

from pydantic import BaseModel

from saathi.llm.base import ChatModel, Usage
from saathi.schemas import Answer, Passage, QAItem

SYSTEM = """You grade answers to questions about Indian government schemes. Questions and \
answers may be in English, Hindi (Devanagari), or Hinglish (Roman-script Hindi-English); \
judge meaning, not language or style.

correctness, compared with the GOLD ANSWER:
  correct            all key facts of the gold answer are present and nothing contradicts it
  partially_correct  some key facts are present, others missing; nothing wrong is stated
  incorrect          a key fact is wrong or contradicts the gold answer, or the answer does
                     not address the question
faithful, compared with the SOURCE PASSAGES only:
  yes  every factual claim in the answer is supported by the source passages
  no   at least one claim is not supported (list it in unsupported_claims)
Extra correct-but-unrequested detail is fine for correctness, but must still be supported
for faithfulness. Be strict about numbers, ages, amounts, and document names."""


class Verdict(BaseModel):
    correctness: Literal["correct", "partially_correct", "incorrect"]
    faithful: Literal["yes", "no"]
    unsupported_claims: list[str]
    reason: str


class Judgment(BaseModel):
    run_id: str
    qa_id: str
    verdict: Verdict
    judge_model: str
    usage: Usage


def needs_judging(item: QAItem, pred: Answer) -> bool:
    return item.answerable and not pred.refused


def build_prompt(item: QAItem, pred: Answer, gold: list[Passage], context: list[Passage]) -> str:
    def block(passages: list[Passage]) -> str:
        return "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in passages)

    sources = context if context else gold
    return (
        f"QUESTION:\n{item.question}\n\n"
        f"GOLD ANSWER:\n{item.answer}\n\n"
        f"GOLD PASSAGES:\n{block(gold)}\n\n"
        f"SOURCE PASSAGES (what the answer must be supported by):\n{block(sources)}\n\n"
        f"ANSWER TO GRADE:\n{pred.text}"
    )


def judge(
    model: ChatModel,
    run_id: str,
    item: QAItem,
    pred: Answer,
    gold: list[Passage],
    context: list[Passage],
) -> Judgment:
    result = model.chat(
        SYSTEM,
        [{"role": "user", "content": build_prompt(item, pred, gold, context)}],
        schema=Verdict,
        max_tokens=8000,
    )
    return Judgment(
        run_id=run_id,
        qa_id=item.qa_id,
        judge_model=result.usage.served_model,
        verdict=Verdict.model_validate_json(result.text),
        usage=result.usage,
    )
