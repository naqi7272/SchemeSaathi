"""Code-based scoring for one prediction. DRAFT: metric definitions await sign-off
(docs/EVAL_METRICS.md). Each function is small on purpose; read them with the spec.

None means "not applicable" (e.g. citation precision for a closed-book arm, or for a
refused answer) and is excluded from averages, never counted as 0 or 1.
"""

import re

from pydantic import BaseModel

from saathi.schemas import Answer, Language, QAItem

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")
# Devanagari digits ०-९ -> 0-9
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def refusal_correct(item: QAItem, pred: Answer) -> bool:
    """True when the model refuses exactly when the question is unanswerable."""
    return pred.refused == (not item.answerable)


def citation_precision(item: QAItem, pred: Answer) -> float | None:
    """Share of cited IDs that are gold. N/A if unanswerable, refused, or nothing cited."""
    if not item.answerable or pred.refused or not pred.citations:
        return None
    cited = set(pred.citations)
    return len(cited & set(item.citations)) / len(cited)


def citation_recall(item: QAItem, pred: Answer) -> float | None:
    """Share of gold IDs that were cited. N/A if unanswerable or refused."""
    if not item.answerable or pred.refused:
        return None
    gold = set(item.citations)
    return len(set(pred.citations) & gold) / len(gold)


def citations_grounded(pred: Answer, retrieved: list[str]) -> bool | None:
    """True if every cited ID was actually in the context shown to the model.
    Catches invented IDs. N/A when nothing is cited."""
    if not pred.citations:
        return None
    return set(pred.citations) <= set(retrieved)


def retrieval_recall(item: QAItem, retrieved: list[str]) -> float | None:
    """Share of gold passages present in the retrieved context. N/A without retrieval."""
    if not item.answerable or not retrieved:
        return None
    gold = set(item.citations)
    return len(set(retrieved) & gold) / len(gold)


def numbers(text: str) -> set[str]:
    """Numbers in a text, with Devanagari digits and thousands separators normalised.
    '₹6,000' -> {'6000'}; '१८ वर्ष' -> {'18'}."""
    text = text.translate(DEVANAGARI_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    return set(NUMBER.findall(text))


def number_recall(item: QAItem, pred: Answer) -> float | None:
    """Share of numbers in the gold answer that also appear in the prediction.
    A cheap, code-only check of key facts (amounts, ages, durations). N/A when the gold
    answer has no numbers, the item is unanswerable, or the model refused."""
    if not item.answerable or pred.refused:
        return None
    gold = numbers(item.answer)
    if not gold:
        return None
    return len(gold & numbers(pred.text)) / len(gold)


def script_match(item: QAItem, pred: Answer) -> bool | None:
    """Hindi questions should be answered mostly in Devanagari; English and Hinglish mostly in
    Latin script (scheme names in the other script are fine). Majority rule on letters.
    Checks script only: it cannot tell English from Hinglish."""
    letters_dev = len(DEVANAGARI.findall(pred.text))
    letters_lat = len(LATIN.findall(pred.text))
    total = letters_dev + letters_lat
    if total == 0:
        return None
    share_dev = letters_dev / total
    if item.language == Language.HI:
        return share_dev >= 0.5
    return share_dev < 0.5


class ItemScore(BaseModel):
    qa_id: str
    arm: str
    language: Language
    answerable: bool
    refusal_correct: bool
    citation_precision: float | None
    citation_recall: float | None
    citations_grounded: bool | None
    retrieval_recall: float | None
    number_recall: float | None
    has_gold_numbers: bool = False  # default keeps older scores.jsonl readable
    script_match: bool | None
    parse_error: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


def score_item(
    item: QAItem,
    arm: str,
    pred: Answer,
    retrieved: list[str],
    latency_s: float,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    parse_error: bool,
) -> ItemScore:
    return ItemScore(
        qa_id=item.qa_id,
        arm=arm,
        language=item.language,
        answerable=item.answerable,
        refusal_correct=refusal_correct(item, pred),
        citation_precision=citation_precision(item, pred),
        citation_recall=citation_recall(item, pred),
        citations_grounded=citations_grounded(pred, retrieved),
        retrieval_recall=retrieval_recall(item, retrieved),
        number_recall=number_recall(item, pred),
        has_gold_numbers=bool(numbers(item.answer)),
        script_match=script_match(item, pred),
        parse_error=parse_error,
        latency_s=latency_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
    )
