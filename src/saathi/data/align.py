"""Is an answer supported by a passage? And which passage supports it?

Code-only checks, used by the open-dataset importer (to find a supporting passage for
items that have none) and by the hand-written question validator.

support() is deliberately conservative:
  1. every number in the answer (amounts, ages, years) must appear in the passage, after
     normalising Devanagari digits and Indian comma grouping;
  2. if answer and passage are in the same script, at least `min_overlap` of the answer's
     content words must appear in the passage.
For a Hindi answer over an English passage only check 1 is possible, so the result says
"numbers only" and such items are best checked by hand.
"""

from dataclasses import dataclass, field

from saathi.eval.score import DEVANAGARI, LATIN, numbers
from saathi.retrieval.tokenize import tokenize
from saathi.schemas import Passage

# Very common words that say nothing about support. Kept short and visible on purpose.
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "on", "be", "by",
    "with", "as", "at", "this", "that", "it", "can", "will", "from", "under", "per", "their",
    "you", "your", "who", "which", "have", "has", "not", "all", "any",
    "है", "हैं", "के", "की", "का", "को", "में", "से", "और", "या", "पर", "भी", "यह", "इस",
}  # fmt: skip


def _script(text: str) -> str:
    dev, lat = len(DEVANAGARI.findall(text)), len(LATIN.findall(text))
    if dev == lat == 0:
        return "none"
    return "dev" if dev > lat else "latin"


def content_words(text: str) -> set[str]:
    return {t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS and not t.isdigit()}


@dataclass
class Support:
    ok: bool
    overlap: float | None  # share of answer content words found; None if scripts differ
    missing_numbers: list[str] = field(default_factory=list)
    note: str = ""


def support(
    answer: str, passage_text: str, min_overlap: float = 0.5, check_words: bool = True
) -> Support:
    """check_words=False: numbers only. Use it for Hinglish answers, which share the Latin
    script with English passages but not their words."""
    missing = sorted(numbers(answer) - numbers(passage_text))
    if not check_words:
        return Support(
            ok=not missing,
            overlap=None,
            missing_numbers=missing,
            note="numbers only (Hinglish answer vs English passage)",
        )
    if _script(answer) != _script(passage_text) or _script(answer) == "none":
        return Support(
            ok=not missing,
            overlap=None,
            missing_numbers=missing,
            note="numbers only (answer and passage in different scripts)",
        )
    words = content_words(answer)
    overlap = len(words & content_words(passage_text)) / len(words) if words else 1.0
    return Support(
        ok=not missing and overlap >= min_overlap, overlap=overlap, missing_numbers=missing
    )


def support_any(
    answer: str, passages: list[Passage], min_overlap: float = 0.5, check_words: bool = True
) -> Support:
    """Support from the cited passages taken together (an answer may combine two)."""
    return support(answer, "\n".join(p.text for p in passages), min_overlap, check_words)


def align(
    question: str,
    answer: str,
    passages: list[Passage],
    k: int = 5,
    min_overlap: float = 0.5,
) -> list[str]:
    """Supporting passage IDs for an item that has none: BM25 over the scheme's passages
    with question+answer as the query, then the best-ranked passage that passes support().
    Returns [] when nothing passes (the caller drops or flags the item)."""
    if not passages:
        return []
    from saathi.retrieval.search import BM25Index

    ranked = BM25Index(passages).search(f"{question} {answer}", k)
    by_id = {p.passage_id: p for p in passages}
    for pid in ranked:
        s = support(answer, by_id[pid].text, min_overlap)
        if s.ok and s.overlap is not None:  # same-script, word-level evidence required
            return [pid]
    return []
