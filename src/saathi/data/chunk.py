"""Turn cleaned scheme sections into passages with stable IDs."""

from saathi.schemas import Passage, Scheme


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph on whitespace into pieces of at most max_chars."""
    pieces, current = [], ""
    for word in paragraph.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def chunk_section(text: str, max_chars: int = 800) -> list[str]:
    """Greedily pack paragraphs (blank-line separated) into chunks of at most max_chars."""
    paragraphs = []
    for p in text.split("\n\n"):
        p = " ".join(p.split())
        if p:
            paragraphs.extend(_split_long(p, max_chars) if len(p) > max_chars else [p])

    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + 2 + len(p) > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def chunk_scheme(scheme: Scheme, max_chars: int = 800) -> list[Passage]:
    passages = []
    for section, text in scheme.sections.items():
        for i, chunk in enumerate(chunk_section(text, max_chars)):
            passages.append(
                Passage(
                    passage_id=f"{scheme.scheme_id}:{section}:{i}",
                    scheme_id=scheme.scheme_id,
                    section=section,
                    text=chunk,
                    source_url=scheme.source_url,
                    fetched_at=scheme.fetched_at,
                )
            )
    return passages
