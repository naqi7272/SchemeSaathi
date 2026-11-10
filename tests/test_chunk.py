from saathi.data.chunk import chunk_scheme, chunk_section
from saathi.schemas import Scheme
from tests.conftest import FETCHED


def test_short_paragraphs_are_packed_together():
    assert chunk_section("one\n\ntwo\n\nthree", max_chars=100) == ["one\n\ntwo\n\nthree"]


def test_chunks_respect_max_chars():
    text = "\n\n".join(["word " * 30] * 5)
    chunks = chunk_section(text, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_oversized_paragraph_is_split_without_losing_words():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_section(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_devanagari_text_is_preserved():
    text = "पात्रता: किसान परिवार।\n\nलाभ: ₹6000 प्रति वर्ष।"
    assert chunk_section(text) == ["पात्रता: किसान परिवार।\n\nलाभ: ₹6000 प्रति वर्ष।"]


def test_empty_paragraphs_are_dropped():
    assert chunk_section("\n\n  \n\nhello\n\n\n\n") == ["hello"]


def test_passage_ids_are_stable_and_unique():
    scheme = Scheme(
        scheme_id="pm-kisan",
        name="PM-KISAN",
        source_url="https://example.gov.in/pm-kisan",
        fetched_at=FETCHED,
        sections={"eligibility": "a\n\nb", "benefits": "x " * 500},
    )
    first = chunk_scheme(scheme, max_chars=300)
    second = chunk_scheme(scheme, max_chars=300)
    ids = [p.passage_id for p in first]
    assert ids == [p.passage_id for p in second]
    assert len(ids) == len(set(ids))
    assert ids[0] == "pm-kisan:eligibility:0"
    assert all(p.source_url == scheme.source_url and p.fetched_at == FETCHED for p in first)
