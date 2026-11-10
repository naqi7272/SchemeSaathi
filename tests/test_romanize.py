import csv
import importlib.util

import pytest

from saathi.data.romanize import casual_word, export_sample, make_variants, romanize
from saathi.schemas import Language, QAItem

# Tiny fake ITRANS so the core logic is tested without the optional dependency.
FAKE = {"किसान": "kisAna", "में": "meM", "पैसा": "paisA", "कितना": "kitanA", "है": "hai"}


def fake(word: str) -> str:
    return FAKE[word]


def test_casual_rules_with_fake_transliterator():
    assert casual_word("किसान", fake) == "kisan"  # final schwa dropped, long a -> a
    assert casual_word("में", fake) == "men"  # anusvara -> n, no schwa to drop (ends in matra)
    assert romanize("किसान कितना पैसा है। ₹६,०००", fake) == "kisan kitana paisa hai. ₹6,000"
    assert romanize("किसान", fake) == "kisan"
    assert "6,000" in romanize("₹६,००० है", fake)


def item(gid: str, lang: Language, q: str = "किसान") -> QAItem:
    return QAItem(qa_id=f"{gid}-{lang}", group_id=gid, scheme_id="s", language=lang,
                  qtype="benefits", question=q, answer="किसान", citations=["s:b:0"],
                  answerable=True)  # fmt: skip


def test_variants_share_group_and_are_marked():
    items = [
        item("g1", Language.EN, "How much?"),
        item("g1", Language.HI),
        item("g2", Language.HI),
        item("g2", Language.HINGLISH, "kisan"),
    ]
    variants, skipped = make_variants(items, fake)
    assert skipped == 1  # g2 already has a human hinglish item
    (v,) = variants
    assert v.qa_id == "g1-hi-rom" and v.group_id == "g1"
    assert v.language == Language.HINGLISH and v.machine_made
    assert v.question == "kisan" and v.origin.startswith("machine_translit")


def test_export_sample(tmp_path):
    src = item("g1", Language.HI)
    (v,) = make_variants([src], fake)[0]
    out = tmp_path / "check.csv"
    export_sample([v], {src.qa_id: src}, n=5, seed=0, out=out)
    with out.open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    assert rows[0][:3] == ["qa_id", "hindi_question", "romanized_question"]
    assert rows[1][:3] == ["g1-hi-rom", "किसान", "kisan"]


@pytest.mark.skipif(
    importlib.util.find_spec("indic_transliteration") is None, reason="translit extra missing"
)
def test_real_transliterator():
    assert romanize("पीएम किसान योजना में कितना पैसा मिलता है?") == (
        "piem kisan yojana men kitana paisa milata hai?"
    )
