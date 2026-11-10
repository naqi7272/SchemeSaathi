import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from saathi.data.ingest_handwritten import COLUMNS, read_rows, validate
from saathi.io import read_jsonl, write_jsonl
from saathi.schemas import Language, Passage, QAItem, Scheme
from saathi.verify.cli import sample_groups

T = datetime(2026, 10, 1, tzinfo=UTC)
PASSAGES = {
    "pmk:benefits:0": Passage(
        passage_id="pmk:benefits:0",
        scheme_id="pmk",
        section="benefits",
        text="Eligible farmer families receive Rs 6,000 per year in three instalments.",
        source_url="u",
        fetched_at=T,
    ),
    "apy:benefits:0": Passage(
        passage_id="apy:benefits:0",
        scheme_id="apy",
        section="benefits",
        text="A monthly pension of Rs 1,000 to Rs 5,000 from age 60.",
        source_url="u",
        fetched_at=T,
    ),
}
SCHEMES = {"pmk", "apy"}


def row(**kw) -> dict:
    base = {
        "group_id": "g1",
        "scheme_id": "pmk",
        "language": "en",
        "qtype": "benefits",
        "question": "How much do farmers get?",
        "answer": "Farmer families receive Rs 6,000 per year.",
        "passage_ids": "pmk:benefits:0",
        "notes": "",
    }
    return base | kw


def run(rows):
    return validate(list(enumerate(rows, start=2)), SCHEMES, PASSAGES)


def test_valid_group_in_two_languages():
    items, errors, manual = run(
        [
            row(),
            row(language="hi", question="कितना पैसा?", answer="किसान परिवारों को ₹6,000 मिलते हैं।"),
            row(group_id="g2", qtype="unanswerable", passage_ids="", answer="Not stated."),
        ]
    )
    assert errors == []
    assert [it.qa_id for it in items] == ["hw-g1-en", "hw-g1-hi", "hw-g2-en"]
    assert all(it.origin == "human:handwritten" for it in items)
    assert items[2].answerable is False and items[2].citations == []
    assert len(manual) == 1 and "line 3" in manual[0]  # Hindi answer: numbers only


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"language": "fr"}, "language"),
        ({"qtype": "trivia"}, "qtype"),
        ({"scheme_id": "nope"}, "not in schemes.jsonl"),
        ({"passage_ids": "pmk:benefits:9"}, "does not exist"),
        ({"passage_ids": "apy:benefits:0"}, "another scheme"),
        ({"passage_ids": ""}, "needs at least one passage_id"),
        ({"qtype": "unanswerable"}, "must have empty passage_ids"),
        ({"answer": "Farmer families receive Rs 8,000 per year."}, "numbers not in passage"),
        ({"answer": "Pension is paid monthly after retirement."}, "answer words"),
        ({"question": ""}, "question is empty"),
    ],
)
def test_errors_name_the_line(bad, message):
    items, errors, _ = run([row(**bad)])
    assert items == []
    assert errors and errors[0].startswith("line 2") and message in " ".join(errors)


def test_hinglish_answer_is_number_checked_only():
    ok = row(language="hinglish", answer="Kisan parivaar ko saal mein Rs 6,000 milte hain.")
    items, errors, manual = run([ok])
    assert errors == [] and len(items) == 1 and "Hinglish" in manual[0]
    bad = row(language="hinglish", answer="Kisan parivaar ko saal mein Rs 9,000 milte hain.")
    assert run([bad])[1]  # a wrong number is still an error


def test_drafted_origin():
    items, errors, _ = validate([(2, row())], SCHEMES, PASSAGES, origin="drafted:claude-opus-5-5")
    assert errors == [] and items[0].origin == "drafted:claude-opus-5-5"


def test_group_consistency():
    _, errors, _ = run([row(), row(scheme_id="apy", passage_ids="apy:benefits:0",
                                   answer="A monthly pension of Rs 1,000.")])  # fmt: skip
    assert any("mixes schemes" in e for e in errors) and any("twice" in e for e in errors)


def test_blank_lines_ignored():
    items, errors, _ = run([row(), {c: "" for c in COLUMNS}])
    assert len(items) == 1 and errors == []


def test_template_header_and_excel_bom(tmp_path):
    template = Path("data/qa/handwritten_template.csv")
    with template.open(encoding="utf-8") as f:
        assert next(csv.reader(f)) == COLUMNS
    path = tmp_path / "x.csv"
    path.write_text("﻿" + ",".join(COLUMNS) + "\n" + ",".join(row().values()) + "\n",
                    encoding="utf-8")  # fmt: skip
    assert read_rows(path)[0][1]["group_id"] == "g1"
    bad = tmp_path / "bad.csv"
    bad.write_text("question,answer\nq,a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        read_rows(bad)


def test_unquoted_comma_gives_a_clear_error(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        ",".join(COLUMNS) + "\ng1,pmk,en,benefits,Q?,Rs 6,000 per year,pmk:benefits:0,\n",
        encoding="utf-8",
    )
    _, errors, _ = validate(read_rows(path), SCHEMES, PASSAGES)
    assert errors == [
        "line 2: more cells than columns; put any text containing a comma in double quotes "
        '("like, this"), or save the file from Excel/Sheets as CSV'
    ]


def test_cli_end_to_end(tmp_path):
    csv_path = tmp_path / "hw.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerow(row())
        w.writerow(row(group_id="g9", language="xx"))
    schemes, passages = tmp_path / "s.jsonl", tmp_path / "p.jsonl"
    write_jsonl(schemes, [Scheme(scheme_id=s, name=s, source_url="u", fetched_at=T, sections={})
                          for s in SCHEMES])  # fmt: skip
    write_jsonl(passages, PASSAGES.values())
    out, manifest = tmp_path / "hw.jsonl", tmp_path / "m.jsonl"
    cmd = [sys.executable, "-m", "saathi.data.ingest_handwritten", str(csv_path),
           "--schemes", str(schemes), "--passages", str(passages), "--out", str(out),
           "--manifest", str(manifest)]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0 and "line 3" in res.stdout and not out.exists()
    res = subprocess.run([*cmd, "--skip-invalid"], capture_output=True, text=True,
                         encoding="utf-8")  # fmt: skip
    assert res.returncode == 0, res.stderr
    assert len(read_jsonl(out, QAItem)) == 1
    rec = json.loads(manifest.read_text(encoding="utf-8"))
    assert rec["provenance"] == "human" and rec["license"] == "project-own"


# ------------------------------------------------------------- saathi-verify --groups any
def _item(group: str, lang: Language) -> QAItem:
    return QAItem(qa_id=f"{group}-{lang}", group_id=group, scheme_id="pmk", language=lang,
                  qtype="benefits", question="q", answer="a", citations=["pmk:benefits:0"],
                  answerable=True)  # fmt: skip


def test_sample_any_groups_keeps_groups_whole_and_respects_n():
    items = [_item(f"g{i}", Language.EN) for i in range(50)]
    items += [_item(f"t{i}", lang) for i in range(20) for lang in (Language.EN, Language.HI)]
    picked = sample_groups(items, 60, seed=1, complete_only=False)
    assert len(picked) <= 60 and len(picked) >= 58
    groups = {it.group_id for it in picked}
    for g in groups:  # whole groups only
        assert sum(it.group_id == g for it in picked) == sum(it.group_id == g for it in items)
    assert sample_groups(items, 60, seed=1, complete_only=False) == picked  # deterministic
    with pytest.raises(ValueError, match="only 90 exist"):
        sample_groups(items, 100, seed=1, complete_only=False)
    with pytest.raises(ValueError, match="complete"):
        sample_groups(items, 60, seed=1)  # default still needs 3-language groups
