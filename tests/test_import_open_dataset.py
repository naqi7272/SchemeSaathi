import csv
import json
from datetime import UTC, datetime

import pytest
import yaml

from saathi.data.align import align, support
from saathi.data.build_splits import build
from saathi.data.import_open_dataset import convert, load_config, run
from saathi.data.provenance import LicenseError, check_license
from saathi.io import read_jsonl, write_jsonl
from saathi.schemas import Language, Passage, QAItem, Scheme

T = datetime(2026, 10, 1, tzinfo=UTC)


def P(pid: str, text: str) -> Passage:
    return Passage(
        passage_id=pid,
        scheme_id=pid.split(":")[0],
        section=pid.split(":")[1],
        text=text,
        source_url="https://example.gov.in",
        fetched_at=T,
    )


PASSAGES = [
    P("pm-kisan:benefits:0", "Eligible farmer families receive Rs 6,000 per year in three "
      "equal instalments of Rs 2,000 each, paid directly into their bank accounts."),
    P("pm-kisan:eligibility:0", "All landholding farmer families are eligible. Institutional "
      "landholders and income tax payers are excluded."),
    P("apy:benefits:0", "Subscribers get a guaranteed monthly pension between Rs 1,000 and "
      "Rs 5,000 from the age of 60."),
    P("apy:eligibility:0", "Any citizen aged 18 to 40 years with a savings bank account can join."),
]  # fmt: skip
SCHEMES = [
    Scheme(scheme_id="pm-kisan", name="PM Kisan Samman Nidhi", source_url="u", fetched_at=T,
           sections={}),
    Scheme(scheme_id="apy", name="Atal Pension Yojana", source_url="u", fetched_at=T, sections={}),
]  # fmt: skip


# ------------------------------------------------------------- license gate
def test_license_gate():
    assert check_license("cc-by-4.0", "https://huggingface.co/x", "human") == []
    with pytest.raises(LicenseError, match="unclear"):
        check_license("", "https://x", "human")
    with pytest.raises(LicenseError, match="unclear"):
        check_license("unknown", "https://x", "human")
    with pytest.raises(LicenseError, match="not in the known open list"):
        check_license("my-custom-license", "https://x", "human")
    with pytest.raises(LicenseError, match="forbid bots"):
        check_license("cc-by-4.0", "https://www.myscheme.gov.in/schemes/x", "scraped")
    assert "non-commercial" in check_license("cc-by-nc-4.0", "https://x", "human")[0]
    assert "ORIGINAL site" in check_license("cc-by-4.0", "https://x", "scraped")[0]


# ------------------------------------------------------------- support / align
def test_support_numbers_and_words():
    text = PASSAGES[0].text
    assert support("Farmer families get Rs 6000 per year.", text).ok
    bad = support("Farmer families get Rs 8000 per year.", text)
    assert not bad.ok and bad.missing_numbers == ["8000"]
    assert not support("The pension is paid monthly from sixty.", text).ok  # words absent


def test_support_cross_script_is_numbers_only():
    s = support("किसान परिवारों को हर साल ₹6,000 मिलते हैं।", PASSAGES[0].text)
    assert s.ok and s.overlap is None and "numbers only" in s.note
    assert not support("किसान परिवारों को हर साल ₹9,000 मिलते हैं।", PASSAGES[0].text).ok


def test_align_picks_supporting_passage_or_nothing():
    pk = [p for p in PASSAGES if p.scheme_id == "pm-kisan"]
    assert align("How much money?", "Farmer families receive Rs 6,000 per year.", pk) == [
        "pm-kisan:benefits:0"
    ]
    assert align("Helpline?", "Call the helpline 155261 for help.", pk) == []


# ------------------------------------------------------------- convert
def cfg_dict(**over) -> dict:
    base = {
        "name": "toy",
        "format": "csv",
        "path": "x.csv",
        "license": "cc-by-4.0",
        "source_url": "https://example.org/toy",
        "provenance": "human",
        "columns": {"question": "q", "answer": "a", "scheme": "scheme"},
        "language_default": None,
    }
    return base | over


def make_cfg(tmp_path, **over):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg_dict(**over)), encoding="utf-8")
    return load_config(path)[0]


def test_convert_resolves_schemes_aligns_and_counts_drops(tmp_path):
    cfg = make_cfg(tmp_path, scheme_map={"Kisan": "pm-kisan"})
    rows = [
        {"q": "How much do farmers get?", "a": "Farmer families receive Rs 6,000 per year.",
         "scheme": "PM Kisan Samman Nidhi"},
        {"q": "Who can join?", "a": "Any citizen aged 18 to 40 with a savings bank account.",
         "scheme": "apy"},
        {"q": "Instalment?", "a": "Three instalments of Rs 2,000 each.", "scheme": "Kisan"},
        {"q": "Helpline?", "a": "Call 1800 000 0000 anytime.", "scheme": "apy"},
        {"q": "What?", "a": "Something.", "scheme": "Unknown Scheme"},
        {"q": "", "a": "x", "scheme": "apy"},
    ]  # fmt: skip
    items, drops = convert(cfg, rows, SCHEMES, PASSAGES)
    assert [it.scheme_id for it in items] == ["pm-kisan", "apy", "pm-kisan"]
    assert items[0].citations == ["pm-kisan:benefits:0"]
    assert items[1].citations == ["apy:eligibility:0"]
    assert all(it.language == Language.EN and it.origin == "open:toy" for it in items)
    assert drops == {
        "could not align to a passage": 1,
        "scheme not in our corpus": 1,
        "empty question or answer": 1,
    }


def test_convert_with_given_passages_languages_and_groups(tmp_path):
    cfg = make_cfg(
        tmp_path,
        columns={"question": "q", "answer": "a", "scheme": "s", "language": "lang",
                 "passages": "pids", "group": "g"},
        language_map={"English": "en", "Hindi": "hi"},
    )  # fmt: skip
    rows = [
        {"q": "How much?", "a": "Rs 6,000 per year for farmer families.", "s": "pm-kisan",
         "lang": "English", "pids": "pm-kisan:benefits:0", "g": "7"},
        {"q": "कितना पैसा?", "a": "किसान परिवारों को सालाना ₹6,000।", "s": "pm-kisan",
         "lang": "Hindi", "pids": ["pm-kisan:benefits:0"], "g": "7"},
        {"q": "How much?", "a": "Rs 9,000.", "s": "pm-kisan", "lang": "English",
         "pids": "pm-kisan:benefits:0", "g": "8"},
        {"q": "Q", "a": "A", "s": "pm-kisan", "lang": "English", "pids": "apy:benefits:0",
         "g": "9"},
        {"q": "Q", "a": "A", "s": "pm-kisan", "lang": "Klingon", "g": "10"},
    ]  # fmt: skip
    items, drops = convert(cfg, rows, SCHEMES, PASSAGES)
    assert len(items) == 2 and items[0].group_id == items[1].group_id
    assert {it.language for it in items} == {Language.EN, Language.HI}
    assert drops == {
        "answer not supported by cited passage": 1,
        "cited passage not in our corpus": 1,
        "unknown language": 1,
    }


def test_run_writes_items_and_manifest_and_refuses_bad_license(tmp_path):
    data = tmp_path / "toy.csv"
    with data.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["q", "a", "scheme"])
        w.writeheader()
        w.writerow({"q": "How much?", "a": "Farmer families receive Rs 6,000 per year.",
                    "scheme": "pm-kisan"})  # fmt: skip
    schemes, passages = tmp_path / "schemes.jsonl", tmp_path / "passages.jsonl"
    write_jsonl(schemes, SCHEMES)
    write_jsonl(passages, PASSAGES)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(cfg_dict(path=str(data), generator_model="qwen2.5-72b-instruct")),
        encoding="utf-8",
    )
    out, manifest = tmp_path / "gen", tmp_path / "manifest.jsonl"
    items, _ = run(cfg, schemes, passages, out, manifest)
    assert read_jsonl(out / "open-toy.jsonl", QAItem) == items
    rec = json.loads(manifest.read_text(encoding="utf-8"))
    assert rec["license"] == "cc-by-4.0" and rec["n_items_out"] == 1
    assert "same family" in rec["generator_family_warning"]

    cfg.write_text(yaml.safe_dump(cfg_dict(path=str(data), license="")), encoding="utf-8")
    with pytest.raises(LicenseError):
        run(cfg, schemes, passages, out, manifest)


def test_dry_run_writes_nothing(tmp_path):
    data = tmp_path / "toy.jsonl"
    data.write_text(json.dumps({"q": "Q?", "a": "Rs 6,000 per year to farmer families.",
                                "scheme": "pm-kisan"}) + "\n", encoding="utf-8")  # fmt: skip
    schemes, passages = tmp_path / "s.jsonl", tmp_path / "p.jsonl"
    write_jsonl(schemes, SCHEMES)
    write_jsonl(passages, PASSAGES)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(cfg_dict(format="jsonl", path=str(data))), encoding="utf-8")
    items, _ = run(cfg, schemes, passages, tmp_path / "gen", tmp_path / "m.jsonl", dry_run=True)
    assert len(items) == 1
    assert not (tmp_path / "gen").exists() and not (tmp_path / "m.jsonl").exists()


def test_bad_config_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        make_cfg(tmp_path, name="Bad Name!")


# ------------------------------------------------------------- forced test schemes
def test_build_forces_handwritten_schemes_into_test(tmp_path, corpus):
    summary = build(corpus, 0.3, 13, tmp_path, force_test=["s0", "s1"])
    assert {"s0", "s1"} <= set(summary["test_schemes"])
    assert not {"s0", "s1"} & set(summary["train_schemes"])
    train_seen = read_jsonl(tmp_path / "train_seen.jsonl", QAItem)
    assert not {it.scheme_id for it in train_seen} & {"s0", "s1"}
    with pytest.raises(ValueError, match="nothing left"):
        build(corpus, 0.3, 13, tmp_path, force_test=[f"s{i}" for i in range(10)])
