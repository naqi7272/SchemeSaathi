"""OPTIONAL: romanized-Hindi variants of Hindi (Devanagari) questions, for $0.

    uv sync --extra translit
    uv run python -m saathi.data.romanize make --qa data/qa/handwritten.jsonl
    uv run python -m saathi.data.romanize sample --qa data/qa/handwritten-romanized.jsonl --n 30

`make` writes <input stem>-romanized.jsonl next to the input. Each variant:
  - language "hinglish", same group_id as its source (translations stay together, so the
    leakage checks and paired comparisons treat them as one question);
  - qa_id "<source qa_id>-rom", machine_made=True, origin "machine_translit:...";
  - skipped if the group already has a hinglish item (a human one is better).
`sample` exports a CSV for you to hand-check (columns ok / fix left blank).

IMPORTANT: this is rule-based transliteration (indic-transliteration, MIT, pure Python)
plus simple casual-spelling rules. It changes the SCRIPT, not the language: real Hinglish
mixes in English words ("form bharna hai", "eligible hoon kya") and spells freely. Results
on these items must be reported separately and described as "romanized Hindi", not
"natural Hinglish".
"""

import argparse
import csv
import random
import re
from collections.abc import Callable
from pathlib import Path

from saathi.data.provenance import MANIFEST, ManifestRecord, now, record
from saathi.eval.score import DEVANAGARI_DIGITS
from saathi.io import read_jsonl, write_jsonl
from saathi.schemas import Language, QAItem

TOOL = "indic-transliteration (ITRANS) + casual rules"
DEV_WORD = re.compile(r"[ऀ-ॿ]+")
# Consonants (with nukta forms). A word ending in one of these has an unwritten final "a".
CONSONANT = re.compile(r"[क-हक़-य़]$")
# ITRANS -> casual Roman, applied in order, before lower-casing.
CASUAL = [
    ("j~n", "gy"),
    (".N", "n"),
    (".n", "n"),
    ("~", "n"),
    ("M", "n"),
    (".h", ""),
    ("R^i", "ri"),
    ("RRi", "ri"),
    ("A", "a"),
    ("I", "i"),
    ("U", "u"),
    ("Ch", "chh"),
    ("Sh", "sh"),
    ("w", "v"),
]


def _itrans(word: str) -> str:
    try:
        from indic_transliteration import sanscript
    except ImportError as err:
        raise SystemExit("romanizing needs the `translit` extra: uv sync --extra translit") from err
    return sanscript.transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)


def casual_word(word: str, to_itrans: Callable[[str], str] = _itrans) -> str:
    r = to_itrans(word)
    if len(word) > 1 and CONSONANT.search(word) and r.endswith("a"):
        r = r[:-1]  # drop the unwritten word-final vowel: "kisAna" -> "kisAn"
    for a, b in CASUAL:
        r = r.replace(a, b)
    return r.lower()


def romanize(text: str, to_itrans: Callable[[str], str] = _itrans) -> str:
    text = text.translate(DEVANAGARI_DIGITS).replace("।", ".")
    return DEV_WORD.sub(lambda m: casual_word(m.group(0), to_itrans), text)


def make_variants(
    items: list[QAItem], to_itrans: Callable[[str], str] = _itrans
) -> tuple[list[QAItem], int]:
    """Variants for Hindi items whose group has no hinglish item. Returns (variants, skipped)."""
    has_hinglish = {it.group_id for it in items if it.language == Language.HINGLISH}
    out, skipped = [], 0
    for it in items:
        if it.language != Language.HI:
            continue
        if it.group_id in has_hinglish:
            skipped += 1
            continue
        out.append(
            it.model_copy(
                update={
                    "qa_id": f"{it.qa_id}-rom",
                    "language": Language.HINGLISH,
                    "question": romanize(it.question, to_itrans),
                    "answer": romanize(it.answer, to_itrans),
                    "machine_made": True,
                    "origin": f"machine_translit:{TOOL}",
                }
            )
        )
    return out, skipped


def export_sample(variants: list[QAItem], sources: dict[str, QAItem], n: int, seed: int, out: Path):
    picked = sorted(variants, key=lambda it: it.qa_id)
    random.Random(seed).shuffle(picked)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: opens in Excel
        w = csv.writer(f)
        w.writerow(["qa_id", "hindi_question", "romanized_question", "ok (y/n)", "fix"])
        for it in picked[:n]:
            src = sources.get(it.qa_id.removesuffix("-rom"))
            w.writerow([it.qa_id, src.question if src else "", it.question, "", ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--qa", type=Path, required=True)
    m.add_argument("--manifest", type=Path, default=MANIFEST)
    s = sub.add_parser("sample")
    s.add_argument("--qa", type=Path, required=True, help="the *-romanized.jsonl file")
    s.add_argument("--sources", type=Path, help="original file (default: name without suffix)")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, default=Path("data/labels/romanize_check.csv"))
    args = ap.parse_args()

    if args.cmd == "make":
        items = read_jsonl(args.qa, QAItem)
        variants, skipped = make_variants(items)
        out = args.qa.with_name(f"{args.qa.stem}-romanized.jsonl")
        write_jsonl(out, variants)
        record(
            ManifestRecord(
                name=f"{args.qa.stem}-romanized",
                kind="machine_variant",
                license="project-own",
                source=args.qa.as_posix(),
                provenance="machine_translit",
                generator_model=TOOL,
                out_file=out.as_posix(),
                n_rows_in=sum(it.language == Language.HI for it in items),
                n_items_out=len(variants),
                n_dropped=skipped,
                drop_reasons={"group already has hinglish": skipped} if skipped else {},
                warnings=["machine-made romanized Hindi, not natural Hinglish"],
                created_at=now(),
            ),
            args.manifest,
        )
        print(f"wrote {len(variants)} romanized variants to {out} ({skipped} skipped)")
        return
    src_path = args.sources or args.qa.with_name(args.qa.name.replace("-romanized", ""))
    sources = {it.qa_id: it for it in read_jsonl(src_path, QAItem)}
    export_sample(read_jsonl(args.qa, QAItem), sources, args.n, args.seed, args.out)
    print(f"wrote {args.out}: fill in 'ok' and 'fix', then correct the jsonl by hand")


if __name__ == "__main__":
    main()
