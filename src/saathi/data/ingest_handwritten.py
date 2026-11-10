"""Ingest hand-written test questions from a CSV template.

Template: data/qa/handwritten_template.csv (copy it to data/qa/handwritten.csv). Columns:
    group_id     same id for translations of ONE question (e.g. pmk-01); required
    scheme_id    one of our scheme IDs (data/processed/schemes.jsonl)
    language     en | hi | hinglish
    qtype        eligibility | benefits | documents | how_to_apply | unanswerable
    question     the question, as a user would ask it
    answer       the answer, using only facts from the passage(s)
    passage_ids  our passage ID(s), ";"-separated; empty for unanswerable questions
    notes        optional, ignored

Validation (every problem is printed with its CSV line number):
  - required fields present, language and qtype valid, scheme exists;
  - every passage ID exists and belongs to that scheme; unanswerable rows cite nothing;
  - one group = one scheme and one qtype, at most one row per language;
  - the answer is supported by the cited passage text (saathi.data.align.support):
    all numbers present, and enough words in common when the scripts match.
    Hindi/Hinglish answers over English passages can only be number-checked: they are
    listed as "check by hand", not rejected.

Output: data/qa/handwritten.jsonl (origin "human:handwritten") and a manifest record.
Nothing is written if any row is invalid, unless --skip-invalid is given.

    uv run python -m saathi.data.ingest_handwritten data/qa/handwritten.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from saathi.data.align import support_any
from saathi.data.provenance import MANIFEST, ManifestRecord, now, record, sha256_text
from saathi.io import read_jsonl, write_jsonl
from saathi.schemas import Language, Passage, QAItem, QuestionType, Scheme

COLUMNS = [
    "group_id",
    "scheme_id",
    "language",
    "qtype",
    "question",
    "answer",
    "passage_ids",
    "notes",
]


def read_rows(path: Path) -> list[tuple[int, dict]]:
    """(CSV line number, row). utf-8-sig so files saved by Excel load too."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS[:-1] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; use the template header")
        rows = []
        for i, row in enumerate(reader):
            clean = {k: (v or "").strip() for k, v in row.items() if k}
            if row.get(None):  # more cells than header columns: an unquoted comma
                clean["__extra__"] = "1"
            rows.append((i + 2, clean))
        return rows


def validate(
    rows: list[tuple[int, dict]],
    scheme_ids: set[str],
    passages: dict[str, Passage],
    min_overlap: float = 0.5,
    origin: str = "human:handwritten",
) -> tuple[list[QAItem], list[str], list[str]]:
    """Returns (valid items, errors, check-by-hand notes)."""
    items, errors, manual = [], [], []
    groups: dict[str, list[tuple[int, QAItem]]] = defaultdict(list)
    for line, r in rows:
        if not any(r.get(c) for c in COLUMNS[:-1]):
            continue  # blank line
        where = f"line {line}"
        if r.get("__extra__"):
            errors.append(
                f"{where}: more cells than columns; put any text containing a comma in "
                'double quotes ("like, this"), or save the file from Excel/Sheets as CSV'
            )
            continue
        problems = []
        for c in ("group_id", "scheme_id", "language", "qtype", "question", "answer"):
            if not r.get(c):
                problems.append(f"{c} is empty")
        if r.get("language") and r["language"] not in set(Language):
            problems.append(f"language {r['language']!r} not in en|hi|hinglish")
        if r.get("qtype") and r["qtype"] not in set(QuestionType):
            problems.append(f"qtype {r['qtype']!r} unknown")
        if r.get("scheme_id") and r["scheme_id"] not in scheme_ids:
            problems.append(f"scheme_id {r['scheme_id']!r} not in schemes.jsonl")
        pids = [p.strip() for p in r.get("passage_ids", "").split(";") if p.strip()]
        answerable = r.get("qtype") != QuestionType.UNANSWERABLE
        for pid in pids:
            if pid not in passages:
                problems.append(f"passage {pid!r} does not exist")
            elif passages[pid].scheme_id != r.get("scheme_id"):
                problems.append(f"passage {pid!r} belongs to another scheme")
        if answerable and not pids:
            problems.append("answerable question needs at least one passage_id")
        if not answerable and pids:
            problems.append("unanswerable question must have empty passage_ids")
        if problems:
            errors += [f"{where}: {p}" for p in problems]
            continue
        if answerable:
            s = support_any(
                r["answer"],
                [passages[p] for p in pids],
                min_overlap,
                check_words=r["language"] != Language.HINGLISH,
            )
            if not s.ok:
                why = []
                if s.missing_numbers:
                    why.append(f"numbers not in passage: {s.missing_numbers}")
                if s.overlap is not None and s.overlap < min_overlap:
                    why.append(f"only {s.overlap:.0%} of answer words are in the passage")
                errors.append(f"{where}: answer not supported ({'; '.join(why)})")
                continue
            if s.overlap is None:
                manual.append(f"{where} [{r['language']}]: {s.note}; check by hand")
        gid = f"hw-{r['group_id']}"
        try:
            item = QAItem(
                qa_id=f"{gid}-{r['language']}",
                group_id=gid,
                scheme_id=r["scheme_id"],
                language=r["language"],
                qtype=r["qtype"],
                question=r["question"],
                answer=r["answer"],
                citations=sorted(set(pids)),
                answerable=answerable,
                origin=origin,
            )
        except ValidationError as err:
            errors.append(f"{where}: {err.errors()[0]['msg']}")
            continue
        groups[gid].append((line, item))
        items.append(item)
    bad_groups = set()
    for gid, members in groups.items():
        lines = ", ".join(str(n) for n, _ in members)
        if len({it.scheme_id for _, it in members}) > 1:
            errors.append(f"group {gid} (lines {lines}): mixes schemes")
            bad_groups.add(gid)
        if len({it.qtype for _, it in members}) > 1:
            errors.append(f"group {gid} (lines {lines}): mixes question types")
            bad_groups.add(gid)
        langs = [it.language for _, it in members]
        if len(langs) != len(set(langs)):
            errors.append(f"group {gid} (lines {lines}): a language appears twice")
            bad_groups.add(gid)
    items = [it for it in items if it.group_id not in bad_groups]
    return items, errors, manual


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--schemes", type=Path, default=Path("data/processed/schemes.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/handwritten.jsonl"))
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--skip-invalid", action="store_true", help="write the valid rows anyway")
    ap.add_argument(
        "--drafted-by",
        help="if a model drafted the rows (you only checked them), name it, e.g. "
        "claude-opus-5-5. Items get origin 'drafted:<model>' and the manifest says synthetic.",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    scheme_ids = {s.scheme_id for s in read_jsonl(args.schemes, Scheme)}
    passages = {p.passage_id: p for p in read_jsonl(args.passages, Passage)}
    origin = f"drafted:{args.drafted_by}" if args.drafted_by else "human:handwritten"
    family_warning = None
    if args.drafted_by:
        from saathi.llm.factory import same_family_warning

        family_warning = same_family_warning(args.drafted_by, "question-drafting")
        if family_warning:
            print(family_warning)
    items, errors, manual = validate(rows, scheme_ids, passages, args.min_overlap, origin)
    for e in errors:
        print(f"ERROR   {e}")
    for m in manual:
        print(f"CHECK   {m}")
    print(f"{len(items)} valid items, {len(errors)} errors, {len(manual)} to check by hand")
    if errors and not args.skip_invalid:
        raise SystemExit("fix the errors above (or pass --skip-invalid); nothing written")
    write_jsonl(args.out, items)
    record(
        ManifestRecord(
            name="handwritten",
            kind="handwritten",
            license="project-own",
            source=args.csv.as_posix(),
            provenance="synthetic" if args.drafted_by else "human",
            generator_model=args.drafted_by,
            generator_family_warning=family_warning,
            out_file=args.out.as_posix(),
            n_rows_in=len(rows),
            n_items_out=len(items),
            n_dropped=len(rows) - len(items),
            drop_reasons={"validation error": len(errors)} if errors else {},
            config_sha256=sha256_text(args.csv.read_text(encoding="utf-8-sig")),
            warnings=manual,
            created_at=now(),
        ),
        args.manifest,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
