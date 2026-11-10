"""Import an open Q&A dataset (Hugging Face, CSV, JSON, or JSONL) into the data/qa schema.

A YAML mapping says which column holds what. See configs/datasets/TEMPLATE.yaml.

Steps:
  1. License gate (saathi.data.provenance): unclear licenses and myscheme.gov.in sources are
     refused before any row is read.
  2. Each row -> QAItem: scheme resolved to one of OUR scheme IDs (by id, by the mapping, or
     by exact normalised name), language from a column / constant / detection.
  3. Rows with no supporting passage are aligned to our processed passages (BM25 + string
     match, saathi.data.align). Rows that cannot be aligned are dropped and counted.
  4. Output: data/qa/generated/open-<name>.jsonl (picked up by build_splits, which applies
     the scheme / passage / group leakage checks), plus a manifest record.

Usage:
    uv run python -m saathi.data.import_open_dataset configs/datasets/<name>.yaml
    uv run python -m saathi.data.import_open_dataset configs/datasets/<name>.yaml --dry-run
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from saathi.agent.language import detect_language
from saathi.data.align import align, support_any
from saathi.data.provenance import (
    ManifestRecord,
    Provenance,
    check_license,
    now,
    record,
    sha256_text,
)
from saathi.io import read_jsonl, write_jsonl
from saathi.llm.factory import same_family_warning
from saathi.schemas import Language, Passage, QAItem, QuestionType, Scheme


class Columns(BaseModel):
    question: str
    answer: str
    scheme: str  # column holding a scheme id or name
    language: str | None = None
    qtype: str | None = None
    passages: str | None = None  # column with passage IDs (list, or ";"-separated)
    group: str | None = None  # column grouping translations of one question
    answerable: str | None = None


class DatasetConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    format: Literal["hf", "csv", "json", "jsonl"]
    path: str  # HF dataset id, or a local file path
    hf_split: str = "train"
    hf_config: str | None = None
    license: str
    source_url: str
    provenance: Provenance
    generator_model: str | None = None  # if the questions were made by a model
    columns: Columns
    language_default: Language | None = None  # used when there is no language column
    language_map: dict[str, Language] = Field(default_factory=dict)
    qtype_default: QuestionType = QuestionType.BENEFITS
    qtype_map: dict[str, QuestionType] = Field(default_factory=dict)
    scheme_map: dict[str, str] = Field(default_factory=dict)  # dataset value -> our scheme_id
    min_overlap: float = 0.5
    limit: int | None = None


def load_config(path: Path) -> tuple[DatasetConfig, str]:
    text = Path(path).read_text(encoding="utf-8")
    return DatasetConfig.model_validate(yaml.safe_load(text)), sha256_text(text)


def load_rows(cfg: DatasetConfig) -> list[dict]:
    if cfg.format == "hf":
        try:
            from datasets import load_dataset
        except ImportError as err:
            raise SystemExit(
                "Hugging Face import needs the `open-data` extra: uv sync --extra open-data"
            ) from err
        ds = load_dataset(cfg.path, cfg.hf_config, split=cfg.hf_split)
        rows = [dict(r) for r in ds]
    else:
        path = Path(cfg.path)
        if cfg.format == "csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
        elif cfg.format == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("data", [])
        else:
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return rows[: cfg.limit] if cfg.limit else rows


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", name.lower()).strip()


def scheme_resolver(cfg: DatasetConfig, schemes: list[Scheme]):
    by_id = {s.scheme_id for s in schemes}
    by_name = {_norm(s.name): s.scheme_id for s in schemes}

    def resolve(value: str) -> str | None:
        value = str(value or "").strip()
        if value in cfg.scheme_map:
            return cfg.scheme_map[value]
        if value in by_id:
            return value
        return by_name.get(_norm(value))

    return resolve


def _passage_ids(value) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).replace(",", ";").split(";") if v.strip()]


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def convert(
    cfg: DatasetConfig, rows: list[dict], schemes: list[Scheme], passages: list[Passage]
) -> tuple[list[QAItem], Counter]:
    """Rows -> validated QAItems. Returns (items, drop reasons)."""
    resolve = scheme_resolver(cfg, schemes)
    by_pid = {p.passage_id: p for p in passages}
    by_scheme: dict[str, list[Passage]] = defaultdict(list)
    for p in passages:
        by_scheme[p.scheme_id].append(p)
    col = cfg.columns
    items, drops = [], Counter()
    for i, row in enumerate(rows):
        question = str(row.get(col.question) or "").strip()
        answer = str(row.get(col.answer) or "").strip()
        if not question or not answer:
            drops["empty question or answer"] += 1
            continue
        scheme_id = resolve(row.get(col.scheme))
        if scheme_id is None or scheme_id not in by_scheme:
            drops["scheme not in our corpus"] += 1
            continue
        if col.language:
            raw = str(row.get(col.language) or "").strip()
            lang = cfg.language_map.get(raw) or (raw if raw in set(Language) else None)
        else:
            lang = cfg.language_default or detect_language(question)
        if lang is None:
            drops["unknown language"] += 1
            continue
        qtype = cfg.qtype_default
        if col.qtype:
            raw_q = str(row.get(col.qtype) or "").strip()
            qtype = cfg.qtype_map.get(raw_q) or (
                QuestionType(raw_q) if raw_q in set(QuestionType) else cfg.qtype_default
            )
        answerable = qtype != QuestionType.UNANSWERABLE
        if col.answerable:
            answerable = _truthy(row.get(col.answerable))
            if not answerable:
                qtype = QuestionType.UNANSWERABLE
        cites: list[str] = []
        if answerable:
            cites = _passage_ids(row.get(col.passages)) if col.passages else []
            if cites:
                if any(c not in by_pid or by_pid[c].scheme_id != scheme_id for c in cites):
                    drops["cited passage not in our corpus"] += 1
                    continue
                ok = support_any(
                    answer,
                    [by_pid[c] for c in cites],
                    cfg.min_overlap,
                    check_words=lang != Language.HINGLISH,
                ).ok
                if not ok:
                    drops["answer not supported by cited passage"] += 1
                    continue
            else:
                cites = align(question, answer, by_scheme[scheme_id], min_overlap=cfg.min_overlap)
                if not cites:
                    drops["could not align to a passage"] += 1
                    continue
        group = str(row.get(col.group)) if col.group and row.get(col.group) else f"{i:05d}"
        group_id = f"open-{cfg.name}-{scheme_id}-{group}"
        try:
            items.append(
                QAItem(
                    qa_id=f"{group_id}-{lang}",
                    group_id=group_id,
                    scheme_id=scheme_id,
                    language=lang,
                    qtype=qtype,
                    question=question,
                    answer=answer,
                    citations=sorted(set(cites)),
                    answerable=answerable,
                    origin=f"open:{cfg.name}",
                    machine_made=cfg.provenance in ("machine_translit", "machine_translation"),
                )
            )
        except ValidationError:
            drops["failed schema validation"] += 1
    dup = Counter(it.qa_id for it in items)
    if any(n > 1 for n in dup.values()):
        seen: set[str] = set()
        kept = []
        for it in items:
            if it.qa_id in seen:
                drops["duplicate (same group and language)"] += 1
                continue
            seen.add(it.qa_id)
            kept.append(it)
        items = kept
    return items, drops


def run(
    config_path: Path,
    schemes_path: Path,
    passages_path: Path,
    out_dir: Path,
    manifest: Path,
    dry_run: bool = False,
) -> tuple[list[QAItem], Counter]:
    cfg, cfg_sha = load_config(config_path)
    warnings = check_license(cfg.license, cfg.source_url, cfg.provenance)
    family_warning = None
    if cfg.generator_model:
        family_warning = same_family_warning(cfg.generator_model, "generator")
        if family_warning:
            warnings.append(family_warning)
    for w in warnings:
        print(f"WARNING: {w}")
    rows = load_rows(cfg)
    items, drops = convert(
        cfg, rows, read_jsonl(schemes_path, Scheme), read_jsonl(passages_path, Passage)
    )
    print(f"{cfg.name}: {len(rows)} rows -> {len(items)} items, {sum(drops.values())} dropped")
    for reason, n in drops.most_common():
        print(f"  dropped {n:5d}  {reason}")
    if dry_run:
        return items, drops
    out = out_dir / f"open-{cfg.name}.jsonl"
    write_jsonl(out, items)
    record(
        ManifestRecord(
            name=cfg.name,
            kind="open_dataset",
            license=cfg.license,
            source=cfg.source_url if cfg.format != "hf" else f"hf:{cfg.path} ({cfg.source_url})",
            provenance=cfg.provenance,
            generator_model=cfg.generator_model,
            generator_family_warning=family_warning,
            out_file=out.as_posix(),
            n_rows_in=len(rows),
            n_items_out=len(items),
            n_dropped=sum(drops.values()),
            drop_reasons=dict(drops),
            config_sha256=cfg_sha,
            warnings=warnings,
            created_at=now(),
        ),
        manifest,
    )
    print(f"wrote {out} and a manifest record in {manifest}")
    return items, drops


def main() -> None:
    from saathi.data.provenance import MANIFEST, LicenseError

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--schemes", type=Path, default=Path("data/processed/schemes.jsonl"))
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/qa/generated"))
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--dry-run", action="store_true", help="convert and report, write nothing")
    args = ap.parse_args()
    try:
        run(args.config, args.schemes, args.passages, args.out, args.manifest, args.dry_run)
    except LicenseError as err:
        raise SystemExit(f"REFUSED: {err}") from err


if __name__ == "__main__":
    main()
