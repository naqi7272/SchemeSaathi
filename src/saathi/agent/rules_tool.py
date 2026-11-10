"""Write eligibility rules by hand ($0) and validate them.

    uv run python -m saathi.agent.rules_tool template --scheme pm-kisan
    uv run python -m saathi.agent.rules_tool validate

`template` writes data/rules/<scheme>.yaml with verified: false, the scheme's passage IDs
(so you can cite them) and the allowed fields and operators as comments. Fill in the
rules from the official passage text, run `validate`, check each rule against its passage
once more, and only then set verified: true. The agent refuses unverified files.

`validate` checks every rules file:
  - YAML matches the schema (SchemeRules), file name == scheme_id;
  - fields, operators and value types are valid (eligibility.validate_rules), and each
    value is a value the profile field can hold (e.g. gender in female|male|other);
  - every cited passage_id exists in passages.jsonl and belongs to that scheme;
  - lists which files are still unverified.
Exit code 1 if any file has errors.
"""

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from saathi.agent.eligibility import PROFILE_FIELDS, SchemeRules, UserProfile, validate_rules
from saathi.io import read_jsonl
from saathi.schemas import Passage

OPS = "==, !=, <, <=, >, >=, in, not_in"


def check_file(path: Path, passages: dict[str, Passage]) -> tuple[SchemeRules | None, list[str]]:
    """Returns (rules if they parse, list of errors)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for note in ("unsupported", "drafted_by"):  # draft_rules notes, not part of the schema
            data.pop(note, None)
        scheme = SchemeRules.model_validate(data)
    except (yaml.YAMLError, ValidationError) as err:
        return None, [f"schema: {err}"]
    errors = []
    if path.stem != scheme.scheme_id:
        errors.append(f"file name {path.name} does not match scheme_id {scheme.scheme_id!r}")
    try:
        validate_rules(scheme)
    except ValueError as err:
        errors.append(str(err))
    for i, rule in enumerate(scheme.rules):
        where = f"rule {i + 1} ({rule.field} {rule.op} {rule.value!r})"
        if rule.field in PROFILE_FIELDS:
            values = rule.value if isinstance(rule.value, list) else [rule.value]
            for v in values:
                try:
                    UserProfile.model_validate({rule.field: v}, strict=False)
                except ValidationError:
                    errors.append(f"{where}: {v!r} is not a valid value for {rule.field}")
        if rule.passage_id not in passages:
            errors.append(f"{where}: passage {rule.passage_id!r} not in passages.jsonl")
        elif passages[rule.passage_id].scheme_id != scheme.scheme_id:
            errors.append(f"{where}: passage {rule.passage_id!r} belongs to another scheme")
    return scheme, errors


def validate_dir(rules_dir: Path, passages: dict[str, Passage]) -> tuple[int, list[str]]:
    """Prints a report. Returns (number of files with errors, unverified scheme ids)."""
    bad, unverified = 0, []
    files = sorted(Path(rules_dir).glob("*.yaml"))
    if not files:
        print(f"no rule files in {rules_dir}")
    for path in files:
        scheme, errors = check_file(path, passages)
        if errors:
            bad += 1
            print(f"FAIL  {path}")
            for e in errors:
                print(f"      - {e}")
            continue
        assert scheme is not None
        flag = "verified" if scheme.verified else "NOT verified (the agent will refuse it)"
        print(f"OK    {path}: {len(scheme.rules)} rules, {flag}")
        if not scheme.verified:
            unverified.append(scheme.scheme_id)
    return bad, unverified


def template_text(scheme_id: str, passage_ids: list[str]) -> str:
    fields = UserProfile.model_json_schema()["properties"]
    field_lines = "\n".join(f"#   {name}: {_type(spec)}" for name, spec in fields.items())
    pid_lines = "\n".join(f"#   {p}" for p in passage_ids) or "#   (none found: run ingest first)"
    first = passage_ids[0] if passage_ids else f"{scheme_id}:eligibility:0"
    return f"""# Eligibility rules for {scheme_id}. ALL rules must hold together (AND).
# OR over values of ONE field: op "in" with a list. OR across different fields is not
# supported: leave such a scheme out. Write only what the cited passage says explicitly.
#
# Fields and types:
{field_lines}
# Operators: {OPS}
# Passages of this scheme you can cite:
{pid_lines}
#
# When every rule is checked against its passage, change verified to true.
scheme_id: {scheme_id}
verified: false
rules:
  - field: age
    op: ">="
    value: 18
    passage_id: {first}
    note: "EXAMPLE - replace with a rule quoted from the passage"
"""


def _type(spec: dict) -> str:
    options = spec.get("anyOf", [spec])
    out = []
    for o in options:
        if "enum" in o:
            out.append(" | ".join(map(str, o["enum"])))
        elif o.get("type") != "null":
            out.append(o.get("type", "?"))
    return " | ".join(out) + " (or leave unknown)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--scheme", required=True)
    v = sub.add_parser("validate")
    for p in (t, v):
        p.add_argument("--rules", type=Path, default=Path("data/rules"))
        p.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    args = ap.parse_args()
    passages = (
        {p.passage_id: p for p in read_jsonl(args.passages, Passage)}
        if args.passages.exists()
        else {}
    )
    if args.cmd == "template":
        out = args.rules / f"{args.scheme}.yaml"
        if out.exists():
            raise SystemExit(f"{out} exists; not overwriting")
        pids = sorted(p for p, x in passages.items() if x.scheme_id == args.scheme)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(template_text(args.scheme, pids), encoding="utf-8")
        print(f"wrote {out}; fill it in, then run: python -m saathi.agent.rules_tool validate")
        return
    if not passages:
        print(f"WARNING: {args.passages} not found; every passage check will fail")
    bad, unverified = validate_dir(args.rules, passages)
    print(f"{bad} file(s) with errors; {len(unverified)} unverified: {unverified}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
