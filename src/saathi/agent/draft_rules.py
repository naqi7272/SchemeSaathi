"""OPTIONAL: draft eligibility rule files with an LLM, for hand verification.

The $0 path writes rules by hand instead (saathi.agent.rules_tool template / validate).
If you do draft with a model, any backend works (--backend, --model); a Claude model
needs the `claude` extra and an API key. For each requested scheme, the model reads the
scheme's passages and proposes rules over the fixed UserProfile fields, each citing a
passage. The draft is written with verified: false; the agent refuses to use it until a
human checks every rule against its passage and flips the flag. Criteria that cannot be
expressed (OR across fields, documents, vague wording) are listed in `unsupported` and
the scheme should then be left out of the rules subset.

Usage:
    uv run python -m saathi.agent.draft_rules --schemes pm-kisan apy \
        --backend openai_compat --model <model>
Then: uv run python -m saathi.agent.rules_tool validate
"""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from saathi.agent.eligibility import PROFILE_FIELDS, Rule, SchemeRules, UserProfile, validate_rules
from saathi.io import read_jsonl
from saathi.llm.base import ChatModel
from saathi.schemas import Passage

SYSTEM = f"""You convert government scheme eligibility text into machine-checkable rules.
Profile fields (use these names only): {", ".join(PROFILE_FIELDS)}.
Field types: {UserProfile.model_json_schema()["properties"]}
Operators: ==, !=, <, <=, >, >=, in, not_in. All rules must hold together (AND).
Use "in" for a choice among values of ONE field. Every rule cites the passage ID that
states it. Only encode criteria the passages state explicitly. Anything you cannot express
(OR across fields, documents to submit, vague wording, fields not in the list) goes in
"unsupported", word for word."""


class Draft(BaseModel):
    rules: list[Rule]
    unsupported: list[str]


def draft(
    model: ChatModel, scheme_id: str, passages: list[Passage]
) -> tuple[SchemeRules | None, list[str], float]:
    """Returns (rules or None if nothing is expressible, unsupported criteria, cost in USD)."""
    body = "\n\n".join(f"[{p.passage_id}]\n{p.text}" for p in passages)
    result = model.chat(
        SYSTEM,
        [{"role": "user", "content": f"Passages:\n\n{body}"}],
        schema=Draft,
        max_tokens=16000,
    )
    d = Draft.model_validate_json(result.text)
    if not d.rules:
        return None, d.unsupported, result.usage.cost_usd
    rules = SchemeRules(scheme_id=scheme_id, verified=False, rules=d.rules)
    validate_rules(rules)
    return rules, d.unsupported, result.usage.cost_usd


def main() -> None:
    from saathi.llm.factory import BACKENDS, make_chat, warn_same_family

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--schemes", nargs="+", required=True)
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/rules"))
    ap.add_argument("--backend", choices=BACKENDS, default="openai_compat")
    ap.add_argument("--model", required=True, help="drafting model (record it; prefer non-Qwen)")
    ap.add_argument("--max-usd", type=float, default=1.0)
    args = ap.parse_args()
    load_dotenv()
    by_scheme: dict[str, list[Passage]] = defaultdict(list)
    for p in read_jsonl(args.passages, Passage):
        by_scheme[p.scheme_id].append(p)
    warn_same_family(args.model, "rule-drafting")
    model = make_chat(args.backend, args.model, effort="high")
    spent = 0.0
    for scheme_id in args.schemes:
        if spent >= args.max_usd:
            raise SystemExit(f"budget reached (${spent:.2f})")
        out = args.out / f"{scheme_id}.yaml"
        if out.exists():
            print(f"skip {scheme_id}: {out} exists")
            continue
        rules, unsupported, cost = draft(model, scheme_id, by_scheme[scheme_id])
        spent += cost
        if rules is None:
            print(f"{scheme_id}: no expressible rules; leave it out. Unsupported: {unsupported}")
            continue
        data = rules.model_dump(mode="json") | {
            "unsupported": unsupported,
            "drafted_by": f"{args.backend}:{args.model}",
        }
        out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"{scheme_id}: {len(rules.rules)} rules, {len(unsupported)} unsupported -> {out}")


if __name__ == "__main__":
    main()
