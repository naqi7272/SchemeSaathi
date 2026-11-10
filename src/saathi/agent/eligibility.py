"""Deterministic eligibility rules.

Each scheme's rules live in data/rules/<scheme_id>.yaml. A scheme's rules are a list of
conditions that must ALL hold (AND). An OR over values of one field is written with "in".
OR across different fields is not supported; such schemes are left out of the rules subset.

Every rule cites the passage it comes from. Rules are drafted from passages and must be
hand-checked; unverified rule files are refused unless explicitly allowed.

Evaluation is three-valued:
  not_eligible  some known profile field fails a rule (missing fields do not matter then)
  unknown       no rule fails, but some needed fields are missing -> ask the user
  eligible      every rule holds
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Fields the agent can collect. None means not yet known."""

    age: int | None = None
    gender: Literal["female", "male", "other"] | None = None
    annual_family_income_inr: int | None = None
    state: str | None = None
    area: Literal["rural", "urban"] | None = None
    occupation: str | None = None  # e.g. "farmer", "student", "street_vendor", "unemployed"
    land_hectares: float | None = None
    social_category: Literal["general", "obc", "sc", "st"] | None = None
    is_income_tax_payer: bool | None = None
    has_bank_account: bool | None = None
    is_government_employee: bool | None = None


PROFILE_FIELDS = tuple(UserProfile.model_fields)

Op = Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in"]


class Rule(BaseModel):
    field: str
    op: Op
    value: int | float | bool | str | list[str]
    passage_id: str
    note: str = ""


class SchemeRules(BaseModel):
    scheme_id: str
    verified: bool = False
    rules: list[Rule] = Field(min_length=1)


class Status(StrEnum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    UNKNOWN = "unknown"


class EligibilityResult(BaseModel):
    scheme_id: str
    status: Status
    failed: list[Rule] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


def check_rule(rule: Rule, actual: object) -> bool:
    """Apply one rule to a known (non-None) value."""
    v = rule.value
    match rule.op:
        case "==":
            return actual == v
        case "!=":
            return actual != v
        case "<":
            return actual < v
        case "<=":
            return actual <= v
        case ">":
            return actual > v
        case ">=":
            return actual >= v
        case "in":
            return actual in v
        case "not_in":
            return actual not in v
    raise ValueError(f"unknown op {rule.op}")


def evaluate(profile: UserProfile, scheme: SchemeRules) -> EligibilityResult:
    failed, missing = [], []
    for rule in scheme.rules:
        actual = getattr(profile, rule.field)
        if actual is None:
            if rule.field not in missing:
                missing.append(rule.field)
        elif not check_rule(rule, actual):
            failed.append(rule)
    if failed:
        status = Status.NOT_ELIGIBLE
    elif missing:
        status = Status.UNKNOWN
    else:
        status = Status.ELIGIBLE
    return EligibilityResult(
        scheme_id=scheme.scheme_id, status=status, failed=failed, missing_fields=missing
    )


def validate_rules(scheme: SchemeRules) -> None:
    """Check field names and that each value has a type the field can compare against."""
    for rule in scheme.rules:
        if rule.field not in PROFILE_FIELDS:
            raise ValueError(f"{scheme.scheme_id}: unknown profile field {rule.field!r}")
        if rule.op in ("in", "not_in") and not isinstance(rule.value, list):
            raise ValueError(f"{scheme.scheme_id}: {rule.op} needs a list value")
        if rule.op in ("<", "<=", ">", ">=") and (
            isinstance(rule.value, bool) or not isinstance(rule.value, int | float)
        ):
            raise ValueError(f"{scheme.scheme_id}: {rule.op} needs a number")
        if not rule.passage_id.startswith(f"{scheme.scheme_id}:"):
            raise ValueError(f"{scheme.scheme_id}: rule cites another scheme's passage")


def load_rules(path: Path, allow_unverified: bool = False) -> SchemeRules:
    scheme = SchemeRules.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    validate_rules(scheme)
    if not scheme.verified and not allow_unverified:
        raise ValueError(f"{path}: rules not hand-verified (set verified: true after checking)")
    return scheme


def load_all_rules(rules_dir: Path, allow_unverified: bool = False) -> dict[str, SchemeRules]:
    out = {}
    for path in sorted(Path(rules_dir).glob("*.yaml")):
        scheme = load_rules(path, allow_unverified)
        out[scheme.scheme_id] = scheme
    return out
