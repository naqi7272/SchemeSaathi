"""Multi-turn evaluation of the agent (arm 5).

Ground truth comes from code, not from an LLM:
- A hidden user profile is sampled from a scheme's verified rules, either satisfying every
  rule (eligible) or violating exactly one (not eligible). The rules engine confirms it.
- The opening message reveals a random subset of those facts. Whether a clarifying question
  is NEEDED is decided by the rules engine on the revealed facts alone.
By default ($0) the opening message comes from templates and a scripted simulator plays the
user (saathi.agent.scripted). Optionally an LLM can write openings and play the user
(--openings llm / --simulator llm, any backend). All metrics are scored in code.

    uv run saathi-agent-eval scenarios --n 80
    uv run --extra models saathi-agent-eval run --run-id agent-sft-q8 --backend ollama \
        --model qwen3-1.7b-saathi:q8_0                      # --simulator scripted (default)
    uv run saathi-agent-eval report --runs agent-sft-q8
"""

import argparse
import random
import time
import typing
import uuid
from pathlib import Path

from langgraph.types import Command
from pydantic import BaseModel

from saathi.agent.eligibility import (
    SchemeRules,
    Status,
    UserProfile,
    check_rule,
    evaluate,
    load_all_rules,
)
from saathi.agent.language import detect_language
from saathi.agent.tracing import run_config
from saathi.io import append_jsonl, read_jsonl, write_jsonl
from saathi.llm.base import ChatModel, Usage
from saathi.schemas import Language

SCENARIOS = Path("data/conversations/scenarios.jsonl")
AGENT_RUNS = Path("results/agent_runs")

DEFAULT_CANDIDATES: dict[str, list] = {
    "age": [12, 17, 25, 35, 45, 59, 61, 70],
    "annual_family_income_inr": [60_000, 150_000, 250_000, 450_000, 900_000, 2_000_000],
    "land_hectares": [0.4, 1.0, 1.8, 2.5, 5.0],
    "occupation": [
        "farmer",
        "student",
        "teacher",
        "street_vendor",
        "unemployed",
        "construction_worker",
    ],
    "state": ["Bihar", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Assam"],
}


def candidates(field: str, rules: list) -> list:
    """Values worth trying for a field: defaults, rule values, and values just around
    every numeric threshold (so boundaries are exercised)."""
    if field in DEFAULT_CANDIDATES:
        values = list(DEFAULT_CANDIDATES[field])
    else:
        annotation = UserProfile.model_fields[field].annotation
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args == [bool]:
            values = [True, False]
        else:
            values = [v for a in args for v in typing.get_args(a)]  # Literal options
    for r in rules:
        vals = r.value if isinstance(r.value, list) else [r.value]
        for v in vals:
            if isinstance(v, bool) or not isinstance(v, int | float):
                values.append(v)
            else:
                step = 1 if isinstance(v, int) else 0.1
                values += [v - step, v, v + step]
    seen, out = set(), []
    for v in values:
        if (type(v), v) not in seen:
            seen.add((type(v), v))
            out.append(v)
    return out


def sample_profile(scheme: SchemeRules, eligible: bool, rng: random.Random) -> UserProfile:
    by_field: dict[str, list] = {}
    for r in scheme.rules:
        by_field.setdefault(r.field, []).append(r)
    fields = sorted(by_field)
    broken = None if eligible else rng.choice(fields)
    values = {}
    for f in fields:
        options = candidates(f, by_field[f])
        ok = [v for v in options if all(check_rule(r, v) for r in by_field[f])]
        bad = [v for v in options if not all(check_rule(r, v) for r in by_field[f])]
        pool = bad if f == broken else ok
        if not pool:
            raise ValueError(
                f"{scheme.scheme_id}: no candidate value for {f} "
                f"({'violating' if f == broken else 'satisfying'})"
            )
        values[f] = rng.choice(pool)
    profile = UserProfile(**values)
    expected = Status.ELIGIBLE if eligible else Status.NOT_ELIGIBLE
    assert evaluate(profile, scheme).status == expected
    return profile


class Scenario(BaseModel):
    conv_id: str
    language: Language
    scheme_id: str
    scheme_name: str
    true_profile: UserProfile  # hidden from the agent; known to the simulated user
    revealed: list[str]  # fields stated in the opening message
    opening: str
    expected_status: Status  # eligible or not_eligible
    needs_clarification: bool


def plan_scenario(
    scheme: SchemeRules, conv_id: str, language: Language, eligible: bool, rng: random.Random
) -> tuple[UserProfile, list[str], bool]:
    """Profile, revealed fields, and whether a clarifying question is needed."""
    profile = sample_profile(scheme, eligible, rng)
    fields = sorted({r.field for r in scheme.rules})
    if rng.random() < 1 / 3:
        revealed = fields  # everything stated up front: asking would be unnecessary
    else:
        revealed = sorted(rng.sample(fields, rng.randrange(len(fields))))
    shown = UserProfile(**{f: getattr(profile, f) for f in revealed})
    needs = evaluate(shown, scheme).status == Status.UNKNOWN
    return profile, revealed, needs


class Opening(BaseModel):
    message: str


class Reply(BaseModel):
    message: str


LANG_DESC = {
    Language.EN: "English",
    Language.HI: "Hindi in Devanagari script",
    Language.HINGLISH: "Hinglish (Hindi-English mix in Roman script, WhatsApp style)",
}


def write_opening(
    model: ChatModel, scheme_name: str, language: Language, facts: dict
) -> tuple[str, Usage]:
    system = (
        f"Write the first message a person sends to a government-scheme helpline chat, "
        f"in {LANG_DESC[language]}. They ask whether they are eligible for the scheme. "
        "Mention EXACTLY the facts given about themselves (naturally, in your own words) "
        "and no other facts about themselves. 1-3 sentences."
    )
    result = model.chat(
        system,
        [{"role": "user", "content": f"Scheme: {scheme_name}\nFacts: {facts or 'none'}"}],
        schema=Opening,
        max_tokens=2000,
    )
    return Opening.model_validate_json(result.text).message, result.usage


class Simulator:
    """Plays the user: answers clarifying questions truthfully from the hidden profile."""

    def __init__(self, model: ChatModel):
        self.model = model

    def reply(self, scenario: Scenario, question: str) -> tuple[str, Usage]:
        facts = {k: v for k, v in scenario.true_profile.model_dump().items() if v is not None}
        system = (
            f"You are role-playing a person asking about a government scheme. Your facts: "
            f"{facts}. Reply in {LANG_DESC[scenario.language]} to the assistant's "
            "question, giving only the facts it asks for, truthfully. If it asks for "
            "something not in your facts, say you are not sure. One or two short sentences."
        )
        result = self.model.chat(
            system, [{"role": "user", "content": question}], schema=Reply, max_tokens=2000
        )
        return Reply.model_validate_json(result.text).message, result.usage


class ConversationResult(BaseModel):
    run_id: str
    conv_id: str
    transcript: list[dict]
    final_answer: dict | None
    final_profile: dict
    steps: list[dict]
    agent_usage: list[Usage]
    simulator_usage: list[Usage]
    latency_s: float
    hit_turn_limit: bool
    error: str | None = None  # the conversation crashed (e.g. invalid model output)


def run_conversation(
    app, scenario: Scenario, simulator: Simulator, run_id: str, max_user_turns: int = 4
) -> ConversationResult:
    config = run_config(thread_id=f"{run_id}-{scenario.conv_id}-{uuid.uuid4().hex[:6]}")
    sim_usage: list[Usage] = []
    start = time.perf_counter()
    state = app.invoke({"messages": [{"role": "user", "content": scenario.opening}]}, config)
    turns = 1
    while "__interrupt__" in state and turns < max_user_turns:
        value = state["__interrupt__"][0].value
        question = value["question"]
        if getattr(simulator, "uses_fields", False):
            reply, usage = simulator.reply(scenario, question, value.get("fields"))
        else:
            reply, usage = simulator.reply(scenario, question)
        sim_usage.append(usage)
        state = app.invoke(Command(resume=reply), config)
        turns += 1
    final = app.get_state(config).values
    return ConversationResult(
        run_id=run_id,
        conv_id=scenario.conv_id,
        transcript=final.get("messages", []),
        final_answer=final.get("answer"),
        final_profile=final.get("profile") or {},
        steps=final.get("steps", []),
        agent_usage=[Usage.model_validate(u) for u in final.get("usage", [])],
        simulator_usage=sim_usage,
        latency_s=time.perf_counter() - start,
        hit_turn_limit="__interrupt__" in state,
    )


class ConversationScore(BaseModel):
    run_id: str
    conv_id: str
    language: Language
    task_success: bool
    needs_clarification: bool
    asked: bool
    clarify_appropriate: bool
    asked_missing_field: bool | None = None  # asked for a needed, unstated fact (if needed)
    asked_known_field: bool
    search_called: bool
    eligibility_tool_correct: bool
    invalid_tool_calls: int
    tool_calls_valid: bool = True  # no tool call was rejected by validation
    citation_correct: bool | None = None  # cites a rule passage of the right scheme only
    profile_accuracy: float
    user_turns: int
    latency_s: float
    cost_usd: float


def _same(a, b) -> bool:
    if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        return abs(a - b) <= max(0.05, 0.02 * abs(b))  # unit conversions round
    return a == b


def needed_fields(s: Scenario, rules: SchemeRules | None) -> set[str]:
    """Rule fields the opening did not reveal (what a good agent should ask for)."""
    if rules is None:
        return set()
    return {r.field for r in rules.rules} - set(s.revealed)


def score_conversation(
    s: Scenario, r: ConversationResult, rules: SchemeRules | None = None
) -> ConversationScore:
    """rules (the scenario scheme's rules) enable asked_missing_field and citation_correct."""
    verdict = (r.final_answer or {}).get("eligibility_verdict")
    clarify_steps = [st for st in r.steps if st["node"] == "clarify"]
    asked_fields = {f for st in clarify_steps for f in st.get("fields", [])}
    elig_calls = [st for st in r.steps if st["node"] == "check_eligibility"]
    truth = {k: v for k, v in s.true_profile.model_dump().items() if v is not None}
    correct = sum(_same(r.final_profile.get(k), v) for k, v in truth.items())
    invalid = sum("error" in st for st in r.steps)
    asked_missing = None
    citation_ok = None
    if rules is not None:
        if s.needs_clarification:
            asked_missing = bool(asked_fields & needed_fields(s, rules))
        cites = list((r.final_answer or {}).get("citations") or [])
        rule_pids = {x.passage_id for x in rules.rules}
        citation_ok = (
            bool(cites)
            and all(c.startswith(f"{s.scheme_id}:") for c in cites)
            and bool(set(cites) & rule_pids)
        )
    return ConversationScore(
        run_id=r.run_id,
        conv_id=s.conv_id,
        language=s.language,
        task_success=verdict == s.expected_status.value,
        needs_clarification=s.needs_clarification,
        asked=bool(clarify_steps),
        clarify_appropriate=bool(clarify_steps) == s.needs_clarification,
        asked_missing_field=asked_missing,
        asked_known_field=bool(asked_fields & set(s.revealed)),
        search_called=any(st.get("tool") == "search_schemes" for st in r.steps),
        eligibility_tool_correct=bool(elig_calls)
        and elig_calls[0]["args"]["scheme_id"] == s.scheme_id,
        invalid_tool_calls=invalid,
        tool_calls_valid=invalid == 0,
        citation_correct=citation_ok,
        profile_accuracy=correct / len(truth) if truth else 1.0,
        user_turns=sum(m["role"] == "user" for m in r.transcript),
        latency_s=r.latency_s,
        cost_usd=sum(u.cost_usd for u in r.agent_usage),
    )


# ---------------------------------------------------------------- CLI
def make_scenarios(
    rules: dict[str, SchemeRules],
    names: dict[str, str],
    n: int,
    model: ChatModel | None = None,
    max_usd: float = 0.0,
    seed: int = 0,
) -> list[Scenario]:
    """Balanced over schemes, languages, and eligible/not eligible.
    model=None (default, $0): template openings. With a model: LLM-written openings."""
    from saathi.agent.scripted import template_opening

    rng = random.Random(seed)
    schemes = sorted(rules)
    langs = list(Language)
    out, spent = [], 0.0
    for i in range(n):
        if spent > 0 and spent >= max_usd:  # free/template openings never stop the run
            raise SystemExit(f"budget reached after {i} scenarios (${spent:.2f})")
        scheme = rules[schemes[i % len(schemes)]]
        language = langs[(i // len(schemes)) % len(langs)]
        eligible = (i // (len(schemes) * len(langs))) % 2 == 0
        profile, revealed, needs = plan_scenario(scheme, f"c{i:03d}", language, eligible, rng)
        facts = {f: getattr(profile, f) for f in revealed}
        name = names.get(scheme.scheme_id, scheme.scheme_id)
        if model is None:
            opening, _ = template_opening(name, language, facts, rng)
        else:
            for _ in range(2):  # one retry if the message comes back in the wrong language
                opening, usage = write_opening(model, name, language, facts)
                spent += usage.cost_usd
                if detect_language(opening) == language:
                    break
        out.append(
            Scenario(
                conv_id=f"c{i:03d}",
                language=language,
                scheme_id=scheme.scheme_id,
                scheme_name=names.get(scheme.scheme_id, scheme.scheme_id),
                true_profile=profile,
                revealed=revealed,
                opening=opening,
                expected_status=Status.ELIGIBLE if eligible else Status.NOT_ELIGIBLE,
                needs_clarification=needs,
            )
        )
    print(f"{len(out)} scenarios, ${spent:.2f}")
    return out


def main() -> None:
    from dotenv import load_dotenv

    from saathi.agent.scripted import ScriptedSimulator
    from saathi.eval.run import RunConfig, make_model
    from saathi.eval.stats import bootstrap_ci
    from saathi.llm.factory import BACKENDS, make_chat, warn_same_family

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--n", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rules", type=Path, default=Path("data/rules"))
    s.add_argument("--openings", choices=["template", "llm"], default="template")
    s.add_argument("--backend", choices=BACKENDS, default="openai_compat", help="--openings llm")
    s.add_argument("--model", help="--openings llm: model writing the openings (not Qwen)")
    s.add_argument("--max-usd", type=float, default=2.0)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--backend", choices=BACKENDS, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--simulator", choices=["scripted", "llm"], default="scripted")
    r.add_argument("--sim-backend", choices=BACKENDS, default="openai_compat")
    r.add_argument("--sim-model", help="--simulator llm: model playing the user (not Qwen)")
    r.add_argument("--rules", type=Path, default=Path("data/rules"))
    r.add_argument(
        "--rerank", action="store_true", help="use the cross-encoder reranker (needs ~2 GB RAM)"
    )
    r.add_argument("--max-usd", type=float, default=5.0)
    rep = sub.add_parser("report")
    rep.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    load_dotenv()

    if args.cmd == "scenarios":
        from saathi.io import read_jsonl as rj
        from saathi.schemas import Scheme

        rules = load_all_rules(args.rules)
        names = {sc.scheme_id: sc.name for sc in rj(Path("data/processed/schemes.jsonl"), Scheme)}
        model = None
        if args.openings == "llm":
            if not args.model:
                raise SystemExit("--openings llm needs --model")
            warn_same_family(args.model, "opening-writer")
            model = make_chat(args.backend, args.model)
        scenarios = make_scenarios(rules, names, args.n, model, args.max_usd, seed=args.seed)
        write_jsonl(SCENARIOS, scenarios)
        print(f"wrote {len(scenarios)} scenarios ({args.openings} openings) to {SCENARIOS}")
        return

    if args.cmd == "run":
        from saathi.agent.runtime import build_agent
        from saathi.retrieval.models import CrossEncoderReranker, E5Embedder
        from saathi.retrieval.search import HybridRetriever
        from saathi.schemas import Passage

        cfg = RunConfig(
            run_id=args.run_id,
            arm="agent",
            backend=args.backend,
            model=args.model,
            rules_dir=str(args.rules),
        )
        if args.simulator == "llm":
            if not args.sim_model:
                raise SystemExit("--simulator llm needs --sim-model")
            warn_same_family(args.sim_model, "user-simulator")
            simulator = Simulator(make_chat(args.sim_backend, args.sim_model))
        else:
            simulator = ScriptedSimulator()
        passages = read_jsonl(Path(cfg.passages_file), Passage)
        reranker = CrossEncoderReranker() if args.rerank else None
        retriever = HybridRetriever(passages, E5Embedder(), reranker)
        app = build_agent(make_model(cfg), retriever, passages, rules_dir=args.rules)
        out_dir = AGENT_RUNS / args.run_id
        path = out_dir / "conversations.jsonl"
        done = {c.conv_id for c in read_jsonl(path, ConversationResult)} if path.exists() else set()
        spent = 0.0
        for sc in read_jsonl(SCENARIOS, Scenario):
            if sc.conv_id in done:
                continue
            if spent >= args.max_usd:
                print(f"budget reached (${spent:.2f}); rerun to continue")
                break
            start = time.perf_counter()
            try:
                res = run_conversation(app, sc, simulator, args.run_id)
            except Exception as err:  # a crash is a failed conversation, not a failed run
                res = ConversationResult(
                    run_id=args.run_id,
                    conv_id=sc.conv_id,
                    transcript=[{"role": "user", "content": sc.opening}],
                    final_answer=None,
                    final_profile={},
                    steps=[],
                    agent_usage=[],
                    simulator_usage=[],
                    latency_s=time.perf_counter() - start,
                    hit_turn_limit=False,
                    error=f"{err.__class__.__name__}: {str(err)[:300]}",
                )
                print(f"{sc.conv_id} FAILED (counted as a failure): {res.error[:120]}")
            spent += sum(u.cost_usd for u in res.agent_usage + res.simulator_usage)
            append_jsonl(path, res)
            print(f"{sc.conv_id} {res.latency_s:.0f}s ${spent:.2f}")
        return

    scenarios = {sc.conv_id: sc for sc in read_jsonl(SCENARIOS, Scenario)}
    all_rules = load_all_rules(Path("data/rules"))
    metrics = [
        "task_success",
        "clarify_appropriate",
        "asked_missing_field",
        "asked_known_field",
        "tool_calls_valid",
        "eligibility_tool_correct",
        "citation_correct",
        "profile_accuracy",
        "user_turns",
        "latency_s",
    ]
    lines = [
        "| run | language | n | " + " | ".join(metrics) + " |",
        "|---|---|---|" + "---|" * len(metrics),
    ]
    for run_id in args.runs:
        results = read_jsonl(AGENT_RUNS / run_id / "conversations.jsonl", ConversationResult)
        n_err = sum(x.error is not None for x in results)
        if n_err:
            print(f"{run_id}: {n_err} conversation(s) crashed; scored as failures")
        scores = [
            score_conversation(
                scenarios[x.conv_id], x, all_rules.get(scenarios[x.conv_id].scheme_id)
            )
            for x in results
        ]
        write_jsonl(AGENT_RUNS / run_id / "scores.jsonl", scores)
        for lang in ["all", *[la.value for la in Language]]:
            subset = [x for x in scores if lang == "all" or x.language == lang]
            if not subset:
                continue
            cells = []
            for m in metrics:
                values = [float(getattr(x, m)) for x in subset if getattr(x, m) is not None]
                if not values:
                    cells.append("n/a")
                    continue
                ci = bootstrap_ci(values)
                pct = m not in ("user_turns", "latency_s")
                k = 100 if pct else 1
                cells.append(f"{k * ci.mean:.1f} [{k * ci.lo:.1f}, {k * ci.hi:.1f}]")
            lines.append(f"| {run_id} | {lang} | {len(subset)} | " + " | ".join(cells) + " |")
    out = Path("results/agent_report.md")
    out.write_text(
        "# Agent conversations (arm 5)\n\nMean [95% bootstrap CI]; percent except "
        "user_turns and latency_s.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
