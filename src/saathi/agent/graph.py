"""Arm 5: LangGraph eligibility agent.

One user turn runs:

  detect_language (code) -> understand (LLM: profile + intent + search query)
    -> search (tool: hybrid retriever)
    -> choose_scheme (LLM tool decision: which scheme to check, or none)
    -> check_eligibility (tool: deterministic rules)
    -> [rules say "unknown" and asking is allowed]
         write_clarification (LLM) -> ask_user (interrupt) -> understand -> check_eligibility ...
    -> answer (LLM: cited answer with eligibility verdict)
    -> self_check (code) -> answer again once if a check failed -> END

Tool calls are structured-output decisions validated in code, so they behave the same on
Claude and on a local model, and every call is logged in `steps` for evaluation.
The LLM call that writes a clarifying question is in its own node because LangGraph
re-runs an interrupted node from the top on resume; `ask_user` only interrupts.

Memory: a checkpointer keeps each conversation (thread_id); a store keeps the user profile
across conversations (configurable.user_id).
"""

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel

from saathi.agent.eligibility import (
    PROFILE_FIELDS,
    SchemeRules,
    Status,
    UserProfile,
    evaluate,
)
from saathi.agent.language import detect_language
from saathi.arms.arms import Retriever
from saathi.llm.base import ChatModel, LLMResult
from saathi.schemas import Language, Passage

LANGUAGE_NAMES = {
    Language.EN: "English",
    Language.HI: "Hindi in Devanagari script",
    Language.HINGLISH: "Hinglish (Hindi-English mix in Roman script)",
}
MAX_CLARIFICATIONS = 2
MAX_REVISIONS = 1
HISTORY_MESSAGES = 8


# ---------------------------------------------------------------- structured outputs
class Understanding(BaseModel):
    profile: UserProfile  # only facts the user stated; null for everything else
    intent: Literal["eligibility", "information", "other"]
    search_query: str


class SchemeChoice(BaseModel):
    scheme_id: str | None  # None = do not call check_eligibility


class Clarification(BaseModel):
    question: str
    fields: list[str]


class AgentAnswer(BaseModel):
    text: str
    citations: list[str]
    refused: bool
    eligibility_verdict: Literal["eligible", "not_eligible", "unknown", "not_applicable"]


# ---------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # whole conversation, kept by checkpointer
    language: str
    profile: dict
    intent: str
    search_query: str
    retrieved: list[dict]
    scheme_id: str | None
    eligibility: dict | None
    clarification: dict | None
    clarifications: int
    answer: dict | None
    feedback: list[str]
    revisions: int
    steps: list[dict]  # this turn's trace: one entry per node, for the demo and evaluation
    usage: list[dict]  # this turn's LLM usage records


@dataclass
class AgentDeps:
    model: ChatModel
    retriever: Retriever
    rules: dict[str, SchemeRules]
    passages_by_id: dict[str, Passage]
    scheme_names: dict[str, str]
    k: int = 5
    allow_clarify: bool = True


def _history(state: AgentState) -> str:
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("messages", [])[-HISTORY_MESSAGES:]
    )


def _llm(
    deps: AgentDeps, system: str, user: str, schema: type[BaseModel], max_tokens: int = 1024
) -> LLMResult:
    return deps.model.chat(
        system, [{"role": "user", "content": user}], schema=schema, max_tokens=max_tokens
    )


def _merge_profile(old: dict, update: UserProfile) -> dict:
    """New non-null facts overwrite old ones; nulls never erase what is known."""
    merged = dict(old)
    merged.update({k: v for k, v in update.model_dump().items() if v is not None})
    return merged


def build_graph(deps: AgentDeps, checkpointer=None, store: BaseStore | None = None):
    # ------------------------------------------------------------ nodes
    def detect(state: AgentState) -> dict:
        """Start of a user turn: detect language and reset per-turn fields."""
        last_user = state["messages"][-1]["content"]
        language = detect_language(last_user)
        return {
            "language": language.value,
            "clarifications": 0,
            "revisions": 0,
            "answer": None,
            "feedback": [],
            "eligibility": None,
            "scheme_id": None,
            "clarification": None,
            "usage": [],
            "steps": [{"node": "detect_language", "language": language.value}],
        }

    def understand(state: AgentState, config: RunnableConfig) -> dict:
        user_id = config.get("configurable", {}).get("user_id")
        profile = dict(state.get("profile") or {})
        if store is not None and user_id:
            saved = store.get(("users", user_id), "profile")
            if saved:
                profile = saved.value | profile
        system = (
            "You read a conversation with a person asking about Indian government schemes. "
            "Extract ONLY facts the person stated about themselves into `profile` (null for "
            "anything not stated; ages in years, income in rupees per year, land in hectares: "
            "1 acre = 0.4047 ha). `intent` is 'eligibility' if they want to know whether they "
            "qualify, 'information' for other questions about a scheme. `search_query` is a "
            "short English query for finding the relevant scheme documents."
        )
        user = (
            f"Known profile: {json.dumps(profile, ensure_ascii=False)}\n\n"
            f"Conversation:\n{_history(state)}"
        )
        result = _llm(deps, system, user, Understanding)
        u = Understanding.model_validate_json(result.text)
        profile = _merge_profile(profile, u.profile)
        if store is not None and user_id:
            store.put(("users", user_id), "profile", profile)
        return {
            "profile": profile,
            "intent": u.intent,
            "search_query": u.search_query,
            "usage": state.get("usage", []) + [result.usage.model_dump()],
            "steps": state.get("steps", [])
            + [
                {
                    "node": "understand",
                    "intent": u.intent,
                    "search_query": u.search_query,
                    "profile": profile,
                }
            ],
        }

    def search(state: AgentState) -> dict:
        passages = deps.retriever.retrieve(state["search_query"], k=deps.k)
        return {
            "retrieved": [p.model_dump(mode="json") for p in passages],
            "steps": state.get("steps", [])
            + [
                {
                    "node": "search",
                    "tool": "search_schemes",
                    "args": {"query": state["search_query"]},
                    "result": [p.passage_id for p in passages],
                }
            ],
        }

    def choose_scheme(state: AgentState) -> dict:
        retrieved_schemes = list(dict.fromkeys(p["scheme_id"] for p in state["retrieved"]))
        options = [s for s in retrieved_schemes if s in deps.rules]
        step = {"node": "choose_scheme", "tool": "check_eligibility", "options": options}
        if state.get("intent") != "eligibility" or not options:
            step |= {"args": None, "reason": "not an eligibility question or no rules"}
            return {"scheme_id": None, "steps": state.get("steps", []) + [step]}
        listing = "\n".join(f"- {s}: {deps.scheme_names.get(s, s)}" for s in options)
        system = (
            "Pick the scheme the person is asking about from the list, to check their "
            "eligibility. Answer with its scheme_id exactly as listed, or null if the "
            "person is not asking about any of them."
        )
        result = _llm(
            deps, system, f"Schemes:\n{listing}\n\nConversation:\n{_history(state)}", SchemeChoice
        )
        choice = SchemeChoice.model_validate_json(result.text).scheme_id
        if choice is not None and choice not in options:
            step |= {"args": {"scheme_id": choice}, "error": "invalid scheme_id"}
            choice = None
        else:
            step |= {"args": {"scheme_id": choice} if choice else None}
        return {
            "scheme_id": choice,
            "usage": state.get("usage", []) + [result.usage.model_dump()],
            "steps": state.get("steps", []) + [step],
        }

    def check_eligibility(state: AgentState) -> dict:
        rules = deps.rules[state["scheme_id"]]
        result = evaluate(UserProfile.model_validate(state.get("profile") or {}), rules)
        # make the rules' source passages available to cite
        have = {p["passage_id"] for p in state["retrieved"]}
        extra = [
            deps.passages_by_id[r.passage_id].model_dump(mode="json")
            for r in rules.rules
            if r.passage_id not in have and r.passage_id in deps.passages_by_id
        ]
        extra = list({p["passage_id"]: p for p in extra}.values())
        return {
            "eligibility": result.model_dump(mode="json"),
            "retrieved": state["retrieved"] + extra,
            "steps": state.get("steps", [])
            + [
                {
                    "node": "check_eligibility",
                    "tool": "check_eligibility",
                    "args": {"scheme_id": state["scheme_id"]},
                    "result": {
                        "status": result.status.value,
                        "missing_fields": result.missing_fields,
                        "failed": [r.field for r in result.failed],
                    },
                }
            ],
        }

    def write_clarification(state: AgentState) -> dict:
        missing = state["eligibility"]["missing_fields"]
        language = LANGUAGE_NAMES[Language(state["language"])]
        system = (
            f"Write ONE short, friendly question in {language} asking the person for "
            "exactly these details, so their eligibility can be checked. Do not ask for "
            "anything else. List the field names you ask about in `fields`."
        )
        result = _llm(
            deps,
            system,
            f"Missing fields: {missing}\n\nConversation:\n{_history(state)}",
            Clarification,
        )
        c = Clarification.model_validate_json(result.text)
        c.fields = [f for f in c.fields if f in PROFILE_FIELDS]
        return {
            "clarification": c.model_dump(),
            "usage": state.get("usage", []) + [result.usage.model_dump()],
            "steps": state.get("steps", [])
            + [
                {
                    "node": "clarify",
                    "question": c.question,
                    "fields": c.fields,
                    "missing_fields": missing,
                }
            ],
        }

    def ask_user(state: AgentState) -> dict:
        reply = interrupt(state["clarification"])  # resumes with the user's answer
        return {
            "messages": [
                {"role": "assistant", "content": state["clarification"]["question"]},
                {"role": "user", "content": reply},
            ],
            "clarifications": state.get("clarifications", 0) + 1,
        }

    def answer(state: AgentState) -> dict:
        language = LANGUAGE_NAMES[Language(state["language"])]
        passages = "\n\n".join(f"[{p['passage_id']}]\n{p['text']}" for p in state["retrieved"])
        elig = state.get("eligibility")
        elig_text = "not checked (eligibility_verdict must be 'not_applicable')"
        if elig:
            elig_text = (
                f"scheme {elig['scheme_id']}: {elig['status']}; failed rules: "
                f"{[r['field'] + ' ' + r['op'] + ' ' + str(r['value']) for r in elig['failed']]}; "
                f"missing: {elig['missing_fields']}. eligibility_verdict must be "
                f"'{elig['status']}'."
            )
        feedback = state.get("feedback") or []
        system = (
            f"You help people understand Indian government schemes. Reply in {language}. "
            "Use ONLY the passages; cite the IDs of the passages you rely on in `citations`. "
            "If an eligibility result is given, state it plainly and explain the reason using "
            "the rules and passages; never contradict it. If the passages do not answer the "
            "question, set refused to true and say so. Be brief (2-5 sentences)."
        )
        user = (
            f"Passages:\n\n{passages}\n\nEligibility check: {elig_text}\n\n"
            f"Profile: {json.dumps(state.get('profile') or {}, ensure_ascii=False)}\n\n"
            f"Conversation:\n{_history(state)}"
        )
        if feedback:
            user += "\n\nYour previous answer failed these checks; fix them:\n- " + "\n- ".join(
                feedback
            )
        # Devanagari takes many tokens; 1024 cut Hindi answers off mid-JSON
        result = _llm(deps, system, user, AgentAnswer, max_tokens=2048)
        a = AgentAnswer.model_validate_json(result.text)
        return {
            "answer": a.model_dump(),
            "usage": state.get("usage", []) + [result.usage.model_dump()],
            "steps": state.get("steps", []) + [{"node": "answer", "answer": a.model_dump()}],
        }

    def self_check(state: AgentState) -> dict:
        a = AgentAnswer.model_validate(state["answer"])
        problems = check_answer(a, state)
        retry = bool(problems) and state.get("revisions", 0) < MAX_REVISIONS
        update: dict = {
            "feedback": problems,
            "steps": state.get("steps", [])
            + [
                {"node": "self_check", "passed": not problems, "problems": problems, "retry": retry}
            ],
        }
        if retry:
            update["revisions"] = state.get("revisions", 0) + 1
        else:
            update["messages"] = [{"role": "assistant", "content": a.text}]
        return update

    # ------------------------------------------------------------ routing
    def after_understand(state: AgentState) -> str:
        # after a clarification the scheme is already chosen: go straight to the rules
        if state.get("clarifications", 0) > 0 and state.get("scheme_id"):
            return "check_eligibility"
        return "search"

    def after_choose(state: AgentState) -> str:
        return "check_eligibility" if state.get("scheme_id") else "answer"

    def after_eligibility(state: AgentState) -> str:
        unknown = state["eligibility"]["status"] == Status.UNKNOWN
        if deps.allow_clarify and unknown and state.get("clarifications", 0) < MAX_CLARIFICATIONS:
            return "write_clarification"
        return "answer"

    def after_self_check(state: AgentState) -> str:
        return "answer" if state["steps"][-1]["retry"] else END

    g = StateGraph(AgentState)
    for name, fn in [
        ("detect_language", detect),
        ("understand", understand),
        ("search", search),
        ("choose_scheme", choose_scheme),
        ("check_eligibility", check_eligibility),
        ("write_clarification", write_clarification),
        ("ask_user", ask_user),
        ("answer", answer),
        ("self_check", self_check),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "detect_language")
    g.add_edge("detect_language", "understand")
    g.add_conditional_edges("understand", after_understand, ["search", "check_eligibility"])
    g.add_edge("search", "choose_scheme")
    g.add_conditional_edges("choose_scheme", after_choose, ["check_eligibility", "answer"])
    g.add_conditional_edges(
        "check_eligibility", after_eligibility, ["write_clarification", "answer"]
    )
    g.add_edge("write_clarification", "ask_user")
    g.add_edge("ask_user", "understand")
    g.add_edge("answer", "self_check")
    g.add_conditional_edges("self_check", after_self_check, ["answer", END])
    return g.compile(checkpointer=checkpointer, store=store)


def check_answer(a: AgentAnswer, state: AgentState) -> list[str]:
    """Code-only checks on a draft answer. Returns problems (empty list = passed)."""
    problems = []
    retrieved = {p["passage_id"] for p in state.get("retrieved", [])}
    invented = [c for c in a.citations if c not in retrieved]
    if invented:
        problems.append(f"citations not in the passages: {invented}")
    if not a.refused and not a.citations:
        problems.append("the answer cites no passage")
    elig = state.get("eligibility")
    expected = elig["status"] if elig else "not_applicable"
    if a.eligibility_verdict != expected:
        problems.append(
            f"eligibility_verdict is '{a.eligibility_verdict}' but the rules say '{expected}'"
        )
    if detect_language(a.text) == Language.HI and state["language"] != Language.HI:
        problems.append(f"answer is in Hindi script but the user wrote in {state['language']}")
    if state["language"] == Language.HI and detect_language(a.text) != Language.HI:
        problems.append("the user wrote in Hindi; answer in Devanagari")
    return problems
