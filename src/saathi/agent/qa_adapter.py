"""Run the agent on the single-question Q&A test set, like arms 1-4.

Each question is a fresh conversation with clarifying questions turned off (there is no
user to answer them), so the agent must answer directly. Multi-turn behaviour is evaluated
separately in conversations.py.
"""

import time
import uuid
from pathlib import Path

from saathi.agent.runtime import build_agent
from saathi.agent.tracing import run_config
from saathi.arms.arms import ArmOutput, Retriever
from saathi.io import read_jsonl
from saathi.llm.base import ChatModel, Usage
from saathi.schemas import Answer, Passage


def output_from_state(state: dict, latency_s: float) -> ArmOutput:
    a = state.get("answer") or {}
    return ArmOutput(
        answer=Answer(
            text=a.get("text", ""),
            citations=a.get("citations", []),
            refused=a.get("refused", False),
        ),
        retrieved=[p["passage_id"] for p in state.get("retrieved", [])],
        usage=[Usage.model_validate(u) for u in state.get("usage", [])],
        latency_s=latency_s,
        parse_error=None if a else "agent produced no answer",
    )


class AgentArm:
    def __init__(
        self,
        name: str,
        model: ChatModel,
        retriever: Retriever,
        rules_dir: Path,
        k: int = 5,
        passages_file: Path = Path("data/processed/passages.jsonl"),
    ):
        self.name = name
        passages = read_jsonl(passages_file, Passage)
        self.app = build_agent(
            model, retriever, passages, rules_dir=rules_dir, allow_clarify=False, k=k
        )

    def answer(self, question: str) -> ArmOutput:
        start = time.perf_counter()
        state = self.app.invoke(
            {"messages": [{"role": "user", "content": question}]},
            run_config(thread_id=str(uuid.uuid4())),
        )
        return output_from_state(state, time.perf_counter() - start)
