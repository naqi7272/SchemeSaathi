"""Build a ready-to-run agent: real retriever, verified rules, SQLite memory."""

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from saathi.agent.eligibility import load_all_rules
from saathi.agent.graph import AgentDeps, build_graph
from saathi.arms.arms import Retriever
from saathi.io import read_jsonl
from saathi.llm.base import ChatModel
from saathi.schemas import Passage, Scheme


def build_agent(
    model: ChatModel,
    retriever: Retriever,
    passages: list[Passage],
    rules_dir: Path = Path("data/rules"),
    schemes_file: Path = Path("data/processed/schemes.jsonl"),
    db_path: Path | str = ":memory:",
    allow_clarify: bool = True,
    k: int = 5,
):
    """db_path=':memory:' for evaluation runs; a file path for the demo's persistent memory."""
    names = (
        {s.scheme_id: s.name for s in read_jsonl(schemes_file, Scheme)}
        if Path(schemes_file).exists()
        else {}
    )
    deps = AgentDeps(
        model=model,
        retriever=retriever,
        rules=load_all_rules(rules_dir),
        passages_by_id={p.passage_id: p for p in passages},
        scheme_names=names,
        k=k,
        allow_clarify=allow_clarify,
    )
    saver = SqliteSaver(sqlite3.connect(str(db_path), check_same_thread=False))
    store = SqliteStore(
        sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
    )
    store.setup()
    return build_graph(deps, checkpointer=saver, store=store)
