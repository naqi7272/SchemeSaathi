"""Gradio demo with two modes.

  Q&A          one question -> cited answer from the RAG arm, with the source passages
  Agent chat   multi-turn eligibility agent; every graph step is shown as it happens,
               and clarifying questions arrive as chat messages you answer

    uv run --extra models --extra demo python -m saathi.demo.app --backend ollama \
        --model qwen3-1.7b-saathi:q8_0
"""

import argparse
import json
import uuid
from pathlib import Path

from langgraph.types import Command

from saathi.agent.tracing import run_config
from saathi.schemas import Answer, Passage

DISCLAIMER = (
    "Research demo. Answers can be wrong; always confirm on the official scheme "
    "website before applying."
)


def format_answer(answer: Answer, passages: list[Passage]) -> tuple[str, str]:
    """(answer markdown, sources markdown). Cited passages first, then the rest shown."""
    head = answer.text
    if answer.refused:
        head = f"*{answer.text}*"
    cited = [p for p in passages if p.passage_id in answer.citations]
    other = [p for p in passages if p.passage_id not in answer.citations]
    lines = []
    for title, group in (("Cited sources", cited), ("Other retrieved passages", other)):
        if group:
            lines.append(f"### {title}")
            for p in group:
                lines.append(
                    f"**[{p.passage_id}]** ({p.source_url}, fetched "
                    f"{p.fetched_at:%Y-%m-%d})\n\n> {p.text}\n"
                )
    return head, "\n".join(lines) or "_No passages retrieved._"


def format_step(node: str, update: dict) -> str:
    """One line per graph step for the trace panel."""
    steps = (update or {}).get("steps") or []
    step = steps[-1] if steps else {}
    if node == "detect_language":
        return f"detect_language: {step.get('language')}"
    if node == "understand":
        return (
            f"understand: intent={step.get('intent')}, query={step.get('search_query')!r}, "
            f"profile={json.dumps(step.get('profile'), ensure_ascii=False)}"
        )
    if node == "search":
        return f"search_schemes({step.get('args', {}).get('query')!r}) -> {step.get('result')}"
    if node == "choose_scheme":
        return f"choose_scheme -> check_eligibility{step.get('args')}" + (
            f" ERROR: {step['error']}" if "error" in step else ""
        )
    if node == "check_eligibility":
        return f"check_eligibility{step.get('args')} -> {step.get('result')}"
    if node == "write_clarification":
        return f"write_clarification: asks for {step.get('fields')}"
    if node == "ask_user":
        return "ask_user: got the reply"
    if node == "answer":
        return f"answer: verdict={step.get('answer', {}).get('eligibility_verdict')}"
    if node == "self_check":
        return (
            "self_check: passed"
            if step.get("passed")
            else f"self_check: {step.get('problems')}"
            + (" -> revising" if step.get("retry") else "")
        )
    if node == "__interrupt__":
        return "(waiting for your reply)"
    return node


class AgentSession:
    """One browser session: a conversation thread and whether the agent awaits a reply."""

    def __init__(self, app, user_id: str = "demo-user"):
        self.app = app
        self.user_id = user_id
        self.new_thread()

    def new_thread(self) -> None:
        self.thread_id = uuid.uuid4().hex
        self.waiting = False

    def send(self, message: str):
        """Yield (trace line, reply or None) as the graph runs. The last reply is the
        agent's message: either a clarifying question or the final answer."""
        config = run_config(self.thread_id, self.user_id)
        payload = (
            Command(resume=message)
            if self.waiting
            else {"messages": [{"role": "user", "content": message}]}
        )
        self.waiting = False
        reply = None
        for chunk in self.app.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    self.waiting = True
                    reply = update[0].value["question"]
                yield format_step(node, update if isinstance(update, dict) else {}), None
        if not self.waiting:
            answer = self.app.get_state(config).values.get("answer") or {}
            reply = answer.get("text", "")
            if answer.get("citations"):
                reply += "\n\nSources: " + ", ".join(answer["citations"])
        yield None, reply


def build_ui(rag_arm, session_factory):
    import gradio as gr

    with gr.Blocks(title="SchemeSaathi") as demo:
        gr.Markdown(
            f"# SchemeSaathi\nAsk about Indian government schemes in English, Hindi, "
            f"or Hinglish.\n\n_{DISCLAIMER}_"
        )
        with gr.Tabs():
            with gr.Tab("Q&A"):
                question = gr.Textbox(
                    label="Question", placeholder="PM-KISAN mein kitna paisa milta hai?"
                )
                ask = gr.Button("Ask", variant="primary")
                answer_md = gr.Markdown()
                sources_md = gr.Markdown()

                def on_ask(q):
                    out = rag_arm.answer(q)
                    passages = [rag_arm.retriever.by_id[i] for i in out.retrieved]
                    head, sources = format_answer(out.answer, passages)
                    return f"{head}\n\n_{out.latency_s:.1f}s_", sources

                ask.click(on_ask, question, [answer_md, sources_md])
                question.submit(on_ask, question, [answer_md, sources_md])

            with gr.Tab("Agent chat"):
                session = gr.State(None)
                with gr.Row():
                    with gr.Column(scale=3):
                        chat = gr.Chatbot(height=460, label="Conversation")
                        box = gr.Textbox(
                            label="Message",
                            placeholder="Main 35 saal ka kisan hoon, kya mujhe PM-KISAN milega?",
                        )
                        reset = gr.Button("New conversation")
                    with gr.Column(scale=2):
                        trace = gr.Textbox(label="Graph steps", lines=22, interactive=False)

                def on_send(message, history, sess, trace_text):
                    sess = sess or session_factory()
                    history = (history or []) + [{"role": "user", "content": message}]
                    lines = (trace_text or "").splitlines() + [f"--- you: {message}"]
                    yield "", history, sess, "\n".join(lines)
                    for line, reply in sess.send(message):
                        if line:
                            lines.append(line)
                        if reply is not None:
                            history = history + [{"role": "assistant", "content": reply}]
                        yield "", history, sess, "\n".join(lines)

                def on_reset(sess):
                    if sess:
                        sess.new_thread()
                    return [], sess, ""

                box.submit(on_send, [box, chat, session, trace], [box, chat, session, trace])
                reset.click(on_reset, session, [chat, session, trace])
    return demo


def main() -> None:
    from dotenv import load_dotenv

    from saathi.agent.runtime import build_agent
    from saathi.arms.arms import RagArm
    from saathi.eval.run import RunConfig, make_model
    from saathi.io import read_jsonl
    from saathi.retrieval.models import CrossEncoderReranker, E5Embedder
    from saathi.retrieval.search import HybridRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "openai_compat", "claude"], default="ollama")
    ap.add_argument("--model", default="qwen3-1.7b-saathi:q8_0")
    ap.add_argument("--passages", type=Path, default=Path("data/processed/passages.jsonl"))
    ap.add_argument("--memory", type=Path, default=Path("demo_memory.sqlite"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    load_dotenv()

    model = make_model(RunConfig(run_id="demo", arm="rag", backend=args.backend, model=args.model))
    passages = read_jsonl(args.passages, Passage)
    retriever = HybridRetriever(passages, E5Embedder(), CrossEncoderReranker())
    rag_arm = RagArm("demo", model, retriever)
    app = build_agent(model, retriever, passages, db_path=args.memory)
    build_ui(rag_arm, lambda: AgentSession(app)).launch(server_port=args.port)


if __name__ == "__main__":
    main()
