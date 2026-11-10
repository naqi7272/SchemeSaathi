"""Tracing for the agent. Pick one in .env:

LangSmith: LANGSMITH_TRACING=true and LANGSMITH_API_KEY. LangGraph traces automatically;
           no code needed here.
Langfuse:  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY (and LANGFUSE_BASE_URL if self-hosted).
           Passed as a callback on each graph call.
"""

import os


def langfuse_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def run_config(thread_id: str, user_id: str | None = None) -> dict:
    """Config for one graph call, with tracing callbacks and session metadata."""
    config: dict = {"configurable": {"thread_id": thread_id}}
    if user_id:
        config["configurable"]["user_id"] = user_id
    if langfuse_enabled():
        from langfuse.langchain import CallbackHandler

        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {
            "langfuse_session_id": thread_id,
            "langfuse_user_id": user_id or "anonymous",
        }
    return config


def flush() -> None:
    """Send buffered traces before the process exits."""
    if langfuse_enabled():
        from langfuse import get_client

        get_client().flush()
