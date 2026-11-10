"""Pick a chat backend by name, and warn about same-family bias.

Backends:
    ollama         local Ollama via its native API (arms 1-4, local arm 5). Free.
    openai_compat  any OpenAI-compatible endpoint (see llm/openai_compat.py). Free unless a
                   price is set.
    claude         paid Claude API. Optional: needs `uv sync --extra claude` and
                   ANTHROPIC_API_KEY.
"""

import re
import warnings

from saathi.llm.base import ChatModel

BACKENDS = ("ollama", "openai_compat", "claude")
MODEL_UNDER_TEST = "qwen3-1.7b"

# Checked in order; the first pattern found anywhere in the lower-cased name wins.
FAMILIES: list[tuple[str, str]] = [
    ("qwen", r"qwen|qwq|saathi"),  # our fine-tuned model is Qwen3
    ("claude", r"claude"),
    ("gpt", r"\bgpt|openai|\bo[134]\b"),
    ("gemini", r"gemini|gemma"),
    ("llama", r"llama"),
    ("mistral", r"mistral|mixtral|ministral"),
    ("deepseek", r"deepseek"),
    ("phi", r"\bphi"),
    ("sarvam", r"sarvam"),
    ("airavata", r"airavata"),
]


def model_family(name: str) -> str:
    lowered = name.lower()
    for family, pattern in FAMILIES:
        if re.search(pattern, lowered):
            return family
    return "unknown"


def same_family_warning(model: str, role: str, under_test: str = MODEL_UNDER_TEST) -> str | None:
    """Message if `model` (used as generator/judge/simulator) shares a family with the model
    under test, or if its family cannot be told. None when they clearly differ."""
    fam, test_fam = model_family(model), model_family(under_test)
    if fam == "unknown":
        return (
            f"WARNING: cannot tell the model family of {role} model {model!r}; check by hand "
            f"that it is not the same family as the model under test ({under_test})."
        )
    if fam == test_fam:
        return (
            f"WARNING: {role} model {model!r} is the same family ({fam}) as the model under "
            f"test ({under_test}). Same-family {role}s can favour its outputs (bias risk)."
        )
    return None


def warn_same_family(model: str, role: str) -> str | None:
    msg = same_family_warning(model, role)
    if msg:
        warnings.warn(msg, stacklevel=2)
        print(msg)
    return msg


def make_chat(backend: str, model: str, *, effort: str = "low", **kwargs) -> ChatModel:
    if backend == "ollama":
        from saathi.llm.local import OllamaChat

        return OllamaChat(model=model, **kwargs)
    if backend == "openai_compat":
        from saathi.llm.openai_compat import OpenAICompatChat

        return OpenAICompatChat(model=model or None, **kwargs)
    if backend == "claude":
        from saathi.llm.claude import ClaudeChat

        return ClaudeChat(model=model, effort=effort)  # type: ignore[arg-type]
    raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
