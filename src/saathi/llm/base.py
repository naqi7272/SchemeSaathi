"""Common interface for every model the project calls (Claude API or local Ollama).

Every call returns the text plus a Usage record, so latency, tokens, and cost are logged the
same way for all arms.
"""

import copy
from typing import Any, Protocol

from pydantic import BaseModel

# USD per million tokens (input, output). First-party Claude API list prices, checked
# 2026-10-23. Thinking tokens are billed as output tokens. Local models cost 0.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class Usage(BaseModel):
    requested_model: str
    served_model: str  # differs from requested when a server-side fallback answered
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


class LLMResult(BaseModel):
    text: str
    usage: Usage


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResult: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of one call. Unknown non-Claude models (local) are free; unknown Claude models raise."""
    if model not in PRICES_PER_MTOK:
        if model.startswith("claude"):
            raise KeyError(f"no price for {model}; add it to PRICES_PER_MTOK")
        return 0.0
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema with every property required and no extra properties.

    Structured-output engines (Claude, Ollama/llama.cpp grammars) need closed objects.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    return schema
