"""Claude API client (OPTIONAL, paid; install with `uv sync --extra claude`).

Uses the beta Messages endpoint so server-side refusal fallbacks ("default" routing) are on.
When a fallback model answers, Usage.served_model records it and cost uses its price.
The current API rejects sampling parameters (temperature etc.), so none are sent;
run-to-run variation is handled by repeated runs in the analysis, not by temperature=0.
"""

import time
from typing import Literal

from pydantic import BaseModel

from saathi.llm.base import LLMResult, Usage, cost_usd, strict_json_schema

try:
    import anthropic
except ImportError as err:  # optional extra: the free path never needs it
    raise ImportError(
        "The Claude backend is optional and needs the `claude` extra: "
        "`uv sync --extra claude`, plus ANTHROPIC_API_KEY in .env. "
        "The default $0 path uses --backend ollama or openai_compat instead."
    ) from err

FALLBACK_BETA = "server-side-fallback-2026-07-01"

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeRefusal(RuntimeError):
    pass


class ClaudeTruncated(RuntimeError):
    pass


class ClaudeChat:
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        client: anthropic.Anthropic | None = None,
    ):
        self.name = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResult:
        output_config: dict = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": strict_json_schema(schema)}
        start = time.perf_counter()
        response = self.client.beta.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        latency = time.perf_counter() - start
        if response.stop_reason == "refusal":
            raise ClaudeRefusal(f"{self.name} refused: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise ClaudeTruncated(f"hit max_tokens={max_tokens}")
        text = "".join(b.text for b in response.content if b.type == "text")
        served = response.model
        return LLMResult(
            text=text,
            usage=Usage(
                requested_model=self.name,
                served_model=served,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=latency,
                cost_usd=cost_usd(
                    served, response.usage.input_tokens, response.usage.output_tokens
                ),
            ),
        )

    def count_input_tokens(self, system: str, messages: list[dict[str, str]]) -> int:
        """Exact input token count (free endpoint), used for cost estimates before a run."""
        return self.client.messages.count_tokens(
            model=self.name, system=system, messages=messages
        ).input_tokens
