"""Local model via Ollama (llama.cpp under the hood). Used by arms 1-4 and arm 5's local run.

Greedy decoding (temperature 0) with a fixed seed so runs are repeatable on one machine.
Structured output uses Ollama's JSON-schema grammar, so the model cannot emit invalid JSON.
"""

import time

from ollama import Client
from pydantic import BaseModel

from saathi.llm.base import LLMResult, Usage, strict_json_schema


class OllamaChat:
    def __init__(
        self,
        model: str,
        num_ctx: int = 8192,
        seed: int = 0,
        client: Client | None = None,
        think: bool | None = False,
    ):
        """think=False turns off Qwen3's hidden reasoning (hybrid models such as
        Qwen3-1.7B think by default, which is slow on CPU). None = server default."""
        self.name = model
        self.num_ctx = num_ctx
        self.seed = seed
        self.think = think
        self.client = client or Client()  # host from OLLAMA_HOST, default localhost:11434

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        start = time.perf_counter()
        response = self.client.chat(
            model=self.name,
            messages=[{"role": "system", "content": system}, *messages],
            format=strict_json_schema(schema) if schema else None,
            **({"think": self.think} if self.think is not None else {}),
            options={
                "temperature": 0,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        )
        latency = time.perf_counter() - start
        return LLMResult(
            text=response.message.content or "",
            usage=Usage(
                requested_model=self.name,
                served_model=self.name,
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
                latency_s=latency,
                cost_usd=0.0,
            ),
        )
