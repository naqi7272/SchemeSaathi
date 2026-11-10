"""Any OpenAI-compatible chat endpoint (Ollama's /v1, llama.cpp server, vLLM, free hosted
providers). Plain HTTP via httpx, no extra SDK.

Configured by arguments or environment variables (arguments win):
    SAATHI_LLM_BASE_URL   default http://localhost:11434/v1 (local Ollama)
    SAATHI_LLM_API_KEY    optional; sent as a Bearer token. Never logged or written to disk.
    SAATHI_LLM_MODEL      model name on that endpoint
    SAATHI_LLM_PRICE_IN   USD per million input tokens  (default 0 = free backend)
    SAATHI_LLM_PRICE_OUT  USD per million output tokens (default 0 = free backend)
    SAATHI_LLM_JSON_MODE  json_schema (default) | json_object | prompt
        json_schema  server enforces the schema (Ollama, vLLM, many providers)
        json_object  server only guarantees JSON; the schema is put in the system prompt
        prompt       no server support; schema in the prompt, JSON extracted from the text

Rate limits (HTTP 429) and transient errors (5xx, network) are retried with exponential
backoff, honouring Retry-After. Other errors raise at once.

Cost: if a price is set, cost is computed from it; otherwise the backend is treated as free
($0), which is what the --max-usd guards then see. Set the price for any paid endpoint.
"""

import json
import os
import re
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from saathi.llm.base import LLMResult, Usage, strict_json_schema

DEFAULT_BASE_URL = "http://localhost:11434/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
JSON_MODES = ("json_schema", "json_object", "prompt")


class BackendError(RuntimeError):
    pass


def _env_float(name: str) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else 0.0


def extract_json(text: str) -> str:
    """Pull a JSON object out of model text: strips ```json fences and surrounding prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # HTTP-date form; fall back to our own backoff


class OpenAICompatChat:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
        json_mode: str | None = None,
        seed: int = 0,
        timeout_s: float = 300.0,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        max_backoff_s: float = 60.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.name = model or os.getenv("SAATHI_LLM_MODEL", "")
        if not self.name:
            raise ValueError("no model: pass model= or set SAATHI_LLM_MODEL")
        self.base_url = (base_url or os.getenv("SAATHI_LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self._api_key = api_key if api_key is not None else os.getenv("SAATHI_LLM_API_KEY", "")
        self.price_in = price_in if price_in is not None else _env_float("SAATHI_LLM_PRICE_IN")
        self.price_out = price_out if price_out is not None else _env_float("SAATHI_LLM_PRICE_OUT")
        self.json_mode = json_mode or os.getenv("SAATHI_LLM_JSON_MODE") or "json_schema"
        if self.json_mode not in JSON_MODES:
            raise ValueError(f"json_mode must be one of {JSON_MODES}, got {self.json_mode!r}")
        self.seed = seed
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self.client = client or httpx.Client(timeout=timeout_s)
        self._sleep = sleep

    @property
    def is_free(self) -> bool:
        return self.price_in == 0 and self.price_out == 0

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatChat(model={self.name!r}, base_url={self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None,
        max_tokens: int,
    ) -> dict:
        if schema is not None and self.json_mode != "json_schema":
            system = (
                f"{system}\n\nReply with ONE JSON object only, no other text, matching this "
                f"JSON schema:\n{json.dumps(strict_json_schema(schema), ensure_ascii=False)}"
            )
        payload: dict = {
            "model": self.name,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": self.seed,
        }
        if schema is not None and self.json_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_json_schema(schema),
                },
            }
        elif schema is not None and self.json_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            wait = min(self.max_backoff_s, self.backoff_s * 2**attempt)
            try:
                response = self.client.post(url, json=payload, headers=self._headers())
            except httpx.TransportError as err:
                if attempt == self.max_retries:
                    raise BackendError(f"{url}: network error after retries: {err!r}") from err
                print(f"network error ({err.__class__.__name__}); retry in {wait:.0f}s")
                self._sleep(wait)
                continue
            if response.status_code in RETRY_STATUS and attempt < self.max_retries:
                wait = _retry_after(response) or wait
                print(f"HTTP {response.status_code} from {self.base_url}; retry in {wait:.0f}s")
                self._sleep(wait)
                continue
            if response.status_code >= 400:
                raise BackendError(f"HTTP {response.status_code} from {url}: {response.text[:500]}")
            return response.json()
        raise AssertionError("unreachable")

    def chat(
        self,
        system: str,
        messages: list[dict[str, str]],
        schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
    ) -> LLMResult:
        start = time.perf_counter()
        data = self._post(self._payload(system, messages, schema, max_tokens))
        latency = time.perf_counter() - start
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as err:
            raise BackendError(f"unexpected response shape: {str(data)[:300]}") from err
        if choice.get("finish_reason") == "length":
            raise BackendError(f"hit max_tokens={max_tokens}")
        if schema is not None and self.json_mode != "json_schema":
            text = extract_json(text)
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        return LLMResult(
            text=text,
            usage=Usage(
                requested_model=self.name,
                served_model=data.get("model") or self.name,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                latency_s=latency,
                cost_usd=(tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000,
            ),
        )
