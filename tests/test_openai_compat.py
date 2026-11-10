"""OpenAI-compatible backend, tested against a fake HTTP server (no network, no keys)."""

import json

import httpx
import pytest
from pydantic import BaseModel

from saathi.llm.openai_compat import BackendError, OpenAICompatChat, extract_json


class Out(BaseModel):
    answer: str


def ok_body(content: str, model: str = "served-model", finish: str = "stop") -> dict:
    return {
        "model": model,
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def make(handler, sleeps: list | None = None, **kwargs) -> OpenAICompatChat:
    record = sleeps if sleeps is not None else []
    return OpenAICompatChat(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=record.append,
        **{"model": "test-model", "base_url": "http://fake/v1", **kwargs},
    )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "SAATHI_LLM_BASE_URL",
        "SAATHI_LLM_API_KEY",
        "SAATHI_LLM_MODEL",
        "SAATHI_LLM_PRICE_IN",
        "SAATHI_LLM_PRICE_OUT",
        "SAATHI_LLM_JSON_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_basic_call_sends_schema_and_parses_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body('{"answer": "6000"}'))

    model = make(handler)
    result = model.chat("sys", [{"role": "user", "content": "q"}], schema=Out)
    assert seen["url"] == "http://fake/v1/chat/completions"
    assert seen["auth"] is None  # no key configured -> no header
    body = seen["body"]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["temperature"] == 0 and body["seed"] == 0
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
    assert Out.model_validate_json(result.text).answer == "6000"
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20
    assert result.usage.served_model == "served-model"
    assert result.usage.requested_model == "test-model"
    assert result.usage.cost_usd == 0.0
    assert model.is_free


def test_env_configuration_and_bearer_key(monkeypatch):
    monkeypatch.setenv("SAATHI_LLM_BASE_URL", "http://envhost/v1/")
    monkeypatch.setenv("SAATHI_LLM_MODEL", "env-model")
    monkeypatch.setenv("SAATHI_LLM_API_KEY", "sk-test-secret")
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=ok_body("hi"))

    model = OpenAICompatChat(client=httpx.Client(transport=httpx.MockTransport(handler)))
    model.chat("s", [{"role": "user", "content": "q"}])
    assert seen == {
        "url": "http://envhost/v1/chat/completions",
        "auth": "Bearer sk-test-secret",
        "model": "env-model",
    }
    assert "sk-test-secret" not in repr(model)


def test_missing_model_is_an_error():
    with pytest.raises(ValueError, match="SAATHI_LLM_MODEL"):
        OpenAICompatChat(base_url="http://fake/v1")


def test_price_makes_cost_nonzero():
    model = make(lambda r: httpx.Response(200, json=ok_body("x")), price_in=1.0, price_out=2.0)
    result = model.chat("s", [{"role": "user", "content": "q"}])
    assert result.usage.cost_usd == pytest.approx((100 * 1.0 + 20 * 2.0) / 1e6)
    assert not model.is_free


def test_price_from_env(monkeypatch):
    monkeypatch.setenv("SAATHI_LLM_PRICE_IN", "0.5")
    model = make(lambda r: httpx.Response(200, json=ok_body("x")))
    assert model.price_in == 0.5 and not model.is_free


def test_retries_429_honouring_retry_after_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow down")
        return httpx.Response(200, json=ok_body("done"))

    sleeps: list[float] = []
    result = make(handler, sleeps).chat("s", [{"role": "user", "content": "q"}])
    assert result.text == "done"
    assert len(calls) == 3
    assert sleeps == [7.0, 7.0]


def test_exponential_backoff_on_5xx_and_network_errors():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=ok_body("ok"))

    sleeps: list[float] = []
    assert make(handler, sleeps, backoff_s=1.0).chat("s", []).text == "ok"
    assert sleeps == [1.0, 2.0]


def test_backoff_is_capped():
    sleeps: list[float] = []
    model = make(
        lambda r: httpx.Response(500), sleeps, backoff_s=10, max_backoff_s=15, max_retries=3
    )
    with pytest.raises(BackendError, match="HTTP 500"):
        model.chat("s", [])
    assert sleeps == [10, 15, 15]


def test_gives_up_after_max_retries():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    with pytest.raises(BackendError, match="429"):
        make(handler, max_retries=2).chat("s", [])
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, text="bad key")

    with pytest.raises(BackendError, match="401"):
        make(handler).chat("s", [])
    assert len(calls) == 1


def test_truncated_output_raises():
    model = make(lambda r: httpx.Response(200, json=ok_body('{"ans', finish="length")))
    with pytest.raises(BackendError, match="max_tokens"):
        model.chat("s", [], schema=Out)


def test_bad_response_shape_raises():
    model = make(lambda r: httpx.Response(200, json={"error": "nope"}))
    with pytest.raises(BackendError, match="unexpected response"):
        model.chat("s", [])


@pytest.mark.parametrize("mode", ["json_object", "prompt"])
def test_fallback_json_modes_put_schema_in_prompt_and_extract(mode):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body('Sure!\n```json\n{"answer": "हाँ"}\n```'))

    result = make(handler, json_mode=mode).chat("sys", [], schema=Out)
    assert Out.model_validate_json(result.text).answer == "हाँ"
    system = seen["body"]["messages"][0]["content"]
    assert system.startswith("sys") and '"answer"' in system
    if mode == "json_object":
        assert seen["body"]["response_format"] == {"type": "json_object"}
    else:
        assert "response_format" not in seen["body"]


def test_invalid_json_mode():
    with pytest.raises(ValueError, match="json_mode"):
        OpenAICompatChat(model="m", json_mode="xml")


def test_extract_json():
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here: {"a": {"b": 2}} done') == '{"a": {"b": 2}}'
    assert extract_json("no json") == "no json"
