import pytest
from pydantic import BaseModel

from saathi.llm.base import cost_usd, strict_json_schema
from saathi.schemas import Answer


def test_cost_for_known_claude_model():
    # Opus 5.5: $4 in / $20 out per million tokens
    assert cost_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert cost_usd("claude-opus-5-5", 1000, 500) == pytest.approx(0.004 + 0.01)


def test_local_models_are_free():
    assert cost_usd("qwen3-4b-sft:q8_0", 10_000, 10_000) == 0.0


def test_unknown_claude_model_raises():
    with pytest.raises(KeyError):
        cost_usd("claude-unknown-9", 1, 1)


class Inner(BaseModel):
    x: int = 0


class Outer(BaseModel):
    inner: Inner
    tags: list[str] = []


def test_strict_schema_closes_all_objects():
    schema = strict_json_schema(Outer)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["inner", "tags"]
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False and inner["required"] == ["x"]


def test_strict_schema_does_not_mutate_pydantic_cache():
    strict_json_schema(Answer)
    assert "additionalProperties" not in Answer.model_json_schema()


def test_ollama_think_is_off_by_default():
    from saathi.llm.local import OllamaChat

    class FakeClient:
        def chat(self, **kw):
            self.kw = kw
            from types import SimpleNamespace

            return SimpleNamespace(
                message=SimpleNamespace(content="ok"), prompt_eval_count=1, eval_count=1
            )

    c = FakeClient()
    OllamaChat("qwen3-1.7b-base:q8_0", client=c).chat("s", [])
    assert c.kw["think"] is False
    OllamaChat("m", client=c, think=None).chat("s", [])
    assert "think" not in c.kw
