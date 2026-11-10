import importlib.util

import pytest

from saathi.eval.run import RunConfig, make_model
from saathi.llm.factory import make_chat, model_family, same_family_warning
from saathi.llm.local import OllamaChat
from saathi.llm.openai_compat import OpenAICompatChat


@pytest.mark.parametrize(
    "name, family",
    [
        ("qwen3-1.7b-base:q8_0", "qwen"),
        ("qwen3-1.7b-saathi:q8_0", "qwen"),
        ("Qwen/Qwen2.5-72B-Instruct", "qwen"),
        ("claude-opus-5-5", "claude"),
        ("llama-3.3-70b-versatile", "llama"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
        ("gemma2:9b", "gemini"),
        ("mistral-small", "mistral"),
        ("gpt-4o-mini", "gpt"),
        ("sarvam-m", "sarvam"),
        ("something-new", "unknown"),
    ],
)
def test_model_family(name, family):
    assert model_family(name) == family


def test_same_family_warning():
    assert "same family (qwen)" in same_family_warning("qwen2.5:32b", "judge")
    assert same_family_warning("llama3.1:8b", "judge") is None
    assert "cannot tell" in same_family_warning("mystery-7b", "generator")


def test_make_chat_backends():
    assert isinstance(make_chat("ollama", "qwen3-1.7b-base:q8_0"), OllamaChat)
    m = make_chat("openai_compat", "llama3.1:8b", base_url="http://x/v1")
    assert isinstance(m, OpenAICompatChat) and m.base_url == "http://x/v1"
    with pytest.raises(ValueError, match="unknown backend"):
        make_chat("gpt-magic", "m")


def test_run_config_openai_compat_needs_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SAATHI_LLM_API_KEY", raising=False)
    cfg = RunConfig(
        run_id="t", arm="closed_book", backend="openai_compat", model="m", base_url="http://h/v1"
    )
    model = make_model(cfg)
    assert isinstance(model, OpenAICompatChat) and model.base_url == "http://h/v1"


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None, reason="claude extra installed"
)
def test_claude_backend_without_extra_gives_clear_error():
    with pytest.raises(ImportError, match="extra claude"):
        make_chat("claude", "claude-opus-5-5")
