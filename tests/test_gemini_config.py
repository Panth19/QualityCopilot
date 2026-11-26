from __future__ import annotations

import sys
import types

from qualitycopilot import graph


def test_gemini_model_uses_key_model_and_structured_schemas(monkeypatch):
    created = {}

    class FakeChatGoogleGenerativeAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def with_structured_output(self, schema):
            created.setdefault("schemas", []).append(schema)
            return object()

    module = types.ModuleType("langchain_google_genai")
    module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI
    monkeypatch.setitem(sys.modules, "langchain_google_genai", module)

    model = graph.GeminiModel(api_key="test-gemini-key", model="test-model")

    assert created["api_key"] == "test-gemini-key"
    assert created["model"] == "test-model"
    assert created["temperature"] == 0
    assert created["schemas"] == [graph.Analysis, graph.SupportCheck]
    assert model.analyzer is not None
    assert model.checker is not None


def test_default_model_reads_gemini_key_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setattr(graph, "GeminiModel", lambda **kwargs: kwargs)

    assert graph.default_model() == {"api_key": "env-key", "model": "gemini-test"}


def test_default_model_keeps_no_key_demo_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert isinstance(graph.default_model(), graph.DemoModel)
