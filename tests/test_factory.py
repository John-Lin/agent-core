from __future__ import annotations

import pytest

from agent_core import ClaudeAgent
from agent_core import OpenAIAgent
from agent_core import build_agent


@pytest.fixture
def stub_instructions(monkeypatch):
    monkeypatch.setattr("agent_core.openai_provider._load_instructions", lambda: "stub instructions")
    monkeypatch.setattr("agent_core.anthropic_provider._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("SESSION_DB_PATH", raising=False)


@pytest.fixture(autouse=True)
def _mock_openai_model(monkeypatch):
    from unittest.mock import create_autospec

    from agents.models.interface import Model

    monkeypatch.setattr("agent_core.openai_provider._get_model", lambda model_name, api_type: create_autospec(Model))


class TestBuildAgent:
    def test_default_provider_is_openai(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {})
        assert isinstance(agent, OpenAIAgent)

    def test_explicit_openai_provider(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {"provider": {"type": "openai"}})
        assert isinstance(agent, OpenAIAgent)

    def test_anthropic_provider(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {"provider": {"type": "anthropic"}})
        assert isinstance(agent, ClaudeAgent)

    def test_unknown_provider_type_raises(self, stub_instructions):  # noqa: ARG002
        with pytest.raises(ValueError, match="Unknown provider type"):
            build_agent("t", {"provider": {"type": "gemini"}})

    def test_provider_must_be_dict(self, stub_instructions):  # noqa: ARG002
        with pytest.raises(ValueError, match="must be a dict"):
            build_agent("t", {"provider": "openai"})

    def test_provider_must_have_type(self, stub_instructions):  # noqa: ARG002
        with pytest.raises(ValueError, match="must be a dict"):
            build_agent("t", {"provider": {"model": "x"}})
