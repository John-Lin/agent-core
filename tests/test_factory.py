from __future__ import annotations

import pytest

from agent_core import ClaudeAgent
from agent_core import OpenAIAgent
from agent_core import build_agent


@pytest.fixture
def stub_instructions(monkeypatch):
    monkeypatch.setattr("agent_core.agent._load_instructions", lambda: "stub instructions")
    monkeypatch.setattr("agent_core.claude._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("SHELL_ENABLED", raising=False)
    monkeypatch.delenv("SESSION_DB_PATH", raising=False)


@pytest.fixture(autouse=True)
def _mock_openai_model(monkeypatch):
    from unittest.mock import create_autospec

    from agents.models.interface import Model

    monkeypatch.setattr("agent_core.agent._get_model", lambda model_name, api_type: create_autospec(Model))


class TestBuildAgent:
    def test_default_provider_is_openai(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {})
        assert isinstance(agent, OpenAIAgent)

    def test_explicit_openai_provider(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {"provider": "openai"})
        assert isinstance(agent, OpenAIAgent)

    def test_claude_provider(self, stub_instructions):  # noqa: ARG002
        agent = build_agent("t", {"provider": "claude"})
        assert isinstance(agent, ClaudeAgent)

    def test_unknown_provider_raises(self, stub_instructions):  # noqa: ARG002
        with pytest.raises(ValueError, match="Unknown provider"):
            build_agent("t", {"provider": "gemini"})
