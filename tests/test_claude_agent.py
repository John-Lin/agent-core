from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import TextBlock

from agent_core.claude import MAX_TURNS
from agent_core.claude import ClaudeAgent


@dataclass
class QueryCall:
    prompt: str
    options: ClaudeAgentOptions


@dataclass
class FakeQuery:
    """Replacement for claude_agent_sdk.query used in tests.

    Records each call's prompt and options, and yields a configurable script
    of messages ending with a ResultMessage that carries ``session_id``.
    """

    session_id: str = "session-uuid-1"
    result_text: str = "hello"
    calls: list[QueryCall] = field(default_factory=list)

    def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        self.calls.append(QueryCall(prompt=prompt, options=options))
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        yield AssistantMessage(
            content=[TextBlock(text=self.result_text)],
            model="claude",
        )
        yield ResultMessage(
            subtype="success",
            duration_ms=0,
            duration_api_ms=0,
            is_error=False,
            num_turns=1,
            session_id=self.session_id,
            result=self.result_text,
        )


@pytest.fixture
def stub_instructions(monkeypatch):
    monkeypatch.setattr("agent_core.claude._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("SHELL_ENABLED", raising=False)
    monkeypatch.delenv("SESSION_DB_PATH", raising=False)


@pytest.fixture
def fake_query(monkeypatch):
    fq = FakeQuery()
    monkeypatch.setattr("agent_core.claude.query", fq)
    return fq


class TestRun:
    @pytest.mark.anyio
    async def test_returns_result_text_from_result_message(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            out = await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert out == "hello"

    @pytest.mark.anyio
    async def test_first_call_has_no_resume(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert fake_query.calls[0].options.resume is None

    @pytest.mark.anyio
    async def test_second_call_resumes_stored_session(self, fake_query):
        fake_query.session_id = "session-abc"
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            await agent.run("chat-1", "hi")
            await agent.run("chat-1", "again")
        finally:
            await agent.cleanup()
        assert fake_query.calls[1].options.resume == "session-abc"

    @pytest.mark.anyio
    async def test_different_chats_have_independent_sessions(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            fake_query.session_id = "sess-A"
            await agent.run("chat-A", "hi")
            fake_query.session_id = "sess-B"
            await agent.run("chat-B", "hi")
            await agent.run("chat-A", "again")
            await agent.run("chat-B", "again")
        finally:
            await agent.cleanup()
        assert fake_query.calls[0].options.resume is None
        assert fake_query.calls[1].options.resume is None
        assert fake_query.calls[2].options.resume == "sess-A"
        assert fake_query.calls[3].options.resume == "sess-B"

    @pytest.mark.anyio
    async def test_prompt_and_system_prompt_are_passed(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="you are a bot")
        try:
            await agent.run("chat-1", "what is 2+2?")
        finally:
            await agent.cleanup()
        call = fake_query.calls[0]
        assert call.prompt == "what is 2+2?"
        assert call.options.system_prompt == "you are a bot"


class TestFromDict:
    def test_instructions_loaded_from_file(self, stub_instructions, fake_query, monkeypatch):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._instructions == "stub instructions"

    def test_model_default_is_none(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._model_name is None

    def test_model_from_config(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {"model": "claude-sonnet-4-6"})
        assert agent._model_name == "claude-sonnet-4-6"

    def test_max_turns_default(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._max_turns == MAX_TURNS

    def test_max_turns_from_config(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {"maxTurns": 3})
        assert agent._max_turns == 3

    def test_shell_disabled_yields_no_allowed_tools(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._allowed_tools == []
        assert agent._setting_sources is None

    def test_shell_enabled_allows_read_only_plus_bash(self, stub_instructions, fake_query, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SHELL_ENABLED", "1")
        agent = ClaudeAgent.from_dict("t", {})
        assert set(agent._allowed_tools) == {"Bash", "Read", "Glob", "Grep"}
        assert agent._setting_sources == ["project"]

    def test_config_allowed_tools_extend_defaults(self, stub_instructions, fake_query, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SHELL_ENABLED", "1")
        agent = ClaudeAgent.from_dict("t", {"allowedTools": ["WebFetch", "Write"]})
        assert "WebFetch" in agent._allowed_tools
        assert "Write" in agent._allowed_tools
        assert "Bash" in agent._allowed_tools  # base kept

    def test_config_allowed_tools_without_shell_still_applied(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {"allowedTools": ["WebFetch"]})
        assert agent._allowed_tools == ["WebFetch"]

    def test_mcp_stdio_server_transformed(self, stub_instructions, fake_query):  # noqa: ARG002
        config = {
            "mcpServers": {
                "my-stdio": {
                    "command": "python",
                    "args": ["-m", "srv"],
                    "env": {"FOO": "bar"},
                }
            }
        }
        agent = ClaudeAgent.from_dict("t", config)
        srv = agent._mcp_servers["my-stdio"]
        assert srv["type"] == "stdio"
        assert srv["command"] == "python"
        assert srv["args"] == ["-m", "srv"]
        assert srv["env"] == {"FOO": "bar"}

    def test_mcp_http_server_transformed(self, stub_instructions, fake_query):  # noqa: ARG002
        config = {
            "mcpServers": {
                "my-http": {
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer x"},
                }
            }
        }
        agent = ClaudeAgent.from_dict("t", config)
        srv = agent._mcp_servers["my-http"]
        assert srv["type"] == "http"
        assert srv["url"] == "https://example.com/mcp"
        assert srv["headers"] == {"Authorization": "Bearer x"}

    def test_mcp_disabled_server_skipped(self, stub_instructions, fake_query):  # noqa: ARG002
        config = {
            "mcpServers": {
                "off": {"command": "x", "enabled": False},
                "on": {"command": "y"},
            }
        }
        agent = ClaudeAgent.from_dict("t", config)
        assert "off" not in agent._mcp_servers
        assert "on" in agent._mcp_servers


class TestOptionsWiring:
    @pytest.mark.anyio
    async def test_mcp_and_tools_flow_into_options(self, stub_instructions, fake_query, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SHELL_ENABLED", "1")
        agent = ClaudeAgent.from_dict(
            "t",
            {
                "mcpServers": {"srv": {"command": "x"}},
                "allowedTools": ["WebFetch"],
            },
        )
        try:
            await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        opts = fake_query.calls[0].options
        assert "srv" in opts.mcp_servers
        assert "WebFetch" in opts.allowed_tools
        assert "Bash" in opts.allowed_tools
        assert opts.setting_sources == ["project"]

    @pytest.mark.anyio
    async def test_max_turns_flows_into_options(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="sys", max_turns=5)
        try:
            await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert fake_query.calls[0].options.max_turns == 5

    @pytest.mark.anyio
    async def test_model_flows_into_options(self, fake_query):
        agent = ClaudeAgent(name="t", instructions="sys", model_name="claude-sonnet-4-6")
        try:
            await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert fake_query.calls[0].options.model == "claude-sonnet-4-6"


class TestConnectCleanup:
    @pytest.mark.anyio
    async def test_connect_is_no_op(self, fake_query):  # noqa: ARG002
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            await agent.connect()  # must not raise
        finally:
            await agent.cleanup()

    @pytest.mark.anyio
    async def test_cleanup_closes_mapping(self, fake_query):  # noqa: ARG002
        agent = ClaudeAgent(name="t", instructions="sys")
        await agent.cleanup()
        # After cleanup, the mapping DB connection is closed; further ops raise.
        with pytest.raises(Exception):  # noqa: B017
            agent._session_map.get("chat-1")
