from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import TextBlock

from agent_core import AgentError
from agent_core.anthropic_provider import ClaudeAgent


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
    is_error: bool = False
    subtype: str = "success"
    empty_stream: bool = False
    calls: list[QueryCall] = field(default_factory=list)

    def __call__(self, *, prompt: str, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        self.calls.append(QueryCall(prompt=prompt, options=options))
        return self._gen()

    async def _gen(self) -> AsyncIterator[Any]:
        if self.empty_stream:
            return
        yield AssistantMessage(
            content=[TextBlock(text=self.result_text)],
            model="claude",
        )
        yield ResultMessage(
            subtype=self.subtype,
            duration_ms=0,
            duration_api_ms=0,
            is_error=self.is_error,
            num_turns=1,
            session_id=self.session_id,
            result=self.result_text,
        )


@pytest.fixture
def stub_instructions(monkeypatch):
    monkeypatch.setattr("agent_core.anthropic_provider._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv("SESSION_DB_PATH", raising=False)


@pytest.fixture
def fake_query(monkeypatch):
    fq = FakeQuery()
    monkeypatch.setattr("agent_core.anthropic_provider.query", fq)
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


class TestErrorHandling:
    @pytest.mark.anyio
    async def test_is_error_raises_agent_error(self, fake_query):
        fake_query.is_error = True
        fake_query.result_text = "Credit balance is too low"
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            with pytest.raises(AgentError) as exc_info:
                await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert "Credit balance is too low" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_error_exposes_subtype_and_session_id(self, fake_query):
        fake_query.is_error = True
        fake_query.subtype = "error_max_turns"
        fake_query.session_id = "sess-err"
        fake_query.result_text = "max turns hit"
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            with pytest.raises(AgentError) as exc_info:
                await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert exc_info.value.subtype == "error_max_turns"
        assert exc_info.value.session_id == "sess-err"
        assert exc_info.value.provider == "anthropic"

    @pytest.mark.anyio
    async def test_errored_session_id_is_not_saved_to_mapping(self, fake_query):
        """A failed run must not overwrite the prior good session id,
        so that retrying resumes the last successful conversation."""
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            # First call succeeds and stores "good-session".
            fake_query.session_id = "good-session"
            await agent.run("chat-1", "hi")
            assert agent._session_map.get("chat-1") == "good-session"

            # Second call errors with a different id.
            fake_query.is_error = True
            fake_query.session_id = "bad-session"
            with pytest.raises(AgentError):
                await agent.run("chat-1", "again")
            assert agent._session_map.get("chat-1") == "good-session"
        finally:
            await agent.cleanup()

    @pytest.mark.anyio
    async def test_first_call_error_leaves_mapping_empty(self, fake_query):
        fake_query.is_error = True
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            with pytest.raises(AgentError):
                await agent.run("chat-1", "hi")
            assert agent._session_map.get("chat-1") is None
        finally:
            await agent.cleanup()


class TestFromDict:
    def test_instructions_loaded_from_file(self, stub_instructions, fake_query, monkeypatch):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._instructions == "stub instructions"

    def test_model_default_is_none(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._model_name is None

    def test_model_from_config(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict(
            "t",
            {"provider": {"type": "anthropic", "model": "claude-sonnet-4-6"}},
        )
        assert agent._model_name == "claude-sonnet-4-6"

    def test_max_turns_default_is_none(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._max_turns is None

    def test_no_allowed_tools_by_default(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._allowed_tools == []

    def test_setting_sources_always_scoped_to_project(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict("t", {})
        assert agent._setting_sources == ["project"]

    def test_constructor_default_setting_sources_is_project(self, fake_query):  # noqa: ARG002
        agent = ClaudeAgent(name="t", instructions="sys")
        assert agent._setting_sources == ["project"]

    def test_config_allowed_tools_applied(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict(
            "t",
            {"provider": {"type": "anthropic", "allowedTools": ["Bash", "Read", "WebFetch"]}},
        )
        assert agent._allowed_tools == ["Bash", "Read", "WebFetch"]
        assert agent._setting_sources == ["project"]

    def test_config_allowed_tools_deduplicated_preserving_order(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict(
            "t",
            {"provider": {"type": "anthropic", "allowedTools": ["Bash", "Read", "Bash", "WebFetch", "Read"]}},
        )
        assert agent._allowed_tools == ["Bash", "Read", "WebFetch"]

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
    async def test_mcp_and_tools_flow_into_options(self, stub_instructions, fake_query):  # noqa: ARG002
        agent = ClaudeAgent.from_dict(
            "t",
            {
                "mcpServers": {"srv": {"command": "x"}},
                "provider": {"type": "anthropic", "allowedTools": ["Bash", "WebFetch"]},
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


class TestMcpTransform:
    def test_malformed_server_raises_value_error(self, fake_query):  # noqa: ARG002
        """A config entry with neither 'url' nor 'command' must raise a clear error."""
        from agent_core.anthropic_provider import _transform_mcp_servers

        with pytest.raises(ValueError, match="must have either"):
            _transform_mcp_servers({"bad": {"enabled": True}})


class TestEmptyStream:
    @pytest.mark.anyio
    async def test_empty_stream_returns_empty_string(self, fake_query, caplog):
        """When SDK yields no ResultMessage, run() returns '' and logs a warning."""
        import logging

        fake_query.empty_stream = True
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            with caplog.at_level(logging.WARNING, logger="agent_core.anthropic_provider"):
                result = await agent.run("chat-1", "hi")
        finally:
            await agent.cleanup()
        assert result == ""
        assert any("no ResultMessage" in r.message for r in caplog.records)

    @pytest.mark.anyio
    async def test_empty_stream_does_not_update_mapping(self, fake_query):
        fake_query.empty_stream = True
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            await agent.run("chat-1", "hi")
            assert agent._session_map.get("chat-1") is None
        finally:
            await agent.cleanup()


class TestConcurrency:
    @pytest.mark.anyio
    async def test_concurrent_runs_same_chat_id_are_serialized(self, fake_query):
        """Two concurrent run() calls for the same chat_id must be serialized by
        the per-chat lock, so the second call sees the session_id stored by the first."""
        fake_query.session_id = "sess-shared"
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            results = await asyncio.gather(
                agent.run("chat-1", "msg1"),
                agent.run("chat-1", "msg2"),
            )
        finally:
            await agent.cleanup()
        assert results == ["hello", "hello"]
        # Whichever call ran second must have resumed the session from the first.
        assert fake_query.calls[1].options.resume == "sess-shared"

    @pytest.mark.anyio
    async def test_concurrent_runs_different_chat_ids_run_independently(self, fake_query):
        """Different chat_ids must not block each other."""
        fake_query.session_id = "sess-x"
        agent = ClaudeAgent(name="t", instructions="sys")
        try:
            results = await asyncio.gather(
                agent.run("chat-A", "hi"),
                agent.run("chat-B", "hi"),
            )
        finally:
            await agent.cleanup()
        assert results == ["hello", "hello"]
