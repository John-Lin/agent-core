from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
from agents import ShellTool
from agents import SQLiteSession
from agents import TResponseInputItem
from agents.exceptions import InputGuardrailTripwireTriggered
from agents.exceptions import MaxTurnsExceeded
from agents.exceptions import MCPToolCancellationError
from agents.exceptions import ModelBehaviorError
from agents.exceptions import OutputGuardrailTripwireTriggered
from agents.exceptions import ToolInputGuardrailTripwireTriggered
from agents.exceptions import ToolOutputGuardrailTripwireTriggered
from agents.exceptions import ToolTimeoutError
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.interface import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from agent_core.openai_provider import HISTORY_TURNS_DEFAULT
from agent_core.openai_provider import MCP_SESSION_TIMEOUT_SECONDS
from agent_core.openai_provider import OpenAIAgent
from agent_core.openai_provider import _get_model
from agent_core.openai_provider import _parse_skill_description
from agent_core.openai_provider import _shell_executor
from agent_core.openai_provider import _turn_truncate


@pytest.fixture
def _stub_instructions(monkeypatch):
    """Stub out instructions.md loading for from_dict tests."""
    monkeypatch.setattr("agent_core.openai_provider._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Prevent tests from constructing a real OpenAI client."""
    monkeypatch.setattr("agent_core.openai_provider._get_model", lambda model_name, api_type: create_autospec(Model))


class TestGetModel:
    def test_returns_responses_model_by_default(self):
        with patch("agent_core.openai_provider.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model("gpt-4o", "responses")
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_responses_model_when_api_type_is_responses(self):
        with patch("agent_core.openai_provider.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model("gpt-4o", "responses")
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_chat_completions_model_when_api_type_is_chat_completions(self):
        with patch("agent_core.openai_provider.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model("gpt-4o", "chat_completions")
        assert isinstance(model, OpenAIChatCompletionsModel)


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictModel:
    def test_default_model_when_not_in_config(self):
        captured = {}

        def fake_get_model(model_name, api_type):
            captured["model_name"] = model_name
            captured["api_type"] = api_type
            return create_autospec(Model)

        with patch("agent_core.openai_provider._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict("test", {"mcp": {}})

        from agent_core.openai_provider import OPENAI_API_TYPE_DEFAULT
        from agent_core.openai_provider import OPENAI_MODEL_DEFAULT

        assert captured["model_name"] == OPENAI_MODEL_DEFAULT
        assert captured["api_type"] == OPENAI_API_TYPE_DEFAULT

    def test_custom_model_from_config(self):
        captured = {}

        def fake_get_model(model_name, api_type):
            captured["model_name"] = model_name
            captured["api_type"] = api_type
            return create_autospec(Model)

        with patch("agent_core.openai_provider._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict(
                "test",
                {"mcp": {}, "provider": {"type": "openai", "model": "gpt-4o-mini"}},
            )

        assert captured["model_name"] == "gpt-4o-mini"

    def test_custom_api_type_from_config(self):
        captured = {}

        def fake_get_model(model_name, api_type):
            captured["model_name"] = model_name
            captured["api_type"] = api_type
            return create_autospec(Model)

        with patch("agent_core.openai_provider._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict(
                "test",
                {"mcp": {}, "provider": {"type": "openai", "apiType": "chat_completions"}},
            )

        assert captured["api_type"] == "chat_completions"


class TestTurnTruncate:
    def test_returns_all_items_when_under_limit(self):
        items: list[TResponseInputItem] = cast(
            list[TResponseInputItem],
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
        )
        assert _turn_truncate(items, max_turns=10) == items

    def test_returns_all_items_when_exactly_at_limit(self):
        items: list[TResponseInputItem] = []
        for i in range(3):
            items += cast(
                list[TResponseInputItem],
                [
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
            )
        assert _turn_truncate(items, max_turns=3) == items

    def test_truncates_oldest_turns(self):
        items: list[TResponseInputItem] = []
        for i in range(5):
            items += cast(
                list[TResponseInputItem],
                [
                    {"role": "user", "content": f"u{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
            )

        result: list[dict[str, Any]] = cast(list[dict[str, Any]], _turn_truncate(items, max_turns=2))

        user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0]["content"] == "u3"
        assert user_msgs[-1]["content"] == "u4"

    def test_preserves_tool_messages_within_turn(self):
        items: list[TResponseInputItem] = cast(
            list[TResponseInputItem],
            [
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "tc1"}]},
                {"role": "tool", "content": "tool-result", "tool_call_id": "tc1"},
                {"role": "assistant", "content": "a1"},
            ],
        )

        result: list[dict[str, Any]] = cast(list[dict[str, Any]], _turn_truncate(items, max_turns=1))

        assert result[0].get("role") == "user"
        assert result[0].get("content") == "u1"
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_empty_list_returns_empty(self):
        assert _turn_truncate([], max_turns=10) == []

    def test_does_not_mutate_input(self):
        items: list[TResponseInputItem] = cast(
            list[TResponseInputItem],
            [
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
        )
        original = list(items)
        _turn_truncate(items, max_turns=0)
        assert items == original


class TestInstructions:
    def test_custom_instructions(self):
        agent = OpenAIAgent(name="test", instructions="Be a HN bot.")
        assert agent.agent.instructions == "Be a HN bot."

    def test_from_dict_loads_instructions_from_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "instructions.md").write_text("From file prompt.", encoding="utf-8")
        agent = OpenAIAgent.from_dict("test", {"mcp": {}})
        assert agent.agent.instructions == "From file prompt."

    def test_from_dict_fails_fast_when_instructions_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Instructions file not found"):
            OpenAIAgent.from_dict("test", {"mcp": {}})


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictMcpServers:
    def test_remote_creates_streamable_http_server(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "remote",
                    "url": "http://localhost:8000/mcp",
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStreamableHttp)

    def test_remote_passes_headers(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "remote",
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer token"},
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        server = agent.agent.mcp_servers[0]
        assert isinstance(server, MCPServerStreamableHttp)

    def test_local_creates_stdio_server(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "local",
                    "command": ["npx", "-y", "some-mcp-server"],
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStdio)

    def test_mixed_servers(self):
        config = {
            "mcp": {
                "remote": {"type": "remote", "url": "http://localhost:8000/mcp"},
                "local": {"type": "local", "command": ["npx", "-y", "server"]},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        types = {type(s) for s in agent.agent.mcp_servers}
        assert types == {MCPServerStreamableHttp, MCPServerStdio}

    def test_disabled_remote_server_is_skipped(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "remote",
                    "url": "http://localhost:8000/mcp",
                    "enabled": False,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 0

    def test_disabled_local_server_is_skipped(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "local",
                    "command": ["npx", "-y", "server"],
                    "enabled": False,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 0

    def test_enabled_true_server_is_included(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "remote",
                    "url": "http://localhost:8000/mcp",
                    "enabled": True,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1

    def test_mixed_enabled_disabled_servers(self):
        config = {
            "mcp": {
                "active": {"type": "remote", "url": "http://localhost:8000/mcp", "enabled": True},
                "inactive": {"type": "local", "command": ["npx"], "enabled": False},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStreamableHttp)

    def test_default_session_timeout_used_when_not_specified(self):
        config = {
            "mcp": {
                "my-server": {"type": "remote", "url": "http://localhost:8000/mcp"},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == MCP_SESSION_TIMEOUT_SECONDS

    def test_per_server_timeout_remote(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "remote",
                    "url": "http://localhost:8000/mcp",
                    "timeout": 60.0,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == 60.0

    def test_per_server_timeout_local(self):
        config = {
            "mcp": {
                "my-server": {
                    "type": "local",
                    "command": ["npx", "-y", "server"],
                    "timeout": 120.0,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == 120.0

    def test_invalid_type_raises(self):
        config = {"mcp": {"bad": {"type": "unknown"}}}
        with pytest.raises(ValueError, match="'type' must be 'local' or 'remote'"):
            OpenAIAgent.from_dict("test", config)

    def test_local_without_command_list_raises(self):
        config = {"mcp": {"bad": {"type": "local", "command": "npx"}}}
        with pytest.raises(ValueError, match="'command' must be a non-empty list"):
            OpenAIAgent.from_dict("test", config)


class TestHistoryTurnsConstant:
    def test_default_history_turns(self):
        assert HISTORY_TURNS_DEFAULT == 10


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictHistoryTurns:
    def test_default_history_turns_when_not_in_config(self):
        agent = OpenAIAgent.from_dict("test", {"mcp": {}})
        assert agent.history_turns == HISTORY_TURNS_DEFAULT

    def test_custom_history_turns_from_config(self):
        agent = OpenAIAgent.from_dict(
            "test",
            {"mcp": {}, "provider": {"type": "openai", "historyTurns": 5}},
        )
        assert agent.history_turns == 5

    @pytest.mark.anyio
    async def test_custom_history_turns_applied_during_run(self):
        agent = OpenAIAgent.from_dict(
            "test",
            {"mcp": {}, "provider": {"type": "openai", "historyTurns": 3}},
        )
        captured_inputs = []

        async def fake_run(ag, input, **kw):
            captured_inputs.append(list(input))
            mock_result = MagicMock()
            mock_result.final_output = "ok"
            mock_result.to_input_list.return_value = list(input) + [{"role": "assistant", "content": "ok"}]
            return mock_result

        session = agent._get_session(1)
        history = []
        for i in range(6):
            history += [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        await session.add_items(history)

        with patch("agent_core.openai_provider.Runner.run", side_effect=fake_run):
            await agent.run(chat_id=1, message="new")

        sent = captured_inputs[0]
        user_msgs = [m for m in sent if m.get("role") == "user" and m["content"] != "new"]
        assert len(user_msgs) == 3


class TestSessionDbPath:
    def test_default_db_path_is_memory(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        session = agent._get_session(1)
        assert session.db_path == ":memory:"

    def test_custom_db_path_used_for_sessions(self, tmp_path):
        db = str(tmp_path / "test.db")
        agent = OpenAIAgent(name="test", instructions="test-prompt", db_path=db)
        session = agent._get_session(1)
        assert str(session.db_path) == db

    def test_from_dict_defaults_to_memory_when_env_not_set(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "instructions.md").write_text("x", encoding="utf-8")
        monkeypatch.delenv("SESSION_DB_PATH", raising=False)

        agent = OpenAIAgent.from_dict("test", {"mcp": {}})
        session = agent._get_session(1)
        assert session.db_path == ":memory:"

    def test_from_dict_reads_session_db_path_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "instructions.md").write_text("x", encoding="utf-8")
        db = str(tmp_path / "conv.db")
        monkeypatch.setenv("SESSION_DB_PATH", db)

        agent = OpenAIAgent.from_dict("test", {"mcp": {}})
        session = agent._get_session(1)
        assert str(session.db_path) == db


class TestRunWithSession:
    def _make_run_result(self, input_items, reply="ok"):
        """Return a mock Runner.run result whose to_input_list() = input_items + [assistant reply]."""
        output_items = list(input_items) + [{"role": "assistant", "content": reply}]
        mock_result = MagicMock()
        mock_result.final_output = reply
        mock_result.to_input_list.return_value = output_items
        return mock_result

    @pytest.mark.anyio
    async def test_run_returns_agent_reply(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")

        async def fake_run(ag, input, **kw):
            return self._make_run_result(input, reply="hello")

        with patch("agent_core.openai_provider.Runner.run", side_effect=fake_run):
            result = await agent.run(chat_id=1, message="hi")

        assert result == "hello"

    @pytest.mark.anyio
    async def test_run_stores_new_items_in_session(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        input_sent = []

        async def fake_run(ag, input, **kw):
            input_sent.extend(input)
            return self._make_run_result(input, reply="pong")

        with patch("agent_core.openai_provider.Runner.run", side_effect=fake_run):
            await agent.run(chat_id=1, message="ping")

        session = agent._get_session(1)
        saved: list[dict[str, Any]] = cast(list[dict[str, Any]], await session.get_items())
        assert any(m["content"] == "ping" for m in saved if m.get("role") == "user")
        assert any(m["content"] == "pong" for m in saved if m.get("role") == "assistant")

    @pytest.mark.anyio
    async def test_run_separate_chat_ids_have_independent_sessions(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")

        async def fake_run(ag, input, **kw):
            return self._make_run_result(input, reply="reply")

        with patch("agent_core.openai_provider.Runner.run", side_effect=fake_run):
            await agent.run(chat_id=100, message="from 100")
            await agent.run(chat_id=200, message="from 200")

        items_100 = await agent._get_session(100).get_items()
        items_200 = await agent._get_session(200).get_items()
        assert all("from 200" not in str(m) for m in items_100)
        assert all("from 100" not in str(m) for m in items_200)

    @pytest.mark.anyio
    async def test_run_sends_truncated_history_to_runner(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        captured_inputs = []

        async def fake_run(ag, input, **kw):
            captured_inputs.append(list(input))
            return self._make_run_result(input, reply="ok")

        # Pre-fill session with more than HISTORY_TURNS_DEFAULT of history
        session = agent._get_session(1)
        history = []
        for i in range(HISTORY_TURNS_DEFAULT + 3):
            history += [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        await session.add_items(history)

        with patch("agent_core.openai_provider.Runner.run", side_effect=fake_run):
            await agent.run(chat_id=1, message="new")

        sent = captured_inputs[0]
        user_msgs = [m for m in sent if m.get("role") == "user" and m["content"] != "new"]
        assert len(user_msgs) == HISTORY_TURNS_DEFAULT


class TestRunErrorMapping:
    """Runner.run failures are mapped to AgentError(subtype=..., provider="openai")."""

    @staticmethod
    def _api_status_exc(cls: type, message: str = "err"):
        """Build an openai APIStatusError subclass instance with the required kwargs."""
        response = MagicMock()
        response.request = MagicMock()
        return cls(message, response=response, body=None)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("exc", "expected_subtype"),
        [
            (MaxTurnsExceeded("turns"), "error_max_turns"),
            (ModelBehaviorError("bad"), "model_behavior"),
            (InputGuardrailTripwireTriggered(MagicMock()), "guardrail"),
            (OutputGuardrailTripwireTriggered(MagicMock()), "guardrail"),
            (ToolInputGuardrailTripwireTriggered(MagicMock(), MagicMock()), "tool_guardrail"),
            (ToolOutputGuardrailTripwireTriggered(MagicMock(), MagicMock()), "tool_guardrail"),
            (ToolTimeoutError("my_tool", 30.0), "tool_timeout"),
            (MCPToolCancellationError("cancelled"), "mcp_cancelled"),
        ],
    )
    async def test_agents_exceptions_mapped(self, exc, expected_subtype):
        from agent_core import AgentError

        agent = OpenAIAgent(name="t", instructions="sys")

        async def boom(ag, input, **kw):
            raise exc

        with patch("agent_core.openai_provider.Runner.run", side_effect=boom), pytest.raises(AgentError) as exc_info:
            await agent.run(chat_id=1, message="hi")
        assert exc_info.value.subtype == expected_subtype
        assert exc_info.value.provider == "openai"
        assert exc_info.value.session_id is None

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("exc_name", "expected_subtype"),
        [
            ("RateLimitError", "rate_limit"),
            ("AuthenticationError", "auth"),
            ("BadRequestError", "bad_request"),
            # Fallback: any other APIStatusError subclass collapses to api_status.
            ("InternalServerError", "api_status"),
        ],
    )
    async def test_openai_status_errors_mapped(self, exc_name, expected_subtype):
        import openai

        from agent_core import AgentError

        exc_cls = getattr(openai, exc_name)
        agent = OpenAIAgent(name="t", instructions="sys")

        async def boom(ag, input, **kw):
            raise self._api_status_exc(exc_cls)

        with patch("agent_core.openai_provider.Runner.run", side_effect=boom), pytest.raises(AgentError) as exc_info:
            await agent.run(chat_id=1, message="hi")
        assert exc_info.value.subtype == expected_subtype
        assert exc_info.value.provider == "openai"

    @pytest.mark.anyio
    async def test_api_timeout_mapped_to_timeout(self):
        import openai

        from agent_core import AgentError

        agent = OpenAIAgent(name="t", instructions="sys")

        async def boom(ag, input, **kw):
            raise openai.APITimeoutError(request=MagicMock())

        with patch("agent_core.openai_provider.Runner.run", side_effect=boom), pytest.raises(AgentError) as exc_info:
            await agent.run(chat_id=1, message="hi")
        assert exc_info.value.subtype == "timeout"
        assert exc_info.value.provider == "openai"

    @pytest.mark.anyio
    async def test_api_connection_mapped_to_connection(self):
        import openai

        from agent_core import AgentError

        agent = OpenAIAgent(name="t", instructions="sys")

        async def boom(ag, input, **kw):
            raise openai.APIConnectionError(request=MagicMock())

        with patch("agent_core.openai_provider.Runner.run", side_effect=boom), pytest.raises(AgentError) as exc_info:
            await agent.run(chat_id=1, message="hi")
        assert exc_info.value.subtype == "connection"
        assert exc_info.value.provider == "openai"

    @pytest.mark.anyio
    async def test_unknown_exception_propagates_unwrapped(self):
        agent = OpenAIAgent(name="t", instructions="sys")

        async def boom(ag, input, **kw):
            raise RuntimeError("???")

        with patch("agent_core.openai_provider.Runner.run", side_effect=boom), pytest.raises(RuntimeError):
            await agent.run(chat_id=1, message="hi")


class TestCleanupClosesSessions:
    @pytest.mark.anyio
    async def test_cleanup_calls_close_on_all_sessions(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        # Create two sessions
        s1 = agent._get_session(1)
        s2 = agent._get_session(2)

        closed = []
        original_close = SQLiteSession.close

        def tracking_close(self):
            closed.append(self.session_id)
            original_close(self)

        with patch.object(SQLiteSession, "close", tracking_close):
            await agent.cleanup()

        assert s1.session_id in closed
        assert s2.session_id in closed


@pytest.mark.usefixtures("_stub_instructions")
class TestShellToolConfiguration:
    def _get_shell_tools(self, agent: OpenAIAgent) -> list[ShellTool]:
        return [t for t in agent.agent.tools if isinstance(t, ShellTool)]

    def _config(self, *, enabled: bool, skills_dir: Path | None = None) -> dict:
        shell: dict = {"enabled": enabled}
        if skills_dir is not None:
            shell["skillsDir"] = str(skills_dir)
        return {"mcp": {}, "provider": {"type": "openai", "shell": shell}}

    def _make_skill(self, parent: Path, name: str, description: str = "desc") -> Path:
        skill_dir = parent / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n")
        return skill_dir

    def test_disabled_by_default(self, tmp_path):
        """enabled=false means no ShellTool, even if skillsDir is set."""
        self._make_skill(tmp_path, "my-skill")

        agent = OpenAIAgent.from_dict("test", self._config(enabled=False, skills_dir=tmp_path))

        assert self._get_shell_tools(agent) == []

    def test_missing_shell_config_means_disabled(self):
        agent = OpenAIAgent.from_dict("test", {"mcp": {}})

        assert self._get_shell_tools(agent) == []

    @pytest.mark.parametrize("bad_value", ["true", "false", 1, 0, None])
    def test_non_bool_enabled_rejected(self, bad_value):
        config = {"mcp": {}, "provider": {"type": "openai", "shell": {"enabled": bad_value}}}
        with pytest.raises(ValueError, match="provider.shell.enabled must be a bool"):
            OpenAIAgent.from_dict("test", config)

    def test_orphaned_skills_dir_without_shell_enabled_warns(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="root"):
            agent = OpenAIAgent.from_dict("test", self._config(enabled=False, skills_dir=tmp_path))

        assert self._get_shell_tools(agent) == []
        assert any(
            "provider.shell.skillsDir" in record.message and "provider.shell.enabled" in record.message
            for record in caplog.records
        )

    def test_shell_enabled_without_skills_dir_adds_bare_shell(self):
        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=None))

        shell_tools = self._get_shell_tools(agent)
        assert len(shell_tools) == 1
        assert shell_tools[0].environment == {"type": "local"}

    def test_shell_enabled_with_skills_dir_mounts_skills(self, tmp_path):
        skill_dir = self._make_skill(tmp_path, "my-skill", description="A test skill")

        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=tmp_path))

        shell_tool = self._get_shell_tools(agent)[0]
        env = cast(dict[str, Any], shell_tool.environment)
        skill = env["skills"][0]
        assert skill["name"] == "my-skill"
        assert skill["description"] == "A test skill"
        assert skill["path"] == str(skill_dir)

    def test_skills_dir_missing_falls_back_to_bare_shell_with_warning(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger="root"):
            agent = OpenAIAgent.from_dict(
                "test", self._config(enabled=True, skills_dir=tmp_path / "nonexistent")
            )

        shell_tools = self._get_shell_tools(agent)
        assert len(shell_tools) == 1
        assert shell_tools[0].environment == {"type": "local"}
        assert any("yielded no skills" in record.message for record in caplog.records)

    def test_multiple_skills_all_mounted(self, tmp_path):
        self._make_skill(tmp_path, "skill-a")
        self._make_skill(tmp_path, "skill-b")

        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=tmp_path))

        shell_tool = self._get_shell_tools(agent)[0]
        assert len(cast(dict[str, Any], shell_tool.environment)["skills"]) == 2

    def test_directory_without_skill_md_is_skipped(self, tmp_path):
        (tmp_path / "not-a-skill").mkdir()
        self._make_skill(tmp_path, "real-skill")

        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=tmp_path))

        shell_tool = self._get_shell_tools(agent)[0]
        skills = cast(dict[str, Any], shell_tool.environment)["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "real-skill"

    def test_mcp_servers_and_shell_skills_coexist(self, tmp_path):
        self._make_skill(tmp_path, "s")

        config = self._config(enabled=True, skills_dir=tmp_path)
        config["mcp"] = {"my-mcp": {"type": "local", "command": ["uvx", "something"]}}
        agent = OpenAIAgent.from_dict("test", config)

        assert len(agent.agent.mcp_servers) == 1
        assert len(self._get_shell_tools(agent)) == 1

    def test_unreadable_utf8_skill_file_is_skipped(self, tmp_path):
        bad = tmp_path / "bad-skill"
        bad.mkdir()
        (bad / "SKILL.md").write_bytes(b"\xff\xfe\x00\x00")
        self._make_skill(tmp_path, "good-skill", description="good")

        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=tmp_path))

        shell_tool = self._get_shell_tools(agent)[0]
        skills = cast(dict[str, Any], shell_tool.environment)["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "good-skill"

    def test_oserror_reading_skill_file_is_skipped(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad-skill"
        bad.mkdir()
        bad_file = bad / "SKILL.md"
        bad_file.write_text("---\nname: bad\ndescription: bad\n---\n")
        self._make_skill(tmp_path, "good-skill", description="good")

        original_read_text = Path.read_text

        def _read_text(self: Path, *args, **kwargs):
            if self == bad_file:
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _read_text)

        agent = OpenAIAgent.from_dict("test", self._config(enabled=True, skills_dir=tmp_path))

        shell_tool = self._get_shell_tools(agent)[0]
        skills = cast(dict[str, Any], shell_tool.environment)["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "good-skill"


class TestParseSkillDescription:
    def test_unquoted_description(self):
        content = "---\nname: my-skill\ndescription: A test skill\n---\nBody text."
        assert _parse_skill_description(content) == "A test skill"

    def test_double_quoted_description(self):
        content = '---\nname: my-skill\ndescription: "A quoted skill"\n---\n'
        assert _parse_skill_description(content) == "A quoted skill"

    def test_single_quoted_description(self):
        content = "---\nname: my-skill\ndescription: 'Single quoted'\n---\n"
        assert _parse_skill_description(content) == "Single quoted"

    def test_no_frontmatter(self):
        content = "Just a plain markdown file."
        assert _parse_skill_description(content) == ""

    def test_no_description_field(self):
        content = "---\nname: my-skill\n---\nBody."
        assert _parse_skill_description(content) == ""

    def test_unclosed_frontmatter(self):
        content = "---\nname: my-skill\ndescription: never closed"
        assert _parse_skill_description(content) == ""

    def test_empty_description(self):
        content = "---\nname: my-skill\ndescription:\n---\n"
        assert _parse_skill_description(content) == ""

    def test_description_with_colon_in_value(self):
        content = "---\ndescription: Run this: do stuff\n---\n"
        assert _parse_skill_description(content) == "Run this: do stuff"


def _make_shell_request(commands: list[str], timeout_ms: int | None = None):
    """Build a minimal object matching the ShellCommandRequest interface."""
    from types import SimpleNamespace

    action = SimpleNamespace(commands=commands, timeout_ms=timeout_ms)
    data = SimpleNamespace(action=action)
    return SimpleNamespace(data=data)


class TestShellExecutor:
    @pytest.mark.anyio
    async def test_single_command_returns_stdout(self):
        request = _make_shell_request(["echo hello"])
        result = await _shell_executor(request)
        assert result.strip() == "hello"

    @pytest.mark.anyio
    async def test_multiple_commands_combined(self):
        request = _make_shell_request(["echo first", "echo second"])
        result = await _shell_executor(request)
        assert "first" in result
        assert "second" in result
        assert result.index("first") < result.index("second")

    @pytest.mark.anyio
    async def test_stderr_merged_into_stdout(self):
        request = _make_shell_request(["echo err >&2"])
        result = await _shell_executor(request)
        assert result.strip() == "err"

    @pytest.mark.anyio
    async def test_command_timeout_kills_process(self):
        request = _make_shell_request(["sleep 30"], timeout_ms=100)
        result = await _shell_executor(request)
        assert "timed out" in result.lower()

    @pytest.mark.anyio
    async def test_nonzero_exit_code_appends_exit_code(self):
        request = _make_shell_request(["echo failing && exit 1"])
        result = await _shell_executor(request)
        assert "failing" in result
        assert "[exit code: 1]" in result

    @pytest.mark.anyio
    async def test_zero_exit_code_no_suffix(self):
        request = _make_shell_request(["echo ok"])
        result = await _shell_executor(request)
        assert "exit code" not in result

    @pytest.mark.anyio
    async def test_timeout_ms_none_uses_default(self):
        """When timeout_ms is None, SHELL_TIMEOUT is used (command completes fine)."""
        request = _make_shell_request(["echo ok"], timeout_ms=None)
        result = await _shell_executor(request)
        assert result.strip() == "ok"

    @pytest.mark.anyio
    async def test_timeout_stops_remaining_commands(self):
        """After a timeout, subsequent commands are not executed."""
        request = _make_shell_request(["sleep 30", "echo should-not-run"], timeout_ms=100)
        result = await _shell_executor(request)
        assert "timed out" in result.lower()
        assert "should-not-run" not in result

    @pytest.mark.anyio
    async def test_subprocess_oserror_returns_error_message(self, monkeypatch):
        """When create_subprocess_shell raises OSError, return error text instead of crashing."""
        import asyncio as _asyncio

        async def _failing_shell(*args, **kwargs):
            raise OSError("fork failed")

        monkeypatch.setattr(_asyncio, "create_subprocess_shell", _failing_shell)

        request = _make_shell_request(["echo hello"])
        result = await _shell_executor(request)
        assert "fork failed" in result
