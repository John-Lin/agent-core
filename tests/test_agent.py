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
from agents.mcp import MCPServerStdio
from agents.mcp import MCPServerStreamableHttp
from agents.models.interface import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_responses import OpenAIResponsesModel

from agent_core.agent import MAX_TURNS
from agent_core.agent import MCP_SESSION_TIMEOUT_SECONDS
from agent_core.agent import OpenAIAgent
from agent_core.agent import _get_model
from agent_core.agent import _parse_skill_description
from agent_core.agent import _shell_executor
from agent_core.agent import _turn_truncate


@pytest.fixture
def _stub_instructions(monkeypatch):
    """Stub out instructions.md loading for from_dict tests."""
    monkeypatch.setattr("agent_core.agent._load_instructions", lambda: "stub instructions")


@pytest.fixture(autouse=True)
def _mock_model(monkeypatch):
    """Prevent tests from constructing a real OpenAI client and isolate shell env vars."""
    monkeypatch.setattr("agent_core.agent._get_model", lambda model_name, api_type: create_autospec(Model))
    monkeypatch.delenv("SHELL_ENABLED", raising=False)
    monkeypatch.delenv("SHELL_SKILLS_DIR", raising=False)


class TestGetModel:
    def test_returns_responses_model_by_default(self):
        with patch("agent_core.agent.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model("gpt-4o", "responses")
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_responses_model_when_api_type_is_responses(self):
        with patch("agent_core.agent.AsyncOpenAI", return_value=MagicMock()):
            model = _get_model("gpt-4o", "responses")
        assert isinstance(model, OpenAIResponsesModel)

    def test_returns_chat_completions_model_when_api_type_is_chat_completions(self):
        with patch("agent_core.agent.AsyncOpenAI", return_value=MagicMock()):
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

        with patch("agent_core.agent._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict("test", {"mcpServers": {}})

        from agent_core.agent import OPENAI_API_TYPE_DEFAULT
        from agent_core.agent import OPENAI_MODEL_DEFAULT

        assert captured["model_name"] == OPENAI_MODEL_DEFAULT
        assert captured["api_type"] == OPENAI_API_TYPE_DEFAULT

    def test_custom_model_from_config(self):
        captured = {}

        def fake_get_model(model_name, api_type):
            captured["model_name"] = model_name
            captured["api_type"] = api_type
            return create_autospec(Model)

        with patch("agent_core.agent._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict("test", {"mcpServers": {}, "model": "gpt-4o-mini"})

        assert captured["model_name"] == "gpt-4o-mini"

    def test_custom_api_type_from_config(self):
        captured = {}

        def fake_get_model(model_name, api_type):
            captured["model_name"] = model_name
            captured["api_type"] = api_type
            return create_autospec(Model)

        with patch("agent_core.agent._get_model", side_effect=fake_get_model):
            OpenAIAgent.from_dict("test", {"mcpServers": {}, "apiType": "chat_completions"})

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
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        assert agent.agent.instructions == "From file prompt."

    def test_from_dict_fails_fast_when_instructions_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="Instructions file not found"):
            OpenAIAgent.from_dict("test", {"mcpServers": {}})


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictMcpServers:
    def test_url_creates_streamable_http_server(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStreamableHttp)

    def test_url_passes_headers(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                    "headers": {"Authorization": "Bearer token"},
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        server = agent.agent.mcp_servers[0]
        assert isinstance(server, MCPServerStreamableHttp)

    def test_command_creates_stdio_server(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "some-mcp-server"],
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStdio)

    def test_mixed_servers(self):
        config = {
            "mcpServers": {
                "remote": {"url": "http://localhost:8000/mcp"},
                "local": {"command": "npx", "args": ["-y", "server"]},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        types = {type(s) for s in agent.agent.mcp_servers}
        assert types == {MCPServerStreamableHttp, MCPServerStdio}

    def test_disabled_http_server_is_skipped(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                    "enabled": False,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 0

    def test_disabled_stdio_server_is_skipped(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "server"],
                    "enabled": False,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 0

    def test_enabled_true_server_is_included(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                    "enabled": True,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1

    def test_mixed_enabled_disabled_servers(self):
        config = {
            "mcpServers": {
                "active": {"url": "http://localhost:8000/mcp", "enabled": True},
                "inactive": {"command": "npx", "args": [], "enabled": False},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert len(agent.agent.mcp_servers) == 1
        assert isinstance(agent.agent.mcp_servers[0], MCPServerStreamableHttp)

    def test_default_session_timeout_used_when_not_specified(self):
        config = {
            "mcpServers": {
                "my-server": {"url": "http://localhost:8000/mcp"},
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == MCP_SESSION_TIMEOUT_SECONDS

    def test_per_server_timeout_http(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:8000/mcp",
                    "timeout": 60.0,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == 60.0

    def test_per_server_timeout_stdio(self):
        config = {
            "mcpServers": {
                "my-server": {
                    "command": "npx",
                    "args": ["-y", "server"],
                    "timeout": 120.0,
                }
            },
        }
        agent = OpenAIAgent.from_dict("test", config)
        assert cast(Any, agent.agent.mcp_servers[0]).client_session_timeout_seconds == 120.0


class TestMaxTurnsConstant:
    def test_default_max_turns(self):
        assert MAX_TURNS == 10


@pytest.mark.usefixtures("_stub_instructions")
class TestFromDictMaxTurns:
    def test_default_max_turns_when_not_in_config(self):
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        assert agent.max_turns == MAX_TURNS

    def test_custom_max_turns_from_config(self):
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}, "maxTurns": 5})
        assert agent.max_turns == 5

    @pytest.mark.anyio
    async def test_custom_max_turns_applied_during_run(self):
        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}, "maxTurns": 3})
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

        with patch("agent_core.agent.Runner.run", side_effect=fake_run):
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

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
        session = agent._get_session(1)
        assert session.db_path == ":memory:"

    def test_from_dict_reads_session_db_path_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "instructions.md").write_text("x", encoding="utf-8")
        db = str(tmp_path / "conv.db")
        monkeypatch.setenv("SESSION_DB_PATH", db)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})
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

        with patch("agent_core.agent.Runner.run", side_effect=fake_run):
            result = await agent.run(chat_id=1, message="hi")

        assert result == "hello"

    @pytest.mark.anyio
    async def test_run_stores_new_items_in_session(self):
        agent = OpenAIAgent(name="test", instructions="test-prompt")
        input_sent = []

        async def fake_run(ag, input, **kw):
            input_sent.extend(input)
            return self._make_run_result(input, reply="pong")

        with patch("agent_core.agent.Runner.run", side_effect=fake_run):
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

        with patch("agent_core.agent.Runner.run", side_effect=fake_run):
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

        # Pre-fill session with more than MAX_TURNS of history
        session = agent._get_session(1)
        history = []
        for i in range(MAX_TURNS + 3):
            history += [
                {"role": "user", "content": f"u{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        await session.add_items(history)

        with patch("agent_core.agent.Runner.run", side_effect=fake_run):
            await agent.run(chat_id=1, message="new")

        sent = captured_inputs[0]
        user_msgs = [m for m in sent if m.get("role") == "user" and m["content"] != "new"]
        assert len(user_msgs) == MAX_TURNS


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

    def _configure_env(self, monkeypatch, *, enabled: bool, skills_dir: Path | None = None) -> None:
        if enabled:
            monkeypatch.setenv("SHELL_ENABLED", "1")
        else:
            monkeypatch.delenv("SHELL_ENABLED", raising=False)
        if skills_dir is not None:
            monkeypatch.setenv("SHELL_SKILLS_DIR", str(skills_dir))
        else:
            monkeypatch.delenv("SHELL_SKILLS_DIR", raising=False)

    def _make_skill(self, parent: Path, name: str, description: str = "desc") -> Path:
        skill_dir = parent / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n")
        return skill_dir

    def test_disabled_by_default(self, tmp_path, monkeypatch):
        """SHELL_ENABLED unset means no ShellTool, even if SHELL_SKILLS_DIR is set."""
        self._make_skill(tmp_path, "my-skill")
        self._configure_env(monkeypatch, enabled=False, skills_dir=tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        assert self._get_shell_tools(agent) == []

    def test_orphaned_skills_dir_without_shell_enabled_warns(self, tmp_path, monkeypatch, caplog):
        self._configure_env(monkeypatch, enabled=False, skills_dir=tmp_path)

        with caplog.at_level("WARNING", logger="root"):
            agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        assert self._get_shell_tools(agent) == []
        assert any(
            "SHELL_SKILLS_DIR" in record.message and "SHELL_ENABLED" in record.message for record in caplog.records
        )

    def test_shell_enabled_without_skills_dir_adds_bare_shell(self, tmp_path, monkeypatch):
        self._configure_env(monkeypatch, enabled=True, skills_dir=None)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        shell_tools = self._get_shell_tools(agent)
        assert len(shell_tools) == 1
        assert shell_tools[0].environment == {"type": "local"}

    def test_shell_enabled_with_skills_dir_mounts_skills(self, tmp_path, monkeypatch):
        skill_dir = self._make_skill(tmp_path, "my-skill", description="A test skill")
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        shell_tool = self._get_shell_tools(agent)[0]
        env = cast(dict[str, Any], shell_tool.environment)
        skill = env["skills"][0]
        assert skill["name"] == "my-skill"
        assert skill["description"] == "A test skill"
        assert skill["path"] == str(skill_dir)

    def test_skills_dir_missing_falls_back_to_bare_shell_with_warning(self, tmp_path, monkeypatch, caplog):
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path / "nonexistent")

        with caplog.at_level("WARNING", logger="root"):
            agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        shell_tools = self._get_shell_tools(agent)
        assert len(shell_tools) == 1
        assert shell_tools[0].environment == {"type": "local"}
        assert any("yielded no skills" in record.message for record in caplog.records)

    def test_multiple_skills_all_mounted(self, tmp_path, monkeypatch):
        self._make_skill(tmp_path, "skill-a")
        self._make_skill(tmp_path, "skill-b")
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        shell_tool = self._get_shell_tools(agent)[0]
        assert len(cast(dict[str, Any], shell_tool.environment)["skills"]) == 2

    def test_directory_without_skill_md_is_skipped(self, tmp_path, monkeypatch):
        (tmp_path / "not-a-skill").mkdir()
        self._make_skill(tmp_path, "real-skill")
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

        shell_tool = self._get_shell_tools(agent)[0]
        skills = cast(dict[str, Any], shell_tool.environment)["skills"]
        assert len(skills) == 1
        assert skills[0]["name"] == "real-skill"

    def test_mcp_servers_and_shell_skills_coexist(self, tmp_path, monkeypatch):
        self._make_skill(tmp_path, "s")
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        config = {"mcpServers": {"my-mcp": {"command": "uvx", "args": ["something"]}}}
        agent = OpenAIAgent.from_dict("test", config)

        assert len(agent.agent.mcp_servers) == 1
        assert len(self._get_shell_tools(agent)) == 1

    def test_unreadable_utf8_skill_file_is_skipped(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad-skill"
        bad.mkdir()
        (bad / "SKILL.md").write_bytes(b"\xff\xfe\x00\x00")
        self._make_skill(tmp_path, "good-skill", description="good")
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

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
        self._configure_env(monkeypatch, enabled=True, skills_dir=tmp_path)

        original_read_text = Path.read_text

        def _read_text(self: Path, *args, **kwargs):
            if self == bad_file:
                raise OSError("permission denied")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", _read_text)

        agent = OpenAIAgent.from_dict("test", {"mcpServers": {}})

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
