from __future__ import annotations

import asyncio
import os
from collections.abc import Hashable
from pathlib import Path
from typing import Any
from typing import cast

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import query

from .env import env_flag
from .session_map import ClaudeSessionMap

INSTRUCTIONS_FILE = Path("instructions.md")

MAX_TURNS = 10
DEFAULT_SHELL_TOOLS = ["Bash", "Read", "Glob", "Grep"]


def _load_instructions() -> str:
    try:
        return INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Instructions file not found: {INSTRUCTIONS_FILE.resolve()}. "
            "Create or mount instructions.md with the agent system prompt."
        ) from e


def _transform_mcp_servers(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert the shared mcpServers config shape to claude-agent-sdk format.

    The shared shape (same one OpenAIAgent.from_dict consumes) uses "command"
    for stdio servers and "url" for HTTP. claude-agent-sdk expects an explicit
    "type" key on each entry.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, srv in raw.items():
        if not srv.get("enabled", True):
            continue
        if "url" in srv:
            entry: dict[str, Any] = {"type": "http", "url": srv["url"]}
            if "headers" in srv:
                entry["headers"] = srv["headers"]
            out[name] = entry
        else:
            entry = {"type": "stdio", "command": srv["command"]}
            if "args" in srv:
                entry["args"] = srv["args"]
            if srv.get("env") is not None:
                entry["env"] = srv["env"]
            out[name] = entry
    return out


class ClaudeAgent:
    """A wrapper for claude-agent-sdk with MCP + project skills support.

    Conversation history is managed by the SDK on disk (one JSONL per
    session, keyed by UUID). We keep a small SQLite mapping from the
    transport's ``chat_id`` to the SDK's session_id so each chat resumes
    its own conversation across process restarts.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
        allowed_tools: list[str] | None = None,
        db_path: str = ":memory:",
        max_turns: int = MAX_TURNS,
        model_name: str | None = None,
        setting_sources: list[str] | None = None,
    ) -> None:
        self.name = name
        self._instructions = instructions
        self._mcp_servers = mcp_servers if mcp_servers is not None else {}
        self._allowed_tools = allowed_tools if allowed_tools is not None else []
        self._model_name = model_name
        self._max_turns = max_turns
        self._setting_sources = setting_sources
        self._session_map = ClaudeSessionMap(db_path)
        self._locks: dict[Hashable, asyncio.Lock] = {}

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> ClaudeAgent:
        mcp_servers = _transform_mcp_servers(config.get("mcpServers", {}))

        shell_enabled = env_flag("SHELL_ENABLED")
        tools: list[str] = list(DEFAULT_SHELL_TOOLS) if shell_enabled else []
        for extra in config.get("allowedTools", []):
            if extra not in tools:
                tools.append(extra)
        # Always scope settings to the project. Leaving this None would make
        # claude-agent-sdk inherit the host user's ~/.claude/ (MCP servers,
        # skills, subagents, slash commands) which is unsafe and
        # non-reproducible for a bot deployment.
        setting_sources = ["project"]

        instructions = _load_instructions()
        db_path = os.getenv("SESSION_DB_PATH", ":memory:")
        max_turns = config.get("maxTurns", MAX_TURNS)
        model_name = config.get("model")

        return cls(
            name,
            instructions=instructions,
            mcp_servers=mcp_servers,
            allowed_tools=tools,
            db_path=db_path,
            max_turns=max_turns,
            model_name=model_name,
            setting_sources=setting_sources,
        )

    async def connect(self) -> None:
        """No-op. claude-agent-sdk opens the CLI subprocess on each query()."""

    async def run(self, chat_id: Hashable, message: str) -> str:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            resume_id = self._session_map.get(chat_id)
            options = ClaudeAgentOptions(
                system_prompt=self._instructions,
                mcp_servers=cast("Any", self._mcp_servers),
                allowed_tools=self._allowed_tools,
                resume=resume_id,
                max_turns=self._max_turns,
                model=self._model_name,
                setting_sources=cast(
                    "list[Any] | None",
                    self._setting_sources,
                ),
            )
            final_text = ""
            captured_session_id: str | None = None
            async for msg in query(prompt=message, options=options):
                if isinstance(msg, ResultMessage):
                    captured_session_id = msg.session_id
                    if msg.result is not None:
                        final_text = msg.result
            if captured_session_id is not None:
                self._session_map.put(chat_id, captured_session_id)
            return final_text

    async def cleanup(self) -> None:
        self._session_map.close()
