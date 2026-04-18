from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Hashable
from typing import Any
from typing import cast

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage
from claude_agent_sdk import query

from .errors import AgentError
from .instructions import _load_instructions
from .session_map import ClaudeSessionMap


def _transform_mcp_servers(raw: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert the shared opencode-style ``mcp`` config to claude-agent-sdk format.

    Input uses ``type: "local" | "remote"`` as the transport discriminator
    (matching opencode's schema). Output uses claude-agent-sdk's
    ``type: "stdio" | "http"`` shape.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, srv in raw.items():
        if not srv.get("enabled", True):
            continue
        mcp_type = srv.get("type")
        if mcp_type == "local":
            command = srv.get("command")
            if not isinstance(command, list) or not command:
                raise ValueError(f"MCP server {name!r}: 'command' must be a non-empty list")
            entry: dict[str, Any] = {"type": "stdio", "command": command[0]}
            if len(command) > 1:
                entry["args"] = command[1:]
            if srv.get("environment") is not None:
                entry["env"] = srv["environment"]
            out[name] = entry
        elif mcp_type == "remote":
            entry = {"type": "http", "url": srv["url"]}
            if "headers" in srv:
                entry["headers"] = srv["headers"]
            out[name] = entry
        else:
            raise ValueError(
                f"MCP server {name!r}: 'type' must be 'local' or 'remote', got {mcp_type!r}"
            )
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
        max_turns: int | None = None,
        model_name: str | None = None,
        setting_sources: list[str] | None = None,
    ) -> None:
        self.name = name
        self._instructions = instructions
        self._mcp_servers = mcp_servers if mcp_servers is not None else {}
        self._allowed_tools = allowed_tools if allowed_tools is not None else []
        self._model_name = model_name
        self._max_turns = max_turns
        self._setting_sources = setting_sources if setting_sources is not None else ["project"]
        self._session_map = ClaudeSessionMap(db_path)
        self._locks: dict[Hashable, asyncio.Lock] = {}

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> ClaudeAgent:
        mcp_servers = _transform_mcp_servers(config.get("mcp", {}))

        provider_cfg = config.get("provider") or {}
        tools: list[str] = list(dict.fromkeys(provider_cfg.get("allowedTools", [])))
        # Always scope settings to the project. Leaving this None would make
        # claude-agent-sdk inherit the host user's ~/.claude/ (MCP servers,
        # skills, subagents, slash commands) which is unsafe and
        # non-reproducible for a bot deployment.
        setting_sources = ["project"]

        instructions = _load_instructions()
        db_path = os.getenv("SESSION_DB_PATH", ":memory:")
        model_name = provider_cfg.get("model")

        return cls(
            name,
            instructions=instructions,
            mcp_servers=mcp_servers,
            allowed_tools=tools,
            db_path=db_path,
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
                # cast: SDK stub types mcp_servers as a union of specific TypedDicts,
                # but the dict-literal form we build is accepted at runtime.
                mcp_servers=cast("Any", self._mcp_servers),
                allowed_tools=self._allowed_tools,
                resume=resume_id,
                max_turns=self._max_turns,
                model=self._model_name,
                # cast: SDK stub expects list[Literal["user","project","local"]]
                # but plain list[str] is identical at runtime.
                setting_sources=cast("list[Any] | None", self._setting_sources),
            )
            final_text = ""
            captured_session_id: str | None = None
            async for msg in query(prompt=message, options=options):
                if isinstance(msg, ResultMessage):
                    captured_session_id = msg.session_id
                    if msg.is_error:
                        raise AgentError(
                            msg.result or "claude-agent-sdk returned is_error=True",
                            subtype=msg.subtype,
                            provider="anthropic",
                            session_id=msg.session_id,
                        )
                    if msg.result is not None:
                        final_text = msg.result
            if captured_session_id is None:
                logging.warning(
                    "claude-agent-sdk query() ended with no ResultMessage for chat_id=%r",
                    chat_id,
                )
            else:
                self._session_map.put(chat_id, captured_session_id)
            return final_text

    async def cleanup(self) -> None:
        self._session_map.close()
