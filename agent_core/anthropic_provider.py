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

# Read-only built-ins that are always exposed to the agent. They map to
# Anthropic's "No permission required" tools and pose no write or exec risk,
# so they form a safe baseline regardless of provider.allowedTools. Any tool
# that can mutate files or run commands (Write, Edit, Bash, …) must be
# listed explicitly by the caller.
DEFAULT_ALLOWED_TOOLS = ["Read", "Glob", "Grep"]

# Upper bound on a single query's wall-clock time. claude-agent-sdk exposes
# no per-query timeout, so without this the CLI subprocess can hang and
# permanently hold the per-chat asyncio.Lock — every subsequent message
# for that chat piles up invisibly. 5 minutes comfortably covers legitimate
# tool runs while still failing within one oncall window on a real hang.
DEFAULT_QUERY_TIMEOUT_S = 300.0

# Every Claude Code built-in tool name known to us at the time of writing.
# Kept in sync with https://code.claude.com/docs/en/tools-reference — revisit
# when bumping claude-agent-sdk in case new tools have been added.
#
# claude-agent-sdk's `allowed_tools` is an auto-approval list, not a visibility
# filter: unlisted built-ins still appear in the model's toolset and the model
# will happily talk about them. To actually hide a tool, it must be named in
# `disallowed_tools`. We compute that set as KNOWN_BUILTIN_TOOLS minus whatever
# the caller opted into (DEFAULT_ALLOWED_TOOLS + provider.allowedTools).
KNOWN_BUILTIN_TOOLS = frozenset(
    {
        "Agent",
        "AskUserQuestion",
        "Bash",
        "CronCreate",
        "CronDelete",
        "CronList",
        "Edit",
        "EnterPlanMode",
        "EnterWorktree",
        "ExitPlanMode",
        "ExitWorktree",
        "Glob",
        "Grep",
        "ListMcpResourcesTool",
        "LSP",
        "Monitor",
        "NotebookEdit",
        "PowerShell",
        "Read",
        "ReadMcpResourceTool",
        "SendMessage",
        "Skill",
        "TaskCreate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "TaskUpdate",
        "TeamCreate",
        "TeamDelete",
        "TodoWrite",
        "ToolSearch",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)


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
            raise ValueError(f"MCP server {name!r}: 'type' must be 'local' or 'remote', got {mcp_type!r}")
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
        claude_home: str,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
        allowed_tools: list[str] | None = None,
        disallowed_tools: list[str] | None = None,
        db_path: str = ":memory:",
        max_turns: int | None = None,
        model_name: str | None = None,
        setting_sources: list[str] | None = None,
        query_timeout_s: float | None = DEFAULT_QUERY_TIMEOUT_S,
    ) -> None:
        if not claude_home:
            # Required unconditionally — without it the CLI subprocess falls
            # back to the parent's $HOME and reads the host user's personal
            # ~/.claude.json (OAuth session, MCP config, project history).
            # There is no safe implicit default; callers must name a writable
            # directory the agent owns.
            raise ValueError(
                "claude_home is required — point it at a dedicated directory "
                "so the CLI subprocess does not read the host user's ~/.claude.json."
            )
        self.name = name
        self._instructions = instructions
        self._mcp_servers = mcp_servers if mcp_servers is not None else {}
        # DEFAULT_ALLOWED_TOOLS is the read-only baseline every ClaudeAgent must
        # expose (Read/Glob/Grep). Merging here, not in from_dict, means direct
        # constructor callers also get the invariant the README promises.
        caller_tools = allowed_tools if allowed_tools is not None else []
        self._allowed_tools = list(dict.fromkeys(DEFAULT_ALLOWED_TOOLS + list(caller_tools)))
        if disallowed_tools is None:
            disallowed_tools = sorted(KNOWN_BUILTIN_TOOLS - set(self._allowed_tools))
        self._disallowed_tools = disallowed_tools
        self._claude_home = claude_home
        self._model_name = model_name
        self._max_turns = max_turns
        self._setting_sources = setting_sources if setting_sources is not None else ["project"]
        self._query_timeout_s = query_timeout_s
        self._session_map = ClaudeSessionMap(db_path)
        self._locks: dict[Hashable, asyncio.Lock] = {}

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> ClaudeAgent:
        mcp_servers = _transform_mcp_servers(config.get("mcp", {}))

        provider_cfg = config.get("provider") or {}
        tools: list[str] = list(provider_cfg.get("allowedTools", []))
        claude_home = provider_cfg.get("claudeHome")
        if not claude_home:
            raise ValueError(
                "provider.claudeHome is required for the Anthropic provider — "
                "point it at a dedicated directory so the CLI subprocess does "
                "not read the host user's ~/.claude.json."
            )
        # Always scope settings to the project. Leaving this None would make
        # claude-agent-sdk inherit the host user's ~/.claude/ (MCP servers,
        # skills, subagents, slash commands) which is unsafe and
        # non-reproducible for a bot deployment.
        setting_sources = ["project"]

        instructions = _load_instructions()
        db_path = os.getenv("SESSION_DB_PATH", ":memory:")
        model_name = provider_cfg.get("model")
        query_timeout_s = provider_cfg.get("queryTimeoutSeconds", DEFAULT_QUERY_TIMEOUT_S)

        return cls(
            name,
            instructions=instructions,
            mcp_servers=mcp_servers,
            allowed_tools=tools,
            db_path=db_path,
            model_name=model_name,
            setting_sources=setting_sources,
            claude_home=claude_home,
            query_timeout_s=query_timeout_s,
        )

    async def connect(self) -> None:
        """No-op. claude-agent-sdk opens the CLI subprocess on each query()."""

    async def run(self, chat_id: Hashable, message: str) -> str:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        async with lock:
            resume_id = self._session_map.get(chat_id)
            # Overriding HOME redirects the CLI subprocess from the host
            # user's ~/.claude.json (OAuth, projects, personal MCP) to an
            # isolated directory owned by this agent.
            env = {"HOME": self._claude_home}
            options = ClaudeAgentOptions(
                system_prompt=self._instructions,
                # cast: SDK stub types mcp_servers as a union of specific TypedDicts,
                # but the dict-literal form we build is accepted at runtime.
                mcp_servers=cast("Any", self._mcp_servers),
                allowed_tools=self._allowed_tools,
                disallowed_tools=self._disallowed_tools,
                resume=resume_id,
                max_turns=self._max_turns,
                model=self._model_name,
                # cast: SDK stub expects list[Literal["user","project","local"]]
                # but plain list[str] is identical at runtime.
                setting_sources=cast("list[Any] | None", self._setting_sources),
                env=env,
            )
            final_text = ""
            captured_session_id: str | None = None
            try:
                async with asyncio.timeout(self._query_timeout_s):
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
            except TimeoutError as e:
                # No ResultMessage arrived, so we have no session_id to persist.
                # The next message for this chat will start a fresh SDK session
                # rather than resume a partially-completed one.
                raise AgentError(
                    f"claude-agent-sdk query timed out after {self._query_timeout_s}s",
                    subtype="timeout",
                    provider="anthropic",
                ) from e
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
