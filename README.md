# agent-core

Shared agent wrapper extracted from `agentic-telegram-bot`,
`agentic-slackbot`, and `agentic-discord-bot`.

Supports two interchangeable providers behind a common interface:

- **`openai`** (default) — wraps [`openai-agents`](https://pypi.org/project/openai-agents/)
- **`anthropic`** — wraps [`claude-agent-sdk`](https://pypi.org/project/claude-agent-sdk/)

Both providers offer:

- Per-conversation history (any `Hashable` key)
- Per-conversation async lock to serialise concurrent messages
- MCP server construction from a JSON config dict
- Optional local shell / built-in tools gated by env vars
- `instructions.md` in the working directory as the system prompt

## Picking a provider

```python
from agent_core import build_agent

agent = build_agent("MyBot", config)   # dispatches on config["provider"]["type"]
await agent.connect()
reply = await agent.run(chat_id, user_text)   # chat_id: Hashable
await agent.cleanup()
```

`build_agent` returns an `AIAgent` (a `Protocol`), so transports can stay
provider-agnostic. `provider` is a tagged union — a dict whose ``type``
selects the implementation and whose other keys are provider-specific
options. Omit `provider` entirely to default to OpenAI with default
settings.

## OpenAI provider

```python
config = {
    "provider": {
        "type": "openai",
        "model": "gpt-5.4",      # optional, default: gpt-5.4
        "apiType": "responses",  # optional: "responses" (default) or "chat_completions"
        "historyTurns": 10,      # optional, default: 10 — user turns kept in history
    },
    "mcpServers": {
        "my-tool": {
            "url": "http://localhost:8000/mcp",
            "timeout": 30.0,
            "enabled": True,
        }
    },
}
```

History is stored in per-chat `SQLiteSession` instances and truncated
turn-aware before every run so the model never sees orphaned tool
results.

## Anthropic provider

```python
config = {
    "provider": {
        "type": "anthropic",
        "model": "claude-sonnet-4-6",  # optional, default: SDK's default
        "allowedTools": ["WebFetch"],  # optional; added on top of the shell set
    },
    "mcpServers": {
        "my-stdio": {
            "command": "python",
            "args": ["-m", "srv"],
            "env": {"FOO": "bar"},
        },
        "my-http": {
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer x"},
        },
    },
}
```

When the SDK returns an error (billing, rate limit, `error_max_turns`,
etc.), `run()` raises `ClaudeAgentError` instead of returning an empty
string. Catch it in your transport:

```python
from agent_core import ClaudeAgentError

try:
    reply = await agent.run(chat_id, user_text)
except ClaudeAgentError as e:
    # e.subtype: "error_max_turns" | "error_max_budget_usd" | ...
    # e.session_id: SDK session that failed (not stored in mapping)
    reply = f"Agent error: {e}"
```

Differences from the OpenAI provider:

- Session history is stored on disk by `claude-agent-sdk` itself
  (`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`). We keep a
  small `chat_id -> session_id` mapping in SQLite so each chat resumes
  its own conversation across restarts.
- Resume is bound to `cwd`. Deploy with a stable working directory
  (e.g. `WorkingDirectory=` in systemd, `WORKDIR` in Docker) or resume
  will silently fall back to a fresh session.
- `provider.apiType` is ignored.
- Settings are always scoped to `setting_sources=["project"]` so the
  bot only picks up `.claude/` inside its deployment cwd. The host
  user's `~/.claude/` (personal MCP servers, skills, subagents, slash
  commands) is **never** inherited — otherwise a bot would silently
  run with whatever the server's user has configured.
- `SHELL_ENABLED` enables the read-only toolset plus `Bash`. Extra
  tools (e.g. `WebFetch`, `Write`, `Edit`) go in
  `config["provider"]["allowedTools"]`.
- All tool execution happens locally in the CLI subprocess the SDK
  spawns — there is no hosted sandbox.

## Environment

| Variable | OpenAI | Anthropic | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | ✓ | | OpenAI / Azure OpenAI v1 key |
| `OPENAI_BASE_URL` | ✓ | | Optional, for Azure or compatible endpoints |
| `ANTHROPIC_API_KEY` | | ✓ | Anthropic key (consumed by `claude-agent-sdk`) |
| `SHELL_ENABLED` | ✓ | ✓ | Truthy to attach shell tools (OpenAI: `ShellTool`; Anthropic: `Bash`/`Read`/`Glob`/`Grep` + project skills) |
| `SHELL_SKILLS_DIR` | ✓ | | OpenAI-only: directory of `SKILL.md` skills to mount on `ShellTool` |
| `SESSION_DB_PATH` | ✓ | ✓ | SQLite path. OpenAI: conversation history. Anthropic: `chat_id -> session_id` mapping. Default: in-memory |
