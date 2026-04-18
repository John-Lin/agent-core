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
- `instructions.md` in the working directory as the system prompt (override path with `AGENT_INSTRUCTIONS_PATH`)

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
        "shell": {               # optional; omit to disable the local ShellTool
            "enabled": True,     # must be a bool — strings like "true"/"false" are rejected
            "skillsDir": "/app/skills",  # optional path to SKILL.md skills
        },
    },
    "mcp": {
        "my-tool": {
            "type": "remote",
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
        "allowedTools": ["Bash", "Write", "Edit"],  # optional; tools listed here are added on top of the read-only defaults
    },
    "mcp": {
        "my-local": {
            "type": "local",
            "command": ["python", "-m", "srv"],
            "environment": {"FOO": "bar"},
        },
        "my-remote": {
            "type": "remote",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer x"},
        },
    },
}
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
- Read-only built-ins (`Read`, `Glob`, `Grep`) are always on — these
  are Anthropic's "no permission required" tools and carry no write or
  exec risk. Any tool that can mutate files or run commands (`Write`,
  `Edit`, `Bash`, `WebFetch`, …) must be listed explicitly in
  `config["provider"]["allowedTools"]`. This includes `Skill`: if the
  agent needs to run project skills under `.claude/skills/`, add
  `"Skill"` to `allowedTools` — otherwise the Skill tool is blocked.
  Tool names are case-sensitive and validated by the SDK, not by us —
  an unrecognized name (e.g. a typo like `"webSearch"`) is silently
  dropped. See Anthropic's [tools reference](https://code.claude.com/docs/en/tools-reference)
  for the canonical list and exact casing.
- Built-ins the caller does *not* opt into are placed in the SDK's
  `disallowed_tools` list, not merely absent from `allowed_tools`.
  This matters because `allowed_tools` is an auto-approval list, not
  a visibility filter: a tool that is absent from `allowed_tools` but
  also absent from `disallowed_tools` still shows up in the model's
  toolset (so the model can see it and describe it to users), it just
  fails at call time. We maintain a `KNOWN_BUILTIN_TOOLS` list in
  `anthropic_provider.py` and subtract the caller's allowlist from it
  to produce the actual block list.
- All tool execution happens locally in the CLI subprocess the SDK
  spawns — there is no hosted sandbox.

## Error handling

Both providers raise a common `AgentError` when the underlying SDK
reports a failure (billing, rate limits, max turns, guardrails, …).
Catch it in your transport and branch on `subtype`:

```python
from agent_core import AgentError

try:
    reply = await agent.run(chat_id, user_text)
except AgentError as e:
    # e.provider:   "openai" | "anthropic"
    # e.subtype:    normalized tag (see table below)
    # e.session_id: Anthropic-only; None for OpenAI
    reply = f"Agent error ({e.subtype}): {e}"
```

| subtype | OpenAI source | Anthropic source |
|---|---|---|
| `error_max_turns` | `MaxTurnsExceeded` | SDK `error_max_turns` |
| `error_max_budget_usd` | — | SDK `error_max_budget_usd` |
| `rate_limit` | `openai.RateLimitError` | — (surfaces via `is_error`) |
| `auth` | `openai.AuthenticationError` | — |
| `bad_request` | `openai.BadRequestError` | — |
| `timeout` | `openai.APITimeoutError` | — |
| `connection` | `openai.APIConnectionError` | — |
| `api_status` | other `openai.APIStatusError` | — |
| `model_behavior` | `agents.ModelBehaviorError` | — |
| `guardrail` | Input/Output guardrail tripwires | — |
| `tool_guardrail` | Tool-level guardrail tripwires | — |
| `tool_timeout` | `agents.ToolTimeoutError` | — |
| `mcp_cancelled` | `agents.MCPToolCancellationError` | — |
| *(passthrough)* | — | any other SDK `subtype` |

Unmapped exceptions (e.g. `RuntimeError`) propagate unchanged — the
wrapper only normalizes known provider failures.

## Environment

| Variable | OpenAI | Anthropic | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | ✓ | | OpenAI / Azure OpenAI v1 key |
| `OPENAI_BASE_URL` | ✓ | | Optional, for Azure or compatible endpoints |
| `ANTHROPIC_API_KEY` | | ✓ | Anthropic key (consumed by `claude-agent-sdk`) |
| `SESSION_DB_PATH` | ✓ | ✓ | SQLite path. OpenAI: conversation history. Anthropic: `chat_id -> session_id` mapping. Default: in-memory |
| `AGENT_INSTRUCTIONS_PATH` | ✓ | ✓ | Override path to the instructions file. Default: `./instructions.md` |

Shell tool configuration is now per-provider config, not env vars:

- **OpenAI**: `provider.shell.enabled` (bool, default `false`) attaches
  a local `ShellTool`. `provider.shell.skillsDir` (path) mounts
  `SKILL.md` skills discovered under it.
- **Anthropic**: list built-in tools explicitly in
  `provider.allowedTools` (e.g. `["Bash", "Read", "Glob", "Grep"]`).
