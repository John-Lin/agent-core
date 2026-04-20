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

`provider` is a tagged union keyed on `type`. Omit it to default to
OpenAI. `build_agent` returns an `AIAgent` `Protocol`, so transports
stay provider-agnostic.

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
        "claudeHome": "/app/claude-home",  # required; HOME for the CLI subprocess
        "model": "claude-sonnet-4-6",  # optional, default: SDK's default
        "allowedTools": ["Bash", "Write", "Edit"],  # optional; tools listed here are added on top of the read-only defaults
        "queryTimeoutSeconds": 300.0,  # optional, default: 300. Set to null to disable.
        "environment": {                # optional; extra env vars merged into the CLI subprocess
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
        },
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

- **History.** Stored on disk by the SDK at
  `<claudeHome>/.claude/projects/<encoded-cwd>/<session-id>.jsonl`.
  We persist `chat_id -> session_id` in SQLite for resume across
  restarts. Resume is bound to `cwd` — deploy with a stable working
  directory (`WorkingDirectory=` in systemd, `WORKDIR` in Docker) or
  resume silently falls back to a fresh session.
- **Settings scope.** Hardcoded to `setting_sources=["project"]`: only
  `.claude/` inside the deployment cwd is loaded. `provider.apiType`
  is ignored.
- **Tools.** `Read`, `Glob`, `Grep` (Anthropic's "no permission
  required" tools) are always on. Anything that mutates files or runs
  commands — `Write`, `Edit`, `Bash`, `WebFetch`, `Skill`, … — must be
  listed explicitly in `provider.allowedTools`. Tool names are
  case-sensitive and validated by the SDK; typos are silently dropped.
  See Anthropic's [tools reference](https://code.claude.com/docs/en/tools-reference).
  Built-ins the caller does not opt into are added to the SDK's
  `disallowed_tools` (not merely absent from `allowed_tools`) so they
  disappear from the model's visible toolset, not just fail at call
  time. All execution is local in the CLI subprocess — no hosted sandbox.
- **Per-query timeout.** The SDK has no built-in timeout, so we wrap
  each call in `asyncio.timeout(queryTimeoutSeconds)` (default `300`,
  `null` to disable). On timeout: `AgentError(subtype="timeout")`,
  the per-chat lock is released, and the next message starts a fresh
  session.
- **`provider.environment`.** Extra env vars merged into the CLI
  subprocess on top of the parent process env. Use it to switch
  backends (e.g. Bedrock — see below) without polluting global env.
  Values must be strings. `HOME` is always applied last so
  `environment.HOME` cannot override `claudeHome`.
- **`provider.claudeHome`** (required). The bot's state directory.
  Everything the CLI persists lives under it:

  ```
  <claudeHome>/.claude.json                          # auth/cache state
  <claudeHome>/.claude/projects/<enc-cwd>/*.jsonl    # conversation history
  ```

  Mechanically it is passed to the CLI subprocess as `HOME`. On a
  shared host this prevents reading the developer's `~/.claude.json`
  (OAuth, personal MCP, project history). In a container the redirect
  is defensive — but it still picks the path you mount your
  persistence volume on, so history survives restarts. For
  multi-restart deployments, also point `SESSION_DB_PATH` at a
  persistent path — without that mapping, the JSONL files become
  orphans no one can resume.

### Using AWS Bedrock

Inject Claude Code's standard Bedrock env vars via `provider.environment`:

```python
config = {
    "provider": {
        "type": "anthropic",
        "claudeHome": "/app/claude-home",
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",  # inference profile ID
        "environment": {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
            "AWS_ACCESS_KEY_ID": "...",        # or rely on IAM role /
            "AWS_SECRET_ACCESS_KEY": "...",    # IRSA / IMDS auto-discovery
        },
    },
}
```

`claudeHome` rewrites `HOME`, so AWS SDK won't find `~/.aws/credentials`
unless you point at it explicitly via `AWS_SHARED_CREDENTIALS_FILE` /
`AWS_CONFIG_FILE` (or skip the file and use env-var credentials / an
IAM role).

## Error handling

Both providers normalize known SDK failures into `AgentError` with a
`provider`, `subtype`, and (Anthropic-only) `session_id`. Branch on
`subtype` in your transport:

```python
from agent_core import AgentError

try:
    reply = await agent.run(chat_id, user_text)
except AgentError as e:
    reply = f"Agent error ({e.subtype}): {e}"
```

Common subtypes: `error_max_turns`, `error_max_budget_usd`,
`rate_limit`, `auth`, `bad_request`, `timeout`, `connection`,
`guardrail`. See `agent_core/errors.py` and each provider's mapping
for the full list. Unmapped exceptions propagate unchanged.

## Environment

| Variable | OpenAI | Anthropic | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | ✓ | | OpenAI / Azure OpenAI v1 key |
| `OPENAI_BASE_URL` | ✓ | | Optional, for Azure or compatible endpoints |
| `ANTHROPIC_API_KEY` | | ✓ | Anthropic key (consumed by `claude-agent-sdk`) |
| `SESSION_DB_PATH` | ✓ | ✓ | SQLite path. OpenAI: conversation history. Anthropic: `chat_id -> session_id` mapping. Default: in-memory (lost on restart — point at a persistent path to resume across restarts) |
| `AGENT_INSTRUCTIONS_PATH` | ✓ | ✓ | Override path to the instructions file. Default: `./instructions.md` |
