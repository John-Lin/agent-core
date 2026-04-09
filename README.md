# agent-core

Shared OpenAI Agent SDK wrapper extracted from `agentic-telegram-bot`,
`agentic-slackbot`, and `agentic-discord-bot`.

Wraps the [`openai-agents`](https://pypi.org/project/openai-agents/) SDK
with:

- Per-conversation history via SQLiteSession (any `Hashable` key; in-memory by default, persistent with `SESSION_DB_PATH`)
- Per-conversation async lock to serialise concurrent messages
- History truncation to a configurable number of turns
- MCP server construction from a JSON config dict, with per-server timeout and enable/disable
- Optional local `ShellTool` with skill discovery, gated by env vars

## Usage

```python
from agent_core import OpenAIAgent

config = {
    "model": "gpt-5.4",      # optional, default: gpt-5.4
    "maxTurns": 10,           # optional, default: 10
    "mcpServers": {
        "my-tool": {
            "url": "http://localhost:8000/mcp",
            "timeout": 30.0,  # optional, default: 30.0
            "enabled": True,  # optional, default: true
        }
    },
}

# Advanced: set "apiType": "chat_completions" to use the Chat Completions API
# instead of the default Responses API. Rarely needed.

agent = OpenAIAgent.from_dict("MyBot", config)
await agent.connect()
reply = await agent.run(chat_id, user_text)  # chat_id: Hashable
await agent.cleanup()
```

`from_dict` reads the system prompt from `instructions.md` in the current
working directory and fails fast if missing.

## Environment

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI / Azure OpenAI v1 key |
| `OPENAI_BASE_URL` | Optional, for Azure or compatible endpoints |
| `SHELL_ENABLED` | Truthy to attach the local `ShellTool` |
| `SHELL_SKILLS_DIR` | Directory of `SKILL.md` skills to mount on the shell |
| `SESSION_DB_PATH` | SQLite file path for persistent conversation history (default: in-memory) |
