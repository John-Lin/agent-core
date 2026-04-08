# agent-core

Shared OpenAI Agent SDK wrapper extracted from `agentic-telegram-bot`,
`agentic-slackbot`, and `agentic-discord-bot`.

Wraps the [`openai-agents`](https://pypi.org/project/openai-agents/) SDK
with:

- Per-conversation message history (any `Hashable` key)
- Per-conversation async lock to serialise concurrent messages
- History truncation to a fixed number of turns
- MCP server construction from a JSON config dict
- Optional local `ShellTool` with skill discovery, gated by env vars

## Usage

```python
from agent_core import OpenAIAgent

agent = OpenAIAgent.from_dict("MyBot", {"mcpServers": {...}})
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
| `OPENAI_MODEL` | Model name (default `gpt-5.4`) |
| `OPENAI_API_TYPE` | `responses` (default) or `chat_completions` |
| `SHELL_ENABLED` | Truthy to attach the local `ShellTool` |
| `SHELL_SKILLS_DIR` | Directory of `SKILL.md` skills to mount on the shell |
