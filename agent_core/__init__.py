from __future__ import annotations

from typing import Any

from .agent import OpenAIAgent
from .base import AIAgent
from .claude import ClaudeAgent
from .claude import ClaudeAgentError
from .env import env_flag

__all__ = ["ClaudeAgent", "ClaudeAgentError", "OpenAIAgent", "build_agent", "env_flag"]


def build_agent(name: str, config: dict[str, Any]) -> AIAgent:
    """Build an agent from a config dict, dispatching on ``provider``.

    Defaults to ``"openai"`` when the key is absent so existing configs
    keep working unchanged.
    """
    provider = config.get("provider", "openai")
    if provider == "openai":
        return OpenAIAgent.from_dict(name, config)
    if provider == "claude":
        return ClaudeAgent.from_dict(name, config)
    raise ValueError(f"Unknown provider: {provider!r}")
