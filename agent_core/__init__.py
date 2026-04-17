from __future__ import annotations

from typing import Any

from .agent import OpenAIAgent
from .base import AIAgent
from .claude import ClaudeAgent
from .claude import ClaudeAgentError
from .env import env_flag

__all__ = ["ClaudeAgent", "ClaudeAgentError", "OpenAIAgent", "build_agent", "env_flag"]


def build_agent(name: str, config: dict[str, Any]) -> AIAgent:
    """Build an agent from a config dict, dispatching on ``provider.type``.

    ``provider`` is a tagged union: a dict with a ``type`` key plus any
    provider-specific fields (``model``, ``apiType``, ``allowedTools``).
    When absent, defaults to an empty OpenAI provider config.
    """
    provider = config.get("provider", {"type": "openai"})
    if not isinstance(provider, dict) or "type" not in provider:
        raise ValueError("'provider' must be a dict with a 'type' key")
    ptype = provider["type"]
    if ptype == "openai":
        return OpenAIAgent.from_dict(name, config)
    if ptype == "anthropic":
        return ClaudeAgent.from_dict(name, config)
    raise ValueError(f"Unknown provider type: {ptype!r}")
