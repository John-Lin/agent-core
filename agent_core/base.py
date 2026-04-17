from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol
from typing import runtime_checkable


@runtime_checkable
class AIAgent(Protocol):
    """Common surface implemented by OpenAIAgent and ClaudeAgent.

    Transports (Telegram, Slack, ...) should depend on this Protocol so
    they can be wired to any provider via ``build_agent``.
    """

    name: str

    async def connect(self) -> None: ...

    async def run(self, chat_id: Hashable, message: str) -> str: ...

    async def cleanup(self) -> None: ...
