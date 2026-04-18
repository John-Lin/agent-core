from __future__ import annotations


class AgentError(Exception):
    """Provider-agnostic error raised by ``AIAgent.run``.

    ``subtype`` is a short machine-readable tag so transports can branch
    on it without parsing messages. Values common to both providers use
    the same tag (e.g. ``error_max_turns``); provider-specific tags
    propagate through unchanged (e.g. Anthropic's
    ``error_max_budget_usd``, OpenAI's ``rate_limit``).

    ``session_id`` is populated when the provider exposes one for the
    failing turn (Anthropic). It is ``None`` for providers that don't
    surface a per-turn id (OpenAI).
    """

    def __init__(
        self,
        message: str,
        *,
        subtype: str,
        provider: str,
        session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.subtype = subtype
        self.provider = provider
        self.session_id = session_id
