from __future__ import annotations

import os

import pytest

from agent_core.anthropic_provider import ClaudeAgent

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live integration test",
)


@pytest.mark.anyio
async def test_round_trip_through_live_sdk():
    """End-to-end: one query, check we get back some text and a session id."""
    agent = ClaudeAgent(
        name="integration",
        instructions="You are a terse assistant. Reply in one short sentence.",
        max_turns=1,
        claude_home="/tmp/agent-core-integration-home",
    )
    try:
        reply = await agent.run("integration-chat", "Say 'pong' and nothing else.")
        assert isinstance(reply, str)
        assert reply.strip() != ""
        assert agent._session_map.get("integration-chat") is not None
    finally:
        await agent.cleanup()
