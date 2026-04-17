from __future__ import annotations

import pytest

from agent_core.session_map import ClaudeSessionMap


@pytest.fixture
def session_map():
    m = ClaudeSessionMap(":memory:")
    yield m
    m.close()


class TestClaudeSessionMap:
    def test_get_returns_none_for_unknown_chat(self, session_map):
        assert session_map.get("chat-1") is None

    def test_put_then_get_roundtrips(self, session_map):
        session_map.put("chat-1", "session-uuid-abc")
        assert session_map.get("chat-1") == "session-uuid-abc"

    def test_put_overwrites_existing_session_id(self, session_map):
        session_map.put("chat-1", "session-uuid-abc")
        session_map.put("chat-1", "session-uuid-def")
        assert session_map.get("chat-1") == "session-uuid-def"

    def test_different_chats_are_independent(self, session_map):
        session_map.put("chat-1", "session-a")
        session_map.put("chat-2", "session-b")
        assert session_map.get("chat-1") == "session-a"
        assert session_map.get("chat-2") == "session-b"

    def test_chat_id_can_be_any_hashable(self, session_map):
        session_map.put(12345, "session-int")
        session_map.put(("channel", "user"), "session-tuple")
        assert session_map.get(12345) == "session-int"
        assert session_map.get(("channel", "user")) == "session-tuple"

    def test_persists_to_disk(self, tmp_path):
        db = tmp_path / "sessions.db"
        m1 = ClaudeSessionMap(str(db))
        m1.put("chat-1", "session-persisted")
        m1.close()

        m2 = ClaudeSessionMap(str(db))
        try:
            assert m2.get("chat-1") == "session-persisted"
        finally:
            m2.close()

    def test_coexists_with_openai_sqlite_session_in_same_db(self, tmp_path):
        """Mapping table must not collide with openai-agents' SQLiteSession schema."""
        from agents import SQLiteSession

        db = tmp_path / "shared.db"
        oai_session = SQLiteSession("chat-1", str(db))
        m = ClaudeSessionMap(str(db))
        try:
            m.put("chat-1", "session-uuid")
            assert m.get("chat-1") == "session-uuid"
        finally:
            m.close()
            oai_session.close()
