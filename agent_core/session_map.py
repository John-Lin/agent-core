from __future__ import annotations

import sqlite3
from collections.abc import Hashable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_chat_sessions (
    chat_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    updated_at INTEGER NOT NULL
)
"""


class ClaudeSessionMap:
    """Persistent chat_id -> claude-agent-sdk session_id mapping.

    Claude Agent SDK generates its own session id (UUID) and stores the
    conversation on disk. We keep a small mapping table so a long-lived
    transport (Telegram, Slack) can resume the same SDK session for the
    same chat across process restarts.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, chat_id: Hashable) -> str | None:
        cur = self._conn.execute(
            "SELECT session_id FROM claude_chat_sessions WHERE chat_id = ?",
            (str(chat_id),),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, chat_id: Hashable, session_id: str) -> None:
        self._conn.execute(
            """
            INSERT INTO claude_chat_sessions (chat_id, session_id, updated_at)
            VALUES (?, ?, strftime('%s', 'now'))
            ON CONFLICT(chat_id) DO UPDATE SET
                session_id = excluded.session_id,
                updated_at = excluded.updated_at
            """,
            (str(chat_id), session_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
