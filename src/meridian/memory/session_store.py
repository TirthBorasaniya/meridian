"""Cross-session conversation memory store.

Persists conversation turns across API restarts and across the in-session
graph checkpointing that :mod:`meridian.graph.graph` already provides via
``SqliteSaver``. The checkpointer resumes mid-graph execution within a single
run; this store lets a later, unrelated process invocation recall what was
said in a prior session by ``session_id``.
"""

import sqlite3
from datetime import datetime, timezone
from functools import lru_cache

from meridian.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_memory (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);

CREATE TABLE IF NOT EXISTS session_summaries (
    session_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    turn_count INTEGER NOT NULL,
    last_updated TEXT NOT NULL
);
"""


@lru_cache(maxsize=1)
def _get_connection() -> sqlite3.Connection:
    """Return the process-wide cached connection to the session memory database."""
    settings = get_settings()
    connection = sqlite3.connect(settings.session_db_path, check_same_thread=False)
    connection.executescript(_SCHEMA)
    connection.commit()
    return connection


def save_turn(session_id: str, role: str, content: str) -> None:
    """Persist a single conversation turn to the session memory store.

    Parameters
    ----------
    session_id : str
        Identifier grouping turns that belong to the same conversation.
    role : str
        Speaker for this turn, for example ``"user"`` or ``"assistant"``.
    content : str
        The turn's text content.
    """
    connection = _get_connection()
    cursor = connection.execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM session_memory WHERE session_id = ?",
        (session_id,),
    )
    next_turn_index = cursor.fetchone()[0]
    connection.execute(
        "INSERT INTO session_memory (session_id, turn_index, role, content, timestamp_utc) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_id, next_turn_index, role, content, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def get_session_context(session_id: str, max_turns: int = 6) -> str:
    """Retrieve the last ``max_turns`` turns for a session as formatted context.

    Parameters
    ----------
    session_id : str
        Identifier of the conversation to recall.
    max_turns : int, optional
        Maximum number of most recent turns to include. Defaults to 6.

    Returns
    -------
    str
        The recalled turns formatted as ``"role: content"`` lines, most recent
        last. Returns an empty string for new or unknown sessions.
    """
    connection = _get_connection()
    cursor = connection.execute(
        "SELECT role, content FROM session_memory WHERE session_id = ? "
        "ORDER BY turn_index DESC LIMIT ?",
        (session_id, max_turns),
    )
    row_list = cursor.fetchall()
    if not row_list:
        return ""
    row_list.reverse()
    return "\n".join(f"{role}: {content}" for role, content in row_list)
