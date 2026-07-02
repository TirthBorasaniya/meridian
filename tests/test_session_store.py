"""Unit tests for the cross-session conversation memory store.

Each test monkeypatches the connection to a fresh in-memory database so tests
do not share state or touch the configured on-disk session database.
"""

import sqlite3

from meridian.memory import session_store


def _fresh_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(session_store._SCHEMA)
    return connection


def test_get_session_context_empty_for_new_session(monkeypatch):
    monkeypatch.setattr(session_store, "_get_connection", lambda: _fresh_connection())
    assert session_store.get_session_context("unknown-session") == ""


def test_save_and_recall_turns(monkeypatch):
    connection = _fresh_connection()
    monkeypatch.setattr(session_store, "_get_connection", lambda: connection)

    session_store.save_turn("s1", "user", "hello")
    session_store.save_turn("s1", "assistant", "hi there")

    assert session_store.get_session_context("s1") == "user: hello\nassistant: hi there"


def test_get_session_context_respects_max_turns(monkeypatch):
    connection = _fresh_connection()
    monkeypatch.setattr(session_store, "_get_connection", lambda: connection)

    for index in range(4):
        session_store.save_turn("s1", "user", f"turn {index}")

    assert session_store.get_session_context("s1", max_turns=2) == "user: turn 2\nuser: turn 3"


def test_sessions_are_isolated(monkeypatch):
    connection = _fresh_connection()
    monkeypatch.setattr(session_store, "_get_connection", lambda: connection)

    session_store.save_turn("s1", "user", "from s1")
    session_store.save_turn("s2", "user", "from s2")

    assert session_store.get_session_context("s1") == "user: from s1"
    assert session_store.get_session_context("s2") == "user: from s2"
