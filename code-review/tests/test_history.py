import pytest

from improved.history import InMemorySessionHistory
from improved.models import ChatTurn


def test_sessions_are_isolated():
    history = InMemorySessionHistory(max_turns=3)
    history.add("alice", ChatTurn(user="Alice question", assistant="Alice answer"))
    history.add("bob", ChatTurn(user="Bob question", assistant="Bob answer"))

    assert [turn.user for turn in history.get("alice")] == ["Alice question"]
    assert [turn.user for turn in history.get("bob")] == ["Bob question"]


def test_history_is_bounded_to_recent_turns():
    history = InMemorySessionHistory(max_turns=2)
    for index in range(3):
        history.add("session", ChatTurn(user=f"q{index}", assistant=f"a{index}"))

    assert [turn.user for turn in history.get("session")] == ["q1", "q2"]


def test_blank_session_id_is_rejected():
    history = InMemorySessionHistory(max_turns=2)

    with pytest.raises(ValueError, match="session_id"):
        history.get("   ")
