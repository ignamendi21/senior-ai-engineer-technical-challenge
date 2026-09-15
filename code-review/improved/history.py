from collections import defaultdict, deque
from collections.abc import Sequence
from threading import RLock
from typing import Protocol

from improved.models import ChatTurn


class SessionHistory(Protocol):
    def get(self, session_id: str) -> Sequence[ChatTurn]: ...

    def add(self, session_id: str, turn: ChatTurn) -> None: ...


class InMemorySessionHistory:
    def __init__(self, max_turns: int) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self._max_turns = max_turns
        self._sessions: dict[str, deque[ChatTurn]] = defaultdict(
            lambda: deque(maxlen=self._max_turns)
        )
        self._lock = RLock()

    def get(self, session_id: str) -> list[ChatTurn]:
        self._validate_session_id(session_id)
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def add(self, session_id: str, turn: ChatTurn) -> None:
        self._validate_session_id(session_id)
        with self._lock:
            self._sessions[session_id].append(turn)

    def clear(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")
