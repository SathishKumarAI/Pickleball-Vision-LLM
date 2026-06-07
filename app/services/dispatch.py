"""GPU job dispatch.

Abstracts "start the GPU analysis for this job" so the control plane is testable
without Modal. ``ModalDispatcher`` spawns the real Modal function;
``FakeDispatcher`` records spawns (and can synchronously simulate completion) for
offline tests/dev.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from app.services.repo import new_id


class Dispatcher(ABC):
    @abstractmethod
    def spawn(self, payload: Dict[str, Any]) -> str:
        """Start async GPU work; return a call id."""
    @abstractmethod
    def cancel(self, call_id: str) -> bool:
        """Best-effort hard-cancel of a running call."""


class FakeDispatcher(Dispatcher):
    """Records spawns; optional ``on_spawn`` simulates the worker (dev/tests)."""

    def __init__(self, on_spawn: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.spawns: List[Dict[str, Any]] = []
        self.cancelled: List[str] = []
        self._on_spawn = on_spawn

    def spawn(self, payload: Dict[str, Any]) -> str:
        self.spawns.append(payload)
        if self._on_spawn:
            self._on_spawn(payload)
        return f"fake-call-{new_id()[:8]}"

    def cancel(self, call_id: str) -> bool:
        self.cancelled.append(call_id)
        return True


class ModalDispatcher(Dispatcher):
    """Spawns the deployed Modal function. Lazy-imports modal."""

    def __init__(self, app_name: str, function: str):
        self._app_name = app_name
        self._function = function
        self._fn = None

    def _lookup(self):
        if self._fn is None:
            import modal  # lazy
            self._fn = modal.Function.lookup(self._app_name, self._function)
        return self._fn

    def spawn(self, payload: Dict[str, Any]) -> str:
        call = self._lookup().spawn(**payload)
        return call.object_id

    def cancel(self, call_id: str) -> bool:
        import modal  # lazy
        try:
            modal.functions.FunctionCall.from_id(call_id).cancel()
            return True
        except Exception:  # noqa: BLE001
            return False
