"""Application logging — structured, correlated, viewable.

Best-practice logging for a production service:
* **JSON** logs in prod (machine-parseable for any log aggregator).
* **Ansible / CLI-style** colored console logs in dev (human-friendly:
  ``ok:`` / ``warn:`` / ``failed:`` per level, with key=value context).
* **Correlation id** per request (contextvar) on every line + response header.
* **In-memory ring buffer** of recent records, exposed to admins via
  ``GET /admin/logs`` and rendered in the admin UI (terminal-style).

Switch format with ``LOG_JSON=true`` and level with ``LOG_LEVEL``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

# Reserved LogRecord attrs we don't want to duplicate as "extra" context.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {
    "message", "asctime", "taskName",
}

# ANSI colors for the CLI formatter.
_COLOR = {"DEBUG": "\033[90m", "INFO": "\033[32m", "WARNING": "\033[33m",
          "ERROR": "\033[31m", "CRITICAL": "\033[41m"}
_RESET = "\033[0m"
# Ansible-style status word per level.
_STATUS = {"DEBUG": "skip", "INFO": "ok", "WARNING": "warn",
           "ERROR": "failed", "CRITICAL": "FATAL"}


def _extras(record: logging.LogRecord) -> Dict[str, Any]:
    return {k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            **_extras(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Ansible/CLI-style: ``HH:MM:SS ok: [logger] message k=v (req=...)``."""

    def __init__(self, color: bool = True):
        super().__init__()
        self.color = color and os.getenv("NO_COLOR") is None

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        status = _STATUS.get(record.levelname, record.levelname.lower())
        ctx = _extras(record)
        ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
        rid = request_id_var.get()
        rid_str = f" (req={rid})" if rid and rid != "-" else ""
        line = f"{ts} {status:>6}: [{record.name}] {record.getMessage()}"
        if ctx_str:
            line += f"  {ctx_str}"
        line += rid_str
        if self.color:
            c = _COLOR.get(record.levelname, "")
            line = f"{c}{line}{_RESET}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class RingBufferHandler(logging.Handler):
    """Keep the last N records in memory for the admin log viewer."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buffer: Deque[Dict[str, Any]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "request_id": request_id_var.get(),
                **_extras(record),
            })
        except Exception:  # noqa: BLE001 - logging must never raise
            pass

    def records(self, limit: int = 200, level: str | None = None) -> List[Dict[str, Any]]:
        items = list(self.buffer)
        if level:
            wanted = level.upper()
            items = [r for r in items if r["level"] == wanted]
        return items[-limit:][::-1]  # newest first


_ring: RingBufferHandler | None = None


def get_ring() -> RingBufferHandler:
    global _ring
    if _ring is None:
        _ring = RingBufferHandler()
    return _ring


def configure_logging() -> None:
    """Install handlers on the root logger (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_pvllm_configured", False):
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(
        JsonFormatter() if os.getenv("LOG_JSON", "").lower() == "true"
        else ConsoleFormatter()
    )
    root.addHandler(console)
    root.addHandler(get_ring())
    root._pvllm_configured = True  # type: ignore[attr-defined]
