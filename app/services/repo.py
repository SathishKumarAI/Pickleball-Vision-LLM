"""Persistence layer for jobs / subscriptions / usage.

Two implementations behind one interface:

* ``InMemoryRepo`` — dependency-free; used for dev + offline tests.
* ``SupabaseRepo`` — Postgres via the Supabase client (service-role key, so it
  writes across users; RLS still protects the browser path).

Selected at runtime by :func:`app.deps.get_repo` based on whether Supabase is
configured. The interface is intentionally small so swapping is trivial.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def new_id() -> str:
    return uuid.uuid4().hex


class Repo(ABC):
    # --- jobs ---
    @abstractmethod
    def create_job(self, user_id: str, **fields: Any) -> Dict[str, Any]: ...
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def list_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def update_job(self, job_id: str, **fields: Any) -> Optional[Dict[str, Any]]: ...
    @abstractmethod
    def claim_running(self, job_id: str) -> bool:
        """Atomically transition queued -> running. False if already claimed
        (idempotency guard against Modal retries / double-spawn)."""
    @abstractmethod
    def find_job_by_sha(self, user_id: str, sha: str) -> Optional[Dict[str, Any]]: ...
    # --- billing/usage ---
    @abstractmethod
    def get_subscription(self, user_id: str) -> Dict[str, Any]: ...
    @abstractmethod
    def set_subscription(self, user_id: str, **fields: Any) -> Dict[str, Any]: ...
    @abstractmethod
    def get_usage(self, user_id: str, period: str) -> Dict[str, Any]: ...
    @abstractmethod
    def incr_usage(self, user_id: str, period: str, videos: int, seconds: int) -> Dict[str, Any]: ...
    # --- admin / GDPR ---
    @abstractmethod
    def list_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def delete_user_data(self, user_id: str) -> Dict[str, int]:
        """Delete all of a user's app data (GDPR). Returns counts deleted."""


class InMemoryRepo(Repo):
    """Thread-safe in-memory store (dev/tests)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._subs: Dict[str, Dict[str, Any]] = {}
        self._usage: Dict[tuple, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, user_id: str, **fields: Any) -> Dict[str, Any]:
        job = {
            "id": new_id(), "user_id": user_id, "status": "queued",
            "progress": 0.0, "message": "", "error": None,
            "result_object_key": None, "output_video_key": None, "modal_call_id": None,
            **fields,
        }
        with self._lock:
            self._jobs[job["id"]] = job
        return dict(job)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def list_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values() if j["user_id"] == user_id]
        return list(reversed(jobs))[:limit]

    def update_job(self, job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return None
            j.update(fields)
            return dict(j)

    def claim_running(self, job_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j or j["status"] != "queued":
                return False
            j["status"] = "running"
            return True

    def find_job_by_sha(self, user_id: str, sha: str) -> Optional[Dict[str, Any]]:
        if not sha:
            return None
        with self._lock:
            for j in reversed(list(self._jobs.values())):
                if j["user_id"] == user_id and j.get("content_sha256") == sha \
                        and j.get("status") in ("done", "running", "queued"):
                    return dict(j)
        return None

    def get_subscription(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._subs.get(user_id, {"user_id": user_id, "plan": "free", "status": "active"}))

    def set_subscription(self, user_id: str, **fields: Any) -> Dict[str, Any]:
        with self._lock:
            sub = self._subs.setdefault(user_id, {"user_id": user_id, "plan": "free", "status": "active"})
            sub.update(fields)
            return dict(sub)

    def get_usage(self, user_id: str, period: str) -> Dict[str, Any]:
        with self._lock:
            return dict(self._usage.get((user_id, period),
                                        {"user_id": user_id, "period": period,
                                         "videos_processed": 0, "seconds_processed": 0}))

    def incr_usage(self, user_id: str, period: str, videos: int, seconds: int) -> Dict[str, Any]:
        with self._lock:
            u = self._usage.setdefault((user_id, period),
                                       {"user_id": user_id, "period": period,
                                        "videos_processed": 0, "seconds_processed": 0})
            u["videos_processed"] += videos
            u["seconds_processed"] += seconds
            return dict(u)

    def list_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed([dict(j) for j in self._jobs.values()]))[:limit]

    def delete_user_data(self, user_id: str) -> Dict[str, int]:
        with self._lock:
            jids = [jid for jid, j in self._jobs.items() if j["user_id"] == user_id]
            for jid in jids:
                del self._jobs[jid]
            self._subs.pop(user_id, None)
            usage_keys = [k for k in self._usage if k[0] == user_id]
            for k in usage_keys:
                del self._usage[k]
        return {"jobs": len(jids), "usage": len(usage_keys), "subscriptions": 1}


class SupabaseRepo(Repo):
    """Supabase Postgres-backed repo (service-role key). Lazy-imports the client."""

    def __init__(self, url: str, service_key: str) -> None:
        from supabase import create_client  # lazy
        self._sb = create_client(url, service_key)

    def create_job(self, user_id: str, **fields: Any) -> Dict[str, Any]:
        row = {"user_id": user_id, "status": "queued", **fields}
        return self._sb.table("jobs").insert(row).execute().data[0]

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        r = self._sb.table("jobs").select("*").eq("id", job_id).limit(1).execute()
        return r.data[0] if r.data else None

    def list_jobs(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return (self._sb.table("jobs").select("*").eq("user_id", user_id)
                .order("created_at", desc=True).limit(limit).execute().data)

    def update_job(self, job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        r = self._sb.table("jobs").update(fields).eq("id", job_id).execute()
        return r.data[0] if r.data else None

    def claim_running(self, job_id: str) -> bool:
        # Conditional update: only succeeds if still queued (atomic in Postgres).
        r = (self._sb.table("jobs").update({"status": "running"})
             .eq("id", job_id).eq("status", "queued").execute())
        return bool(r.data)

    def find_job_by_sha(self, user_id: str, sha: str) -> Optional[Dict[str, Any]]:
        if not sha:
            return None
        r = (self._sb.table("jobs").select("*").eq("user_id", user_id)
             .eq("content_sha256", sha).in_("status", ["done", "running", "queued"])
             .order("created_at", desc=True).limit(1).execute())
        return r.data[0] if r.data else None

    def get_subscription(self, user_id: str) -> Dict[str, Any]:
        r = self._sb.table("subscriptions").select("*").eq("user_id", user_id).limit(1).execute()
        return r.data[0] if r.data else {"user_id": user_id, "plan": "free", "status": "active"}

    def set_subscription(self, user_id: str, **fields: Any) -> Dict[str, Any]:
        row = {"user_id": user_id, **fields}
        return self._sb.table("subscriptions").upsert(row).execute().data[0]

    def get_usage(self, user_id: str, period: str) -> Dict[str, Any]:
        r = (self._sb.table("usage").select("*").eq("user_id", user_id)
             .eq("period", period).limit(1).execute())
        return r.data[0] if r.data else {"user_id": user_id, "period": period,
                                         "videos_processed": 0, "seconds_processed": 0}

    def incr_usage(self, user_id: str, period: str, videos: int, seconds: int) -> Dict[str, Any]:
        cur = self.get_usage(user_id, period)
        row = {"user_id": user_id, "period": period,
               "videos_processed": cur["videos_processed"] + videos,
               "seconds_processed": cur["seconds_processed"] + seconds}
        return self._sb.table("usage").upsert(row).execute().data[0]

    def list_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        return (self._sb.table("jobs").select("*")
                .order("created_at", desc=True).limit(limit).execute().data)

    def delete_user_data(self, user_id: str) -> Dict[str, int]:
        # ON DELETE CASCADE from auth.users handles jobs/analyses/usage/subscriptions;
        # deleting the auth user is the authoritative GDPR erase.
        self._sb.auth.admin.delete_user(user_id)
        return {"deleted_user": 1}
