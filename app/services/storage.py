"""Object storage (Supabase Storage) — signed upload/download URLs.

The browser uploads the video **directly** to Supabase Storage via a signed URL
minted here, so the API never streams large video bodies. The Modal worker pulls
the object and writes outputs back. ``FakeStorage`` makes this testable offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class Storage(ABC):
    @abstractmethod
    def signed_upload_url(self, bucket: str, key: str, expires_in: int = 900) -> Dict[str, str]: ...
    @abstractmethod
    def signed_download_url(self, bucket: str, key: str, expires_in: int = 900) -> str: ...
    @abstractmethod
    def download_file(self, bucket: str, key: str, dest_path: str) -> str:
        """Download object to a local path (used by the Modal worker)."""
    @abstractmethod
    def upload_file(self, bucket: str, key: str, src_path: str) -> str:
        """Upload a local file; return the object key (used by the Modal worker)."""


class FakeStorage(Storage):
    def __init__(self) -> None:
        self.uploaded: Dict[str, str] = {}

    def signed_upload_url(self, bucket: str, key: str, expires_in: int = 900) -> Dict[str, str]:
        return {"url": f"https://fake.storage/{bucket}/{key}?upload=1", "token": "fake", "key": key}

    def signed_download_url(self, bucket: str, key: str, expires_in: int = 900) -> str:
        return f"https://fake.storage/{bucket}/{key}?download=1"

    def download_file(self, bucket: str, key: str, dest_path: str) -> str:
        return dest_path  # no-op for tests

    def upload_file(self, bucket: str, key: str, src_path: str) -> str:
        self.uploaded[f"{bucket}/{key}"] = src_path
        return key


class SupabaseStorage(Storage):
    """Lazy-imports the Supabase client; uses the service-role key."""

    def __init__(self, url: str, service_key: str):
        from supabase import create_client  # lazy
        self._sb = create_client(url, service_key)

    def signed_upload_url(self, bucket: str, key: str, expires_in: int = 900) -> Dict[str, str]:
        res = self._sb.storage.from_(bucket).create_signed_upload_url(key)
        return {"url": res["signed_url"], "token": res.get("token", ""), "key": key}

    def signed_download_url(self, bucket: str, key: str, expires_in: int = 900) -> str:
        res = self._sb.storage.from_(bucket).create_signed_url(key, expires_in)
        return res["signedURL"]

    def download_file(self, bucket: str, key: str, dest_path: str) -> str:
        data = self._sb.storage.from_(bucket).download(key)
        with open(dest_path, "wb") as f:
            f.write(data)
        return dest_path

    def upload_file(self, bucket: str, key: str, src_path: str) -> str:
        with open(src_path, "rb") as f:
            self._sb.storage.from_(bucket).upload(key, f, {"upsert": "true"})
        return key
