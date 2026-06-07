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


class FakeStorage(Storage):
    def signed_upload_url(self, bucket: str, key: str, expires_in: int = 900) -> Dict[str, str]:
        return {"url": f"https://fake.storage/{bucket}/{key}?upload=1", "token": "fake", "key": key}

    def signed_download_url(self, bucket: str, key: str, expires_in: int = 900) -> str:
        return f"https://fake.storage/{bucket}/{key}?download=1"


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
