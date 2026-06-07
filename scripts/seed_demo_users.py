"""Seed temporary demo users into Supabase (admin API).

Run server-side with the Supabase service-role key:
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python scripts/seed_demo_users.py

Creates 3 confirmed users (admin / coach / player), sets roles + plans, and prints
an Ansible-style recap. Credentials are documented in docs/DEMO_ACCESS.md.

These are DEMO creds for evaluation only — rotate/disable before real launch.
"""

from __future__ import annotations

import os
import sys

DEMO_USERS = [
    {"email": "admin@pvdemo.app",  "password": "Pv-O4JkCbiWb8fGOL", "role": "admin", "plan": "pro",     "name": "Demo Admin"},
    {"email": "coach@pvdemo.app",  "password": "Pv-Dlek2sdKKnfXvn", "role": "user",  "plan": "pro",     "name": "Demo Coach"},
    {"email": "player@pvdemo.app", "password": "Pv-uO0GVeCwPyPC4i", "role": "user",  "plan": "free",    "name": "Demo Player"},
]

OK, CHANGED, FAILED = "\033[32m", "\033[33m", "\033[31m"
RESET = "\033[0m"


def main() -> int:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print(f"{FAILED}fatal: set SUPABASE_URL and SUPABASE_SERVICE_KEY{RESET}")
        return 2
    from supabase import create_client

    sb = create_client(url, key)
    print("\nPLAY [seed demo users] " + "*" * 50 + "\n")
    recap = {"ok": 0, "changed": 0, "failed": 0}

    for u in DEMO_USERS:
        task = f"TASK [create {u['email']}]"
        print(task + " " + "*" * max(2, 60 - len(task)))
        try:
            res = sb.auth.admin.create_user({
                "email": u["email"], "password": u["password"], "email_confirm": True,
                "app_metadata": {"role": u["role"]},
                "user_metadata": {"full_name": u["name"]},
            })
            uid = res.user.id
            # set subscription plan (profile + subscription rows are auto-created by trigger)
            sb.table("subscriptions").upsert(
                {"user_id": uid, "plan": u["plan"], "status": "active"}).execute()
            print(f"{CHANGED}changed: [{u['email']}] => "
                  f"{{role={u['role']}, plan={u['plan']}, id={uid[:8]}}}{RESET}\n")
            recap["changed"] += 1
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "already" in msg.lower() or "registered" in msg.lower():
                print(f"{OK}ok: [{u['email']}] => (already exists, skipped){RESET}\n")
                recap["ok"] += 1
            else:
                print(f"{FAILED}failed: [{u['email']}] => {msg}{RESET}\n")
                recap["failed"] += 1

    print("PLAY RECAP " + "*" * 60)
    print(f"demo-users : ok={recap['ok']}  changed={recap['changed']}  "
          f"failed={recap['failed']}\n")
    return 1 if recap["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
