"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const router = useRouter();
  const supabase = createClient();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function deleteAccount() {
    setBusy(true);
    try {
      await api("/account", { method: "DELETE" });
      await supabase.auth.signOut();
      router.push("/");
    } catch {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Settings</h1>

      <section className="rounded-xl border border-flame bg-paper-2 p-6">
        <h2 className="font-semibold text-flame">Delete account</h2>
        <p className="mt-2 text-sm text-ink/60">
          Permanently erase your account, videos, and analyses (GDPR). This cannot be undone.
        </p>
        {!confirming ? (
          <button onClick={() => setConfirming(true)} className="mt-4 rounded-lg border border-flame px-4 py-2 text-sm font-semibold text-flame">
            Delete my account
          </button>
        ) : (
          <div className="mt-4 flex gap-3">
            <button onClick={deleteAccount} disabled={busy} className="rounded-lg bg-flame px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
              {busy ? "Erasing…" : "Yes, delete everything"}
            </button>
            <button onClick={() => setConfirming(false)} className="text-sm text-ink/50">Cancel</button>
          </div>
        )}
      </section>
    </div>
  );
}
