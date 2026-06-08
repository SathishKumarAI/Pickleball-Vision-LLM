"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/ui";

export default function SettingsPage() {
  const router = useRouter();
  const supabase = createClient();
  const [email, setEmail] = useState<string>("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => setEmail(data.user?.email ?? ""));
  }, [supabase]);

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
      <PageHeader title="Settings" sub="Account and data." />

      <section className="panel p-6">
        <h2 className="field-label">Account</h2>
        <p className="mt-2 text-sm">{email || "—"}</p>
      </section>

      <section className="rounded-xl border-2 border-flame bg-flame/5 p-6">
        <h2 className="font-semibold text-flame">Danger zone</h2>
        <p className="mt-1 max-w-md text-sm text-ink/70">
          Permanently erase your account, uploaded videos, and analyses (GDPR). This
          cannot be undone.
        </p>
        {!confirming ? (
          <button onClick={() => setConfirming(true)}
                  className="mt-4 rounded-lg border-2 border-flame px-4 py-2 text-sm font-semibold text-flame transition hover:bg-flame hover:text-paper">
            Delete my account
          </button>
        ) : (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button onClick={deleteAccount} disabled={busy}
                    className="rounded-lg bg-flame px-4 py-2 text-sm font-semibold text-paper transition hover:bg-flame-dim disabled:opacity-50">
              {busy ? "Erasing…" : "Yes, delete everything"}
            </button>
            <button onClick={() => setConfirming(false)} className="text-sm font-semibold text-ink/60 hover:text-ink">
              Cancel
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
