"use client";

import { useState } from "react";
import { api } from "@/lib/api";

const PLANS = [
  { id: "starter", name: "Starter", price: "$12", limit: "50 videos / month" },
  { id: "pro", name: "Pro", price: "$39", limit: "500 videos / month" },
];

export default function BillingPage() {
  const [busy, setBusy] = useState<string | null>(null);

  async function checkout(plan: string) {
    setBusy(plan);
    try {
      const { url } = await api<{ url: string }>("/billing/checkout", {
        method: "POST",
        body: JSON.stringify({
          plan,
          success_url: `${window.location.origin}/dashboard`,
          cancel_url: `${window.location.origin}/billing`,
        }),
      });
      window.location.href = url;
    } catch {
      setBusy(null);
    }
  }

  async function portal() {
    const { url } = await api<{ url: string }>("/billing/portal", {
      method: "POST",
      body: JSON.stringify({ return_url: `${window.location.origin}/billing` }),
    });
    window.location.href = url;
  }

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Billing</h1>
      <div className="grid gap-6 sm:grid-cols-2">
        {PLANS.map((p) => (
          <div key={p.id} className="panel p-6 text-center">
            <h3 className="font-semibold">{p.name}</h3>
            <p className="mt-2 text-3xl font-bold">{p.price}</p>
            <p className="mt-1 text-sm text-ink/60">{p.limit}</p>
            <button
              onClick={() => checkout(p.id)} disabled={busy === p.id}
              className="mt-4 w-full rounded-lg btn-primary w-full disabled:opacity-50"
            >
              {busy === p.id ? "…" : "Upgrade"}
            </button>
          </div>
        ))}
      </div>
      <button onClick={portal} className="text-sm font-semibold text-flame">Manage existing subscription →</button>
    </div>
  );
}
