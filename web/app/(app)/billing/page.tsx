"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/ui";

const PLANS = [
  { id: "free", name: "Free", price: "$0", limit: "3 videos / month" },
  { id: "starter", name: "Starter", price: "$12", limit: "50 videos / month" },
  { id: "pro", name: "Pro", price: "$39", limit: "500 videos / month" },
];

export default function BillingPage() {
  const [busy, setBusy] = useState<string | null>(null);
  const [plan, setPlan] = useState<string>("free");

  useEffect(() => {
    api<{ plan: string }>("/account/usage").then((u) => setPlan(u.plan)).catch(() => {});
  }, []);

  async function checkout(p: string) {
    setBusy(p);
    try {
      const { url } = await api<{ url: string }>("/billing/checkout", {
        method: "POST",
        body: JSON.stringify({
          plan: p,
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
    try {
      const { url } = await api<{ url: string }>("/billing/portal", {
        method: "POST",
        body: JSON.stringify({ return_url: `${window.location.origin}/billing` }),
      });
      window.location.href = url;
    } catch { /* no customer yet */ }
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Billing" sub={`You're on the ${plan} plan.`} />

      <div className="grid gap-6 md:grid-cols-3">
        {PLANS.map((p) => {
          const current = p.id === plan;
          return (
            <div key={p.id} className={`panel p-6 ${current ? "border-ink ring-2 ring-lime" : ""}`}>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">{p.name}</h3>
                {current && <span className="field-label text-[#2e7d32]">Current</span>}
              </div>
              <p className="mt-2 text-3xl font-bold">{p.price}<span className="text-sm font-normal text-ink/50">/mo</span></p>
              <p className="mt-1 text-sm text-ink/60">{p.limit}</p>
              {p.id !== "free" && (
                <button
                  onClick={() => checkout(p.id)}
                  disabled={current || busy === p.id}
                  className="btn-primary mt-5 w-full disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {current ? "Active" : busy === p.id ? "…" : "Upgrade"}
                </button>
              )}
            </div>
          );
        })}
      </div>

      <button onClick={portal} className="ink-link text-sm">Manage subscription &amp; invoices →</button>
    </div>
  );
}
