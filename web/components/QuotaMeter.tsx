"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Quota = { plan: string; videos_used: number; videos_limit: number; remaining: number };

export default function QuotaMeter() {
  const [q, setQ] = useState<Quota | null>(null);
  useEffect(() => {
    api<Quota>("/account/usage").then(setQ).catch(() => {});
  }, []);
  if (!q) return null;
  const pct = q.videos_limit ? Math.min(100, (q.videos_used / q.videos_limit) * 100) : 0;
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium capitalize">{q.plan} plan</span>
        <span className="text-sm text-slate-500">{q.videos_used} / {q.videos_limit} this month</span>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div className="h-full bg-grad-brand" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
