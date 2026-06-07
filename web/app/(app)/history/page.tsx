"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Job = { id: string; status: string; created_at?: string; backend?: string };

export default function HistoryPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  useEffect(() => {
    api<{ jobs: Job[] }>("/jobs").then((r) => setJobs(r.jobs)).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Match history</h1>
      {jobs.length === 0 ? (
        <p className="text-slate-500">Nothing here yet.</p>
      ) : (
        <ul className="divide-y divide-white/10 card">
          {jobs.map((j) => (
            <li key={j.id} className="flex items-center justify-between px-4 py-3">
              <Link href={j.status === "done" ? `/analyses/${j.id}` : `/jobs/${j.id}`} className="text-court">
                {j.id.slice(0, 8)}…
              </Link>
              <span className="text-xs text-slate-400">{j.created_at?.slice(0, 16).replace("T", " ")}</span>
              <span className="text-sm capitalize text-slate-500">{j.status}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
