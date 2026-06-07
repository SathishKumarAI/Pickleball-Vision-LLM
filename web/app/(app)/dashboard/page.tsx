"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import QuotaMeter from "@/components/QuotaMeter";

type Job = { id: string; status: string; created_at?: string };

export default function Dashboard() {
  const [jobs, setJobs] = useState<Job[]>([]);
  useEffect(() => {
    api<{ jobs: Job[] }>("/jobs").then((r) => setJobs(r.jobs.slice(0, 5))).catch(() => {});
  }, []);

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <Link href="/upload" className="rounded-lg btn-primary">
          Analyze a match
        </Link>
      </div>

      <QuotaMeter />

      <section>
        <h2 className="mb-3 font-semibold">Recent analyses</h2>
        {jobs.length === 0 ? (
          <p className="text-slate-500">No analyses yet — upload your first match.</p>
        ) : (
          <ul className="divide-y divide-white/10 card">
            {jobs.map((j) => (
              <li key={j.id} className="flex items-center justify-between px-4 py-3">
                <Link href={j.status === "done" ? `/analyses/${j.id}` : `/jobs/${j.id}`} className="text-court">
                  {j.id.slice(0, 8)}…
                </Link>
                <span className="text-sm capitalize text-slate-500">{j.status}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
