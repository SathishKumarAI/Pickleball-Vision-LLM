"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import StatusPill from "@/components/StatusPill";
import { PageHeader, SkeletonRows, EmptyState } from "@/components/ui";

type Job = { id: string; status: string; created_at?: string; backend?: string };

export default function HistoryPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  useEffect(() => {
    api<{ jobs: Job[] }>("/jobs").then((r) => setJobs(r.jobs)).catch(() => setJobs([]));
  }, []);

  return (
    <div className="space-y-8">
      <PageHeader title="Match history" sub="Every clip you've analyzed." />

      {jobs === null ? (
        <SkeletonRows rows={6} />
      ) : jobs.length === 0 ? (
        <EmptyState
          title="Nothing here yet"
          body="Your analyzed matches will appear here. Upload a clip to get started."
          action={{ href: "/upload", label: "Analyze a match" }}
        />
      ) : (
        <ul className="panel divide-y divide-ink/10">
          {jobs.map((j) => (
            <li key={j.id}>
              <Link
                href={j.status === "done" ? `/analyses/${j.id}` : `/jobs/${j.id}`}
                className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors hover:bg-paper-2"
              >
                <span className="font-mono text-sm">{j.id.slice(0, 8)}</span>
                <div className="flex items-center gap-4">
                  <span className="hidden text-xs text-ink/45 sm:inline">
                    {j.created_at?.slice(0, 16).replace("T", " ")}
                  </span>
                  <StatusPill status={j.status} />
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
