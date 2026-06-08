"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import QuotaMeter from "@/components/QuotaMeter";
import StatusPill from "@/components/StatusPill";
import { PageHeader, SkeletonRows, EmptyState } from "@/components/ui";

type Job = { id: string; status: string; created_at?: string };

export default function Dashboard() {
  const [jobs, setJobs] = useState<Job[] | null>(null);

  useEffect(() => {
    api<{ jobs: Job[] }>("/jobs")
      .then((r) => setJobs(r.jobs.slice(0, 6)))
      .catch(() => setJobs([]));
  }, []);

  return (
    <div className="space-y-8">
      <PageHeader
        title="Dashboard"
        sub="Your recent analyses and monthly usage."
        action={{ href: "/upload", label: "Analyze a match →" }}
      />

      <QuotaMeter />

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="field-label">Recent analyses</h2>
          {jobs && jobs.length > 0 && (
            <Link href="/history" className="ink-link text-xs">View all →</Link>
          )}
        </div>

        {jobs === null ? (
          <SkeletonRows rows={4} />
        ) : jobs.length === 0 ? (
          <EmptyState
            title="No analyses yet"
            body="Upload a match clip and we'll track the players, the ball, and the rally — then hand you a coaching report."
            action={{ href: "/upload", label: "Upload your first match" }}
          />
        ) : (
          <ul className="panel divide-y divide-ink/10">
            {jobs.map((j) => (
              <li key={j.id}>
                <Link
                  href={j.status === "done" ? `/analyses/${j.id}` : `/jobs/${j.id}`}
                  className="flex items-center justify-between px-4 py-3.5 transition-colors hover:bg-paper-2"
                >
                  <span className="font-mono text-sm">{j.id.slice(0, 8)}</span>
                  <div className="flex items-center gap-4">
                    {j.created_at && (
                      <span className="hidden text-xs text-ink/45 sm:inline">
                        {j.created_at.slice(0, 16).replace("T", " ")}
                      </span>
                    )}
                    <StatusPill status={j.status} />
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
