"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useJobRealtime } from "@/lib/hooks/useJobRealtime";
import StatusPill from "@/components/StatusPill";

const LABEL: Record<string, string> = {
  queued: "Waiting in line…",
  running: "Analyzing your match…",
  done: "Analysis ready",
  error: "Something went wrong",
  cancelling: "Cancelling…",
  cancelled: "Cancelled",
};

export default function JobProgress({ jobId }: { jobId: string }) {
  const job = useJobRealtime(jobId);
  const router = useRouter();
  const status = job?.status ?? "queued";

  useEffect(() => {
    if (job?.status === "done") {
      const t = setTimeout(() => router.push(`/analyses/${jobId}`), 700);
      return () => clearTimeout(t);
    }
  }, [job?.status, jobId, router]);

  const pct = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="panel p-6">
      <div className="flex items-center justify-between">
        <p className="font-medium">{LABEL[status] ?? status}</p>
        <StatusPill status={status} />
      </div>
      <div
        className="mt-4 h-2.5 w-full overflow-hidden rounded-full bg-ink/10"
        role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
      >
        <div
          className={`h-full bg-lime ${status === "running" ? "transition-[width] duration-500 ease-out" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="mt-2 font-mono text-xs text-ink/50">{job?.message || `${pct}%`}</p>
      {status === "error" && (
        <div className="mt-3 rounded-lg border border-flame bg-flame/5 px-4 py-3 text-sm font-semibold text-flame" role="alert">
          {job?.error}
        </div>
      )}
    </div>
  );
}
