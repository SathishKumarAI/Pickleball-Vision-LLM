"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useJobRealtime } from "@/lib/hooks/useJobRealtime";

const LABEL: Record<string, string> = {
  queued: "Queued…", running: "Analyzing your match…", done: "Done!",
  error: "Something went wrong", cancelling: "Cancelling…", cancelled: "Cancelled",
};

export default function JobProgress({ jobId }: { jobId: string }) {
  const job = useJobRealtime(jobId);
  const router = useRouter();

  useEffect(() => {
    if (job?.status === "done") {
      const t = setTimeout(() => router.push(`/analyses/${jobId}`), 800);
      return () => clearTimeout(t);
    }
  }, [job?.status, jobId, router]);

  const pct = Math.round((job?.progress ?? 0) * 100);

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-6">
      <p className="font-medium">{LABEL[job?.status ?? "queued"] ?? job?.status}</p>
      <div className="mt-3 h-3 w-full overflow-hidden rounded-full bg-slate-100">
        <div className="h-full bg-brand transition-all" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-sm text-slate-500">{job?.message || `${pct}%`}</p>
      {job?.status === "error" && <p className="mt-2 text-sm text-red-600">{job.error}</p>}
    </div>
  );
}
