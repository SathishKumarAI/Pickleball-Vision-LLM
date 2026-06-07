"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import JobProgress from "@/components/JobProgress";

export default function JobPage({ params }: { params: { id: string } }) {
  const [cancelled, setCancelled] = useState(false);

  async function cancel() {
    try {
      await api(`/jobs/${params.id}/cancel`, { method: "POST" });
      setCancelled(true);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="mx-auto max-w-xl space-y-6">
      <h1 className="text-2xl font-bold">Processing</h1>
      <JobProgress jobId={params.id} />
      {!cancelled && (
        <button onClick={cancel} className="text-sm text-slate-500 hover:text-red-400">
          Cancel
        </button>
      )}
    </div>
  );
}
