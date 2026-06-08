"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import JobProgress from "@/components/JobProgress";
import { PageHeader } from "@/components/ui";

export default function JobPage({ params }: { params: { id: string } }) {
  const [cancelled, setCancelled] = useState(false);

  async function cancel() {
    setCancelled(true);
    try {
      await api(`/jobs/${params.id}/cancel`, { method: "POST" });
    } catch {
      /* best-effort */
    }
  }

  return (
    <div className="mx-auto max-w-xl space-y-6">
      <PageHeader title="Processing your match" sub="This usually takes a couple of minutes. You can leave this page — we'll keep going." />
      <JobProgress jobId={params.id} />
      {!cancelled && (
        <button onClick={cancel} className="btn-ghost py-2 text-sm">Cancel analysis</button>
      )}
    </div>
  );
}
