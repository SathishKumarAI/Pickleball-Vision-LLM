"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import CoachingReport from "@/components/CoachingReport";
import { PageHeader } from "@/components/ui";

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<{ summary?: string }>(`/jobs/${params.id}/result`).then(setResult).catch((e) => setErr(e.message));
    api<{ url: string }>(`/jobs/${params.id}/video`).then((r) => setVideoUrl(r.url)).catch(() => {});
  }, [params.id]);

  if (err) {
    return (
      <div className="space-y-6">
        <PageHeader title="Match analysis" />
        <div className="panel p-6 text-sm text-ink/70">
          {err}. <Link href="/history" className="ink-link">Back to history →</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Match analysis" sub="Annotated replay + coaching report." />

      {videoUrl ? (
        <video src={videoUrl} controls className="w-full rounded-xl border-2 border-ink bg-black" />
      ) : result ? (
        <div className="flex h-64 items-center justify-center rounded-xl border-2 border-dashed border-ink/30 bg-paper-2 text-sm text-ink/50">
          Annotated video unavailable
        </div>
      ) : (
        <div className="skeleton aspect-video w-full rounded-xl" />
      )}

      {result ? (
        <CoachingReport analysis={result} />
      ) : (
        <div className="space-y-3">
          <div className="skeleton h-24 w-full rounded-xl" />
          <div className="skeleton h-24 w-full rounded-xl" />
        </div>
      )}
    </div>
  );
}
