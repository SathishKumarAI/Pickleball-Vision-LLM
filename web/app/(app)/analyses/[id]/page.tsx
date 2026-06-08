"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import CoachingReport from "@/components/CoachingReport";

export default function AnalysisPage({ params }: { params: { id: string } }) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api<{ summary?: string }>(`/jobs/${params.id}/result`).then(setResult).catch((e) => setErr(e.message));
    api<{ url: string }>(`/jobs/${params.id}/video`).then((r) => setVideoUrl(r.url)).catch(() => {});
  }, [params.id]);

  if (err) return <p className="text-flame">{err}</p>;
  if (!result) return <p className="text-ink/50">Loading…</p>;

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Match analysis</h1>
      {videoUrl ? (
        <video src={videoUrl} controls className="w-full rounded-xl border border-ink/15 bg-black" />
      ) : (
        <div className="flex h-64 items-center justify-center rounded-xl border border-ink/15 bg-ink/10 text-ink/60">
          Annotated video unavailable
        </div>
      )}
      <CoachingReport analysis={result} />
    </div>
  );
}
