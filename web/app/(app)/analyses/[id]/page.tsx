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

  if (err) return <p className="text-red-400">{err}</p>;
  if (!result) return <p className="text-slate-500">Loading…</p>;

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Match analysis</h1>
      {videoUrl ? (
        <video src={videoUrl} controls className="w-full rounded-xl border border-white/10 bg-black" />
      ) : (
        <div className="flex h-64 items-center justify-center rounded-xl border border-white/10 bg-white/10 text-slate-400">
          Annotated video unavailable
        </div>
      )}
      <CoachingReport analysis={result} />
    </div>
  );
}
