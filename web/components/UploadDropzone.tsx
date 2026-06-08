"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { api, ApiError } from "@/lib/api";

const MAX_MB = 300;
const STAGES = ["hashing", "uploading", "queuing"] as const;
const STAGE_LABEL: Record<string, string> = {
  hashing: "Fingerprinting", uploading: "Uploading", queuing: "Queuing analysis",
};

async function sha256(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default function UploadDropzone() {
  const router = useRouter();
  const supabase = createClient();
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [stage, setStage] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");

  const handle = useCallback(async (file: File) => {
    setErr(null);
    if (!file.type.startsWith("video/")) return setErr("That's not a video file.");
    if (file.size > MAX_MB * 1024 * 1024) return setErr(`File is too large (max ${MAX_MB} MB).`);

    setBusy(true);
    setFileName(file.name);
    try {
      const { data: u } = await supabase.auth.getUser();
      const uid = u.user?.id;
      if (!uid) throw new Error("Not signed in.");

      setStage("hashing");
      const hash = await sha256(file);
      const key = `${uid}/${crypto.randomUUID()}/${file.name}`;

      setStage("uploading");
      const { error } = await supabase.storage.from("uploads").upload(key, file, {
        contentType: file.type || "video/mp4",
      });
      if (error) throw error;

      setStage("queuing");
      const res = await api<{ job_id: string }>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          object_key: key, content_sha256: hash, size_bytes: file.size,
          tracker: "supervision", backend: "rule",
        }),
      });
      router.push(`/jobs/${res.job_id}`);
    } catch (e: any) {
      setErr(e instanceof ApiError && e.isQuota
        ? "You've hit your monthly quota — upgrade to keep analyzing."
        : e.message || "Upload failed.");
      setBusy(false);
    }
  }, [router, supabase]);

  if (busy) {
    const idx = STAGES.indexOf(stage as (typeof STAGES)[number]);
    return (
      <div className="panel p-6">
        <p className="text-sm text-ink/60">Working on <span className="font-mono">{fileName}</span></p>
        <ol className="mt-4 space-y-2">
          {STAGES.map((s, i) => (
            <li key={s} className="flex items-center gap-3 text-sm">
              <span className={`pill-dot h-2 w-2 ${i < idx ? "bg-[#2e7d32]" : i === idx ? "bg-lime-dim animate-pulse" : "bg-ink/20"}`} />
              <span className={i <= idx ? "text-ink" : "text-ink/40"}>{STAGE_LABEL[s]}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  return (
    <div>
      <label
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files?.[0]; if (f) handle(f); }}
        className={`flex h-56 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors
          ${drag ? "border-flame bg-lime/10" : "border-ink/40 bg-paper-2 hover:border-ink hover:bg-paper"}`}
      >
        <input type="file" accept="video/*" className="hidden"
               onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])} />
        <div className="text-3xl">📹</div>
        <p className="mt-3 font-semibold">Drop a match video, or click to choose</p>
        <p className="mt-1 font-mono text-xs text-ink/50">MP4 · ≤ {MAX_MB} MB · ≤ 2 min</p>
      </label>
      {err && (
        <div className="mt-3 rounded-lg border border-flame bg-flame/5 px-4 py-3 text-sm font-semibold text-flame" role="alert">
          {err}
        </div>
      )}
    </div>
  );
}
