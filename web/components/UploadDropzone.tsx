"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { api, ApiError } from "@/lib/api";

async function sha256(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default function UploadDropzone() {
  const router = useRouter();
  const supabase = createClient();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [stage, setStage] = useState("");

  async function handle(file: File) {
    setBusy(true);
    setErr(null);
    try {
      const { data: u } = await supabase.auth.getUser();
      const uid = u.user?.id;
      if (!uid) throw new Error("not signed in");

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
          object_key: key,
          content_sha256: hash,
          size_bytes: file.size,
          tracker: "supervision",
          backend: "rule",
        }),
      });
      router.push(`/jobs/${res.job_id}`);
    } catch (e: any) {
      setErr(e instanceof ApiError && e.isQuota ? "You've hit your monthly quota — upgrade to keep analyzing." : e.message);
      setBusy(false);
    }
  }

  return (
    <div>
      <label className="flex h-48 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-ink bg-paper-2 hover:border-flame">
        <input
          type="file" accept="video/*" className="hidden" disabled={busy}
          onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
        />
        <span className="text-ink/50">
          {busy ? `${stage}…` : "Drop a match video or click to choose (≤300 MB, ≤2 min)"}
        </span>
      </label>
      {err && <p className="mt-3 text-sm text-flame">{err}</p>}
    </div>
  );
}
