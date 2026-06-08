"use client";

import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "@/lib/api";

type LogRec = {
  ts: string; level: string; logger: string; msg: string;
  request_id?: string; [k: string]: any;
};
type Job = { id: string; user_id: string; status: string; created_at?: string };

const LEVEL_COLOR: Record<string, string> = {
  DEBUG: "text-paper/40", INFO: "text-lime", WARNING: "text-yellow-300",
  ERROR: "text-flame", CRITICAL: "text-red-300 font-bold",
};
const STATUS_WORD: Record<string, string> = {
  DEBUG: "skip", INFO: "ok", WARNING: "warn", ERROR: "failed", CRITICAL: "FATAL",
};

function extras(r: LogRec) {
  const skip = new Set(["ts", "level", "logger", "msg", "request_id"]);
  return Object.entries(r).filter(([k]) => !skip.has(k))
    .map(([k, v]) => `${k}=${v}`).join(" ");
}

export default function AdminPage() {
  const [logs, setLogs] = useState<LogRec[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [level, setLevel] = useState("");
  const [denied, setDenied] = useState(false);
  const [auto, setAuto] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const q = level ? `?level=${level}&limit=300` : "?limit=300";
      const [l, j] = await Promise.all([
        api<{ logs: LogRec[] }>(`/admin/logs${q}`),
        api<{ jobs: Job[] }>("/admin/jobs"),
      ]);
      setLogs(l.logs);
      setJobs(j.jobs);
    } catch (e) {
      if (e instanceof ApiError && e.status === 403) setDenied(true);
    }
  }, [level]);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!auto) return;
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [auto, refresh]);

  if (denied) return <p className="text-flame">Admin access only.</p>;

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Admin</h1>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
          auto-refresh
        </label>
      </div>

      {/* Logs — terminal / CLI style */}
      <section>
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-semibold">Application logs</h2>
          <select value={level} onChange={(e) => setLevel(e.target.value)} className="rounded border border-ink/20 px-2 py-1 text-sm">
            <option value="">all levels</option>
            {["INFO", "WARNING", "ERROR"].map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>
        <div className="h-96 overflow-auto rounded-lg border-2 border-ink bg-ink p-4 font-mono text-xs leading-relaxed shadow-hard">
          {logs.length === 0 && <p className="text-paper/40">no logs</p>}
          {logs.map((r, i) => (
            <div key={i} className={LEVEL_COLOR[r.level] || "text-paper/80"}>
              <span className="text-paper/40">{r.ts?.slice(11, 19)} </span>
              <span>{(STATUS_WORD[r.level] || r.level).padStart(6)}: </span>
              <span className="text-paper/40">[{r.logger}] </span>
              <span>{r.msg} </span>
              <span className="text-paper/30">{extras(r)}</span>
              {r.request_id && r.request_id !== "-" && (
                <span className="text-paper/40"> (req={r.request_id})</span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* All jobs */}
      <section>
        <h2 className="mb-2 font-semibold">All jobs ({jobs.length})</h2>
        <div className="overflow-auto panel">
          <table className="w-full text-sm">
            <thead className="bg-paper-2 text-left text-ink/50">
              <tr><th className="p-2">Job</th><th className="p-2">User</th><th className="p-2">Status</th><th className="p-2">Created</th></tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="border-t border-ink/10">
                  <td className="p-2 font-mono">{j.id.slice(0, 8)}</td>
                  <td className="p-2 font-mono">{j.user_id?.slice(0, 8)}</td>
                  <td className="p-2 capitalize">{j.status}</td>
                  <td className="p-2 text-ink/60">{j.created_at?.slice(0, 16).replace("T", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
