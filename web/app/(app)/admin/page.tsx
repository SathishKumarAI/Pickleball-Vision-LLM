"use client";

import { useEffect, useState, useCallback } from "react";
import { api, ApiError } from "@/lib/api";

type LogRec = {
  ts: string; level: string; logger: string; msg: string;
  request_id?: string; [k: string]: any;
};
type Job = { id: string; user_id: string; status: string; created_at?: string };

const LEVEL_COLOR: Record<string, string> = {
  DEBUG: "text-slate-500", INFO: "text-green-400", WARNING: "text-yellow-400",
  ERROR: "text-red-400", CRITICAL: "text-red-300 font-bold",
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

  if (denied) return <p className="text-red-600">Admin access only.</p>;

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
          <select value={level} onChange={(e) => setLevel(e.target.value)} className="rounded border border-slate-300 px-2 py-1 text-sm">
            <option value="">all levels</option>
            {["INFO", "WARNING", "ERROR"].map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </div>
        <div className="h-96 overflow-auto rounded-lg bg-slate-950 p-4 font-mono text-xs leading-relaxed">
          {logs.length === 0 && <p className="text-slate-500">no logs</p>}
          {logs.map((r, i) => (
            <div key={i} className={LEVEL_COLOR[r.level] || "text-slate-300"}>
              <span className="text-slate-600">{r.ts?.slice(11, 19)} </span>
              <span>{(STATUS_WORD[r.level] || r.level).padStart(6)}: </span>
              <span className="text-slate-400">[{r.logger}] </span>
              <span>{r.msg} </span>
              <span className="text-slate-500">{extras(r)}</span>
              {r.request_id && r.request_id !== "-" && (
                <span className="text-slate-600"> (req={r.request_id})</span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* All jobs */}
      <section>
        <h2 className="mb-2 font-semibold">All jobs ({jobs.length})</h2>
        <div className="overflow-auto rounded-lg border border-slate-200 bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-slate-500">
              <tr><th className="p-2">Job</th><th className="p-2">User</th><th className="p-2">Status</th><th className="p-2">Created</th></tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="border-t border-slate-100">
                  <td className="p-2 font-mono">{j.id.slice(0, 8)}</td>
                  <td className="p-2 font-mono">{j.user_id?.slice(0, 8)}</td>
                  <td className="p-2 capitalize">{j.status}</td>
                  <td className="p-2 text-slate-400">{j.created_at?.slice(0, 16).replace("T", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
