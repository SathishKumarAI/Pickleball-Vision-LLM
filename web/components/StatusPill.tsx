// Semantic job-status pill (dot + label). Restrained color: accent only for state.
const MAP: Record<string, { label: string; cls: string; dot: string }> = {
  queued:     { label: "Queued",     cls: "border-ink/20 text-ink/60",        dot: "bg-ink/40" },
  running:    { label: "Analyzing",  cls: "border-ink text-ink bg-lime/30",   dot: "bg-lime-dim animate-pulse" },
  done:       { label: "Ready",      cls: "border-ink text-ink",              dot: "bg-[#2e7d32]" },
  error:      { label: "Failed",     cls: "border-flame text-flame",          dot: "bg-flame" },
  cancelling: { label: "Cancelling", cls: "border-ink/20 text-ink/60",        dot: "bg-ink/40" },
  cancelled:  { label: "Cancelled",  cls: "border-ink/15 text-ink/45",        dot: "bg-ink/30" },
};

export default function StatusPill({ status }: { status: string }) {
  const s = MAP[status] ?? { label: status, cls: "border-ink/20 text-ink/60", dot: "bg-ink/40" };
  return (
    <span className={`pill ${s.cls}`}>
      <span className={`pill-dot ${s.dot}`} />
      {s.label}
    </span>
  );
}
