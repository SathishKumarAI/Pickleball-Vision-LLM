"use client";

type Analysis = {
  summary?: string | null;
  metrics?: Record<string, any> | null;
  states?: { action: string }[] | null;
};

export default function CoachingReport({ analysis }: { analysis: Analysis }) {
  const actions = (analysis.states || []).reduce<Record<string, number>>((acc, s) => {
    acc[s.action] = (acc[s.action] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      <section className="rounded-xl border border-slate-200 bg-white p-6">
        <h2 className="font-semibold">Coaching summary</h2>
        <p className="mt-2 whitespace-pre-line text-slate-700">
          {analysis.summary || "No summary available."}
        </p>
      </section>

      {Object.keys(actions).length > 0 && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="font-semibold">Action breakdown</h2>
          <ul className="mt-3 space-y-1 text-sm">
            {Object.entries(actions).map(([action, n]) => (
              <li key={action} className="flex justify-between">
                <span className="capitalize">{action.replace(/-/g, " ")}</span>
                <span className="text-slate-500">{n} frames</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {analysis.metrics && (
        <section className="rounded-xl border border-slate-200 bg-white p-6">
          <h2 className="font-semibold">Court analytics</h2>
          <pre className="mt-2 overflow-x-auto text-xs text-slate-600">
            {JSON.stringify(analysis.metrics, null, 2)}
          </pre>
        </section>
      )}
    </div>
  );
}
