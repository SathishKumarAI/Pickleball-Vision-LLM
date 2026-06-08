// Small product-register primitives: page header, skeleton rows, empty state.
import Link from "next/link";

export function PageHeader({ title, sub, action }: {
  title: string; sub?: string; action?: { href: string; label: string };
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink/15 pb-5">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
        {sub && <p className="mt-1 text-sm text-ink/60">{sub}</p>}
      </div>
      {action && <Link href={action.href} className="btn-primary">{action.label}</Link>}
    </div>
  );
}

export function SkeletonRows({ rows = 4 }: { rows?: number }) {
  return (
    <div className="panel divide-y divide-ink/10">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex items-center justify-between px-4 py-3.5">
          <div className="skeleton h-4 w-28" />
          <div className="skeleton h-5 w-20 rounded-full" />
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ icon = "🏓", title, body, action }: {
  icon?: string; title: string; body: string; action?: { href: string; label: string };
}) {
  return (
    <div className="panel flex flex-col items-center px-6 py-14 text-center">
      <div className="text-4xl">{icon}</div>
      <h3 className="mt-3 text-lg font-semibold">{title}</h3>
      <p className="mt-1 max-w-sm text-sm text-ink/60">{body}</p>
      {action && <Link href={action.href} className="btn-primary mt-5">{action.label}</Link>}
    </div>
  );
}
