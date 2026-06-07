import Link from "next/link";

const FEATURES = [
  ["🎥", "Annotated replay", "Player & ball tracking, IDs, and ball trail rendered onto your video."],
  ["📊", "Court analytics", "Position heatmaps, kitchen usage, rally tempo, and shot labels."],
  ["🧠", "AI coaching", "Actionable feedback on strategy, positioning, and movement."],
];

const PLANS = [
  ["Free", "$0", "3 videos / month", false],
  ["Pro", "$39", "500 videos / month", true],
  ["Starter", "$12", "50 videos / month", false],
];

export default function Landing() {
  return (
    <main className="relative overflow-hidden">
      {/* glow orbs */}
      <div className="pointer-events-none absolute -top-40 left-1/2 h-96 w-96 -translate-x-1/2 rounded-full bg-court/20 blur-[120px]" />
      <div className="pointer-events-none absolute top-40 right-0 h-80 w-80 rounded-full bg-ball/10 blur-[120px]" />

      {/* nav */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <span className="font-display text-lg font-bold">🏓 Pickleball<span className="gradient-text">Vision</span></span>
        <Link href="/login" className="btn-ghost py-2">Sign in</Link>
      </header>

      {/* hero */}
      <section className="mx-auto max-w-6xl px-6 pb-24 pt-12 text-center">
        <div className="badge mx-auto animate-fadeup">⚡ Vision + LLM · results in minutes</div>
        <h1 className="mx-auto mt-6 max-w-3xl animate-fadeup font-display text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl">
          Your match, <span className="gradient-text">analyzed by AI</span>.
          <span className="relative ml-3 inline-block animate-floaty">🏓</span>
        </h1>
        <p className="muted mx-auto mt-6 max-w-2xl animate-fadeup text-lg">
          Upload a clip. Get an annotated replay with player &amp; ball tracking,
          court-aware analytics, and a personalized coaching report.
        </p>
        <div className="mt-9 flex animate-fadeup items-center justify-center gap-4">
          <Link href="/login" className="btn-primary">Analyze my match →</Link>
          <Link href="#pricing" className="btn-ghost">See pricing</Link>
        </div>

        {/* mock app frame */}
        <div className="card mx-auto mt-16 max-w-4xl overflow-hidden p-2 animate-fadeup">
          <div className="rounded-xl border border-white/5 bg-ink-800/60 p-6 text-left">
            <div className="mb-4 flex gap-1.5">
              <span className="h-3 w-3 rounded-full bg-red-400/70" />
              <span className="h-3 w-3 rounded-full bg-yellow-400/70" />
              <span className="h-3 w-3 rounded-full bg-green-400/70" />
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              {["Players: 4", "Ball: tracked", "Action: rally"].map((t) => (
                <div key={t} className="rounded-lg border border-white/10 bg-white/[0.03] p-4 text-sm">
                  <span className="gradient-text font-semibold">{t}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/5">
              <div className="h-full w-2/3 animate-shimmer rounded-full bg-grad-brand bg-[length:200%_100%]" />
            </div>
          </div>
        </div>
      </section>

      {/* features */}
      <section className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid gap-6 sm:grid-cols-3">
          {FEATURES.map(([icon, title, body]) => (
            <div key={title} className="card card-hover p-6">
              <div className="text-3xl">{icon}</div>
              <h3 className="mt-3 font-display font-semibold">{title}</h3>
              <p className="muted mt-2 text-sm">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* pricing */}
      <section id="pricing" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-center font-display text-3xl font-bold">Simple pricing</h2>
        <div className="mt-10 grid items-center gap-6 sm:grid-cols-3">
          {PLANS.map(([name, price, limit, featured]) => (
            <div key={name as string}
                 className={`card p-7 text-center ${featured ? "border-court/40 shadow-glow sm:scale-105" : "card-hover"}`}>
              {featured ? <div className="badge mx-auto mb-3 border-ball/30 text-ball">Most popular</div> : null}
              <h3 className="font-display font-semibold">{name}</h3>
              <p className="mt-2 font-display text-4xl font-bold">{price}</p>
              <p className="muted mt-1 text-sm">{limit}</p>
              <Link href="/login" className={`mt-5 w-full ${featured ? "btn-primary" : "btn-ghost"}`}>
                Get started
              </Link>
            </div>
          ))}
        </div>
      </section>

      <footer className="mx-auto max-w-6xl px-6 py-10 text-center text-sm text-slate-500">
        🏓 PickleballVision — AI game analysis.
      </footer>
    </main>
  );
}
