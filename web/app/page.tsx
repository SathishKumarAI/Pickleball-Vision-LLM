import Link from "next/link";

const MARQUEE = "SERVE · DINK · DRIVE · VOLLEY · LOB · ERNE · KITCHEN · RALLY · ";

const FEATURES = [
  ["01", "Annotated Replay", "Every player and the ball, tracked and labeled — rendered back onto your footage."],
  ["02", "Court Almanac", "Position heatmaps, kitchen time, rally tempo, shot-by-shot breakdown."],
  ["03", "Coach's Notes", "Plain-language feedback on positioning, shot choice, and strategy."],
];

export default function Landing() {
  return (
    <main className="relative">
      {/* top bar */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <span className="font-display text-2xl">PICKLEBALL<span className="text-flame">.</span>VISION</span>
        <Link href="/login" className="btn-ghost py-2">Sign in</Link>
      </header>

      {/* marquee strip */}
      <div className="border-y-2 border-ink bg-ink py-2 text-paper">
        <div className="flex whitespace-nowrap font-mono text-sm font-bold uppercase tracking-[0.3em]">
          <span className="animate-marquee">{MARQUEE.repeat(4)}</span>
          <span className="animate-marquee">{MARQUEE.repeat(4)}</span>
        </div>
      </div>

      {/* HERO — poster */}
      <section className="mx-auto max-w-6xl px-6 pb-10 pt-14">
        <div className="grid items-end gap-8 md:grid-cols-12">
          <div className="md:col-span-8">
            <p className="eyebrow animate-fadeup">Est. 2026 · Vision + Language Model</p>
            <h1 className="mt-4 animate-fadeup font-display leading-[0.85] [text-wrap:balance]"
                style={{ fontSize: "clamp(3rem, 11vw, 6rem)" }}>
              YOUR MATCH,
              <br />
              <span className="bg-lime px-2 text-ink shadow-hard">DECODED</span>
              <span className="ml-2 inline-block animate-floaty">🏓</span>
            </h1>
          </div>
          <div className="md:col-span-4">
            <p className="animate-fadeup font-serif text-xl italic text-ink-soft">
              Upload a clip. Get an annotated replay and a coaching almanac of your
              game — players, ball, court &amp; strategy — in minutes.
            </p>
            <div className="mt-6 flex animate-fadeup gap-3">
              <Link href="/login" className="btn-primary">Analyze a match →</Link>
            </div>
          </div>
        </div>

        {/* stat band */}
        <div className="mt-14 grid grid-cols-2 gap-px overflow-hidden rounded-xl border-2 border-ink bg-ink shadow-hard md:grid-cols-4">
          {[["≤2", "MIN / CLIP"], ["4", "PLAYERS TRACKED"], ["YOLO", "+ ByteTrack"], ["100%", "ON YOUR VIDEO"]].map(([n, l]) => (
            <div key={l} className="bg-paper p-6">
              <div className="font-display text-5xl">{n}</div>
              <div className="mt-1 font-mono text-[10px] font-bold uppercase tracking-widest text-ink/70">{l}</div>
            </div>
          ))}
        </div>
      </section>

      {/* FEATURES — editorial numbered */}
      <section className="mx-auto max-w-6xl px-6 py-12">
        <div className="grid gap-6 md:grid-cols-3">
          {FEATURES.map(([n, title, body], i) => (
            <div key={title} className={`card card-hover p-6 ${i === 1 ? "bg-lime" : ""}`}>
              <div className="font-mono text-sm font-bold text-flame">{n}</div>
              <h3 className="mt-2 font-display text-3xl leading-none">{title}</h3>
              <p className="mt-3 text-sm text-ink/70">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* PRICING */}
      <section id="pricing" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="font-display text-6xl">THE LINEUP</h2>
        <div className="mt-8 grid gap-6 md:grid-cols-3">
          {[
            ["FREE", "$0", "3 videos / mo", "bg-paper"],
            ["PRO", "$39", "500 videos / mo", "bg-flame text-paper"],
            ["STARTER", "$12", "50 videos / mo", "bg-paper"],
          ].map(([name, price, limit, bg], i) => (
            <div key={name} className={`card p-7 ${bg} ${i === 1 ? "shadow-hard-lg md:-translate-y-2" : "card-hover"}`}>
              <h3 className="font-display text-3xl">{name}</h3>
              <p className="mt-3 font-display text-6xl">{price}</p>
              <p className={`mt-1 font-mono text-xs uppercase tracking-widest ${i === 1 ? "text-paper/80" : "text-ink/70"}`}>{limit}</p>
              <Link href="/login" className={`mt-6 w-full ${i === 1 ? "btn-ghost" : "btn-primary"}`}>Pick this →</Link>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t-2 border-ink">
        <div className="mx-auto max-w-6xl px-6 py-8 font-mono text-xs uppercase tracking-widest text-ink/60">
          🏓 Pickleball.Vision — AI game almanac · © 2026
        </div>
      </footer>
    </main>
  );
}
