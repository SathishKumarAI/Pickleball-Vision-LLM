import Link from "next/link";

export default function Landing() {
  return (
    <main className="mx-auto max-w-4xl px-6 py-20">
      <div className="text-center">
        <p className="text-5xl">🏓</p>
        <h1 className="mt-4 text-4xl font-extrabold tracking-tight">
          Your AI Pickleball Coach
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-slate-600">
          Upload a match video. Get back an annotated replay with player &amp; ball
          tracking, court analytics, and personalized coaching — in minutes.
        </p>
        <div className="mt-8 flex justify-center gap-4">
          <Link href="/login" className="rounded-lg bg-brand px-6 py-3 font-semibold text-white hover:bg-brand-dark">
            Get started
          </Link>
          <Link href="/#pricing" className="rounded-lg border border-slate-300 px-6 py-3 font-semibold">
            Pricing
          </Link>
        </div>
      </div>

      <section className="mt-20 grid gap-6 sm:grid-cols-3">
        {[
          ["🎥 Annotated replay", "Boxes, track IDs, and ball trail rendered onto your video."],
          ["📊 Court analytics", "Positioning heatmaps, kitchen usage, rally tempo, shot labels."],
          ["🧠 Coaching report", "Actionable feedback on strategy and movement."],
        ].map(([title, body]) => (
          <div key={title} className="rounded-xl border border-slate-200 bg-white p-6">
            <h3 className="font-semibold">{title}</h3>
            <p className="mt-2 text-sm text-slate-600">{body}</p>
          </div>
        ))}
      </section>

      <section id="pricing" className="mt-20 grid gap-6 sm:grid-cols-3">
        {[
          ["Free", "$0", "3 videos / month"],
          ["Starter", "$12", "50 videos / month"],
          ["Pro", "$39", "500 videos / month"],
        ].map(([name, price, limit]) => (
          <div key={name} className="rounded-xl border border-slate-200 bg-white p-6 text-center">
            <h3 className="font-semibold">{name}</h3>
            <p className="mt-2 text-3xl font-bold">{price}</p>
            <p className="mt-1 text-sm text-slate-600">{limit}</p>
          </div>
        ))}
      </section>
    </main>
  );
}
