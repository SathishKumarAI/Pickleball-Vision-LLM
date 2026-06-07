"use client";

import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export default function LoginPage() {
  const router = useRouter();
  const supabase = createClient();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    const fn =
      mode === "login"
        ? supabase.auth.signInWithPassword({ email, password })
        : supabase.auth.signUp({ email, password });
    const { error } = await fn;
    setBusy(false);
    if (error) return setErr(error.message);
    router.push("/dashboard");
  }

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-6">
      <div className="pointer-events-none absolute left-1/2 top-0 h-96 w-96 -translate-x-1/2 rounded-full bg-court/20 blur-[120px]" />
      <div className="card w-full max-w-md animate-fadeup p-8">
        <Link href="/" className="font-display text-lg font-bold">🏓 Pickleball<span className="gradient-text">Vision</span></Link>
        <h1 className="mt-6 font-display text-2xl font-bold">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h1>
        <p className="muted mt-1 text-sm">Analyze your next match in minutes.</p>
        <form onSubmit={submit} className="mt-6 space-y-3">
          <input type="email" required placeholder="you@email.com" value={email}
                 onChange={(e) => setEmail(e.target.value)} className="input" />
          <input type="password" required placeholder="password (8+ chars)" value={password}
                 onChange={(e) => setPassword(e.target.value)} className="input" />
          {err && <p className="text-sm text-red-400">{err}</p>}
          <button disabled={busy} className="btn-primary w-full">
            {busy ? "…" : mode === "login" ? "Sign in" : "Sign up"}
          </button>
        </form>
        <button onClick={() => setMode(mode === "login" ? "signup" : "login")} className="mt-5 text-sm text-court hover:text-ball">
          {mode === "login" ? "Need an account? Sign up" : "Have an account? Sign in"}
        </button>
      </div>
    </main>
  );
}
