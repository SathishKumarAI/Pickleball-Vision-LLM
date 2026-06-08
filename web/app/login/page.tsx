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
    <main className="flex min-h-screen items-center justify-center px-6">
      <div className="w-full max-w-md animate-fadeup">
        <Link href="/" className="font-display text-2xl">PICKLEBALL<span className="text-flame">.</span>VISION</Link>
        <div className="card mt-6 p-8">
          <p className="eyebrow">{mode === "login" ? "Members" : "New player"}</p>
          <h1 className="mt-2 font-display text-5xl leading-none">
            {mode === "login" ? "WELCOME BACK" : "JOIN THE CLUB"}
          </h1>
          <form onSubmit={submit} className="mt-6 space-y-3">
            <input type="email" required placeholder="you@email.com" value={email}
                   onChange={(e) => setEmail(e.target.value)} className="input" />
            <input type="password" required placeholder="password (8+ chars)" value={password}
                   onChange={(e) => setPassword(e.target.value)} className="input" />
            {err && <p className="font-mono text-sm font-semibold text-flame">{err}</p>}
            <button disabled={busy} className="btn-primary w-full">
              {busy ? "…" : mode === "login" ? "Sign in" : "Sign up"}
            </button>
          </form>
        </div>
        <button onClick={() => setMode(mode === "login" ? "signup" : "login")} className="mt-5 ink-link text-sm">
          {mode === "login" ? "Need an account? Sign up →" : "Have an account? Sign in →"}
        </button>
      </div>
    </main>
  );
}
