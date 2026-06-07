"use client";

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
    <main className="mx-auto flex min-h-screen max-w-md flex-col justify-center px-6">
      <h1 className="text-2xl font-bold">{mode === "login" ? "Sign in" : "Create account"}</h1>
      <form onSubmit={submit} className="mt-6 space-y-4">
        <input
          type="email" required placeholder="you@email.com" value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="w-full rounded-lg border border-slate-300 px-4 py-2"
        />
        <input
          type="password" required placeholder="password (8+ chars)" value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full rounded-lg border border-slate-300 px-4 py-2"
        />
        {err && <p className="text-sm text-red-600">{err}</p>}
        <button disabled={busy} className="w-full rounded-lg bg-brand py-2 font-semibold text-white hover:bg-brand-dark disabled:opacity-50">
          {busy ? "…" : mode === "login" ? "Sign in" : "Sign up"}
        </button>
      </form>
      <button
        onClick={() => setMode(mode === "login" ? "signup" : "login")}
        className="mt-4 text-sm text-brand"
      >
        {mode === "login" ? "Need an account? Sign up" : "Have an account? Sign in"}
      </button>
    </main>
  );
}
