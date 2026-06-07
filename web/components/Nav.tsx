"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { api } from "@/lib/api";

const LINKS = [
  ["/dashboard", "Dashboard"],
  ["/upload", "Upload"],
  ["/history", "History"],
  ["/billing", "Billing"],
  ["/settings", "Settings"],
];

export default function Nav() {
  const pathname = usePathname();
  const router = useRouter();
  const supabase = createClient();
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    api<{ is_admin?: boolean }>("/auth/me").then((u) => setIsAdmin(!!u.is_admin)).catch(() => {});
  }, []);

  async function signOut() {
    await supabase.auth.signOut();
    router.push("/login");
  }

  const cls = (active: boolean) =>
    active ? "text-ball" : "text-slate-400 hover:text-slate-100";

  return (
    <nav className="sticky top-0 z-30 border-b border-white/10 bg-ink/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3.5">
        <Link href="/dashboard" className="font-display font-bold">
          🏓 Pickleball<span className="gradient-text">Vision</span>
        </Link>
        <div className="flex items-center gap-5 text-sm">
          {LINKS.map(([href, label]) => (
            <Link key={href} href={href} className={`transition ${cls(pathname === href)}`}>{label}</Link>
          ))}
          {isAdmin && <Link href="/admin" className={`transition ${cls(pathname === "/admin")}`}>Admin</Link>}
          <button onClick={signOut} className="rounded-lg border border-white/10 px-3 py-1.5 text-slate-400 transition hover:border-white/30 hover:text-slate-100">
            Sign out
          </button>
        </div>
      </div>
    </nav>
  );
}
