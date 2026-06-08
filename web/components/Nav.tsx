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
    active ? "text-ink underline decoration-lime decoration-2 underline-offset-4"
           : "text-ink/60 hover:text-ink";

  return (
    <nav className="sticky top-0 z-30 border-b-2 border-ink bg-paper">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3.5">
        <Link href="/dashboard" className="font-display text-xl">
          PICKLEBALL<span className="text-flame">.</span>VISION
        </Link>
        <div className="flex items-center gap-5 font-mono text-xs font-bold uppercase tracking-wider">
          {LINKS.map(([href, label]) => (
            <Link key={href} href={href} className={`transition ${cls(pathname === href)}`}>{label}</Link>
          ))}
          {isAdmin && <Link href="/admin" className={`transition ${cls(pathname === "/admin")}`}>Admin</Link>}
          <button onClick={signOut} className="rounded-md border-2 border-ink px-3 py-1.5 transition hover:bg-flame hover:text-paper">
            Sign out
          </button>
        </div>
      </div>
    </nav>
  );
}
