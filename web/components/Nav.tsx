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

  return (
    <nav className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-3">
      <Link href="/dashboard" className="font-bold">🏓 Pickleball Vision</Link>
      <div className="flex items-center gap-4 text-sm">
        {LINKS.map(([href, label]) => (
          <Link
            key={href}
            href={href}
            className={pathname === href ? "font-semibold text-brand" : "text-slate-600 hover:text-slate-900"}
          >
            {label}
          </Link>
        ))}
        {isAdmin && (
          <Link href="/admin" className={pathname === "/admin" ? "font-semibold text-brand" : "text-slate-600 hover:text-slate-900"}>
            Admin
          </Link>
        )}
        <button onClick={signOut} className="text-slate-500 hover:text-slate-900">Sign out</button>
      </div>
    </nav>
  );
}
