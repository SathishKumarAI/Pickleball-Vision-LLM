import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import Nav from "@/components/Nav";

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const supabase = createClient();
  const { data } = await supabase.auth.getUser();
  if (!data.user) redirect("/login");

  return (
    <div className="min-h-screen">
      <Nav />
      <div className="mx-auto max-w-5xl px-6 py-8">{children}</div>
    </div>
  );
}
