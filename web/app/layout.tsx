import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pickleball Vision — AI Game Analysis",
  description: "Upload your match, get an annotated video + coaching insights in minutes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
