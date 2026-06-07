import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#0a0f1e", 800: "#0e1424", 700: "#141b30" },
        ball: { DEFAULT: "#d4f000", dim: "#b6cf00" },   // neon pickleball-yellow
        court: { DEFAULT: "#15c6a3", dark: "#0e9c80" },  // teal
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-display)", "var(--font-inter)", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(212,240,0,.25), 0 8px 40px -8px rgba(212,240,0,.35)",
        card: "0 10px 40px -12px rgba(0,0,0,.6)",
      },
      backgroundImage: {
        "grad-brand": "linear-gradient(100deg,#15c6a3,#d4f000)",
        "grad-ink": "radial-gradient(1200px 600px at 70% -10%,rgba(21,198,163,.18),transparent),radial-gradient(900px 500px at 10% 10%,rgba(212,240,0,.10),transparent)",
      },
      keyframes: {
        floaty: { "0%,100%": { transform: "translateY(0) rotate(0)" }, "50%": { transform: "translateY(-14px) rotate(8deg)" } },
        fadeup: { "0%": { opacity: "0", transform: "translateY(12px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        shimmer: { "0%": { backgroundPosition: "200% 0" }, "100%": { backgroundPosition: "-200% 0" } },
      },
      animation: {
        floaty: "floaty 5s ease-in-out infinite",
        fadeup: "fadeup .5s ease-out both",
        shimmer: "shimmer 2.5s linear infinite",
      },
    },
  },
  plugins: [],
};
export default config;
