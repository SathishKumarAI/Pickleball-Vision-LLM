import type { Config } from "tailwindcss";

// "Athletic Editorial" — neo-brutalist vintage sports almanac.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: { DEFAULT: "#f3efe3", 2: "#e9e2cf" }, // warm cream
        ink: { DEFAULT: "#17150f", soft: "#3a362c" }, // near-black
        lime: { DEFAULT: "#c2f000", dim: "#a7d000" },  // electric court chartreuse
        flame: { DEFAULT: "#ea4b2a", dim: "#c93d20" }, // vintage athletic red-orange
      },
      fontFamily: {
        display: ["Anton", "Impact", "sans-serif"],          // poster headlines
        serif: ["Fraunces", "Georgia", "serif"],              // editorial accents
        sans: ["'Spline Sans'", "system-ui", "sans-serif"],   // body
        mono: ["'JetBrains Mono'", "monospace"],              // data / coords
      },
      boxShadow: {
        hard: "4px 4px 0 0 #17150f",
        "hard-lg": "8px 8px 0 0 #17150f",
        "hard-lime": "5px 5px 0 0 #c2f000",
        "hard-flame": "5px 5px 0 0 #ea4b2a",
      },
      keyframes: {
        floaty: { "0%,100%": { transform: "translateY(0) rotate(-6deg)" }, "50%": { transform: "translateY(-16px) rotate(6deg)" } },
        fadeup: { "0%": { opacity: "0", transform: "translateY(16px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        marquee: { "0%": { transform: "translateX(0)" }, "100%": { transform: "translateX(-50%)" } },
        bar: { "0%,100%": { transform: "scaleY(.4)" }, "50%": { transform: "scaleY(1)" } },
      },
      animation: {
        floaty: "floaty 6s ease-in-out infinite",
        fadeup: "fadeup .6s cubic-bezier(.2,.7,.2,1) both",
        marquee: "marquee 22s linear infinite",
        bar: "bar 1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
