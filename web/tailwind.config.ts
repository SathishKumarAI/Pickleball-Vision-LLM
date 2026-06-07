import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: "#16a34a", dark: "#15803d" }, // pickleball green
      },
    },
  },
  plugins: [],
};
export default config;
