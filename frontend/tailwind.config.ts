import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        void: "#0b0e11",
        panel: "#141a22",
        panel2: "#1a222d",
        border: "#243041",
        muted: "#8b9bb4",
        long: "#10b981",
        short: "#f43f5e",
        fsot: "#22d3ee",
        warn: "#fbbf24",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
