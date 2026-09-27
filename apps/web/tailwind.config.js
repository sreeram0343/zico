/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        zico: {
          yellow: {
            DEFAULT: "#F7C948",
            50: "#FFFDF7",
            100: "#FFF6D8",
            200: "#FEED9F",
            300: "#FDE266",
            400: "#F7C948",
            500: "#F5BE22",
            600: "#D99E10",
            700: "#B37D08",
          },
          dark: "#101828",
          navy: "#0F172A",
          muted: "#667085",
          border: "#E5E7EB",
          card: "#FFFFFF",
          soft: "#FAFAF7",
        },
      },
      boxShadow: {
        soft: "0 2px 10px rgba(0, 0, 0, 0.04)",
        card: "0 4px 20px -2px rgba(0, 0, 0, 0.05)",
      },
    },
  },
  plugins: [],
};
